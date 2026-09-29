"""Quick throughput anchors for proposal C (v1 fast tiers). RTX 3090, torch 2.14.

Workloads:
  W1 sparse emitters, full-frame pupil FFT per emitter (fluorescence/SMLM)
  W2 sparse emitters, ROI via matrix DFT (MFT) + scatter-add
  W3 Mie-in-pupil holography (fp64 coeffs, 1-D S1/S2 on theta grid, interp to pupil, phase ramps)
  W4 dense fluorescence, per-z OTF convolution (rfft2)
  W5 multislice through procedural soft ellipsoids, fwd+bwd, with/without checkpoint
All timings: median of several reps after warmup, CUDA synchronized.
"""
import math
import time

import torch
import torch.utils.checkpoint as cp

dev = "cuda"
torch.backends.cuda.matmul.fp32_precision = "ieee"


def timeit(fn, reps=10, warm=3):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    ts.sort()
    return ts[len(ts) // 2]


def peak(fn):
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    fn()
    torch.cuda.synchronize()
    return (torch.cuda.max_memory_allocated() - base) / 2**30


# ---------------- common optics -----------------
wl, NA, n_m = 0.6e-6, 1.3, 1.33


def pupil_grid(N, dx):
    f = torch.fft.fftfreq(N, d=dx, device=dev)
    fy, fx = torch.meshgrid(f, f, indexing="ij")
    return fx, fy


# ---------------- W1: full-frame per emitter -----------------
def w1(B=64, E=32, N=128, dx=100e-9, grad=False):
    fx, fy = pupil_grid(N, dx)
    k = 2 * math.pi
    rho2 = (fx**2 + fy**2) * (wl / NA) ** 2
    P = torch.sigmoid((1 - rho2.sqrt()) * 50).to(torch.complex64)
    kz = k * torch.sqrt(torch.clamp((n_m / wl) ** 2 - fx**2 - fy**2, min=0))
    pos = (torch.rand(B, E, 3, device=dev) * torch.tensor([N * dx, N * dx, 2e-6], device=dev)
           - torch.tensor([0, 0, 1e-6], device=dev)).requires_grad_(grad)
    phot = torch.full((B, E), 1000.0, device=dev)

    def run():
        ph = -k * (fx * pos[..., 0, None, None] + fy * pos[..., 1, None, None]) + kz * pos[..., 2, None, None]
        U = torch.fft.ifft2(P * torch.polar(torch.ones_like(ph), ph))
        I = (U.real**2 + U.imag**2)
        I = I / I.sum((-2, -1), keepdim=True)
        img = (I * phot[..., None, None]).sum(1)
        if grad:
            img.sum().backward()
        return img
    return run


# ---------------- W2: ROI via MFT + scatter-add -----------------
def w2(B=64, E=32, Np=64, R=32, H=128, dx=100e-9, grad=False):
    # pupil sampled on Np x Np over [-NA/wl, NA/wl]; ROI R x R at camera-ish sampling dx
    k = 2 * math.pi
    fmax = NA / wl
    f = torch.linspace(-fmax, fmax, Np, device=dev)
    fy, fx = torch.meshgrid(f, f, indexing="ij")
    rho2 = (fx**2 + fy**2) / fmax**2
    P = torch.sigmoid((1 - rho2.sqrt()) * 50).to(torch.complex64)
    kz = k * torch.sqrt(torch.clamp((n_m / wl) ** 2 - fx**2 - fy**2, min=0))
    pos = torch.rand(B, E, 3, device=dev)
    pos = (pos * torch.tensor([H * dx, H * dx, 2e-6], device=dev) - torch.tensor([0, 0, 1e-6], device=dev)).requires_grad_(grad)
    phot = torch.full((B, E), 1000.0, device=dev)
    xr = (torch.arange(R, device=dev) - R // 2) * dx

    def run():
        anchor = torch.floor(pos[..., :2] / dx).detach()           # integer pixel anchor
        sub = pos[..., :2] - anchor * dx                             # subpixel offset (differentiable)
        # MFT matrices with emitter subpixel offset folded in: A[b,e,r,p] = exp(i 2pi (x_r - sub) f_p)
        ax = torch.polar(torch.ones(1, device=dev), k * (xr[None, None, :, None] - sub[..., 0, None, None]) * f)
        ay = torch.polar(torch.ones(1, device=dev), k * (xr[None, None, :, None] - sub[..., 1, None, None]) * f)
        defoc = torch.polar(torch.ones_like(fx), kz * pos[..., 2, None, None])
        Pe = P * defoc                                               # [B,E,Np,Np]
        U = ay @ Pe @ ax.transpose(-1, -2)                           # [B,E,R,R]
        I = U.real**2 + U.imag**2
        I = I / I.sum((-2, -1), keepdim=True) * phot[..., None, None]
        # scatter-add into padded frame
        img = torch.zeros(B, (H + R) * (H + R), device=dev)
        iy = (anchor[..., 1].long()[..., None, None] + torch.arange(R, device=dev)[:, None])
        ix = (anchor[..., 0].long()[..., None, None] + torch.arange(R, device=dev)[None, :])
        idx = (iy * (H + R) + ix).reshape(B, -1)
        img = img.scatter_add(1, idx, I.reshape(B, -1))
        if grad:
            img.sum().backward()
        return img
    return run


# ---------------- W3: Mie in pupil (holography) -----------------
def mie_coeffs(x, m, nmax):
    """x [P] real fp64, m [P] complex128. Returns a_n, b_n [P, nmax]. Downward D_n; upward psi/xi (timing only)."""
    mx = m * x
    nst = nmax + 16
    D = torch.zeros_like(mx)
    Ds = []
    for n in range(nst, 0, -1):
        D = n / mx - 1 / (D + n / mx)
        if n <= nmax + 1:
            Ds.append(D)
    Ds = torch.stack(Ds[::-1], -1)[:, :nmax]  # D_0..D_{nmax-1} -> use D_n at index n-1 approx
    psi0, psi1 = torch.sin(x), torch.sin(x) / x - torch.cos(x)
    chi0, chi1 = -torch.cos(x), -torch.cos(x) / x - torch.sin(x)
    a, b = [], []
    for n in range(1, nmax + 1):
        xi1 = torch.complex(psi1, -chi1)
        xi0 = torch.complex(psi0, -chi0)
        Dn = Ds[:, n - 1]
        an = ((Dn / m + n / x) * psi1 - psi0) / ((Dn / m + n / x) * xi1 - xi0)
        bn = ((Dn * m + n / x) * psi1 - psi0) / ((Dn * m + n / x) * xi1 - xi0)
        a.append(an); b.append(bn)
        psi0, psi1 = psi1, (2 * n + 1) / x * psi1 - psi0
        chi0, chi1 = chi1, (2 * n + 1) / x * chi1 - chi0
    return torch.stack(a, -1), torch.stack(b, -1)


def w3(B=64, Pn=5, N=256, dx=100e-9, Nth=512, grad=False):
    k = 2 * math.pi * n_m / wl
    P = B * Pn
    radius = (torch.rand(P, device=dev, dtype=torch.float64) * 0.4e-6 + 0.2e-6).requires_grad_(grad)
    nre = torch.full((P,), 1.58, device=dev, dtype=torch.float64)
    pos = torch.rand(B, Pn, 3, device=dev) * torch.tensor([N * dx, N * dx, 10e-6], device=dev)
    pos = pos.requires_grad_(grad)
    fx, fy = pupil_grid(N, dx)
    fr = torch.sqrt(fx**2 + fy**2)
    sin_t = (fr * wl / n_m).clamp(max=1)
    cos_t = torch.sqrt(1 - sin_t**2)
    phi = torch.atan2(fy, fx)
    inNA = torch.sigmoid((NA / wl - fr) * wl * 50).to(torch.complex64)
    theta_grid = torch.linspace(0, math.asin(NA / n_m), Nth, device=dev, dtype=torch.float64)
    tq = torch.asin(sin_t).clamp(max=theta_grid[-1].item()) / theta_grid[-1] * (Nth - 1)
    i0 = tq.floor().long().clamp(max=Nth - 2); w = (tq - i0).float()
    kz = 2 * math.pi * torch.sqrt(torch.clamp((n_m / wl) ** 2 - fx**2 - fy**2, min=0))

    def run():
        x = k * radius
        nmax = int(x.max().item() + 4 * x.max().item() ** (1 / 3) + 2)
        a, b = mie_coeffs(x, torch.complex(nre / n_m, torch.zeros_like(nre)), nmax)
        # pi/tau on theta grid
        mu = torch.cos(theta_grid)
        pi_prev = torch.zeros_like(mu); pi_cur = torch.ones_like(mu)
        S1 = torch.zeros(P, Nth, dtype=torch.complex128, device=dev); S2 = torch.zeros_like(S1)
        for n in range(1, nmax + 1):
            tau = n * mu * pi_cur - (n + 1) * pi_prev
            c = (2 * n + 1) / (n * (n + 1))
            S1 = S1 + c * (a[:, n - 1, None] * pi_cur + b[:, n - 1, None] * tau)
            S2 = S2 + c * (a[:, n - 1, None] * tau + b[:, n - 1, None] * pi_cur)
            pi_prev, pi_cur = pi_cur, ((2 * n + 1) * mu * pi_cur - (n + 1) * pi_prev) / n
        S1 = S1.to(torch.complex64); S2 = S2.to(torch.complex64)
        # interpolate onto pupil (x-polarized illumination, scalar projection)
        s1 = S1[:, i0] * (1 - w) + S1[:, i0 + 1] * w           # [P, N, N]
        s2 = S2[:, i0] * (1 - w) + S2[:, i0 + 1] * w
        Es = (s2 * torch.cos(phi) ** 2 + s1 * torch.sin(phi) ** 2).view(B, Pn, N, N)
        ph = -2 * math.pi * (fx * pos[..., 0, None, None] + fy * pos[..., 1, None, None]) + kz * pos[..., 2, None, None]
        Es = (Es * torch.polar(torch.ones_like(ph), ph) / cos_t.clamp(min=0.1)).sum(1) * inNA * 1e-3
        E = torch.fft.ifft2(Es)
        I = 1 + 2 * E.real + E.real**2 + E.imag**2       # explicit interference term
        if grad:
            I.sum().backward()
        return I
    return run


# ---------------- W4: dense fluorescence per-z OTF -----------------
def w4(B=16, Z=32, N=256, dx=100e-9, dz=200e-9, grad=False):
    fx, fy = pupil_grid(N, dx)
    rho = torch.sqrt(fx**2 + fy**2) * wl / NA
    P = torch.sigmoid((1 - rho) * 50).to(torch.complex64)
    kz = 2 * math.pi * torch.sqrt(torch.clamp((n_m / wl) ** 2 - fx**2 - fy**2, min=0))
    zs = (torch.arange(Z, device=dev) - Z / 2) * dz
    defocus = torch.randn((), device=dev).requires_grad_(grad)
    vol = torch.rand(B, Z, N, N, device=dev)

    def run():
        Pz = P * torch.polar(torch.ones_like(kz), kz * (zs[:, None, None] + defocus * 1e-6))
        psf = torch.fft.ifft2(Pz); psf = psf.real**2 + psf.imag**2
        otf = torch.fft.rfft2(psf / psf.sum((-2, -1), keepdim=True))       # [Z, N, N/2+1]
        img = torch.fft.irfft2((torch.fft.rfft2(vol) * otf).sum(1), s=(N, N))
        if grad:
            img.sum().backward()
        return img
    return run


# ---------------- W5: multislice through procedural soft ellipsoids -----------------
def w5(B=8, S=64, N=256, dx=100e-9, dz=150e-9, n_obj=6, ckpt=0):
    fx, fy = pupil_grid(N, dx)
    kz = 2 * math.pi * torch.sqrt(torch.clamp((n_m / wl) ** 2 - fx**2 - fy**2, min=0))
    H = torch.polar(torch.ones_like(kz), (kz - 2 * math.pi * n_m / wl) * dz)
    xs = (torch.arange(N, device=dev) + 0.5) * dx
    Y, X = torch.meshgrid(xs, xs, indexing="ij")
    c = (torch.rand(B, n_obj, 3, device=dev) * torch.tensor([N * dx, N * dx, S * dz], device=dev)).requires_grad_(True)
    r = (torch.rand(B, n_obj, 3, device=dev) * 2e-6 + 1.5e-6).requires_grad_(True)
    dn = torch.full((B, n_obj), 0.03, device=dev, requires_grad=True)
    k0 = 2 * math.pi / wl
    h = dx

    def slice_phase(s):
        z = (s + 0.5) * dz
        q = (((X - c[..., 0, None, None]) / r[..., 0, None, None]) ** 2
             + ((Y - c[..., 1, None, None]) / r[..., 1, None, None]) ** 2
             + ((z - c[..., 2, None, None]) / r[..., 2, None, None]) ** 2)
        d = (q.sqrt() - 1) * r.min(-1).values[..., None, None]   # first-order sdf
        occ = torch.clamp(0.5 - d / h, 0, 1)
        return k0 * dz * (occ * dn[..., None, None]).sum(1)

    def step(u, s):
        return torch.fft.ifft2(torch.fft.fft2(u * torch.polar(torch.ones_like(u.real), slice_phase(s))) * H)

    def seg(u, s0, s1):
        for s in range(s0, s1):
            u = step(u, s)
        return u

    def run():
        u = torch.ones(B, N, N, dtype=torch.complex64, device=dev)
        if ckpt:
            for s0 in range(0, S, ckpt):
                u = cp.checkpoint(seg, u, s0, min(S, s0 + ckpt), use_reentrant=False)
        else:
            u = seg(u, 0, S)
        I = u.real**2 + u.imag**2
        I.mean().backward()
        return I
    return run


def main():
    print(torch.__version__, torch.cuda.get_device_name())
    res = {}
    for name, mk, per in [
        ("W1 fullframe emitters B64xE32 128^2 fwd", lambda: w1(), 64),
        ("W1 fullframe emitters B64xE32 128^2 fwd+bwd(pos)", lambda: w1(grad=True), 64),
        ("W2 MFT-ROI emitters B64xE32 ROI32 pupil64 fwd", lambda: w2(), 64),
        ("W2 MFT-ROI emitters B64xE32 ROI32 pupil64 fwd+bwd(pos)", lambda: w2(grad=True), 64),
        ("W3 Mie-pupil holography B64x5 256^2 fwd", lambda: w3(), 64),
        ("W3 Mie-pupil holography B64x5 256^2 fwd+bwd(r,pos)", lambda: w3(grad=True), 64),
        ("W4 dense fluo B16 Z32 256^2 fwd", lambda: w4(), 16),
        ("W4 dense fluo B16 Z32 256^2 fwd+bwd(defocus)", lambda: w4(grad=True), 16),
        ("W5 multislice B8 S64 256^2 6 soft ellipsoids fwd+bwd naive", lambda: w5(ckpt=0), 8),
        ("W5 multislice B8 S64 256^2 6 soft ellipsoids fwd+bwd ckpt8", lambda: w5(ckpt=8), 8),
    ]:
        try:
            run = mk()
            t = timeit(run, reps=7, warm=2)
            m = peak(run)
            print(f"{name:62s} {t*1e3:8.2f} ms  {per/t:9.0f} img/s  peak {m:5.2f} GiB")
        except Exception as e:  # noqa: BLE001
            print(f"{name:62s} FAILED: {type(e).__name__}: {e}")
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
