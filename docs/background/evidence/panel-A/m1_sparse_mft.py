"""Architect A measurements.

(1) Sparse per-emitter pupil -> ROI rendering via batched matrix DFT (MFT), scalar and 6-component
    (vectorial dipole basis), forward and forward+backward w.r.t. emitter xyz.
(2) Scatter-add of ROIs into frames.
(3) TF32 influence on complex64 matmul accuracy (MFT) vs complex128 reference.
(4) Dense alternative: full-frame FFT pupil propagation per emitter (for comparison).
(5) Pupil-plane synthesis of a 1-D angular function (Mie-like S(theta)) onto the pupil by interpolation,
    batched over particles.
"""
import math
import time

import torch

dev = "cuda"
torch.manual_seed(0)


def bench(fn, n=20, warm=3):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / n * 1e3


def mft_mats(Np, Nout, dk, dx, x0=0.0, dtype=torch.complex64, device=dev):
    # pupil samples k_j = (j - Np/2) * dk ; output samples x_m = x0 + (m - Nout/2) * dx
    k = (torch.arange(Np, device=device, dtype=torch.float64) - Np / 2) * dk
    x = x0 + (torch.arange(Nout, device=device, dtype=torch.float64) - Nout / 2) * dx
    A = torch.exp(2j * math.pi * x[:, None] * k[None, :]) * dk  # [Nout, Np]
    return A.to(dtype)


# ---------------- (1)+(2) sparse emitters ----------------
lam, NA, n = 0.6, 1.4, 1.518
Np, R, H = 64, 32, 256  # pupil samples across [-NA/lam, NA/lam], ROI, frame
dk = 2 * NA / lam / Np
dx = 0.1  # um camera pixel in object space
kk = (torch.arange(Np, device=dev) - Np / 2) * dk
KY, KX = torch.meshgrid(kk, kk, indexing="ij")
rho2 = (KX**2 + KY**2) * (lam / NA) ** 2
aperture = torch.sigmoid((1 - rho2.sqrt()) * 50.0)
kz = torch.sqrt(torch.clamp((n / lam) ** 2 - KX**2 - KY**2, min=1e-9))
Ay = mft_mats(Np, R, dk, dx)
Ax = mft_mats(Np, R, dk, dx)

for B, E, C in [(64, 50, 1), (64, 50, 6), (256, 20, 1)]:
    xyz = torch.rand(B, E, 3, device=dev)
    xyz[..., :2] = xyz[..., :2] * 0.1 - 0.05  # sub-pixel offset within ROI centre (um)
    xyz[..., 2] = xyz[..., 2] * 2 - 1  # z in um
    anchors = torch.randint(0, H - R, (B, E, 2), device=dev)
    basis = torch.randn(C, Np, Np, device=dev, dtype=torch.complex64) * aperture  # stand-in for 6 dipole pupils

    def render(xyz):
        ph = 2 * math.pi * (
            KX * xyz[..., 0, None, None] + KY * xyz[..., 1, None, None] + kz * xyz[..., 2, None, None]
        )  # [B,E,Np,Np]
        P = basis[None, None] * torch.polar(torch.ones_like(ph), ph)[:, :, None]  # [B,E,C,Np,Np]
        roi = Ay @ P @ Ax.T  # [B,E,C,R,R]
        I = (roi.real**2 + roi.imag**2).sum(2)  # [B,E,R,R]
        # scatter-add into frames
        frame = torch.zeros(B, H * H, device=dev)
        iy = anchors[..., 0, None, None] + torch.arange(R, device=dev)[:, None]
        ix = anchors[..., 1, None, None] + torch.arange(R, device=dev)[None, :]
        idx = (iy * H + ix).reshape(B, -1)
        frame = frame.scatter_add(1, idx, I.reshape(B, -1))
        return frame.view(B, H, H)

    with torch.no_grad():
        tf = bench(lambda: render(xyz))
    xg = xyz.clone().requires_grad_(True)

    def fb():
        out = render(xg)
        out.square().mean().backward()

    torch.cuda.reset_peak_memory_stats()
    tb = bench(fb, n=10)
    mem = torch.cuda.max_memory_allocated() / 2**30
    print(
        f"sparse MFT B={B} E={E} C={C} pupil {Np}^2 ROI {R}^2: fwd {tf:.2f} ms, fwd+bwd {tb:.2f} ms, "
        f"peak {mem:.2f} GiB, imgs/s(fwd) {B/tf*1e3:.0f}"
    )

# ---------------- (3) TF32 influence on complex matmul ----------------
P = (torch.randn(256, Np, Np, device=dev, dtype=torch.complex128) * aperture.double())
A64 = mft_mats(Np, 128, dk, 0.05, dtype=torch.complex128)
ref = A64 @ P @ A64.T
for setting in ["ieee", "tf32"]:
    try:
        torch.backends.cuda.matmul.fp32_precision = setting
    except Exception:
        torch.backends.cuda.matmul.allow_tf32 = setting == "tf32"
    out = A64.to(torch.complex64) @ P.to(torch.complex64) @ A64.T.to(torch.complex64)
    rel = ((out.to(torch.complex128) - ref).norm() / ref.norm()).item()
    # also real matmul for comparison
    Ar = torch.randn(512, 512, device=dev, dtype=torch.float64)
    Br = torch.randn(512, 512, device=dev, dtype=torch.float64)
    rr = ((Ar.float() @ Br.float()).double() - Ar @ Br).norm() / (Ar @ Br).norm()
    print(f"matmul precision={setting}: complex64 MFT rel err {rel:.2e}; float32 real matmul rel err {rr.item():.2e}")
try:
    torch.backends.cuda.matmul.fp32_precision = "ieee"
except Exception:
    torch.backends.cuda.matmul.allow_tf32 = False

# ---------------- (4) dense alternative: per-emitter full-frame FFT ----------------
B, E = 8, 50
Pd = torch.randn(B, E, H, H, device=dev, dtype=torch.complex64)
with torch.no_grad():
    td = bench(lambda: (torch.fft.ifft2(Pd).abs() ** 2).sum(1))
print(f"dense per-emitter FFT {H}^2 B={B} E={E}: fwd {td:.2f} ms -> imgs/s {B/td*1e3:.0f}")

# ---------------- (5) 1-D angular function -> pupil synthesis, batched over particles ----------------
Nt, L = 1024, 60
for Bp in [64 * 10, 64 * 100]:
    an = torch.randn(Bp, L, device=dev, dtype=torch.complex64)
    theta = torch.linspace(0, math.asin(NA / n), Nt, device=dev)
    mu = torch.cos(theta)
    # pi_n, tau_n recurrences (Legendre) on 1-D theta grid
    def S_of_theta(an):
        pi_prev = torch.zeros_like(mu)
        pi_cur = torch.ones_like(mu)
        S = torch.zeros(an.shape[0], Nt, device=dev, dtype=torch.complex64)
        for l in range(1, L + 1):
            tau = l * mu * pi_cur - (l + 1) * pi_prev
            S = S + an[:, l - 1, None] * (pi_cur + tau)
            pi_next = ((2 * l + 1) * mu * pi_cur - (l + 1) * pi_prev) / l
            pi_prev, pi_cur = pi_cur, pi_next
        return S

    sin_t = (KX**2 + KY**2).sqrt() * lam / n
    tq = torch.asin(sin_t.clamp(max=0.999)) / theta[-1] * (Nt - 1)

    def synth(an):
        S = S_of_theta(an)  # [Bp, Nt]
        i0 = tq.floor().long().clamp(0, Nt - 2)
        w = (tq - i0).to(torch.float32)
        Sp = S[:, i0] * (1 - w) + S[:, i0 + 1] * w  # [Bp, Np, Np]
        return Sp * aperture

    with torch.no_grad():
        ts = bench(lambda: synth(an), n=10)
    print(f"1-D S(theta) (L={L}, Nt={Nt}) -> {Np}^2 pupil for {Bp} particles: {ts:.2f} ms")
