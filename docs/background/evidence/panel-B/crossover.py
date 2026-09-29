# Sparse (per-emitter MFT ROI) vs dense (splat to z-strata + per-plane OTF) fluorescence rendering.
import torch, time, math
dev = "cuda"
torch.backends.cuda.matmul.fp32_precision = "ieee"
def timeit(fn, n=10):
    fn(); torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t = time.perf_counter()
    for _ in range(n): fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / n * 1e3, torch.cuda.max_memory_allocated() / 2**20

B, H, W = 32, 256, 256
dx = 0.1  # um, object space
lam, NA, n = 0.6, 1.2, 1.33
Np, R = 64, 32  # pupil samples, ROI size

# pupil grid in spatial frequency (cycles/um), covering |k|<=NA/lam
kmax = NA / lam
k1 = torch.linspace(-kmax, kmax, Np, device=dev, dtype=torch.float64)
KY, KX = torch.meshgrid(k1, k1, indexing="ij")
rho2 = (KX**2 + KY**2)
P0 = (rho2 <= kmax**2).to(torch.complex64)
kz = torch.sqrt(torch.clamp((n / lam) ** 2 - rho2, min=0)).float()
xr = (torch.arange(R, device=dev) - R // 2).double() * dx
A = torch.exp(2j * math.pi * xr[:, None] * k1[None, :]).to(torch.complex64)  # [R, Np]

def sparse(N, pos, z, phot, grad=False):
    # pos [B,N,2] um, z [B,N] um
    anchor = torch.round(pos / dx)
    frac = pos - anchor * dx  # subpixel offset
    ph = 2 * math.pi * (kz[None, None] * z[..., None, None]
                        - KX.float()[None, None] * frac[..., 1, None, None]
                        - KY.float()[None, None] * frac[..., 0, None, None])
    Pj = P0 * torch.polar(torch.ones_like(ph), ph)                # [B,N,Np,Np]
    E = A @ Pj @ A.T                                              # [B,N,R,R]
    I = (E.real**2 + E.imag**2) * phot[..., None, None]
    # scatter-add into frame
    iy = anchor[..., 0].long()[..., None, None] + torch.arange(R, device=dev)[:, None] - R // 2
    ix = anchor[..., 1].long()[..., None, None] + torch.arange(R, device=dev)[None, :] - R // 2
    valid = (iy >= 0) & (iy < H) & (ix >= 0) & (ix < W)
    flat = (torch.arange(B, device=dev)[:, None, None, None] * H * W + iy.clamp(0, H - 1) * W + ix.clamp(0, W - 1))
    out = torch.zeros(B * H * W, device=dev)
    out = out.index_add(0, flat.reshape(-1), (I * valid).reshape(-1))
    return out.view(B, H, W)

# dense: Z strata OTFs precomputed on full grid
def make_otf(Z, zr):
    fy = torch.fft.fftfreq(H, d=dx, device=dev).double(); fx = torch.fft.fftfreq(W, d=dx, device=dev).double()
    FY, FX = torch.meshgrid(fy, fx, indexing="ij")
    r2 = FX**2 + FY**2
    P = (r2 <= kmax**2).to(torch.complex128)
    kzf = torch.sqrt(torch.clamp((n / lam) ** 2 - r2, min=0))
    zs = torch.linspace(-zr, zr, Z, device=dev, dtype=torch.float64)
    Pz = P[None] * torch.exp(2j * math.pi * kzf[None] * zs[:, None, None])
    psf = torch.fft.ifft2(Pz).abs() ** 2
    psf = psf / psf[Z // 2].sum()
    return torch.fft.rfft2(psf).to(torch.complex64), zs.float()

def dense(N, pos, z, phot, otf, zs):
    Z = otf.shape[0]
    dz = (zs[1] - zs[0])
    zi = ((z - zs[0]) / dz).clamp(0, Z - 1.001)
    z0 = zi.floor(); wz = zi - z0
    yi = pos[..., 0] / dx; xi = pos[..., 1] / dx
    y0 = yi.floor(); x0 = xi.floor(); wy = yi - y0; wx = xi - x0
    vol = torch.zeros(B * Z * H * W, device=dev)
    bidx = torch.arange(B, device=dev)[:, None].expand(B, N)
    for dzi, fz in ((0, 1 - wz), (1, wz)):
        for dyi, fy_ in ((0, 1 - wy), (1, wy)):
            for dxi, fx_ in ((0, 1 - wx), (1, wx)):
                idx = ((bidx * Z + (z0.long() + dzi)) * H + (y0.long() + dyi).clamp(0, H - 1)) * W + (x0.long() + dxi).clamp(0, W - 1)
                vol = vol.index_add(0, idx.reshape(-1), (phot * fz * fy_ * fx_).reshape(-1))
    vol = vol.view(B, Z, H, W)
    S = (torch.fft.rfft2(vol) * otf[None]).sum(1)
    return torch.fft.irfft2(S, s=(H, W))

otf, zs = make_otf(32, 1.5)
print(f"B={B}, frame {H}x{W}, ROI {R}^2, pupil {Np}^2, dense Z=32 strata")
print("   N | sparse fwd ms  MB | sparse f+b ms  MB | dense fwd ms  MB | dense f+b ms MB")
for N in [1, 4, 16, 64, 256, 1024]:
    g = torch.Generator(device=dev).manual_seed(0)
    pos = (torch.rand(B, N, 2, device=dev, generator=g) * 0.8 + 0.1) * H * dx
    z = (torch.rand(B, N, device=dev, generator=g) - 0.5) * 2.8
    phot = torch.full((B, N), 1000.0, device=dev)
    with torch.no_grad():
        ts, ms = timeit(lambda: sparse(N, pos, z, phot))
        td, md = timeit(lambda: dense(N, pos, z, phot, otf, zs))
    posg = pos.clone().requires_grad_(True); zg = z.clone().requires_grad_(True)
    def fb_s():
        o = sparse(N, posg, zg, phot); o.square().mean().backward()
    def fb_d():
        o = dense(N, posg, zg, phot, otf, zs); o.square().mean().backward()
    try:
        tsb, msb = timeit(fb_s, n=5)
    except torch.OutOfMemoryError:
        tsb, msb = float('nan'), float('nan'); torch.cuda.empty_cache()
    tdb, mdb = timeit(fb_d, n=5)
    print(f"{N:5d} | {ts:9.2f} {ms:6.0f} | {tsb:9.2f} {msb:6.0f} | {td:9.2f} {md:6.0f} | {tdb:9.2f} {mdb:6.0f}")
