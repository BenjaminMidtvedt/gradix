"""Exp A: throughput of batched sparse (per-emitter ROI) vs dense (depth-strata OTF) fluorescence rendering.

Scalar pupil with soft aperture, defocus (exact kz), 6 Zernike-like modes + pixel phase map.
Sparse: per-emitter pupil -> ifft2 on RxR ROI (subpixel via pupil phase ramp) -> |.|^2 -> scatter_add.
Dense: trilinear splat into Z strata -> per-z OTF (rfft2) -> sum over z.
Also: full-frame Fourier phase-ramp (B x N x H x W) for reference.
"""
import math
import time

import torch

dev = "cuda"
torch.backends.cuda.matmul.allow_tf32 = False


def bench(fn, n=10, warm=3):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / n * 1e3
    peak = (torch.cuda.max_memory_allocated() - base) / 2**30
    return dt, peak


lam, NA, n_imm, dx = 0.6, 1.4, 1.518, 0.1  # um


def pupil_grid(R):
    f = torch.fft.fftfreq(R, d=dx, device=dev)
    FY, FX = torch.meshgrid(f, f, indexing="ij")
    FR = torch.sqrt(FX**2 + FY**2)
    kmax = NA / lam
    ap = torch.sigmoid((kmax - FR) / (0.5 / (R * dx)))
    kz = torch.sqrt(torch.clamp((n_imm / lam) ** 2 - FR**2, min=1e-12)) - n_imm / lam
    rho = (FR / kmax).clamp(max=1.0)
    ph = torch.atan2(FY, FX)
    Z = torch.stack(
        [
            2 * rho**2 - 1,
            rho**2 * torch.cos(2 * ph),
            rho**2 * torch.sin(2 * ph),
            (3 * rho**3 - 2 * rho) * torch.cos(ph),
            (3 * rho**3 - 2 * rho) * torch.sin(ph),
            6 * rho**4 - 6 * rho**2 + 1,
        ]
    )
    return FX, FY, ap, kz, Z


def render_sparse(pos, z, phot, zc, pix, H, W, R=32):
    FX, FY, ap, kz, Z = G[R]
    B, N, _ = pos.shape
    p = pos / dx
    anchor = torch.floor(p).detach()
    shift = (p - anchor) * dx
    anchor = anchor.long()
    P = ap * torch.exp(1j * (torch.einsum("j,jyx->yx", zc, Z) + pix))
    phase = 2 * math.pi * (
        kz * z[..., None, None] - FX * shift[..., 0, None, None] - FY * shift[..., 1, None, None]
    )
    fld = torch.fft.fftshift(torch.fft.ifft2(P * torch.exp(1j * phase)), dim=(-2, -1))
    psf = fld.real**2 + fld.imag**2
    norm = (ap**2).sum() / (R * R)  # Parseval: in-focus total
    roi = psf * (phot / norm)[..., None, None]
    o = torch.arange(R, device=dev) - R // 2
    iy = anchor[..., 1, None, None] + o[:, None]
    ix = anchor[..., 0, None, None] + o[None, :]
    valid = (iy >= 0) & (iy < H) & (ix >= 0) & (ix < W)
    flat = iy.clamp(0, H - 1) * W + ix.clamp(0, W - 1)
    frame = torch.zeros(B, H * W, device=dev)
    frame = frame.scatter_add(1, flat.reshape(B, -1), (roi * valid).reshape(B, -1))
    return frame.view(B, H, W)


def otf_stack(zc, pix, H, W, zs):
    # full-frame pupils for each z stratum (computed once per batch; shared optics)
    FX, FY, ap, kz, Z = Gfull[(H, W)]
    P = ap * torch.exp(1j * (torch.einsum("j,jyx->yx", zc, Z) + pix))
    fld = torch.fft.ifft2(P * torch.exp(1j * 2 * math.pi * kz * zs[:, None, None]))
    psf = fld.real**2 + fld.imag**2
    psf = psf / ((ap**2).sum() / (H * W))
    return torch.fft.rfft2(psf)  # [Z, H, W//2+1]


def render_dense(pos, z, phot, zc, pix, H, W, zs):
    B, N, _ = pos.shape
    Zn = zs.numel()
    dz = (zs[1] - zs[0]).item()
    p = pos / dx
    fz = (z - zs[0]) / dz
    i0 = torch.floor(p).detach().long()
    iz0 = torch.floor(fz).detach().long()
    w = p - i0
    wz = fz - iz0
    vol = torch.zeros(B, Zn * H * W, device=dev)
    for a in (0, 1):
        for b in (0, 1):
            for c in (0, 1):
                wt = (
                    (w[..., 0] if a else 1 - w[..., 0])
                    * (w[..., 1] if b else 1 - w[..., 1])
                    * (wz if c else 1 - wz)
                )
                ix = (i0[..., 0] + a).clamp(0, W - 1)
                iy = (i0[..., 1] + b).clamp(0, H - 1)
                iz = (iz0 + c).clamp(0, Zn - 1)
                vol = vol.scatter_add(1, (iz * H * W + iy * W + ix), wt * phot)
    vol = vol.view(B, Zn, H, W)
    otf = otf_stack(zc, pix, H, W, zs)
    img = torch.fft.irfft2(torch.fft.rfft2(vol) * otf, s=(H, W)).sum(1)
    return img


def render_fullframe(pos, z, phot, zc, pix, H, W):
    FX, FY, ap, kz, Z = Gfull[(H, W)]
    P = ap * torch.exp(1j * (torch.einsum("j,jyx->yx", zc, Z) + pix))
    phase = 2 * math.pi * (
        kz * z[..., None, None] - FX * pos[..., 0, None, None] - FY * pos[..., 1, None, None]
    )
    fld = torch.fft.ifft2(P * torch.exp(1j * phase))
    psf = fld.real**2 + fld.imag**2
    return (psf * (phot / ((ap**2).sum() / (H * W)))[..., None, None]).sum(1)


G = {32: pupil_grid(32)}
Gfull = {}
for H in (64, 128, 256):
    FX, FY, ap, kz, Z = pupil_grid(H)
    Gfull[(H, H)] = (FX, FY, ap, kz, Z)

zs = torch.linspace(-1.0, 1.0, 33, device=dev)

configs = [(256, 20, 64), (64, 50, 128), (64, 200, 256)]
print(f"{'cfg':>22} {'method':>10} {'fwd ms':>8} {'fwd GB':>7} {'f+b ms':>8} {'f+b GB':>7} {'img/s(fwd)':>11}")
for B, N, H in configs:
    W = H
    g = torch.Generator(device=dev).manual_seed(0)
    pos0 = (torch.rand(B, N, 2, device=dev, generator=g) * (H - 8) + 4) * dx
    z0 = (torch.rand(B, N, device=dev, generator=g) * 1.6 - 0.8)
    ph0 = 1000 + 4000 * torch.rand(B, N, device=dev, generator=g)
    zc0 = 0.1 * torch.randn(6, device=dev, generator=g)
    pix0 = torch.zeros(32, 32, device=dev)
    pixF = torch.zeros(H, W, device=dev)

    methods = {
        "sparse": lambda pos, z, ph, zc: render_sparse(pos, z, ph, zc, pix0, H, W),
        "dense33": lambda pos, z, ph, zc: render_dense(pos, z, ph, zc, pixF, H, W, zs),
    }
    if B * N * H * W <= 64 * 50 * 128 * 128:
        methods["fullframe"] = lambda pos, z, ph, zc: render_fullframe(pos, z, ph, zc, pixF, H, W)
    for name, fn in methods.items():
        def fwd():
            with torch.inference_mode():
                return fn(pos0, z0, ph0, zc0)

        pos = pos0.clone().requires_grad_()
        z = z0.clone().requires_grad_()
        ph = ph0.clone().requires_grad_()
        zc = zc0.clone().requires_grad_()

        def fb():
            out = fn(pos, z, ph, zc)
            out.square().mean().backward()

        try:
            tf, mf = bench(fwd)
            tb, mb = bench(fb)
            print(f"B={B:3d} N={N:3d} {H}^2 {name:>10} {tf:8.2f} {mf:7.3f} {tb:8.2f} {mb:7.3f} {B / tf * 1e3:11.0f}")
        except torch.OutOfMemoryError:
            print(f"B={B:3d} N={N:3d} {H}^2 {name:>10} OOM")
            torch.cuda.empty_cache()

# consistency check sparse vs fullframe (same optics, small case)
B, N, H = 2, 3, 64
g = torch.Generator(device=dev).manual_seed(1)
pos = (torch.rand(B, N, 2, device=dev, generator=g) * 32 + 16) * dx
z = torch.rand(B, N, device=dev, generator=g) * 1.0 - 0.5
ph = torch.full((B, N), 1000.0, device=dev)
zc = 0.1 * torch.randn(6, device=dev, generator=g)
a = render_sparse(pos, z, ph, zc, torch.zeros(32, 32, device=dev), H, H, R=32)
G[64] = pupil_grid(64)
a64 = render_sparse(pos, z, ph, zc, torch.zeros(64, 64, device=dev), H, H, R=64)
b = render_fullframe(pos, z, ph, zc, torch.zeros(H, H, device=dev), H, H)
print("sparse R=32 vs fullframe rel L2:", ((a - b).norm() / b.norm()).item())
print("sparse R=64 vs fullframe rel L2:", ((a64 - b).norm() / b.norm()).item())
print("energy captured R=32:", (a.sum() / b.sum()).item())
