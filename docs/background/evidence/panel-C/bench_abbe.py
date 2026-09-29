"""Abbe partial coherence on a thin (projection) sample: cost vs number of source points.
Phase-contrast-like: annular source + phase ring pupil. fwd and fwd+bwd wrt thin phase map."""
import math

import torch
import torch.utils.checkpoint as cp

from bench_v1_tiers import peak, timeit

dev = "cuda"
wl, NA, NAc, n_m = 0.55e-6, 0.4, 0.3, 1.0


def abbe(B=16, N=256, dx=160e-9, K=25, chunk=0, grad=True):
    f = torch.fft.fftfreq(N, d=dx, device=dev)
    fy, fx = torch.meshgrid(f, f, indexing="ij")
    fr = torch.sqrt(fx**2 + fy**2)
    P = torch.sigmoid((NA / wl - fr) * wl * 60)
    ring = torch.sigmoid((fr * wl - 0.27) * 300) * torch.sigmoid((0.33 - fr * wl) * 300)
    P = P.to(torch.complex64) * torch.polar(1 - 0.75 * ring, math.pi / 2 * ring)
    # annular source on a ring at NA_c, snapped to the grid
    ang = torch.arange(K, device=dev) * 2 * math.pi / K
    sx = torch.round(NAc / wl * torch.cos(ang) * N * dx) / (N * dx)
    sy = torch.round(NAc / wl * torch.sin(ang) * N * dx) / (N * dx)
    xs = torch.arange(N, device=dev) * dx
    Y, X = torch.meshgrid(xs, xs, indexing="ij")
    tilt = torch.polar(torch.ones(K, N, N, device=dev), 2 * math.pi * (sx[:, None, None] * X + sy[:, None, None] * Y))
    phase = (torch.rand(B, N, N, device=dev) * 0.5).requires_grad_(grad)

    def chunk_img(t, tl):
        U = torch.fft.ifft2(torch.fft.fft2(t[:, None] * tl[None]) * P)
        return (U.real**2 + U.imag**2).sum(1)

    def run():
        t = torch.polar(torch.ones_like(phase), phase)
        if chunk:
            I = sum(cp.checkpoint(chunk_img, t, tilt[i:i + chunk], use_reentrant=False)
                    for i in range(0, K, chunk))
        else:
            I = chunk_img(t, tilt)
        I = I / K
        if grad:
            I.sum().backward()
        return I
    return run


for K in (1, 9, 25, 81):
    for chunk in (0, 8):
        if chunk and chunk >= K:
            continue
        r = abbe(K=K, chunk=chunk)
        t = timeit(r, reps=5)
        m = peak(r)
        print(f"Abbe thin B16 256^2 K={K:3d} chunk={chunk}: fwd+bwd {t*1e3:7.2f} ms ({16/t:6.0f} img/s) peak {m:5.2f} GiB")
