"""Revision 6: throughput of a diffractive optical network (architecture.md §5.11, §8.3).

A phase-only diffractive stack: L layers, each a learnable phase mask followed by band-limited
angular-spectrum propagation (FFT, transfer function, inverse FFT), on complex64 fields [B, N, N].
Reports forward and forward+backward time (median of repeats) and peak memory, with and without
checkpointing each layer.

Run from the repo root: uv run python docs/background/evidence/review/rev6/d2nn_throughput.py
"""

import math
import time

import torch
from torch.utils.checkpoint import checkpoint

dev = "cuda"
torch.backends.cuda.matmul.allow_tf32 = False


def transfer(n, dx, wavelength, distance):
    """Band-limited angular-spectrum transfer function (evanescent band removed)."""
    f = torch.fft.fftfreq(n, d=dx, device=dev)
    fx, fy = torch.meshgrid(f, f, indexing="ij")
    arg = (1.0 / wavelength) ** 2 - fx**2 - fy**2
    kz = 2 * math.pi * torch.sqrt(arg.clamp(min=0.0))
    # Matsushima-Shimobaba band limit for the sampled transfer function
    f_lim = 1.0 / (wavelength * math.sqrt((2 * distance / (n * dx)) ** 2 + 1))
    band = (fx.abs() < f_lim) & (fy.abs() < f_lim) & (arg > 0)
    return torch.polar(band.float(), kz * distance).to(torch.complex64)


def layer(u, phase, h):
    u = u * torch.polar(torch.ones_like(phase), phase)
    return torch.fft.ifft2(torch.fft.fft2(u) * h)


def run(batch, n, layers, ckpt, reps=20):
    h = transfer(n, dx=0.4, wavelength=0.532, distance=40.0)  # µm: 0.4 µm pixels, 40 µm gaps
    phases = [torch.zeros(n, n, device=dev, requires_grad=True) for _ in range(layers)]
    u0 = torch.randn(batch, n, n, device=dev, dtype=torch.complex64)
    target = torch.rand(batch, n, n, device=dev)

    def forward():
        u = u0
        for p in phases:
            u = checkpoint(layer, u, p, h, use_reentrant=False) if ckpt else layer(u, p, h)
        return u

    def timed(fn):
        for _ in range(3):
            fn()
        torch.cuda.synchronize()
        ts = []
        for _ in range(reps):
            t0 = time.perf_counter()
            fn()
            torch.cuda.synchronize()
            ts.append(time.perf_counter() - t0)
        return sorted(ts)[len(ts) // 2]

    with torch.no_grad():
        t_fwd = timed(forward)

    def step():
        loss = ((forward().abs() ** 2 - target) ** 2).mean()
        loss.backward()

    torch.cuda.reset_peak_memory_stats()
    t_bwd = timed(step)
    mem = torch.cuda.max_memory_allocated() / 2**30
    return t_fwd, t_bwd, mem


print(f"torch {torch.__version__}, {torch.cuda.get_device_name()}")
for batch, n, layers in [(64, 512, 5), (16, 512, 5), (64, 256, 5)]:
    for ckpt in (False, True):
        t_fwd, t_bwd, mem = run(batch, n, layers, ckpt)
        print(f"B={batch:3d} N={n} L={layers} checkpoint={ckpt!s:5s}: fwd {t_fwd * 1e3:6.1f} ms "
              f"({batch / t_fwd:7.0f} img/s), fwd+bwd {t_bwd * 1e3:6.1f} ms "
              f"({batch / t_bwd:6.0f} img/s), "
              f"peak {mem:.2f} GiB")
