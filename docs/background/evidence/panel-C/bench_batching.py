"""Batched vs per-sample loop (DeepTrack-style) and Mie cost split."""
import math

import torch

from bench_v1_tiers import mie_coeffs, timeit, w1, w3, w4

dev = "cuda"


def loop(mk, n):
    runs = [mk() for _ in range(1)]
    r = runs[0]

    def run():
        for _ in range(n):
            r()
    return run


print("per-sample loop vs batched (fwd only)")
for name, mkb, mk1, n in [
    ("W1 emitters 32/img 128^2", lambda: w1(B=64), lambda: w1(B=1), 64),
    ("W3 Mie holo 5/img 256^2", lambda: w3(B=64), lambda: w3(B=1), 64),
    ("W4 dense fluo Z32 256^2", lambda: w4(B=16), lambda: w4(B=1), 16),
]:
    tb = timeit(mkb(), reps=5)
    tl = timeit(loop(mk1, n), reps=3, warm=1)
    print(f"{name:28s} batched {tb*1e3:7.2f} ms   loop x{n} {tl*1e3:8.2f} ms   speedup {tl/tb:5.1f}x")

# Mie coefficient cost alone (launch-bound)
for P in (5, 320, 4096):
    x = torch.rand(P, device=dev, dtype=torch.float64) * 8 + 2
    m = torch.full((P,), 1.19 + 0j, device=dev, dtype=torch.complex128)
    nmax = int(10 + 4 * 10 ** (1 / 3) + 2)
    t = timeit(lambda: mie_coeffs(x, m, nmax), reps=5)
    xc, mc = x.cpu(), m.cpu()
    tc = timeit(lambda: mie_coeffs(xc, mc, nmax), reps=5)
    print(f"Mie coeffs P={P:5d} nmax={nmax}: GPU {t*1e3:6.2f} ms  CPU {tc*1e3:6.2f} ms")

# CUDA graph capture of Mie coefficient recurrence
P = 320
x = torch.rand(P, device=dev, dtype=torch.float64) * 8 + 2
m = torch.full((P,), 1.19 + 0j, device=dev, dtype=torch.complex128)
nmax = 25
s = torch.cuda.Stream()
s.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s):
    for _ in range(3):
        mie_coeffs(x, m, nmax)
torch.cuda.current_stream().wait_stream(s)
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
    out = mie_coeffs(x, m, nmax)
t = timeit(lambda: g.replay(), reps=10)
print(f"Mie coeffs P=320 nmax=25 CUDA-graph replay: {t*1e3:6.3f} ms")
