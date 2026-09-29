"""Exp D: cost of the 'resolve' (scene sampling) phase vs the render phase.

(a) naive: one Python site object per parameter, each doing its own rsample -> ~S*few kernel launches
(b) plate-fused: sites grouped by (distribution family, plate) into one vectorized draw
(c) DT2-style: per-sample Python loop with numpy scalar sampling, then stack + H2D copy
(d) (a) captured in a CUDA graph
"""
import math
import time

import numpy as np
import torch

dev = "cuda"
B, N, S = 64, 50, 40  # batch, object slots, number of random sites (~realistic scene)


def bench(fn, n=50, warm=10):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n * 1e3


g = torch.Generator(device=dev).manual_seed(0)
locs = [torch.nn.Parameter(torch.randn((), device=dev)) for _ in range(S)]
scales = [torch.nn.Parameter(torch.rand((), device=dev) + 0.1) for _ in range(S)]
# half the sites are per-object [B,N], half per-image [B]
shapes = [(B, N) if i % 2 == 0 else (B,) for i in range(S)]


def naive():
    out = {}
    for i in range(S):
        eps = torch.randn(shapes[i], device=dev, generator=g)
        v = locs[i] + torch.nn.functional.softplus(scales[i]) * eps
        if i % 4 == 0:
            v = torch.exp(v)  # LogNormal-like
        out[i] = v
    return out


L = torch.stack(locs)
Sc = torch.stack(scales)
obj_idx = torch.tensor([i for i in range(S) if i % 2 == 0], device=dev)
img_idx = torch.tensor([i for i in range(S) if i % 2 == 1], device=dev)


def fused():
    Lv, Sv = torch.stack(locs), torch.nn.functional.softplus(torch.stack(scales))
    eo = torch.randn(len(obj_idx), B, N, device=dev, generator=g)
    ei = torch.randn(len(img_idx), B, device=dev, generator=g)
    vo = Lv[obj_idx, None, None] + Sv[obj_idx, None, None] * eo
    vi = Lv[img_idx, None] + Sv[img_idx, None] * ei
    return vo, vi


rng = np.random.default_rng(0)
lv = [float(x) for x in L.detach().cpu()]
sv = [float(x) for x in Sc.detach().cpu()]


def dt2_style():
    samples = []
    for b in range(B):
        d = {}
        for i in range(S):
            if shapes[i] == (B, N):
                d[i] = [lv[i] + sv[i] * rng.standard_normal() for _ in range(N)]
            else:
                d[i] = lv[i] + sv[i] * rng.standard_normal()
        samples.append(d)
    arr = np.array([[s[i] if isinstance(s[i], float) else np.mean(s[i]) for i in range(S)] for s in samples])
    return torch.as_tensor(arr, device=dev)


print(f"B={B} N={N} sites={S}")
print(f"naive per-site eager:        {bench(naive):7.3f} ms")
with torch.no_grad():
    print(f"naive per-site eager no_grad:{bench(naive):7.3f} ms")
print(f"plate-fused eager:           {bench(fused):7.3f} ms")
print(f"DT2-style per-sample python: {bench(dt2_style, n=5, warm=1):7.3f} ms")

# CUDA graph capture of naive resolve (forward only; graph-safe generator)
gg = torch.Generator(device=dev).manual_seed(0)
torch.cuda.synchronize()
static = {}


def naive_g():
    out = {}
    for i in range(S):
        eps = torch.randn(shapes[i], device=dev, generator=gg)
        v = locs[i].detach() + torch.nn.functional.softplus(scales[i].detach()) * eps
        if i % 4 == 0:
            v = torch.exp(v)
        out[i] = v
    return out


s = torch.cuda.Stream()
s.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s):
    for _ in range(3):
        naive_g()
torch.cuda.current_stream().wait_stream(s)
graph = torch.cuda.CUDAGraph()
graph.register_generator_state(gg)
with torch.cuda.graph(graph):
    static = naive_g()
print(f"naive resolve as CUDA graph: {bench(graph.replay):7.3f} ms")
