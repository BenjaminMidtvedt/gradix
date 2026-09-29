"""Cheaper counter hash alternative: 'lowbias32'-style avalanche (2 multiply-xorshift rounds) over a combined 32-bit key,
applied twice (key mixing + element). Same workload as counter_rng.py (B=64, 40 sites, 1020 uniforms/image)."""
import time, math, torch
dev = "cuda"; B, N, S = 64, 50, 40; U = (S // 2) * N + (S // 2); M32 = 0xFFFFFFFF
def lowbias32(x):
    x = x ^ (x >> 16); x = (x * 0x7feb352d) & M32
    x = x ^ (x >> 15); x = (x * 0x846ca68b) & M32
    return x ^ (x >> 16)
img = torch.arange(B, device=dev, dtype=torch.int64)[:, None].expand(B, U)
elem = torch.arange(U, device=dev, dtype=torch.int64)[None, :].expand(B, U)
seed = 1234
key_img = lowbias32((img * 0x9E3779B9 + seed) & M32)            # per-image key (could be cached per batch)
def uniforms():
    h = lowbias32((key_img ^ (elem * 0x85EBCA6B)) & M32)
    return (h.to(torch.float32) + 0.5) * (1.0 / 4294967296.0)
loc = torch.randn(S, device=dev); scale = torch.rand(S, device=dev) + 0.1
def resolve():
    u = uniforms(); uo = u[:, : (S // 2) * N].reshape(B, S // 2, N); ui = u[:, (S // 2) * N:]
    eo = math.sqrt(2) * torch.erfinv(2 * uo.clamp(1e-7, 1 - 1e-7) - 1); ei = math.sqrt(2) * torch.erfinv(2 * ui.clamp(1e-7, 1 - 1e-7) - 1)
    return loc[None, :S // 2, None] + scale[None, :S // 2, None] * eo, loc[None, S // 2:] + scale[None, S // 2:] * ei
def bench(fn, n=200, warm=20):
    for _ in range(warm): fn()
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(n): fn()
    torch.cuda.synchronize(); return (time.perf_counter() - t0) / n * 1e3
with torch.no_grad():
    u = uniforms(); print(f"mean {u.mean().item():.4f} var {u.var().item():.4f}; lag-1 corr {torch.corrcoef(torch.stack([u[:, :-1].flatten(), u[:, 1:].flatten()]))[0,1].item():.1e}")
    from torch.profiler import profile, ProfilerActivity
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        resolve(); torch.cuda.synchronize()
    print("kernels:", sum(e.count for e in prof.key_averages() if e.device_type.name == "CUDA"))
    print(f"cheap counter hash + erfinv eager: {bench(resolve):.3f} ms")
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3): resolve()
    torch.cuda.current_stream().wait_stream(s)
    gr = torch.cuda.CUDAGraph()
    with torch.cuda.graph(gr): resolve()
    print(f"cheap counter hash + erfinv graph: {bench(gr.replay):.3f} ms")
