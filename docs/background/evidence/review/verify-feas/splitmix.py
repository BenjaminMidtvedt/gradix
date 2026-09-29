"""FEAS-4 alternative: counter-keyed uniforms from a splitmix64 finaliser in pure torch int64 (wraparound multiply,
logical shifts emulated by masking). Workload as [M-D] expD / reviewer: B=64, 40 sites, 1020 uniforms per image."""
import time, math, torch
dev = "cuda"; B, N, S = 64, 50, 40; U = (S // 2) * N + (S // 2)
def s64(c):  # python int -> signed int64 constant
    c &= (1 << 64) - 1
    return c - (1 << 64) if c >= (1 << 63) else c
M1, M2, GAMMA = s64(0xbf58476d1ce4e5b9), s64(0x94d049bb133111eb), s64(0x9E3779B97F4A7C15)
def lsr(z, k): return (z >> k) & ((1 << (64 - k)) - 1)      # logical shift right on int64
def mix(z):
    z = (z ^ lsr(z, 30)) * M1
    z = (z ^ lsr(z, 27)) * M2
    return z ^ lsr(z, 31)
img = torch.arange(B, device=dev, dtype=torch.int64)[:, None]
elem = torch.arange(U, device=dev, dtype=torch.int64)[None, :]
seed = 1234
key_img = mix(img * GAMMA + seed)                              # [B,1] per-image key (site hash would be folded in per site)
def uniforms():
    z = mix(key_img + elem * GAMMA)                            # counter = element index (includes site offset)
    return (lsr(z, 40).to(torch.float32) + 0.5) * (1.0 / (1 << 24))
loc = torch.randn(S, device=dev); scale = torch.rand(S, device=dev) + 0.1
def resolve():
    u = uniforms(); uo = u[:, : (S // 2) * N].reshape(B, S // 2, N); ui = u[:, (S // 2) * N:]
    eo = math.sqrt(2) * torch.erfinv(2 * uo - 1); ei = math.sqrt(2) * torch.erfinv(2 * ui - 1)
    return loc[None, :S // 2, None] + scale[None, :S // 2, None] * eo, loc[None, S // 2:] + scale[None, S // 2:] * ei
def bench(fn, n=200, warm=20):
    for _ in range(warm): fn()
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(n): fn()
    torch.cuda.synchronize(); return (time.perf_counter() - t0) / n * 1e3
with torch.no_grad():
    u = uniforms()
    print(f"mean {u.mean().item():.4f} var {u.var().item():.4f} lag1 {torch.corrcoef(torch.stack([u[:, :-1].flatten(), u[:, 1:].flatten()]))[0,1].item():.1e} "
          f"img-img corr {torch.corrcoef(torch.stack([u[0], u[1]]))[0,1].item():.1e}")
    # batch-composition invariance: image 17 computed alone equals image 17 in the batch
    k17 = mix(torch.tensor([[17]], device=dev) * GAMMA + seed)
    z17 = mix(k17 + elem * GAMMA); print("image 17 identical alone vs in batch:", torch.equal(z17[0], mix(key_img + elem * GAMMA)[17]))
    from torch.profiler import profile, ProfilerActivity
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        resolve(); torch.cuda.synchronize()
    print("kernels:", sum(e.count for e in prof.key_averages() if e.device_type.name == "CUDA"))
    print(f"splitmix64 + erfinv eager: {bench(resolve):.3f} ms")
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3): resolve()
    torch.cuda.current_stream().wait_stream(s)
    gr = torch.cuda.CUDAGraph()
    with torch.cuda.graph(gr): resolve()
    print(f"splitmix64 + erfinv graph: {bench(gr.replay):.3f} ms")
