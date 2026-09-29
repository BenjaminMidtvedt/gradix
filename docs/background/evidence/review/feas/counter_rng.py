"""FEAS check: cost of a counter-keyed Philox4x32-10 hash implemented in pure torch (int64 emulation of uint32),
followed by inverse-CDF transforms, vs torch.randn with a registered generator (the [M-D] anchor, 0.30 ms graph).
Workload mirrors [M-D] expD: B=64, N=50, 40 sites (20 per-object [B,N], 20 per-image [B]).
"""
import time, math, torch
dev = "cuda"
B, N, S = 64, 50, 40
U = (S // 2) * N + (S // 2)            # uniforms per image = 1020
M32 = 0xFFFFFFFF
PH_M0, PH_M1 = 0xD2511F53, 0xCD9E8D57
PH_W0, PH_W1 = 0x9E3779B9, 0xBB67AE85

def mulhilo(a, m):
    p = a * m                          # int64 wraps mod 2^64; a < 2^32, m < 2^32
    return (p >> 32) & M32, p & M32

def philox(c0, c1, c2, c3, k0, k1, rounds=10):
    for _ in range(rounds):
        hi0, lo0 = mulhilo(c0, PH_M0)
        hi1, lo1 = mulhilo(c2, PH_M1)
        c0, c1, c2, c3 = hi1 ^ c1 ^ k0, lo1, hi0 ^ c3 ^ k1, lo0
        k0 = (k0 + PH_W0) & M32
        k1 = (k1 + PH_W1) & M32
    return c0, c1, c2, c3

img = torch.arange(B, device=dev, dtype=torch.int64)[:, None].expand(B, U)
elem = torch.arange(U, device=dev, dtype=torch.int64)[None, :].expand(B, U)
site = (elem // N) & M32              # stand-in for site hash
seed = 1234

def counter_uniforms(rounds=10):
    c0, c1, c2, c3 = philox(img.clone(), elem.clone(), site.clone(), torch.zeros_like(img),
                            torch.full_like(img, seed & M32), torch.full_like(img, 0), rounds)
    return (c0.to(torch.float32) + 0.5) * (1.0 / 4294967296.0)

loc = torch.randn(S, device=dev); scale = torch.rand(S, device=dev) + 0.1
def counter_resolve(rounds=10):
    u = counter_uniforms(rounds)
    uo = u[:, : (S // 2) * N].reshape(B, S // 2, N)
    ui = u[:, (S // 2) * N:]
    eo = math.sqrt(2) * torch.erfinv(2 * uo.clamp(1e-7, 1 - 1e-7) - 1)
    ei = math.sqrt(2) * torch.erfinv(2 * ui.clamp(1e-7, 1 - 1e-7) - 1)
    vo = loc[None, :S // 2, None] + scale[None, :S // 2, None] * eo
    vi = loc[None, S // 2:] + scale[None, S // 2:] * ei
    return vo, vi

g = torch.Generator(device=dev).manual_seed(0)
def randn_fused():
    eo = torch.randn(B, S // 2, N, device=dev, generator=g)
    ei = torch.randn(B, S // 2, device=dev, generator=g)
    return loc[None, :S // 2, None] + scale[None, :S // 2, None] * eo, loc[None, S // 2:] + scale[None, S // 2:] * ei

def bench(fn, n=200, warm=20):
    for _ in range(warm): fn()
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(n): fn()
    torch.cuda.synchronize(); return (time.perf_counter() - t0) / n * 1e3

def graphed(fn):
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3): fn()
    torch.cuda.current_stream().wait_stream(s)
    gr = torch.cuda.CUDAGraph()
    gr.register_generator_state(g)
    with torch.cuda.graph(gr): fn()
    return gr.replay

# sanity: uniformity of the hash
with torch.no_grad():
    u = counter_uniforms()
    print(f"hash uniforms: mean {u.mean().item():.4f} (0.5)  var {u.var().item():.4f} (0.0833)  distinct {u.unique().numel()}/{u.numel()}")
    # count launches via profiler
    from torch.profiler import profile, ProfilerActivity
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        counter_resolve(); torch.cuda.synchronize()
    nk = sum(e.count for e in prof.key_averages() if e.device_type.name == "CUDA")
    print(f"CUDA kernels per counter_resolve (Philox-10): {nk}")
    for r in (10, 4):
        print(f"counter-keyed Philox-{r} + erfinv, eager : {bench(lambda: counter_resolve(r)):.3f} ms")
        print(f"counter-keyed Philox-{r} + erfinv, graph : {bench(graphed(lambda: counter_resolve(r))):.3f} ms")
    print(f"torch.randn fused (registered gen), eager: {bench(randn_fused):.3f} ms")
    print(f"torch.randn fused (registered gen), graph: {bench(graphed(randn_fused)):.3f} ms")
