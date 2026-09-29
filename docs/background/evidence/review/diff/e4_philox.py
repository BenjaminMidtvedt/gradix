"""E4: counter-keyed Philox4x32-10 in pure torch (int64 emulation of uint32 arithmetic).
Checks: Random123 known-answer test on CPU and CUDA; eager time and kernel count for a [B, U] pool;
CUDA-graph capture/replay; what happens when the seed is a Python int vs a device tensor under replay;
uint32 op support in this torch build."""
import time, torch
M0, M1 = 0xD2511F53, 0xCD9E8D57
W0, W1 = 0x9E3779B9, 0xBB67AE85
MASK = 0xFFFFFFFF

def philox(c0, c1, c2, c3, k0, k1, rounds=10):
    for r in range(rounds):
        p0 = c0 * M0; p1 = c2 * M1                 # < 2^64: wraps in int64, low 64 bits exact
        hi0 = (p0 >> 32) & MASK; lo0 = p0 & MASK
        hi1 = (p1 >> 32) & MASK; lo1 = p1 & MASK
        c0, c1, c2, c3 = hi1 ^ c1 ^ k0, lo1, hi0 ^ c3 ^ k1, lo0
        if r < rounds - 1:
            k0 = (k0 + W0) & MASK; k1 = (k1 + W1) & MASK
    return c0, c1, c2, c3

def kat(dev):
    z = torch.zeros(1, dtype=torch.int64, device=dev)
    out = philox(z, z, z, z, z.clone(), z.clone())
    f = torch.full((1,), MASK, dtype=torch.int64, device=dev)
    out2 = philox(f, f, f, f, f.clone(), f.clone())
    pi = [torch.tensor([v], dtype=torch.int64, device=dev) for v in (0x243f6a88, 0x85a308d3, 0x13198a2e, 0x03707344, 0xa4093822, 0x299f31d0)]
    out3 = philox(*pi)
    return [hex(int(t)) for t in out], [hex(int(t)) for t in out2], [hex(int(t)) for t in out3]

for dev in ("cpu", "cuda"):
    print(dev, "KAT:", kat(dev))
print("expected  :", ['0x6627e8d5', '0xe169c58d', '0xbc57ac4c', '0x9b00dbd8'],
      ['0x408f276d', '0x41c83b0e', '0xa20bc7c6', '0x6d5451fd'], ['0xd16cfe09', '0x94fdcceb', '0x5001e420', '0x24126ea1'])

dev = "cuda"
B, U = 64, 1024          # images x uniforms per image (40 sites, ~25 elements avg) -> 4 uniforms per Philox call
img = torch.arange(B, device=dev, dtype=torch.int64)[:, None].expand(B, U // 4)
elem = torch.arange(U // 4, device=dev, dtype=torch.int64)[None, :].expand(B, U // 4)
site = torch.full_like(img, 0x1234ABCD)
zero = torch.zeros_like(img)
seed_t = torch.tensor([12345, 678], device=dev, dtype=torch.int64)

def pool(seed_k0, seed_k1):
    c = philox(img, site, elem, zero, seed_k0, seed_k1)
    u = torch.stack(c, -1).reshape(B, U)
    return (u.to(torch.float64) + 0.5) * 2.0**-32

def bench(fn, n=200):
    for _ in range(10): fn()
    torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(n): fn()
    torch.cuda.synchronize(); return (time.perf_counter() - t) / n * 1e3

from torch.profiler import profile, ProfilerActivity
with profile(activities=[ProfilerActivity.CUDA]) as prof:
    pool(seed_t[0], seed_t[1]); torch.cuda.synchronize()
nk = sum(1 for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA)
print(f"eager pool [{B},{U}]: {bench(lambda: pool(seed_t[0], seed_t[1])):.3f} ms, CUDA kernels per call: {nk}")

# CUDA graph: seed as device tensor (static input) vs Python int baked into the graph
static_seed = seed_t.clone()
s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s):
    for _ in range(3): pool(static_seed[0], static_seed[1]); pool(12345, 678)
torch.cuda.current_stream().wait_stream(s)
gA = torch.cuda.CUDAGraph()
with torch.cuda.graph(gA): outA = pool(static_seed[0], static_seed[1])
py_seed = 12345
gB = torch.cuda.CUDAGraph()
with torch.cuda.graph(gB): outB = pool(py_seed, 678)
print(f"graph replay: {bench(gA.replay):.3f} ms")
gA.replay(); a1 = outA.clone(); static_seed.copy_(torch.tensor([999, 678], device=dev)); gA.replay(); a2 = outA.clone()
gB.replay(); b1 = outB.clone(); py_seed = 999; gB.replay(); b2 = outB.clone()
print("seed as device tensor: new seed changes pool ->", not torch.equal(a1, a2))
print("seed as Python int   : new seed changes pool ->", not torch.equal(b1, b2))

# uint32 op support
for opname, fn in (("mul", lambda x: x * x), ("xor", lambda x: x ^ x), ("rshift", lambda x: x >> 3), ("and", lambda x: x & x)):
    try:
        fn(torch.ones(4, dtype=torch.uint32, device=dev)); ok = "ok"
    except Exception as e:
        ok = type(e).__name__ + ": " + str(e).splitlines()[0][:80]
    print(f"uint32 {opname} on CUDA: {ok}")
