import torch, time, math
from torch.utils.checkpoint import checkpoint
dev = "cuda"
torch.manual_seed(0)

def bench(fn, n=20, warm=3):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / n * 1e3

# (a) CUDA graph capture of a small multislice forward
N, S = 256, 32
H = torch.exp(1j * torch.rand(N, N, device=dev))
phis = torch.rand(1, S, N, N, device=dev)
u_in = torch.ones(1, N, N, dtype=torch.complex64, device=dev)
def fwd():
    u = u_in
    for k in range(S):
        u = torch.fft.ifft2(torch.fft.fft2(u * torch.exp(1j * phis[:, k])) * H)
    return u
te = bench(fwd)
s = torch.cuda.Stream()
s.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s):
    for _ in range(3):
        fwd()
torch.cuda.current_stream().wait_stream(s)
g = torch.cuda.CUDAGraph()
try:
    with torch.cuda.graph(g):
        out_static = fwd()
    tg = bench(lambda: g.replay())
    ref = fwd()
    print(f"CUDA graph: eager {te:.3f} ms, graph replay {tg:.3f} ms, max diff {(out_static-ref).abs().max().item():.2e}")
except Exception as e:
    print("CUDA graph capture ERR", str(e)[:200])

# (b) per-op overhead on tiny tensors
x = torch.rand(1000, device=dev)
def tiny():
    y = x
    for _ in range(100):
        y = y * 1.0001 + 0.0001
    return y
print(f"100 tiny elementwise pairs: {bench(tiny):.3f} ms -> {bench(tiny)/200*1e3:.1f} us/op (GPU)")
xc = x.cpu()
def tiny_cpu():
    y = xc
    for _ in range(100):
        y = y * 1.0001 + 0.0001
    return y
t = time.perf_counter(); [tiny_cpu() for _ in range(20)]; print(f"same on CPU: {(time.perf_counter()-t)/20*1e3/200*1e3:.1f} us/op")

# (c) Mie log-derivative downward recurrence D_n(mx), batched over particles, float64 vs float32
def mie_D(mx, nmax, nstart):
    D = torch.zeros_like(mx)
    out = [None] * (nmax + 1)
    for n in range(nstart, 0, -1):
        D = n / mx - 1.0 / (D + n / mx)
        if n - 1 <= nmax:
            out[n - 1] = D
    return torch.stack(out[1:], -1)  # D_1..D_nmax
P = 4096
for devx in ["cpu", "cuda"]:
    for dt in [torch.complex128, torch.complex64]:
        x = (torch.rand(P, dtype=torch.float64, device=devx) * 20 + 1)
        m = torch.full((P,), 1.5 + 0.01j, dtype=torch.complex128, device=devx)
        mx = (m * x).to(dt)
        nmax = int(20 + 4 * 20 ** (1 / 3) + 2)
        nstart = int(max(nmax, 1.5 * 21) + 16)
        f = lambda: mie_D(mx, nmax, nstart)
        if devx == "cuda":
            t = bench(f, n=10)
        else:
            t0 = time.perf_counter(); [f() for _ in range(5)]; t = (time.perf_counter() - t0) / 5 * 1e3
        print(f"Mie D_n recurrence P={P} nstart={nstart} {devx} {dt}: {t:.2f} ms")
# accuracy c64 vs c128
x = torch.linspace(1, 50, 200, dtype=torch.float64)
for mm in [1.5 + 0.0j, 1.33 + 0.0j, 0.2 + 3.0j]:
    mx = mm * x.to(torch.complex128)
    nmax = int(50 + 4 * 50 ** (1 / 3) + 2); nstart = int(max(nmax, abs(mm) * 50) + 16)
    Dd = mie_D(mx, nmax, nstart)
    Ds = mie_D(mx.to(torch.complex64), nmax, nstart).to(torch.complex128)
    print(f"m={mm}: max rel err D_n c64 vs c128: {((Ds-Dd).abs()/Dd.abs()).max().item():.2e}")

# (d) incoherent sum over modes, chunked with checkpoint
N, B, M = 512, 8, 64
H = torch.exp(1j * torch.rand(N, N, device=dev)).to(torch.complex64)
obj_phase = (torch.rand(B, N, N, device=dev) * 0.5).requires_grad_()
fx = torch.fft.fftfreq(N, device=dev)
tilts = torch.rand(M, 2, device=dev) * 0.1
X = torch.arange(N, device=dev, dtype=torch.float32)
def modes_image(obj_phase, tilts):
    ill = torch.exp(2j * torch.pi * (tilts[:, 0, None, None] * X[:, None] + tilts[:, 1, None, None] * X[None, :]))  # [m,N,N]
    u = ill[None] * torch.exp(1j * obj_phase)[:, None]  # [B,m,N,N]
    u = torch.fft.ifft2(torch.fft.fft2(u) * H)
    return u.abs().pow(2).sum(1)
W = torch.rand(N, N, device=dev)
def run(chunk, use_ckpt):
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); base = torch.cuda.memory_allocated()
    t = time.perf_counter()
    img = 0
    for c0 in range(0, M, chunk):
        tt = tilts[c0:c0 + chunk]
        img = img + (checkpoint(modes_image, obj_phase, tt, use_reentrant=False) if use_ckpt else modes_image(obj_phase, tt))
    (img * W).mean().backward()
    torch.cuda.synchronize()
    g = obj_phase.grad.clone(); obj_phase.grad = None
    print(f"modes M={M} B={B} chunk={chunk:3d} ckpt={use_ckpt!s:5s}: peak {(torch.cuda.max_memory_allocated()-base)/2**20:7.0f} MB, {(time.perf_counter()-t)*1e3:7.1f} ms")
    return g
run(M, False)
g0 = run(M, False)
g1 = run(8, True); g1 = run(8, True)
g2 = run(1, True)
print("chunked grad rel diff", ((g1-g0).norm()/g0.norm()).item(), ((g2-g0).norm()/g0.norm()).item())
