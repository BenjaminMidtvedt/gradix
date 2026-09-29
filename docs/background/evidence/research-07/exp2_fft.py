import torch, time
dev = "cuda"
torch.manual_seed(0)

def bench(fn, n=20):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / n * 1e3

print("== FFT timings (ms) on", torch.cuda.get_device_name(0))
for B, N in [(1, 512), (64, 512), (64, 1024), (256, 512), (1, 2048)]:
    x = torch.randn(B, N, N, dtype=torch.complex64, device=dev)
    t64 = bench(lambda: torch.fft.fft2(x))
    xh = x.to(torch.complex32)
    th = bench(lambda: torch.fft.fft2(xh))
    xd = x.to(torch.complex128)
    td = bench(lambda: torch.fft.fft2(xd), n=5)
    # elementwise complex multiply for comparison
    h = torch.randn(N, N, dtype=torch.complex64, device=dev)
    tm = bench(lambda: x * h)
    gb = B * N * N * 8 / 1e9
    print(f"B={B:4d} N={N:5d}  c64 fft2 {t64:8.3f}  c32 {th:8.3f}  c128 {td:8.3f}  c64 mul {tm:7.3f}  (field {gb*1e3:.0f} MB)")

# odd sizes
for N in [500, 509, 512, 520, 1000, 1024]:
    x = torch.randn(64, N, N, dtype=torch.complex64, device=dev)
    print(f"N={N} c64 fft2 B=64: {bench(lambda: torch.fft.fft2(x)):.3f} ms")

print("== precision of fft->ifft roundtrip & propagation (rel L2 err vs complex128)")
N = 512
xd = torch.randn(1, N, N, dtype=torch.complex128, device=dev)
fx = torch.fft.fftfreq(N, d=0.1, device=dev, dtype=torch.float64)
KX, KY = torch.meshgrid(fx, fx, indexing="ij")
lam = 0.5
kz = torch.sqrt(torch.clamp(1 / lam**2 - KX**2 - KY**2, min=0))
for z in [1.0, 100.0, 10000.0]:
    Hd = torch.exp(2j * torch.pi * kz * z)
    ref = torch.fft.ifft2(torch.fft.fft2(xd) * Hd)
    for dt, name in [(torch.complex64, "c64"), (torch.complex32, "c32")]:
        x = xd.to(dt)
        H = Hd.to(dt)
        # phase computed in fp32 then cast
        out = torch.fft.ifft2(torch.fft.fft2(x, norm="ortho") * H, norm="ortho")
        err = (out.to(torch.complex128) - ref).norm() / ref.norm()
        print(f"z={z:8.1f} {name}: rel err {err.item():.2e}")
    # phase computed in float32 from kz (precision loss in phase argument)
    kz32 = kz.float()
    H32 = torch.exp(2j * torch.pi * kz32 * z)
    out = torch.fft.ifft2(torch.fft.fft2(xd.to(torch.complex64)) * H32)
    err = (out.to(torch.complex128) - ref).norm() / ref.norm()
    print(f"z={z:8.1f} c64 with fp32 phase arg: rel err {err.item():.2e}")

# 100 slices accumulated error c64 vs c128 with random phase screens
print("== multislice accumulated error, 100 slices")
phis = torch.rand(100, N, N, device=dev, dtype=torch.float64) * 0.1
Hd = torch.exp(2j * torch.pi * kz * 0.2)
def ms(u, H, phis, dt):
    u = u.to(dt); H = H.to(dt)
    for p in phis:
        u = torch.fft.ifft2(torch.fft.fft2(u * torch.exp(1j * p.to(u.real.dtype)), norm="ortho") * H, norm="ortho")
    return u
u0 = torch.ones(1, N, N, dtype=torch.complex128, device=dev)
ref = ms(u0, Hd, phis, torch.complex128)
for dt in [torch.complex64, torch.complex32]:
    try:
        out = ms(u0, Hd, phis, dt)
        print(dt, "rel err", ((out.to(torch.complex128) - ref).norm() / ref.norm()).item())
    except Exception as e:
        print(dt, "ERR", str(e)[:100])
