"""FEAS check: per-slice overhead of the scattered-field march with an analytic PlaneWaves background
(E_s <- P_dz[(t_s - 1) E_b(z_s) + t_s E_s], E_b evaluated analytically per slice) vs a plain total-field march
(u <- P_dz[t_s u]). Inference, precomputed slices, eager torch (no Triton/compile on this Windows box).
Also check that both give the same total field for an on-grid (untilted) background."""
import math, time, torch
dev = "cuda"; torch.backends.cuda.matmul.allow_tf32 = False
B, M, N, NZ = 8, 1, 256, 64
dx, lam, nm, dz = 0.1, 0.532, 1.333, 0.2
f = torch.fft.fftfreq(N, d=dx, device=dev)
FY, FX = torch.meshgrid(f, f, indexing="ij")
kz = 2 * math.pi * torch.sqrt(((nm / lam) ** 2 - FX ** 2 - FY ** 2).clamp(min=0).to(torch.complex64))
H = torch.where((FX ** 2 + FY ** 2) < (nm / lam) ** 2, torch.exp(1j * kz * dz), torch.zeros((), dtype=torch.complex64, device=dev))
g = torch.Generator(device=dev).manual_seed(0)
dn = 0.02 * torch.rand(NZ, B, 1, N, N, device=dev, generator=g)
t = torch.exp(1j * 2 * math.pi / lam * dn * dz).to(torch.complex64)          # [NZ,B,1,N,N]
d = t - 1                                                                   # precomputed (t-1) screens
kz_b = 2 * math.pi * nm / lam                                               # on-axis background
Eb_lat = torch.ones(B, M, N, N, dtype=torch.complex64, device=dev)          # lateral pattern exp(i k_perp r) (on-axis: 1)
ph = torch.tensor([complex(math.cos(kz_b * dz * (s + 1)), math.sin(kz_b * dz * (s + 1))) for s in range(NZ)],
                  dtype=torch.complex64, device=dev)

def total():
    u = Eb_lat.clone()
    for s in range(NZ):
        u = torch.fft.ifft2(torch.fft.fft2(t[s] * u) * H)
    return u

def scattered():
    Es = torch.zeros(B, M, N, N, dtype=torch.complex64, device=dev)
    for s in range(NZ):
        Eb_s = Eb_lat * ph[max(s - 1, 0)] if s else Eb_lat                 # E_b at the slice plane (analytic)
        src = d[s] * (Eb_s + Es) + Es                                       # (t-1)E_b + t E_s
        Es = torch.fft.ifft2(torch.fft.fft2(src) * H)
    return Es, Eb_lat * ph[NZ - 1]

def bench(fn, n=20, warm=3):
    for _ in range(warm): fn()
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(n): fn()
    torch.cuda.synchronize(); return (time.perf_counter() - t0) / n * 1e3

with torch.inference_mode():
    u = total(); Es, Eb = scattered()
    print(f"rel diff total vs Eb+Es: {((Eb + Es) - u).abs().norm() / u.abs().norm():.2e}")
    ta, tb = bench(total), bench(scattered)
    print(f"total-field march     B{B} {N}^2 x {NZ}: {ta:6.2f} ms")
    print(f"scattered + analytic  B{B} {N}^2 x {NZ}: {tb:6.2f} ms  ({tb/ta:.2f}x)")
