"""Verify FEAS-5: best-case (in-place) cost of the plan's scattered-field update E_s <- P[t(E_b+E_s) - E_b]
vs plain total-field march, B64 x 64 slices x 256^2, inference, precomputed transmittance."""
import math, time, torch
dev = "cuda"
wl, n_m = 0.532, 1.33
B, S, N, dx, dz = 64, 64, 256, 0.1, 0.15
f = torch.fft.fftfreq(N, dx, device=dev); fy, fx = torch.meshgrid(f, f, indexing="ij")
kz = 2*math.pi*torch.sqrt(torch.clamp((n_m/wl)**2 - fx**2 - fy**2, min=0))
H = torch.polar(torch.ones_like(kz), (kz - 2*math.pi*n_m/wl)*dz).to(torch.complex64)
pre = torch.randn(B, N, N, device=dev)*0.01
T = torch.polar(torch.ones_like(pre), pre)   # precomputed transmittance (shared by slices for timing)
def bench(fn, reps=7):
    fn(); torch.cuda.synchronize(); ts=[]
    for _ in range(reps):
        t0=time.perf_counter(); fn(); torch.cuda.synchronize(); ts.append(time.perf_counter()-t0)
    ts.sort(); return ts[len(ts)//2]*1e3
@torch.inference_mode()
def plain():
    u = torch.ones(B, N, N, dtype=torch.complex64, device=dev)
    for s in range(S):
        u = torch.fft.ifft2(torch.fft.fft2(u*T)*H)
    return u
@torch.inference_mode()
def plain_inplace():
    u = torch.ones(B, N, N, dtype=torch.complex64, device=dev)
    for s in range(S):
        u.mul_(T); u = torch.fft.fft2(u); u.mul_(H); u = torch.fft.ifft2(u)
    return u
# plan's form, E_b tilted, lateral pattern precomputed per mode; E_b(z_s) lateral = lat (carrier kzb folded into H)
lat = torch.polar(torch.ones(N, N, device=dev), 2*math.pi*0.37*torch.arange(N, device=dev).float()[None,:]*dx).to(torch.complex64).expand(B,N,N).contiguous()
@torch.inference_mode()
def scattered_inplace():
    Es = torch.zeros(B, N, N, dtype=torch.complex64, device=dev)
    for s in range(S):
        Es.add_(lat); Es.mul_(T); Es.sub_(lat)          # t(E_b+E_s) - E_b : 3 elementwise ops
        Es = torch.fft.fft2(Es); Es.mul_(H); Es = torch.fft.ifft2(Es)
    return Es
tp, tpi, tsi = bench(plain), bench(plain_inplace), bench(scattered_inplace)
print(f"plain total-field (out-of-place): {tp:6.1f} ms {B/tp*1e3:6.0f} img/s")
print(f"plain total-field (in-place)    : {tpi:6.1f} ms {B/tpi*1e3:6.0f} img/s")
print(f"scattered form (in-place, 3 ops): {tsi:6.1f} ms {B/tsi*1e3:6.0f} img/s  ({tsi/tpi:.2f}x vs in-place plain)")
