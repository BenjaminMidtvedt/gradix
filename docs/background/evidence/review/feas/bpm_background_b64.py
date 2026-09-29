"""Replicates the judges-engineering/bpm_check.py precomputed-slice anchor (B64, 64 slices, 256^2, inference)
and the same march in the plan's scattered-field form with an analytic background (ADR-04 obligation)."""
import math, time, torch
dev = "cuda"; torch.backends.cuda.matmul.allow_tf32 = False
wl, n_m = 0.532, 1.33
B, S, N, dx, dz = 64, 64, 256, 0.1, 0.15
f = torch.fft.fftfreq(N, dx, device=dev); fy, fx = torch.meshgrid(f, f, indexing="ij")
kz = 2*math.pi*torch.sqrt(torch.clamp((n_m/wl)**2 - fx**2 - fy**2, min=0))
H = torch.polar(torch.ones_like(kz), (kz - 2*math.pi*n_m/wl)*dz)   # carrier-subtracted, as in the anchor
pre = torch.randn(B, N, N, device=dev)*0.01
def bench(fn, reps=5):
    fn(); torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(reps): fn()
    torch.cuda.synchronize(); return (time.perf_counter()-t)/reps*1e3
def anchor():          # judge's fwd_pre
    with torch.inference_mode():
        u = torch.ones(B, N, N, dtype=torch.complex64, device=dev)
        for s in range(S):
            u = torch.fft.ifft2(torch.fft.fft2(u*torch.polar(torch.ones_like(pre), pre))*H)
        return u.real**2+u.imag**2
def scattered():       # E_s <- P[(t-1) E_b + t E_s]; E_b analytic (on-axis, carrier-removed -> lateral 1, phase 1)
    with torch.inference_mode():
        Eb = torch.ones(B, N, N, dtype=torch.complex64, device=dev)
        Es = torch.zeros(B, N, N, dtype=torch.complex64, device=dev)
        for s in range(S):
            tt = torch.polar(torch.ones_like(pre), pre)
            Es = torch.fft.ifft2(torch.fft.fft2((tt - 1)*Eb + tt*Es)*H)
        Ebd = Eb  # background at detection, evaluated analytically
        return (Ebd.real**2+Ebd.imag**2) + 2*(Ebd.conj()*Es).real + (Es.real**2+Es.imag**2)
ta, tb = bench(anchor), bench(scattered)
print(f"anchor total-field  : {ta:6.1f} ms  {B/ta*1e3:6.0f} img/s")
print(f"scattered+analytic  : {tb:6.1f} ms  {B/tb*1e3:6.0f} img/s  ({tb/ta:.2f}x)")
