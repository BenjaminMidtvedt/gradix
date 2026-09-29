"""Training variant of the judge anchor (B8, 64 slices, 256^2, procedural 6 soft ellipsoids, checkpoint every 8):
total-field march (anchor, 41 img/s, 0.90 GiB) vs the plan's scattered-field march with analytic background."""
import math, time, torch
import torch.utils.checkpoint as cp
dev = "cuda"; torch.backends.cuda.matmul.allow_tf32 = False
wl, n_m = 0.532, 1.33
B, S, N, dx, dz, n_obj = 8, 64, 256, 0.1, 0.15, 6
f = torch.fft.fftfreq(N, dx, device=dev); fy, fx = torch.meshgrid(f, f, indexing="ij")
kz = 2*math.pi*torch.sqrt(torch.clamp((n_m/wl)**2 - fx**2 - fy**2, min=0))
H = torch.polar(torch.ones_like(kz), (kz - 2*math.pi*n_m/wl)*dz)
xs = (torch.arange(N, device=dev)+0.5)*dx; Y, X = torch.meshgrid(xs, xs, indexing="ij")
g = torch.Generator(device=dev).manual_seed(0)
c = (torch.rand(B, n_obj, 3, device=dev, generator=g)*torch.tensor([N*dx, N*dx, S*dz], device=dev)).requires_grad_()
r = (torch.rand(B, n_obj, 3, device=dev, generator=g)*2.0+1.5).requires_grad_()
dn = torch.full((B, n_obj), 0.03, device=dev).requires_grad_()
k0 = 2*math.pi/wl
def slice_t(s):
    z = (s+0.5)*dz
    q = (((X-c[...,0,None,None])/r[...,0,None,None])**2 + ((Y-c[...,1,None,None])/r[...,1,None,None])**2
         + ((z-c[...,2,None,None])/r[...,2,None,None])**2)
    d = (q.sqrt()-1)*r.min(-1).values[...,None,None]
    occ = torch.clamp(0.5-d/dx, 0, 1)
    ph = k0*dz*(occ*dn[...,None,None]).sum(1)
    return torch.polar(torch.ones_like(ph), ph)
def seg_total(u, s0, s1):
    for s in range(s0, s1): u = torch.fft.ifft2(torch.fft.fft2(u*slice_t(s))*H)
    return u
Eb = torch.ones(B, N, N, dtype=torch.complex64, device=dev)
def seg_scat(Es, s0, s1):
    for s in range(s0, s1):
        t = slice_t(s); Es = torch.fft.ifft2(torch.fft.fft2((t-1)*Eb + t*Es)*H)
    return Es
def run_total():
    u = torch.ones(B, N, N, dtype=torch.complex64, device=dev)
    for s0 in range(0, S, 8): u = cp.checkpoint(seg_total, u, s0, s0+8, use_reentrant=False)
    (u.real**2+u.imag**2).mean().backward()
def run_scat():
    Es = torch.zeros(B, N, N, dtype=torch.complex64, device=dev)
    for s0 in range(0, S, 8): Es = cp.checkpoint(seg_scat, Es, s0, s0+8, use_reentrant=False)
    I = (Eb.real**2+Eb.imag**2) + 2*(Eb.conj()*Es).real + (Es.real**2+Es.imag**2)
    I.mean().backward()
def bench(fn, reps=3):
    fn(); torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); t = time.perf_counter()
    for _ in range(reps): fn()
    torch.cuda.synchronize(); return (time.perf_counter()-t)/reps*1e3, torch.cuda.max_memory_allocated()/2**30
for name, fn in (("anchor total-field", run_total), ("scattered+analytic", run_scat)):
    t, m = bench(fn); print(f"{name:20s}: {t:6.0f} ms  {B/t*1e3:5.1f} img/s  peak {m:.2f} GiB")
