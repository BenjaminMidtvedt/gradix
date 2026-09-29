# Verify: (1) forward-only BPM throughput at 256^2 x 64 slices (claim in D: ~1e3 img/s),
#         (2) procedural slices + plain checkpointing already avoid the volume wall (C's position).
import math, time, torch
import torch.utils.checkpoint as cp
dev = "cuda"; torch.backends.cuda.matmul.allow_tf32 = False
wl, n_m = 0.532, 1.33   # um
def setup(B, S, N, dx=0.1, dz=0.15, n_obj=6, grad=False):
    f = torch.fft.fftfreq(N, dx, device=dev); fy, fx = torch.meshgrid(f, f, indexing="ij")
    kz = 2*math.pi*torch.sqrt(torch.clamp((n_m/wl)**2 - fx**2 - fy**2, min=0))
    H = torch.polar(torch.ones_like(kz), (kz - 2*math.pi*n_m/wl)*dz)
    xs = (torch.arange(N, device=dev)+0.5)*dx; Y, X = torch.meshgrid(xs, xs, indexing="ij")
    g = torch.Generator(device=dev).manual_seed(0)
    c = torch.rand(B, n_obj, 3, device=dev, generator=g)*torch.tensor([N*dx, N*dx, S*dz], device=dev)
    r = torch.rand(B, n_obj, 3, device=dev, generator=g)*2.0+1.5
    dn = torch.full((B, n_obj), 0.03, device=dev)
    if grad: c.requires_grad_(); r.requires_grad_(); dn.requires_grad_()
    k0 = 2*math.pi/wl
    def slice_phase(s):
        z = (s+0.5)*dz
        q = (((X-c[...,0,None,None])/r[...,0,None,None])**2 + ((Y-c[...,1,None,None])/r[...,1,None,None])**2
             + ((z-c[...,2,None,None])/r[...,2,None,None])**2)
        d = (q.sqrt()-1)*r.min(-1).values[...,None,None]
        occ = torch.clamp(0.5-d/dx, 0, 1)
        return k0*dz*(occ*dn[...,None,None]).sum(1)
    def step(u, s, phase=None):
        ph = slice_phase(s) if phase is None else phase
        return torch.fft.ifft2(torch.fft.fft2(u*torch.polar(torch.ones_like(ph), ph))*H)
    return step, (c, r, dn)

def bench(fn, reps=5):
    fn(); torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(reps): fn()
    torch.cuda.synchronize(); return (time.perf_counter()-t)/reps*1e3

# (1) forward-only, inference mode
B, S, N = 64, 64, 256
step, _ = setup(B, S, N)
pre = torch.randn(B, N, N, device=dev)*0.01
def fwd_proc():
    with torch.inference_mode():
        u = torch.ones(B, N, N, dtype=torch.complex64, device=dev)
        for s in range(S): u = step(u, s)
        return u.real**2+u.imag**2
def fwd_pre():
    with torch.inference_mode():
        u = torch.ones(B, N, N, dtype=torch.complex64, device=dev)
        for s in range(S): u = step(u, s, pre)
        return u.real**2+u.imag**2
t1 = bench(fwd_proc); t2 = bench(fwd_pre)
print(f"fwd inference B{B} S{S} {N}^2: procedural 6 objs {t1:.1f} ms ({B/t1*1e3:.0f} img/s); "
      f"precomputed phase {t2:.1f} ms ({B/t2*1e3:.0f} img/s); per slice-step {t2/S:.3f} ms")

# (2) gradients: naive vs checkpoint every 8, procedural slices, B=8
for ck in [0, 8]:
    B2 = 8
    step, params = setup(B2, S, N, grad=True)
    def run():
        u = torch.ones(B2, N, N, dtype=torch.complex64, device=dev)
        if ck:
            def seg(u, s0, s1):
                for s in range(s0, s1): u = step(u, s)
                return u
            for s0 in range(0, S, ck): u = cp.checkpoint(seg, u, s0, min(S, s0+ck), use_reentrant=False)
        else:
            for s in range(S): u = step(u, s)
        (u.real**2+u.imag**2).mean().backward()
    run(); torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    t = bench(run, reps=3); pk = torch.cuda.max_memory_allocated()/2**30
    g = params[0].grad.abs().sum().item()
    print(f"grad B{B2} S{S} {N}^2 ckpt={ck}: {t:.0f} ms, peak {pk:.2f} GiB, |dL/dc|={g:.3e}")
