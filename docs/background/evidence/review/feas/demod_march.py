"""Is the per-slice analytic-background update needed for seam-free, tilt-differentiable multislice?
Compare, for an OFF-GRID tilted plane wave through a random weak phase object (with padding):
 (1) plan's scattered-field march: E_s <- P[(t-1) E_b(z) + t E_s], E_b analytic;
 (2) demodulated total-field march: u' <- IFFT(H(q + k_b) FFT(t u')), total = e^{i k_b r} u'   (no background bookkeeping);
 (3) naive sampled total-field march of the tilted wave on the periodic grid (the seam case).
Reference = (1) in complex128. Also check d(result)/d(tilt) flows in (2)."""
import math, torch
torch.set_default_dtype(torch.float64)
dev = "cuda"
N, dx, lam, nm, dz, S = 256, 0.1, 0.532, 1.333, 0.2, 16
k = 2*math.pi*nm/lam
f = torch.fft.fftfreq(N, dx, device=dev); FY, FX = torch.meshgrid(f, f, indexing="ij")
xs = torch.arange(N, device=dev)*dx; Y, X = torch.meshgrid(xs, xs, indexing="ij")
g = torch.Generator(device=dev).manual_seed(1)
obj = torch.zeros(S, N, N, device=dev)
win = ((X-12.8)**2 + (Y-12.8)**2 < 6.0**2).double()                         # object inside, padding outside
obj[:] = 0.3*torch.randn(S, N, N, device=dev, generator=g)*win
def kz_of(fx, fy):
    a = (nm/lam)**2 - fx**2 - fy**2
    return torch.where(a > 0, 2*math.pi*torch.sqrt(a.clamp(min=0)), torch.zeros_like(a)), a > 0
def run(dtype, sinx):
    cd = torch.complex128 if dtype == 64 else torch.complex64
    t = torch.polar(torch.ones_like(obj), obj).to(cd)
    fbx = nm*sinx/lam                                                   # off-grid tilt (cycles/um)
    kzb = 2*math.pi*math.sqrt((nm/lam)**2 - fbx**2)
    lat = torch.polar(torch.ones_like(X), 2*math.pi*fbx*X).to(cd)       # analytic lateral carrier
    # (1) scattered-field + analytic background, lab frame, carrier kzb removed from H
    kz, prop = kz_of(FX, FY); H = torch.where(prop, torch.polar(torch.ones_like(kz), (kz-kzb)*dz), 0*kz).to(cd)
    Es = torch.zeros(N, N, dtype=cd, device=dev)
    for s in range(S):
        Es = torch.fft.ifft2(torch.fft.fft2((t[s]-1)*lat + t[s]*Es)*H)
    tot1 = lat + Es
    # (2) demodulated total field: H evaluated at q + k_b
    kz2, prop2 = kz_of(FX + fbx, FY); H2 = torch.where(prop2, torch.polar(torch.ones_like(kz2), (kz2-kzb)*dz), 0*kz2).to(cd)
    u = torch.ones(N, N, dtype=cd, device=dev)
    for s in range(S):
        u = torch.fft.ifft2(torch.fft.fft2(t[s]*u)*H2)
    tot2 = lat*u
    # (3) naive sampled tilted wave in the lab frame
    v = lat.clone()
    for s in range(S):
        v = torch.fft.ifft2(torch.fft.fft2(t[s]*v)*H)
    return tot1, tot2, v
ref, dem, naive = run(64, 0.2037)
inner = ((X-12.8)**2 + (Y-12.8)**2 < 8.0**2)
rel = lambda a, b: ((a-b)[inner].abs().norm()/b[inner].abs().norm()).item()
print(f"demodulated vs scattered+analytic (c128): {rel(dem, ref):.2e}")
print(f"naive sampled tilt vs scattered+analytic : {rel(naive, ref):.2e}")
r32 = run(32, 0.2037)
print(f"demodulated c64 vs reference            : {rel(r32[1].to(torch.complex128), ref):.2e}")
print(f"scattered c64 vs reference              : {rel(r32[0].to(torch.complex128), ref):.2e}")
# gradient wrt tilt through the demodulated march (continuous, no snapping)
sx = torch.tensor(0.2037, device=dev, requires_grad=True)
fbx = nm*sx/lam; kzb = 2*math.pi*torch.sqrt((nm/lam)**2 - fbx**2)
kz2 = 2*math.pi*torch.sqrt(((nm/lam)**2 - (FX+fbx)**2 - FY**2).clamp(min=1e-12))
H2 = torch.polar(torch.ones_like(kz2), (kz2-kzb)*dz)*(((nm/lam)**2 - (FX+fbx)**2 - FY**2) > 0)
t = torch.polar(torch.ones_like(obj), obj)
u = torch.ones(N, N, dtype=torch.complex128, device=dev)
for s in range(S): u = torch.fft.ifft2(torch.fft.fft2(t[s]*u)*H2)
I = (u.abs()**2)[inner].mean(); I.backward()
print(f"dI/d(sin tilt) via demodulated march: {sx.grad.item():.3e} (non-zero, continuous)")
