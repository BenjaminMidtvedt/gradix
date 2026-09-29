# Test 4 (GPU): power test  int PSF = (1-cos th_max)/2  under the plan's apodisation rule, and the NA > n_sample case.
# Pupil sampled on a u-grid (u = n sin th), as an MFT emitter would; int PSF = (2pi)^2 sum |pupil(k)|^2 dk^2 by Parseval.
import math, torch
dev = "cuda"; lam0 = 0.68; k0 = 2*math.pi/lam0
def grid(NA, Np):
    u = torch.linspace(-NA, NA, Np, device=dev, dtype=torch.float64); du = u[1]-u[0]
    UY, UX = torch.meshgrid(u, u, indexing="ij"); ur = torch.sqrt(UX**2+UY**2)
    return ur, (k0*du)**2
def kz(n, ur): return k0*torch.sqrt((n*n - ur*ur).to(torch.complex128))   # branch Im>=0

print("=== index-matched: emitter in oil n=1.518, NA 1.4; plan rule pupil = sqrt(kz/k) * A, A = -1/(2 pi k kz) ===")
n = 1.518; NA = 1.4; k = k0*n
for Np in [64, 128, 512, 2048]:
    ur, dk2 = grid(NA, Np); inside = ur <= NA
    kzz = kz(n, ur); A = -1/(2*math.pi*k*kzz); pup = torch.sqrt(kzz/k)*A
    P = ((2*math.pi)**2*(pup.abs()**2)[inside].sum()*dk2/(4*math.pi/k**2)).item()
    print(f"  Np={Np:5d}: int PSF / emitted = {P:.4f}   target (1-cos)/2 = {(1-math.sqrt(1-(NA/n)**2))/2:.4f}")

print("\n=== emitter in WATER (n_s=1.33) on/above glass, oil objective NA 1.45 (plan example (b)) ===")
ns, ng, NA = 1.33, 1.518, 1.45; kw = k0*ns
for Np in [64, 128, 512, 2048, 8192]:
    ur, dk2 = grid(NA, Np); inside = ur <= NA
    kzw, kzg = kz(ns, ur), kz(ng, ur)
    A = -1/(2*math.pi*kw*kzw)                                  # exact Weyl spectrum of the emitter in water (h = 0)
    # (a) literal plan rule: sqrt(cos th) with kz of the medium where A is defined (water)
    pa = torch.sqrt(kzw/kw)*A
    Pa = ((2*math.pi)**2*(pa.abs()**2)[inside].sum()*dk2/(4*math.pi/kw**2)).item()
    # (b) sqrt(cos th) of the immersion/glass medium, no interface transmission (scalar Gibson-Lanni is phase-only)
    pb = torch.sqrt(kzg/(k0*ng))*A
    Pb = ((2*math.pi)**2*(pb.abs()**2)[inside].sum()*dk2/(4*math.pi/kw**2)).item()
    # (c) with scalar (TE) Fresnel transmission water->glass and flux in glass: F = (kz_g/k0)|t A|^2, emitted = n_w 4pi/k_w^2
    ts = 2*kzw/(kzw + kzg)
    Pc = ((2*math.pi)**2*((kzg.real/k0)*(ts*A).abs()**2)[inside].sum()*dk2/(ns*4*math.pi/kw**2)).item()
    # fraction coming from u in (n_s, NA): evanescent in water = supercritical (SAF) light
    sup = inside & (ur > ns)
    Pc_saf = ((2*math.pi)**2*((kzg.real/k0)*(ts*A).abs()**2)[sup].sum()*dk2/(ns*4*math.pi/kw**2)).item()
    print(f"  Np={Np:5d}: (a) literal {Pa:.4f} | (b) sqrt(cos_imm), no t: {Pb:.4f} | (c) with t_s: {Pc:.4f} (of which u>n_s: {Pc_saf:.4f})")
print(f"  index-matched formula would give (1-cos)/2 with n=1.518: {(1-math.sqrt(1-(NA/ng)**2))/2:.4f}; with NA->n_s in water: 0.5")

print("\n=== defocus 'Propagate(z_ref -> focus)' in the sample medium for u > n_s (emitter 1 um deep, focus 1 um deeper) ===")
ur, dk2 = grid(NA, 512); kzw = kz(ns, ur)
for dz in [-0.5, -1.0, -2.0]:
    H = torch.exp(1j*kzw*dz)
    print(f"  Delta = {dz:+.1f} um: max |H| over pupil = {H.abs().max().item():.3e} (physical: <= 1; evanescent part should decay emitter->coverslip only)")
