# Collected power of an isotropic scalar emitter deep in water (n_s=1.33) through a pupil support |u|<=NA=1.45:
# plan rule: |A|^2 * cos -> density 1/(k_m k_z) per d^2k (propagating part only; evanescent part decays with depth).
# Exact (no Fresnel): 0.5 of 4pi (whole forward hemisphere). Check grid-sampled midpoint sum vs exact.
import torch, math
lam, n_s, NA = 0.68, 1.33, 1.45
km = 2*math.pi*n_s/lam
for N in (64, 96, 128, 192, 256, 384, 512):
    u = (torch.arange(N, dtype=torch.float64) - N/2 + 0.5) * (2*NA/N)   # cell centres
    du = 2*NA/N
    uy, ux = torch.meshgrid(u, u, indexing="ij")
    ur = torch.sqrt(ux**2 + uy**2)
    k = 2*math.pi*ur/lam; dk2 = (2*math.pi*du/lam)**2
    prop = (ur < n_s) & (ur <= NA)
    kz = torch.sqrt(torch.clamp(km**2 - k**2, min=0))
    P = (dk2/(km*kz[prop])).sum().item()/(4*math.pi)
    # clipped at n_s(1-delta) with delta = one cell
    d = du/n_s
    clip = ur < n_s*(1-d)
    Pc = (dk2/(km*kz[clip])).sum().item()/(4*math.pi)
    print(f"N={N}: midpoint sum = {P:.4f} (exact 0.5000, err {100*(P-0.5)/0.5:+.1f} %), clipped 1 cell = {Pc:.4f} (err {100*(Pc-0.5)/0.5:+.1f} %), max w/w0 = {(km/kz[prop]).max().item():.0f}")
