import math, torch
# Rayleigh polarizability of a 10 nm-radius particle (m=1.5 rel.), in SI metres vs micrometres, float32
for unit, a in [("m", 10e-9), ("um", 10e-3)]:
    a = torch.tensor(a, dtype=torch.float32)
    m2 = 1.5**2
    alpha = 4*math.pi*a**3*(m2-1)/(m2+2)
    k = torch.tensor(2*math.pi*1.33/(532e-9 if unit=="m" else 0.532), dtype=torch.float32)
    csca = k**4*alpha**2/(6*math.pi)
    csca_alt = (k**2*alpha)**2/(6*math.pi)
    print(f"{unit:>2}: alpha={alpha.item():.3e}  alpha^2={(alpha**2).item():.3e}  Csca(k^4*a^2 order)={csca.item():.3e}  Csca(reordered)={csca_alt.item():.3e}")
# Learnable position with Adam lr=1e-2: one step moves by ~lr regardless of scale
for unit, x0 in [("m", 1e-6), ("um", 1.0)]:
    p = torch.nn.Parameter(torch.tensor(x0)); opt = torch.optim.Adam([p], lr=1e-2)
    (p*3).backward(); opt.step()
    print(f"{unit:>2}: Adam lr=1e-2 step moved position by {abs(p.item()-x0):.2e} {unit} (relative {abs(p.item()-x0)/x0:.1e})")
