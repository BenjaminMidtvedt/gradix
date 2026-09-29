import math, torch, importlib.util, sys
spec = importlib.util.spec_from_file_location("t3", "t3_multislice.py")
src = open("t3_multislice.py").read().split("# (i) identity check")[0]
ns = {}; exec(src, ns)
march, scalar_coeffs, S_scalar, k, lam0, n_m, dev = ns["march"], ns["scalar_coeffs"], ns["S_scalar"], ns["k"], ns["lam0"], ns["n_m"], ns["dev"]
def err(a, dn, NA, dz, obliq):
    A, FX, FY, kz = march(a, dn, dz=dz, obliq=obliq)
    sel = torch.sqrt(FX**2+FY**2)*lam0 <= NA
    m = 1 + dn/n_m; c = scalar_coeffs(a, m, int(k*a*m + 4*(k*a*m)**(1/3) + 8))
    Aex = torch.tensor(-S_scalar((kz.real[sel]/k).cpu().numpy(), c), device=dev)/(2*math.pi*k*kz.real[sel])
    return (torch.linalg.norm(A[sel]-Aex)/torch.linalg.norm(Aex)).item()
lm = lam0/n_m
for dn, x in [(0.05, 10.0), (0.05, 3.0), (0.1, 10.0)]:
    a = x/k
    print(f"dn={dn} x={x}: " + " | ".join(f"dz=lam_m/{int(round(lm/dz))}: plain {err(a, dn, 0.8, dz, False):.3f} obliq {err(a, dn, 0.8, dz, True):.3f}" for dz in [lm/2, lm/4, lm/8]))
