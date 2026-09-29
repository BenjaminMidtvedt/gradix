# Numerical failure mode: pupil support |u| <= NA with NA > n_medium (oil objective, water sample).
# Plan rule (sec 4.1): collection x sqrt(cos theta) = sqrt(k_z/k_m), k_z in the medium where A is defined;
# far-field -> spectrum A = -S/(2 pi k_m k_z) ... ; k_z = 2 pi sqrt((n/l)^2 - f^2), Im k_z >= 0.
import torch, math
torch.manual_seed(0)
lam, n_m, NA = 0.68, 1.33, 1.45
for dev in (["cpu","cuda"] if torch.cuda.is_available() else ["cpu"]):
    for N in (64, 128, 256):
        u = torch.linspace(-NA, NA, N, dtype=torch.float32, device=dev)
        uy, ux = torch.meshgrid(u, u, indexing="ij")
        ur = torch.sqrt(ux**2 + uy**2)
        sup = ur <= NA
        f = ur[sup] / lam                                   # cycles/um
        kz2 = (n_m/lam)**2 - f**2
        kz = 2*math.pi*torch.sqrt(kz2.to(torch.complex64))  # principal branch: Im>=0 for negative arg
        km = 2*math.pi*n_m/lam
        coll = torch.sqrt(kz/km)                            # sqrt(cos theta)
        inv = 1.0/kz                                        # far-field -> angular spectrum factor
        frac_evan = (f > n_m/lam).float().mean().item()
        print(f"{dev} N={N}: support samples={sup.sum().item()}, frac |u|>n_m = {frac_evan:.3f}, "
              f"max|1/kz| / (1/kz at axis) = {(inv.abs().max()*(2*math.pi*n_m/lam)).item():.1f}, "
              f"sqrt(cos) complex-valued samples = {(coll.imag.abs()>0).sum().item()}, "
              f"nan/inf in 1/kz: {(~torch.isfinite(inv)).sum().item()}")
# gradient failure: sqrt at exactly the critical radius (e.g. learnable NA or n_m hitting k_z = 0)
n = torch.tensor(1.33, requires_grad=True)
fr = torch.tensor(1.33/lam)
kz = 2*math.pi*torch.sqrt((n/lam)**2 - fr**2)
kz.backward()
print("d kz / d n at k_z = 0:", n.grad.item())
n2 = torch.tensor(1.33, requires_grad=True)
kzc = 2*math.pi*torch.sqrt(((n2/lam)**2 - fr**2).to(torch.complex64))
(1.0/kzc).abs().backward()
print("d |1/kz| / d n at k_z = 0 (complex path):", n2.grad)
