"""E11: gradients at the forward-scattering direction k = k_in (pupil-support sample f = 0 under on-axis
illumination / a centre quadrature node). Mie: S(theta) from a 1-D theta table; born_ff: F(|k - k_in|)."""
import math, torch
dev = "cuda"
lam, n_m, NA = 0.532, 1.33, 0.8
km = 2 * math.pi * n_m / lam
f = torch.tensor([[0.0, 0.0], [0.3, 0.0], [0.0, 1.2]], device=dev)          # pupil-support samples (cycles/um)
# illumination direction depends on a learnable (condenser NA / LED position / tilt); centre node has u = s * 0
s = torch.tensor(0.4, device=dev, requires_grad=True)
u_node = s * torch.tensor([0.0, 0.0], device=dev)                           # centre node: 0 for any NA
kin_perp = 2 * math.pi * u_node / lam
kin = torch.cat([kin_perp, torch.sqrt(km**2 - kin_perp.square().sum()).reshape(1)])
kp = 2 * math.pi * f
k = torch.cat([kp, torch.sqrt(km**2 - kp.square().sum(-1, keepdim=True))], -1)
theta_tab = torch.linspace(0, math.pi, 1024, device=dev)
S_tab = torch.cos(3 * theta_tab) + 2.0                                     # stand-in for |S1| table
def interp(th):
    x = th / math.pi * 1023; i = x.floor().clamp(0, 1022).long(); w = x - i
    return S_tab[i] * (1 - w) + S_tab[i + 1] * w
cos_t = (k @ kin) / km**2
theta = torch.acos(cos_t.clamp(-1, 1))
A = interp(theta)
(gs,) = torch.autograd.grad(A.sum(), s)
print("Mie-like S(theta) via acos + theta table: cos_t =", cos_t.tolist(), " d/dNA =", gs.item())
# tabulate on mu = cos(theta) instead
mu_tab = torch.linspace(-1, 1, 1024, device=dev); S_mu = torch.cos(3 * torch.acos(mu_tab)) + 2.0
def interp_mu(mu):
    x = (mu + 1) / 2 * 1023; i = x.floor().clamp(0, 1022).long(); w = x - i
    return S_mu[i] * (1 - w) + S_mu[i + 1] * w
(gs2,) = torch.autograd.grad(interp_mu(cos_t).sum(), s)
print("same, table indexed by mu = cos(theta):                       d/dNA =", gs2.item())
# Born form factor of a ball at q = |k - k_in| = 0
R = torch.tensor(0.3, device=dev, requires_grad=True)
q = (k - kin).norm(dim=-1)
F_naive = 4 * math.pi * (torch.sin(q * R) - q * R * torch.cos(q * R)) / q**3
small = q * R < 1e-2
F_where = torch.where(small, 4 / 3 * math.pi * R**3 * torch.ones_like(q), F_naive)
(gR,) = torch.autograd.grad(F_where.sum(), R)
print("born_ff ball F(q) with single torch.where series branch at q=0: F =", F_where.tolist(), " dF/dR =", gR.item())
qs = torch.where(small, torch.ones_like(q), q)
F_safe = torch.where(small, 4 / 3 * math.pi * R**3 * (1 - (q * R)**2 / 10), 4 * math.pi * (torch.sin(qs * R) - qs * R * torch.cos(qs * R)) / qs**3)
(gR2,) = torch.autograd.grad(F_safe.sum(), R)
print("double-where (safe denominator in the unused branch):           dF/dR =", gR2.item())
