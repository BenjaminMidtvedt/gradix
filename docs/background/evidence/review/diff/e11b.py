import math, torch
dev = "cuda"
for label, val in (("cos_t exactly 1", 1.0), ("cos_t = 1 + 1ulp (rounding)", 1.0000001)):
    t = torch.tensor(0.0, device=dev, requires_grad=True)      # learnable tilt; cos_t = c(val) + 0*t
    x = torch.tensor(val, device=dev) + 0.0 * t
    th = torch.acos(x.clamp(-1, 1))
    (g,) = torch.autograd.grad(th, t)
    print(f"{label:<30} theta={th.item():.3e} d theta/d tilt = {g.item()}")
R = torch.tensor(0.3, device=dev, requires_grad=True)
q = torch.tensor([0.0, 1.0], device=dev)
small = q * R < 1e-2
F_naive = 4 * math.pi * (torch.sin(q * R) - q * R * torch.cos(q * R)) / q**3
F = torch.where(small, 4 / 3 * math.pi * R**3, F_naive)
(gR,) = torch.autograd.grad(F.sum(), R)
qs = torch.where(small, torch.ones_like(q), q)
F2 = torch.where(small, 4 / 3 * math.pi * R**3 * (1 - (q * R)**2 / 10), 4 * math.pi * (torch.sin(qs * R) - qs * R * torch.cos(qs * R)) / qs**3)
(gR2,) = torch.autograd.grad(F2.sum(), R)
print("ball form factor at q=0, single where:", gR.item(), "| double-where:", gR2.item())
