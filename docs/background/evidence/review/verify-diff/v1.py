import torch, math
print(torch.__version__)
for dev in ("cpu","cuda"):
    x = torch.tensor([-6.28,-5.0,-4.0, 6.2, 9.0], device=dev)
    print(dev, "ndtr fp32", torch.special.ndtr(x).tolist(), "fp64", torch.special.ndtr(x.double()).tolist())
    print(dev, "0.5*erfc(-x/sqrt2) fp32", (0.5*torch.erfc(-x/math.sqrt(2))).tolist())
# textbook truncated normal, window 6.2..9 sigma, fp32
mu = torch.tensor(0.0, requires_grad=True); sig = torch.tensor(1.0)
u = torch.rand(10000)
a, b = 6.2, 9.0
Pa = torch.special.ndtr((a-mu)/sig); Pb = torch.special.ndtr((b-mu)/sig)
x = mu + sig*torch.special.ndtri(Pa + u*(Pb-Pa))
print("tail window fp32 nonfinite:", (~torch.isfinite(x)).sum().item(), "Pa,Pb", Pa.item(), Pb.item())
# lower-tail mirrored version fp32 using erfc
za, zb = -(b-mu)/sig, -(a-mu)/sig   # mirror
Pa2 = 0.5*torch.erfc(-za/math.sqrt(2)); Pb2 = 0.5*torch.erfc(-zb/math.sqrt(2))
x2 = mu - sig*torch.special.ndtri(Pa2 + u*(Pb2-Pa2))
print("mirrored fp32 nonfinite:", (~torch.isfinite(x2)).sum().item(), "range", x2.min().item(), x2.max().item())
# _standard_gamma at 0
c = torch.tensor([0.0, 1.0, 5.0], requires_grad=True)
torch.manual_seed(0)
y = torch._standard_gamma(c)
y.sum().backward()
print("std_gamma", y.tolist(), c.grad.tolist())
c.grad=None
n = c
safe = torch.where(n>0, n, torch.ones_like(n))
y2 = torch.where(n>0, torch._standard_gamma(safe), torch.zeros_like(n))
y2.sum().backward(); print("double-where", y2.tolist(), c.grad.tolist())
c.grad=None
y3 = torch.where(n>0, torch._standard_gamma(n), torch.zeros_like(n)); y3.sum().backward(); print("single where", c.grad.tolist())
import inspect
print(torch._standard_gamma.__doc__)
g = torch.Generator().manual_seed(1)
try:
    print(torch._standard_gamma(torch.tensor([2.0]), generator=g))
except Exception as e: print("gen kw fails", e)
# normal zero-grad overloads
mu = torch.tensor([1.0,2.0], requires_grad=True)
for name, f in [("normal(mu,2.0)", lambda: torch.normal(mu, 2.0)), ("normal(t,mu)", lambda: torch.normal(torch.zeros(2), mu)),]:
    mu.grad=None; f().sum().backward(); print(name, mu.grad.tolist())
import torch.distributions as D
print("Normal.sample grad_fn", D.Normal(mu,1.).sample().grad_fn)
print("default validate", D.Distribution._validate_args)
