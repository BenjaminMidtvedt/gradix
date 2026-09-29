"""E1: scaled straight-through Poisson estimator as written in the plan (sec 6.4):
   y = lam + sqrt(lam) * stopgrad((n - lam)/sqrt(lam))
Checks: (a) lam = 0 pixels (NaN forward/backward); (b) forward exactness (integer?);
(c) bias vs the exact derivative d/dlam E[L(n)] = E[L(n+1) - L(n)] for nonlinear losses at low lam;
(d) a linear-functional probe cannot see plain-ST's failure on variance losses.
"""
import math
import torch

torch.set_printoptions(precision=5)
dev = "cuda"
g = torch.Generator(device=dev).manual_seed(0)


def scaled_st_plan(lam, n):
    s = lam.sqrt()
    return lam + s * ((n - lam) / s).detach()


def scaled_st_fixed(lam, n):
    # exact forward (y == n bitwise), identical gradient (n+lam)/(2 lam) = 1 + eps/(2 sqrt lam), 1/2 at lam = 0
    lam_d = lam.detach()
    gfac = torch.where(lam_d > 0, (n + lam_d) / (2 * lam_d.clamp_min(torch.finfo(lam.dtype).tiny)),
                       torch.full_like(lam_d, 0.5))
    return n + (lam - lam_d) * gfac


print("(a) lam = 0 pixel (e.g. no background, outside every emitter ROI)")
lam = torch.tensor([0.0, 0.5, 3.0], device=dev, requires_grad=True)
n = torch.poisson(lam.detach(), generator=g)
y = scaled_st_plan(lam, n)
y.sum().backward()
print("   plan formula forward:", y.detach().tolist(), " grad:", lam.grad.tolist())
lam.grad = None
y2 = scaled_st_fixed(lam, n)
y2.sum().backward()
print("   fixed formula forward:", y2.detach().tolist(), " grad:", lam.grad.tolist())

print("(b) forward exactness: fraction of samples where plan-formula y != n (fp32)")
for L0 in (0.3, 7.0, 300.0, 3000.0, 3e4):
    lam = torch.full((1_000_000,), L0, device=dev)
    lam = lam * (1 + 0.01 * torch.rand(lam.shape, device=dev, generator=g))
    n = torch.poisson(lam, generator=g)
    y = scaled_st_plan(lam, n)
    frac = (y != n).float().mean().item()
    print(f"   lam~{L0:>7}: frac(y != n) = {frac:.3f}, max|y-n| = {(y - n).abs().max().item():.2e}")

print("(c) bias of scaled-ST vs exact d/dlam E[L(n)] (exact sums over the Poisson pmf, fp64)")


def pmf(lam, kmax):
    k = torch.arange(kmax, dtype=torch.float64)
    return k, torch.exp(k * math.log(lam) - lam - torch.lgamma(k + 1)) if lam > 0 else (k == 0).double()


losses = {
    "anscombe 2sqrt(y+3/8)": (lambda y: 2 * torch.sqrt(y + 0.375), lambda y: 1 / torch.sqrt(y + 0.375)),
    "log1p(y)": (lambda y: torch.log1p(y), lambda y: 1 / (1 + y)),
    "exp(-y/2)": (lambda y: torch.exp(-y / 2), lambda y: -0.5 * torch.exp(-y / 2)),
    "(y-lam0)^2 (quadratic)": (lambda y: (y - 5.0) ** 2, lambda y: 2 * (y - 5.0)),
}
print(f"   {'loss':>24} " + " ".join(f"{'lam=' + str(l):>16}" for l in (0.2, 1.0, 3.0, 10.0, 40.0)))
for name, (L, dL) in losses.items():
    row = []
    for lam in (0.2, 1.0, 3.0, 10.0, 40.0):
        k, p = pmf(lam, 400)
        true = (p * (L(k + 1) - L(k))).sum().item()
        st = (p * dL(k) * (k + lam) / (2 * lam)).sum().item()
        row.append(f"{st / true - 1:+16.3%}")
    print(f"   {name:>24} " + " ".join(row))
print("   (entries = relative bias of the scaled-ST expected gradient)")

print("(c2) per-pixel variance of the scaled-ST derivative dy/dlam = (n+lam)/(2lam): Var = 1/(4 lam)")
for lam in (0.01, 0.1, 1.0, 10.0):
    k, p = pmf(lam, 200)
    d = (k + lam) / (2 * lam)
    m = (p * d).sum()
    v = (p * (d - m) ** 2).sum()
    print(f"   lam={lam:>5}: E[dy/dlam]={m.item():.4f}  Var={v.item():.3g}  (1/(4lam)={1 / (4 * lam):.3g})")

print("(d) gradient_report probe = backward of a random linear functional <w, y>")
M = 4096
w = torch.randn(M, device=dev, generator=g)
theta = torch.tensor(40.0, device=dev, requires_grad=True)
lam = theta.expand(M)
n = torch.poisson(lam.detach(), generator=g)
y_plain = lam + (n - lam).detach()
(gp,) = torch.autograd.grad((w * y_plain).sum(), theta)
y_var_loss = (y_plain.var() - 50) ** 2
(gv,) = torch.autograd.grad(y_var_loss, theta)
print(f"   plain-ST: probe |g| = {gp.abs().item():.3g} (non-zero -> 'ok'); variance-loss grad = {gv.item():.3g}")
