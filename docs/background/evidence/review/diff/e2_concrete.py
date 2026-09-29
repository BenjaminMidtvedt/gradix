"""E2: straight-through Concrete (binary Gumbel-sigmoid), tau = 0.5, for presence ~ Bernoulli(p).
(1) Linear loss: exact expected-gradient ratio E[ST grad]/true, by quadrature over the logistic noise.
(2) The [M-D Exp B] quadratic count loss, re-run over p and K (incl. K = 8 of example (a), p in the
    example's learnable bounds [0.05, 0.95]), against the analytic gradient and score + LOO.
(3) Does the hard sample of ST-Concrete coincide with the 'u < p' presence rule of sec 6.5 for the same u?
"""
import math
import torch

dev = "cuda"
torch.set_printoptions(precision=4)

print("(1) linear loss L = sum(presence * w): E[ST-Concrete grad] / true grad (quadrature, fp64)")
g = torch.linspace(-40, 40, 400001, dtype=torch.float64)
dens = torch.sigmoid(g) * torch.sigmoid(-g)  # logistic density
dg = (g[1] - g[0]).item()
print(f"   {'p':>6} " + " ".join(f"{'tau=' + str(t):>10}" for t in (0.1, 0.25, 0.5, 1.0)))
for p in (0.02, 0.05, 0.25, 0.5, 0.75, 0.95):
    l = math.log(p / (1 - p))
    row = []
    for tau in (0.1, 0.25, 0.5, 1.0):
        z = (l + g) / tau
        # d soft / dp = sigmoid'(z)/tau * dlogit/dp ; dlogit/dp = 1/(p(1-p)); true d E[b]/dp = 1
        est = ((torch.sigmoid(z) * torch.sigmoid(-z) / tau) * dens).sum().item() * dg / (p * (1 - p))
        row.append(f"{est:10.3f}")
    print(f"   {p:6.2f} " + " ".join(row))

print("(2) quadratic count loss L = (sum_i b_i w_i - target)^2, w ~ U(0.5,1.5)")
gen = torch.Generator(device=dev).manual_seed(2)


def run(K, p0, target, reps=200_000):
    w = 0.5 + torch.rand(reps, K, device=dev, generator=gen)
    sw, sw2 = w.sum(-1), (w ** 2).sum(-1)
    true = (2 * (p0 * sw - target) * sw + (1 - 2 * p0) * sw2).mean().item()
    u = torch.rand(reps, K, device=dev, generator=gen).clamp(1e-7, 1 - 1e-7)
    logistic = torch.log(u) - torch.log1p(-u)
    out = {}
    for tau in (0.5, 1.0):
        p = torch.tensor(p0, device=dev, requires_grad=True)
        soft = torch.sigmoid((torch.log(p) - torch.log1p(-p) + logistic) / tau)
        hard = (soft > 0.5).float() + soft - soft.detach()
        # per-replicate gradients via manual chain rule (loss is separable per replicate)
        for nm, pres in ((f"ST tau={tau}", hard), (f"soft tau={tau}", soft)):
            C = (pres * w).sum(-1)
            dLdC = 2 * (C.detach() - target)
            dsoft = (soft * (1 - soft) / tau / (p0 * (1 - p0))).detach()
            per = dLdC * (dsoft * w).sum(-1)
            out[nm] = per
    b = (u < p0).float()  # NOTE: rule 'u < p' (sec 6.5) ...
    L = ((b * w).sum(-1) - target) ** 2
    base = (L.sum() - L) / (reps - 1)
    per = (L - base) * (b / p0 - (1 - b) / (1 - p0)).sum(-1)
    out["score+LOO"] = per
    return true, out, sw.mean().item()


for K, p0, target in ((64, 0.25, 20.0), (8, 0.6, 3.0), (8, 0.6, 6.0), (8, 0.05, 1.0), (8, 0.95, 6.0), (64, 0.05, 8.0)):
    true, out, _ = run(K, p0, target)
    s = " | ".join(f"{k}: {v.mean().item():8.2f} (se {v.std().item() / math.sqrt(v.numel()):.2f})" for k, v in out.items())
    print(f"   K={K:>2} p={p0:4.2f} t={target:4.1f} true={true:8.2f} || {s}")

print("(3) ST-Concrete hard sample vs 'u < p' presence rule, same counter uniform u")
u = torch.rand(1_000_000, device=dev, generator=gen)
p0 = 0.3
hard_concrete = ((math.log(p0 / (1 - p0)) + torch.log(u) - torch.log1p(-u)) > 0)
rule = u < p0
print(f"   P(hard)={hard_concrete.float().mean().item():.4f}, P(u<p)={rule.float().mean().item():.4f}, "
      f"agreement={(hard_concrete == rule).float().mean().item():.4f}")
