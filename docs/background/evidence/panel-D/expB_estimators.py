"""Exp B: bias/variance of gradient estimators for (1) Poisson shot noise and (2) relaxed discrete counts.

(1) y ~ Poisson(mu) on M pixels. Losses: L1 = (mean(y) - 50)^2 (mean-sensitive),
    L2 = (var(y) - 50)^2 (noise-level-sensitive). True gradient by finite differences of E[L] with
    common random numbers is noisy for Poisson, so use analytic: E[L1]' = 2(mu-50) + 1/M,
    E[L2]' ~= 2(mu-50) (var(y) ~ mu, small-sample corrections negligible at M=4096).
(2) K=64 slots, presence ~ Bernoulli(p). count = sum(presence * w) with w ~ U(0.5,1.5) brightness.
    L = (count - 20)^2. True dE[L]/dp computed analytically.
"""
import torch

dev = "cuda"
torch.manual_seed(0)
M, mu0, reps = 4096, 40.0, 2000


def poisson_estimators(mu, g):
    lam = mu.expand(reps, M)
    n = torch.poisson(lam.detach(), generator=g)
    eps = torch.randn(reps, M, device=dev, generator=g)
    out = {}
    out["gauss_reparam"] = lam + lam.sqrt() * eps
    out["plain_ST"] = lam + (n - lam).detach()
    out["scaled_ST"] = lam + lam.sqrt() * ((n - lam) / lam.sqrt()).detach()
    out["_n"] = n
    return out


def run_poisson():
    g = torch.Generator(device=dev).manual_seed(1)
    print("Poisson shot noise, mu=40, M=4096 pixels, 2000 replicates")
    print(f"{'estimator':>16} {'loss':>5} {'grad mean':>10} {'grad std':>10} {'true':>8}")
    for lname, lossf, true in (
        ("L1", lambda y: (y.mean(-1) - 50) ** 2, 2 * (mu0 - 50) + 1 / M),
        ("L2", lambda y: (y.var(-1) - 50) ** 2, 2 * (mu0 - 50)),
    ):
        for est in ("gauss_reparam", "plain_ST", "scaled_ST", "score_fn", "score_fn_baseline"):
            mu = torch.tensor(mu0, device=dev, requires_grad=True)
            ys = poisson_estimators(mu, g)
            if est.startswith("score"):
                n = ys["_n"]
                L = lossf(n)  # [reps]
                logp = (n * torch.log(mu) - mu).sum(-1)  # Poisson log-prob, up to const
                b = L.mean().detach() if est.endswith("baseline") else 0.0
                # leave-one-out baseline to keep it unbiased
                if est.endswith("baseline"):
                    b = (L.sum() - L).detach() / (reps - 1)
                surrogate = ((L - b).detach() * logp)
                grads = torch.autograd.grad(surrogate.sum(), mu, is_grads_batched=False)
                # per-replicate grads: d/dmu logp = sum(n/mu - 1)
                per = ((L - b).detach() * (n / mu - 1).sum(-1)).detach()
            else:
                L = lossf(ys[est])
                # per-replicate gradient via autograd of sum (independent replicates)
                (gsum,) = torch.autograd.grad(L.sum(), mu)
                # individual grads: recompute via jacobian trick: grad of each L wrt mu
                per = torch.autograd.functional.jacobian(
                    lambda m: lossf(poisson_like(est, m, ys)), mu
                )
            print(f"{est:>16} {lname:>5} {per.mean().item():10.3f} {per.std().item():10.3f} {true:8.2f}")


def poisson_like(est, m, ys):
    n = ys["_n"]
    lam = m.expand(reps, M)
    if est == "gauss_reparam":
        eps = (ys["gauss_reparam"].detach() - mu0) / mu0**0.5
        return lam + lam.sqrt() * eps
    if est == "plain_ST":
        return lam + (n - lam).detach()
    if est == "scaled_ST":
        return lam + lam.sqrt() * ((n - lam) / lam.sqrt()).detach()
    raise ValueError


def run_counts():
    K, p0, target, reps2 = 64, 0.25, 20.0, 4000
    g = torch.Generator(device=dev).manual_seed(2)
    w = 0.5 + torch.rand(reps2, K, device=dev, generator=g)  # brightness, not learned
    # true: count = sum b_i w_i, b_i~Bern(p). E[(C-t)^2] = Var + (E-t)^2
    # E = p*sum w ; Var = p(1-p) sum w^2 ; per-replicate w differs -> average over w
    sw, sw2 = w.sum(-1), (w**2).sum(-1)
    true = (2 * (p0 * sw - target) * sw + (1 - 2 * p0) * sw2).mean().item()
    print(f"\nRelaxed counts: K={K} slots, p={p0}, target={target}, {reps2} replicates; true dE[L]/dp = {true:.2f}")
    print(f"{'estimator':>22} {'grad mean':>10} {'grad std':>10}")
    u = torch.rand(reps2, K, device=dev, generator=g).clamp(1e-6, 1 - 1e-6)
    logistic = torch.log(u) - torch.log1p(-u)
    for tau in (0.1, 0.5, 1.0):
        for hard in (False, True):
            p = torch.tensor(p0, device=dev, requires_grad=True)
            logit = torch.log(p) - torch.log1p(-p)
            soft = torch.sigmoid((logit + logistic) / tau)
            pres = soft
            if hard:
                pres = (soft > 0.5).float() + soft - soft.detach()
            C = (pres * w).sum(-1)
            per = torch.autograd.functional.jacobian(
                lambda pp: (((torch.sigmoid(((torch.log(pp) - torch.log1p(-pp)) + logistic) / tau)
                              if not hard else
                              ((torch.sigmoid(((torch.log(pp) - torch.log1p(-pp)) + logistic) / tau) > 0.5).float()
                               + torch.sigmoid(((torch.log(pp) - torch.log1p(-pp)) + logistic) / tau)
                               - torch.sigmoid(((torch.log(pp) - torch.log1p(-pp)) + logistic) / tau).detach()))
                             * w).sum(-1) - target) ** 2,
                p,
            )
            nm = f"concrete tau={tau}{' ST' if hard else ''}"
            print(f"{nm:>22} {per.mean().item():10.2f} {per.std().item():10.2f}")
    # score function with leave-one-out baseline
    p = torch.tensor(p0, device=dev)
    b = (u < p).float()
    L = ((b * w).sum(-1) - target) ** 2
    base = (L.sum() - L) / (reps2 - 1)
    dlogp = (b / p - (1 - b) / (1 - p)).sum(-1)
    per = (L - base) * dlogp
    print(f"{'score fn + LOO baseline':>22} {per.mean().item():10.2f} {per.std().item():10.2f}")
    # expected-value (mean-field) surrogate: presence = p (deterministic), no noise
    pp = torch.tensor(p0, device=dev, requires_grad=True)
    per = torch.autograd.functional.jacobian(lambda q: ((q * w).sum(-1) - target) ** 2, pp)
    print(f"{'mean-field (pres=p)':>22} {per.mean().item():10.2f} {per.std().item():10.2f}")


run_poisson()
run_counts()
