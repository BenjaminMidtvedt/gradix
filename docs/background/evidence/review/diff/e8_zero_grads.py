"""E8: which sampling calls silently return zero / no gradient (plan bans only torch.poisson,
torch.normal(Tensor, Tensor) and torch.bernoulli)."""
import torch, torch.distributions as D
dev = "cuda"
def check(label, f):
    mu = torch.tensor([3.0, 4.0], device=dev, requires_grad=True)
    try:
        y = f(mu)
        if not y.requires_grad:
            print(f"{label:<44} output has no grad_fn (graph silently cut)"); return
        (gr,) = torch.autograd.grad(y.sum(), mu, allow_unused=True)
        print(f"{label:<44} grad = {None if gr is None else gr.tolist()}")
    except Exception as e:
        print(f"{label:<44} {type(e).__name__}: {str(e).splitlines()[0][:90]}")
check("torch.poisson(mu)", lambda m: torch.poisson(m))
check("torch.normal(mu, std_tensor)", lambda m: torch.normal(m, torch.ones_like(m)))
check("torch.normal(mu, 2.0)   [Tensor, float]", lambda m: torch.normal(m, 2.0))
check("torch.normal(0.0, mu)   [float, Tensor]", lambda m: torch.normal(torch.zeros_like(m), m) * 1.0)
check("torch.bernoulli(mu/10)", lambda m: torch.bernoulli(m / 10))
check("torch.binomial(count, mu/10)", lambda m: torch.binomial(torch.full_like(m, 10.0), m / 10))
check("D.Normal(mu,1).sample()", lambda m: D.Normal(m, 1.0).sample())
check("D.Gamma(mu,1).sample()", lambda m: D.Gamma(m, 1.0).sample())
check("D.Normal(mu,1).rsample()", lambda m: D.Normal(m, 1.0).rsample())
check("D.Gamma(mu,1).rsample()", lambda m: D.Gamma(m, 1.0).rsample())
check("mu.new_empty(2).normal_() * mu", lambda m: m * m.new_empty(2).normal_())
# Gamma at concentration 0 (EMCCD: Gamma(n_electrons, G) with n = 0)
n = torch.tensor([0.0, 1.0, 5.0], device=dev, requires_grad=True)
x = torch._standard_gamma(n)
(gn,) = torch.autograd.grad(x.sum(), n)
print("_standard_gamma(conc=[0,1,5]) ->", x.tolist(), " d/dconc ->", gn.tolist())
import inspect
print("torch._standard_gamma signature:", torch._standard_gamma.__doc__ if torch._standard_gamma.__doc__ else "(no doc)")
try:
    gg = torch.Generator(device=dev).manual_seed(0)
    torch._standard_gamma(torch.ones(2, device=dev), generator=gg); print("_standard_gamma accepts generator=")
except Exception as e:
    print("_standard_gamma generator=:", type(e).__name__, str(e)[:100])
try:
    D.Gamma(torch.ones(2, device=dev), 1.0).rsample(generator=gg)
except Exception as e:
    print("D.Gamma.rsample(generator=...):", type(e).__name__, str(e)[:100])
