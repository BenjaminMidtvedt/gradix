"""E3c: tail-safe truncated Normal: open-interval uniforms, erfc-based CDF in the tail nearest the window, fp64."""
import torch, math
dev = "cuda"
g = torch.Generator(device=dev).manual_seed(0)
SQ2 = math.sqrt(2.0)
def lower_cdf(z):  # Phi(z), accurate for z << 0
    return 0.5 * torch.special.erfc(-z / SQ2)
def trunc_fixed(mu, sig, a, b, u):
    za, zb = (a - mu) / sig, (b - mu) / sig
    flip = (za + zb) > 0                     # window right of the mean: mirror into the lower tail
    lo = torch.where(flip, -zb, za); hi = torch.where(flip, -za, zb)
    A, B = lower_cdf(lo), lower_cdf(hi)
    z = torch.special.ndtri(A + u * (B - A))
    z = torch.maximum(torch.minimum(z, hi), lo)  # guards fp rounding of the *uniform* only (not a sample clamp)
    return mu + sig * torch.where(flip, -z, z)
cases = [(5.0, 0.5, -1.86, 1.86, "both bounds in lower tail (-13.7..-6.3 sigma)"),
         (0.0, 0.1, 0.62, 0.9, "upper-tail window (6.2..9 sigma)"),
         (0.0, 1.0, -30.0, 30.0, "wide user truncation")]
N = 1_000_000
k = torch.randint(0, 2**24, (N,), device=dev, generator=g, dtype=torch.int64)
k[0] = 0
u = (k.double() + 0.5) * 2.0**-24          # open interval: never 0 or 1
for mu0, s0, a, b, desc in cases:
    mu = torch.tensor(mu0, device=dev, dtype=torch.float64, requires_grad=True)
    sig = torch.tensor(s0, device=dev, dtype=torch.float64, requires_grad=True)
    x = trunc_fixed(mu, sig, torch.tensor(a, device=dev, dtype=torch.float64), torch.tensor(b, device=dev, dtype=torch.float64), u)
    gm, gs = torch.autograd.grad(x.sum(), (mu, sig))
    print(f"{desc:<48} nonfinite {(~torch.isfinite(x)).sum().item()}  range [{x.min().item():.4f}, {x.max().item():.4f}]  "
          f"dE[x]/dmu {gm.item()/N:.4f}  dE[x]/dsig {gs.item()/N:.4f}")
# the u == 0 poisoning with the textbook formula, fp32
mu = torch.tensor(0.0, device=dev, requires_grad=True)
u32 = torch.rand(N, device=dev, generator=g); u32[0] = 0.0
x = mu + math.sqrt(2) * torch.erfinv(2 * u32 - 1)
loss = x[torch.isfinite(x)].square().mean()
(gm,) = torch.autograd.grad(loss, mu)
print("textbook erfinv, one u==0 among 1e6, loss masks non-finite:", "grad =", gm.item())
