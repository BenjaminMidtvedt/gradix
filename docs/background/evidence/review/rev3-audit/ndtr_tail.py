"""Revision-3 audit: the cookbook's truncated-normal recipe (architecture.md §10.3, Appendix F.2).

1. torch.special.ndtr computes 1/2 (1 + erf(z / sqrt 2)) and loses the lower tail even in fp64.
2. The recipe (erfc form, windows right of the mean mirrored) matches scipy.stats.truncnorm
   in both tails and gives finite gradients to the location and scale.

Run from the repo root: uv run python docs/background/evidence/review/rev3-audit/ndtr_tail.py
"""

import math

import torch
from scipy import special, stats

dev = "cuda" if torch.cuda.is_available() else "cpu"
print(f"torch {torch.__version__} on {dev}")

z = torch.tensor([-6.0, -8.0, -9.0, -30.0], dtype=torch.float64, device=dev)
torch_ndtr = torch.special.ndtr(z).cpu().numpy()
erfc_form = (0.5 * torch.special.erfc(-z / math.sqrt(2))).cpu().numpy()
exact = special.ndtr(z.cpu().numpy())
for zi, t, e, x in zip(z.tolist(), torch_ndtr, erfc_form, exact, strict=True):
    print(f"z = {zi:6.1f}: ndtr {t:.4e}  erfc form {e:.4e}  scipy {x:.4e}  ndtr/scipy {t / x:.4f}")

B, N = 256, 64
g = torch.Generator(device=dev).manual_seed(0)


def truncated(m, s, a, b):
    """The §10.3 recipe: finite window [a, b], fp64 inverse CDF, never clamping."""
    Phi = lambda t: 0.5 * torch.special.erfc(-t / math.sqrt(2))  # noqa: E731
    za, zb = ((a - m) / s).double(), ((b - m) / s).double()
    right = za > 0
    lo, hi = Phi(torch.where(right, -zb, za)), Phi(torch.where(right, -za, zb))
    u = lo + (hi - lo) * torch.rand(B, N, generator=g, device=dev, dtype=torch.float64)
    return m + s * (torch.where(right, -1.0, 1.0) * torch.special.ndtri(u)).float()


for mv, sv, a, b in [(1.0, 0.5, 0.2, 3.0), (0.0, 1.0, -1.0, 2.0), (0.0, 1.0, 8.0, 9.0),
                     (0.0, 1.0, -9.0, -8.0), (0.0, 1.0, 30.0, 31.0), (0.0, 1.0, -31.0, -30.0)]:
    m = torch.tensor(mv, device=dev, requires_grad=True)
    s = torch.tensor(sv, device=dev, requires_grad=True)
    x = truncated(m, s, a, b)
    x.mean().backward()
    xs = x.detach().double().cpu().numpy().ravel()
    ref = stats.truncnorm((a - mv) / sv, (b - mv) / sv, loc=mv, scale=sv)
    print(f"[{a:5.1f}, {b:5.1f}] m={mv} s={sv}: range [{xs.min():.4f}, {xs.max():.4f}] "
          f"mean {xs.mean():.4f} (exact {ref.mean():.4f}), "
          f"KS p = {stats.kstest(xs, ref.cdf).pvalue:.3f}, "
          f"d mean/dm = {m.grad.item():.3f}, d mean/ds = {s.grad.item():.3f}")
