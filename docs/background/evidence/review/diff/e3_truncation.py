"""E3: truncated Normal by inverse CDF on counter uniforms (plan sec 6.5: 'Normal (erfinv)', sec 6.6).
x = mu + sigma * Phi^-1( Phi(a') + u (Phi(b') - Phi(a')) ),  a' = (a-mu)/sigma, b' = (b-mu)/sigma
Cases: (i) textbook erfinv formula in fp32; (ii) ndtri + survival-function (upper tail mirrored) in fp32;
(iii) same in fp64. Report non-finite samples, samples outside [a, b], and non-finite gradients."""
import torch, math
dev = "cuda"
g = torch.Generator(device=dev).manual_seed(0)
SQ2 = math.sqrt(2.0)

def Phi(z): return 0.5 * (1 + torch.erf(z / SQ2))
def Phi_inv_erfinv(p): return SQ2 * torch.erfinv(2 * p - 1)

def trunc_erfinv(mu, sig, a, b, u):
    A, B = Phi((a - mu) / sig), Phi((b - mu) / sig)
    return mu + sig * Phi_inv_erfinv(A + u * (B - A))

def trunc_stable(mu, sig, a, b, u):
    # work in the lower tail when the interval lies left of the mean, mirror otherwise
    za, zb = (a - mu) / sig, (b - mu) / sig
    flip = (za + zb) > 0
    lo = torch.where(flip, -zb, za); hi = torch.where(flip, -za, zb)
    A, B = torch.special.ndtr(lo), torch.special.ndtr(hi)
    # log-space interpolation would be better still; this keeps the example short
    z = torch.special.ndtri(A + u * (B - A))
    z = torch.where(flip, -z, z)
    return mu + sig * z

cases = [  # (mu, sigma, a, b, description)
    (0.0, 1.0, -3.72, 3.72, "auto-truncation at (1e-4, 1-1e-4), theta at init"),
    (5.0, 0.5, -1.86, 1.86, "same bounds after mu drifted to 5 (both bounds in lower tail)"),
    (0.0, 0.1, 0.62, 0.9, "upper-tail window (6.2..9 sigma)"),
    (0.0, 1.0, -30.0, 30.0, "wide user truncation"),
]
N = 1_000_000
u = torch.rand(N, device=dev, generator=g, dtype=torch.float64)
u[0] = 0.0  # a counter hash producing k * 2^-24 can emit exactly 0
for dtype in (torch.float32, torch.float64):
    for mu0, s0, a, b, desc in cases:
        for name, fn in (("erfinv", trunc_erfinv), ("ndtri+mirror", trunc_stable)):
            mu = torch.tensor(mu0, device=dev, dtype=dtype, requires_grad=True)
            sig = torch.tensor(s0, device=dev, dtype=dtype, requires_grad=True)
            x = fn(mu, sig, torch.tensor(a, device=dev, dtype=dtype), torch.tensor(b, device=dev, dtype=dtype), u.to(dtype))
            nonfinite = (~torch.isfinite(x)).sum().item()
            fin = torch.isfinite(x)
            outside = ((x[fin] < a - 1e-6) | (x[fin] > b + 1e-6)).sum().item()
            gm, gs = torch.autograd.grad(x[fin].sum() if fin.any() else x.sum(), (mu, sig))
            print(f"{str(dtype)[6:]:>8} {name:>13} | {desc:<58} | nonfinite {nonfinite:>7} outside [a,b] {outside:>7} "
                  f"| dsum/dmu {gm.item():>10.4g} dsum/dsig {gs.item():>10.4g}")
