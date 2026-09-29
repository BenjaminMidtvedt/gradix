"""E15: common-random-number coupling of the detector under replay(mode='noise'): same generator key, lambda
perturbed by a relative 1e-4 (one L-BFGS line-search step). Fraction of pixels whose Poisson count changes, and
|change|, for torch.poisson (rejection sampler) vs an inverse-CDF sampler on stored uniforms (monotone coupling)."""
import torch
dev = "cuda"
M = 1_000_000
for L0 in (3.0, 30.0, 300.0):
    lam = torch.full((M,), L0, device=dev, dtype=torch.float64)
    n1 = torch.poisson(lam, generator=torch.Generator(device=dev).manual_seed(5))
    n2 = torch.poisson(lam * (1 + 1e-4), generator=torch.Generator(device=dev).manual_seed(5))
    # inverse-CDF coupling: n = min{k : F(k; lam) >= u} with fixed u (gammaincc(k+1, lam) = F(k; lam))
    u = torch.rand(M, device=dev, dtype=torch.float64, generator=torch.Generator(device=dev).manual_seed(5))
    def icdf(l):
        k = torch.arange(int(L0 + 12 * L0**0.5 + 20), device=dev, dtype=torch.float64)
        F = torch.special.gammaincc(k + 1, l[:1].expand(k.numel()))
        return torch.searchsorted(F, u)
    m1, m2 = icdf(lam), icdf(lam * (1 + 1e-4))
    print(f"lam={L0:>5}: torch.poisson changed {((n1 != n2).double().mean().item()):.4f} of pixels "
          f"(mean |dn| {(n1 - n2).abs().mean().item():.3f});  inverse-CDF changed {((m1 != m2).double().mean().item()):.2e}")
