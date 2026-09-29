"""Camera kernels: noise samplers, gradient estimators and likelihoods (§5.5, §6.2, §6.3).

The samplers draw with explicit keys only: an int64 ``Tensor[B]`` image key through the counter
hash of :mod:`gradix._core.keys`, or an ``int`` batch key through a :class:`torch.Generator`
seeded from it. This module and ``_core/keys.py`` are the only places allowed to call torch's
generator-based samplers (§3.2).

Gradient estimators attach gradients to the (integer) Poisson draw:

- :func:`scaled_st` (default): ``y = n + (λ − sg(λ))·sg(g)``, ``g = (n + λ)/(2·max(λ, tiny))``
  for λ > 0 and 1 at λ = 0. Unbiased for losses up to quadratic in counts; finite at λ = 0.
- :func:`straight_through` (plain ST, ``g = 1``): exactly zero gradient of variance-sensitive
  losses; the camera estimator suite flags it (§11.6). Provided for comparison only.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

from gradix._core import keys as _keys

__all__ = [
    "gamma",
    "normal",
    "poisson",
    "poisson_gaussian_log_prob",
    "poisson_log_prob",
    "scaled_st",
    "shifted_poisson_log_prob",
    "st_round",
    "straight_through",
]

_TINY = 1e-12
_POISSON_CAP = float(2**23)
"""Batch-keyed rates above this are drawn at the cap and their fluctuation rescaled to √λ (a
Gaussian limit: at these rates the relative noise is below 4·10⁻⁴)."""


_LAM0 = 1e-30
"""The infinitesimal rate at which likelihoods evaluate λ = 0 (keeps its derivative)."""


def poisson(lam: Tensor, key: int | Tensor, stream: str) -> Tensor:
    """Draw Poisson counts with an explicit key (no gradient).

    Parameters
    ----------
    lam : Tensor
        Rates ``[B, ...]``; negative values (including −inf) count as 0; NaN and +inf give NaN.
    key : int or Tensor
        A batch key (``int``) or image keys (int64 ``Tensor[B]``, moved to ``lam``'s device).
    stream : str
        Stream name; distinct stages use distinct streams.

    Returns
    -------
    Tensor
        Integer-valued counts in ``lam``'s dtype, same shape (``[B, ...]`` for image keys and
        shared rates). Image keys draw exactly, with a counter-based sampler that is
        reproducible across devices. Batch keys use ``torch.poisson`` for speed: exact on CPU,
        but on CUDA (curand) only approximate above λ ≈ 1000. At λ = 4000 the variance is
        0.9 % low and the third moment 1.8× high, and above λ ≈ 5000 the draws are normal (no
        skewness); means stay within 0.01 %. An exact sampler as fast is planned before 1.0.
        Rates above 2²³ are drawn at the cap with their fluctuation rescaled to √λ.
    """
    rate = torch.clamp(lam.detach(), min=0.0)
    if isinstance(key, Tensor):
        return _keys.poisson(key, stream, rate).to(lam.dtype)
    bad = ~torch.isfinite(rate)
    capped = torch.where(bad, torch.zeros_like(rate), torch.clamp(rate, max=_POISSON_CAP))
    counts = torch.poisson(capped, generator=_keys.generator(key, stream, rate.device))
    huge = rate > _POISSON_CAP
    scale = torch.sqrt(torch.where(huge, rate, _POISSON_CAP) / _POISSON_CAP)
    counts = torch.where(huge, torch.round(rate + (counts - _POISSON_CAP) * scale), counts)
    return torch.where(bad, torch.full_like(counts, float("nan")), counts)


def normal(like: Tensor, key: int | Tensor, stream: str) -> Tensor:
    """Draw standard normal values shaped like a tensor, with an explicit key.

    Parameters
    ----------
    like : Tensor
        ``[B, ...]`` tensor whose shape, dtype and device the draw takes.
    key : int or Tensor
        A batch key (``int``) or image keys (int64 ``Tensor[B]``).
    stream : str
        Stream name.

    Returns
    -------
    Tensor
        Standard normal values, same shape, dtype and device as ``like``.
    """
    if isinstance(key, Tensor):
        key = key.to(like.device)
        if key.shape[0] != like.shape[0]:
            like = like.expand(key.shape[0], *like.shape[1:])
        return _keys.normal(key, stream, tuple(like.shape[1:])).to(like.dtype)
    gen = _keys.generator(key, stream, like.device)
    return torch.randn(like.shape, generator=gen, dtype=like.dtype, device=like.device)


def gamma(concentration: Tensor, key: int, stream: str) -> Tensor:
    """Draw Gamma(concentration, 1) variates with an explicit batch key.

    Gradients with respect to the concentration are implicit-reparameterisation gradients.
    Image keys are not supported yet (1.x): the draw uses a keyed ``torch.Generator``.

    Parameters
    ----------
    concentration : Tensor
        Positive shape parameters.
    key : int
        A batch key.
    stream : str
        Stream name.

    Returns
    -------
    Tensor
        The variates, same shape and dtype as ``concentration``.
    """
    gen = _keys.generator(key, stream, concentration.device)
    # torch has no public keyed Gamma sampler with reparameterisation gradients (risk 14)
    return torch._standard_gamma(concentration, generator=gen)


def scaled_st(lam: Tensor, counts: Tensor) -> Tensor:
    """Attach the scaled straight-through gradient to Poisson counts.

    Parameters
    ----------
    lam : Tensor
        Rates the counts were drawn with; gradients flow to them.
    counts : Tensor
        Poisson draws (no gradient), same shape.

    Returns
    -------
    Tensor
        Equal to ``counts`` in value; ``d/dλ = (n + λ)/(2λ)`` for λ > 0 and 1 at λ = 0.
    """
    lam_d = lam.detach()
    g = torch.where(lam_d > 0, (counts + lam_d) / (2.0 * torch.clamp(lam_d, min=_TINY)), 1.0)
    return counts + (lam - lam_d) * g


def straight_through(lam: Tensor, counts: Tensor) -> Tensor:
    """Attach the plain straight-through gradient (``d/dλ = 1``) to Poisson counts.

    Plain ST gives exactly zero gradient of variance-sensitive losses (§6.2); it exists so that
    the camera estimator suite can flag it.

    Parameters
    ----------
    lam : Tensor
        Rates the counts were drawn with.
    counts : Tensor
        Poisson draws, same shape.

    Returns
    -------
    Tensor
        Equal to ``counts`` in value, with unit gradient to ``lam``.
    """
    return counts + (lam - lam.detach())


def st_round(y: Tensor) -> Tensor:
    """Round to integers with a straight-through gradient (the ADC).

    Parameters
    ----------
    y : Tensor
        Values to round.

    Returns
    -------
    Tensor
        ``round(y)`` in value, identity gradient.
    """
    return y + (torch.round(y) - y).detach()


def poisson_log_prob(k: Tensor, lam: Tensor) -> Tensor:
    """Return the Poisson log-likelihood ``k·log λ − λ − log Γ(k + 1)``, continued to real k.

    Parameters
    ----------
    k : Tensor
        Observed counts (real values allowed).
    lam : Tensor
        Rates, broadcastable to ``k``.

    Returns
    -------
    Tensor
        Log-likelihood per element: at λ = 0 it is 0 for k = 0 (with gradient −1) and −inf for
        k > 0; for λ < 0, outside the domain, it is −inf (with zero gradient), so a line search
        that steps below zero backs off instead of running away.
    """
    k_safe = torch.clamp(k, min=0.0)
    positive = lam > 0
    lam_safe = torch.where(positive, lam, torch.ones_like(lam))
    inside = torch.xlogy(k_safe, lam_safe) - lam - torch.lgamma(k_safe + 1.0)
    minus_inf = torch.full_like(inside, -math.inf)
    at_zero = torch.where(k_safe == 0, -lam * torch.ones_like(inside), minus_inf)
    out = torch.where(positive, inside, torch.where(lam == 0, at_zero, minus_inf))
    return torch.where(torch.isnan(lam), lam, out)  # NaN rates stay visible


def poisson_gaussian_log_prob(
    k: Tensor, lam: Tensor, variance: Tensor | float, *, terms: int | None = None
) -> Tensor:
    """Return the Poisson ⊛ Gaussian log-likelihood by a truncated convolution.

    ``log Σ_n Pois(n; λ)·N(k; n, σ²)`` is summed over ``terms`` consecutive counts n around
    the mode of n given k (Newton steps from the Gaussian approximation, so outliers far from
    λ keep their mass). The dropped terms lie more than ``terms/2`` counts from it, many widths
    ``w ≤ min(√λ, σ)`` away: their relative weight is below 1e-9 while ``w ≤ terms/12``. Exact
    otherwise, gradients included (with respect to λ, σ² and k, so to gain and offset through
    k).

    Parameters
    ----------
    k : Tensor
        Observed values in electrons (real).
    lam : Tensor
        Expected electrons, broadcastable to ``k``.
    variance : Tensor or float
        Read-noise variance σ² in electrons², broadcastable to ``k``.
    terms : int, optional
        Counts summed per element; by default ``max(64, 12·σ_max)`` rounded up to a multiple
        of 8 (the largest read noise is read back once, so the window holds every posterior).

    Returns
    -------
    Tensor
        Log-density per element (per electron), in a floating dtype. For λ < 0 it is −inf with
        zero gradient. λ = 0 is evaluated at an infinitesimal rate (1e-30), which keeps the
        derivative ``exp((2k − 1)/(2σ²)) − 1`` of the n = 1 term (the value differs from the
        n = 0 term alone only for outliers ``k > 69σ²``).
    """
    real = k.dtype if k.is_floating_point() else torch.get_default_dtype()  # counts may be ints
    if isinstance(lam, Tensor) and lam.is_floating_point():
        real = torch.promote_types(real, lam.dtype)
    if isinstance(variance, Tensor):
        var_t = variance.to(device=k.device, dtype=real)
    else:  # a Python number, built at the working precision
        var_t = torch.tensor(float(variance), dtype=real, device=k.device)
    lam_t = torch.as_tensor(lam, device=k.device).to(real)
    k, lam, var = torch.broadcast_tensors(k.to(real), lam_t, var_t)
    var = torch.clamp(var, min=_TINY)
    if terms is None:  # the posterior width is at most σ: a window of 12σ holds it
        finite = var.detach()[torch.isfinite(var.detach())]  # NaN read noise gives NaN, below
        sigma = math.sqrt(float(finite.max())) if finite.numel() else 1.0
        terms = max(64, 8 * math.ceil(12.0 * sigma / 8.0))
    infinite = torch.isposinf(lam)
    lam = torch.where(infinite, torch.ones_like(lam), lam)  # computed finitely, returned −inf
    lam_eff = lam + (lam == 0).to(real) * _LAM0  # keeps the n = 1 derivative at λ = 0
    with torch.no_grad():  # the window is placed without gradient; the sum is exact inside it
        rate = torch.clamp(lam_eff, min=_LAM0)
        mode = torch.clamp((rate * var + k * rate) / (rate + var), min=0.0)
        mode = torch.nan_to_num(mode, nan=0.0, posinf=0.0, neginf=0.0)
        log_rate = torch.log(rate)
        for _ in range(8):  # Newton on n·log λ − log Γ(n + 1) − (k − n)²/(2σ²): concave in n
            slope = log_rate - torch.digamma(mode + 1.0) + (k - mode) / var
            curve = -torch.polygamma(1, mode + 1.0) - 1.0 / var
            mode = torch.clamp(mode - slope / curve, min=0.0)
        mode = torch.nan_to_num(mode, nan=0.0, posinf=0.0, neginf=0.0)
        start = torch.clamp(torch.floor(mode) - (terms // 2 - 1), min=0.0)
    n = start[..., None] + torch.arange(terms, dtype=real, device=k.device)
    log_pois = poisson_log_prob(n, lam_eff[..., None])
    outside = (lam < 0)[..., None]
    log_pois = torch.where(outside, torch.zeros_like(log_pois), log_pois)  # no NaN gradients
    v = var[..., None]
    log_gauss = -0.5 * ((k[..., None] - n) ** 2 / v + torch.log(2.0 * math.pi * v))
    total = torch.logsumexp(log_pois + log_gauss, dim=-1)
    outside = (lam < 0) | infinite  # no probability at negative or infinite rates
    return torch.where(outside, torch.full_like(total, -math.inf), total)


def shifted_poisson_log_prob(k: Tensor, lam: Tensor, variance: Tensor | float) -> Tensor:
    """Return the shifted-Poisson approximation of the Poisson ⊛ Gaussian log-likelihood.

    ``k + σ²`` is treated as Poisson with rate ``λ + σ²`` (§5.5); the choice of default
    approximation is settled by the M1 benchmark (Q6).

    Parameters
    ----------
    k : Tensor
        Observed values in electrons.
    lam : Tensor
        Expected electrons, broadcastable to ``k``.
    variance : Tensor or float
        Read-noise variance σ² in electrons².

    Returns
    -------
    Tensor
        Log-likelihood per element.
    """
    return poisson_log_prob(k + variance, lam + variance)
