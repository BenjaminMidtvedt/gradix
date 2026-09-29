"""The sCMOS likelihood benchmark (Q6, §11.6): Poisson ⊛ Gaussian approximations against exact.

An sCMOS pixel reports ``y = g·(n + σ·z) + o`` ADU for ``n ~ Poisson(λ)`` photoelectrons and
``z ~ N(0, 1)``. Each approximation of ``log p(y; λ, g, o, σ)`` is judged on the exact
distribution, by deterministic quadrature over a fine grid of y (no sampling noise):

- the error of the log-likelihood, ``E|log p_A − log p|`` (nats);
- for each parameter θ (the others held at their true values), the shift of its
  maximum-likelihood estimate: the root θ* of the expected score ``E[∂θ log p_A(θ)] = 0``
  (θ units; the same for any number of pixels), also in units of the one-pixel Cramér–Rao
  standard error. The root is found directly, so likelihoods with kinks (the shifted Poisson's
  clamp) are judged correctly;
- the variance efficiency ``J² / (I·K)`` at θ*, the Cramér–Rao bound over the sandwich variance
  (``J`` the slope of the expected score, ``K = Var[∂θ log p_A]``, ``I`` the exact Fisher
  information): 1 is optimal for unbiased approximations, and biased ones may exceed it (judge
  those by their shift);
- the Cramér–Rao bound the approximation implies, ``I / J`` times the exact one.

``run`` sweeps light levels and read noises; ``speed`` times a forward and backward pass.
"""

from __future__ import annotations

import dataclasses
import math
import time
from collections.abc import Callable, Mapping, Sequence

import torch
from torch import Tensor

from gradix.ops import detect as ops

__all__ = ["APPROXIMATIONS", "PARAMETERS", "LikelihoodResult", "evaluate", "run", "speed"]


def _gaussian(k: Tensor, lam: Tensor, var: Tensor) -> Tensor:
    total = torch.clamp(torch.clamp(lam, min=0.0) + var, min=1e-12)
    return -0.5 * ((k - lam) ** 2 / total + torch.log(2.0 * math.pi * total))


APPROXIMATIONS: dict[str, Callable[[Tensor, Tensor, Tensor], Tensor]] = {
    "gaussian": _gaussian,
    "shifted_poisson": ops.shifted_poisson_log_prob,
    "convolution": ops.poisson_gaussian_log_prob,
}
"""Electron-space log-likelihoods ``(k, λ, σ²) → log p`` under test (``PoissonGaussian``'s)."""

PARAMETERS: tuple[str, ...] = ("lam", "gain", "offset", "read")
"""Parameters whose estimates are judged: photoelectrons, ADU/e⁻, ADU and e⁻ rms."""


@dataclasses.dataclass(frozen=True)
class LikelihoodResult:
    """One approximation at one light level and read noise.

    Parameters
    ----------
    approximation : str
        Name in :data:`APPROXIMATIONS`.
    lam : float
        Expected photoelectrons λ.
    read : float
        Read-noise standard deviation σ, electrons.
    ll_error : float
        ``E|log p_A − log p|``, nats.
    bias : Mapping[str, float]
        Shift of each parameter's maximum-likelihood estimate (the expected score's root), in
        its units.
    bias_se : Mapping[str, float]
        The shift in units of the one-pixel Cramér–Rao standard error.
    efficiency : Mapping[str, float]
        Cramér–Rao bound over the sandwich variance at the root, per parameter (1 is optimal
        for unbiased approximations; biased ones may exceed it).
    crlb_ratio : Mapping[str, float]
        The approximation's implied Cramér–Rao bound over the exact one, per parameter.
    """

    approximation: str
    lam: float
    read: float
    ll_error: float
    bias: Mapping[str, float]
    bias_se: Mapping[str, float]
    efficiency: Mapping[str, float]
    crlb_ratio: Mapping[str, float]


def _grid(lam: float, read: float) -> tuple[Tensor, float]:
    """Return a grid of electron values that holds the exact density to double precision."""
    spread = math.sqrt(lam + read * read)
    lo = min(-8.0 * read - 5.0, lam - 12.0 * spread)
    hi = lam + 12.0 * spread + 8.0 * read + 5.0
    step = min(read, 1.0) / 16.0
    count = math.ceil((hi - lo) / step) + 1
    return torch.linspace(lo, hi, count, dtype=torch.float64), (hi - lo) / (count - 1)


def _terms(
    log_prob: Callable[[Tensor, Tensor, Tensor], Tensor],
    y: Tensor,
    params: Mapping[str, float],
) -> tuple[Tensor, dict[str, Tensor]]:
    """Return the log-density of observations y (ADU) and every parameter's per-point score."""
    count = y.shape[0]
    theta = {
        name: torch.full((count,), params[name], dtype=torch.float64, requires_grad=True)
        for name in PARAMETERS
    }
    electrons = (y - theta["offset"]) / theta["gain"]
    values = log_prob(electrons, theta["lam"], theta["read"] ** 2) - torch.log(theta["gain"])
    grads = torch.autograd.grad(values.sum(), list(theta.values()))
    return values.detach(), dict(zip(PARAMETERS, grads, strict=True))


def _root(
    score: Callable[[float], float], start: float, scale: float, lower: float, floor: float
) -> float:
    """Return the root of an expected score near ``start``, or ±inf if it never crosses zero.

    A score within ``floor`` of zero at ``start`` (rounding noise at an exact root) counts as
    a root there; elsewhere only a verified sign change is bisected, so a score that decays
    without crossing zero gives ±inf (the estimate diverges). Steps grow geometrically away
    from ``start`` and approach ``lower`` geometrically; an estimate pinned at the bound
    returns ``lower``.
    """
    s0 = score(start)
    if abs(s0) <= floor:
        return start
    up = s0 > 0  # a positive score says the estimate lies higher
    step = 0.05 * scale
    here = start
    for _ in range(48):
        if up:
            there = here + step
        else:
            there = here - step
            if there <= lower:
                there = lower + 0.5 * (here - lower)  # approach the bound, never cross it
        value = score(there)
        if abs(value) > floor and (value > 0) != up:  # a real crossing: bisect
            a, b = (here, there) if here < there else (there, here)
            sa = score(a)
            for _ in range(60):
                mid = 0.5 * (a + b)
                sm = score(mid)
                if abs(sm) <= floor:
                    return mid
                if (sm > 0) == (sa > 0):
                    a, sa = mid, sm
                else:
                    b = mid
            return 0.5 * (a + b)
        if not up and there - lower <= 1e-9 * scale:
            return lower  # the estimate is pinned at the bound
        here, step = there, step * 2.0
    return math.inf if up else -math.inf  # the score never crosses: the estimate diverges


def evaluate(
    approximation: str, lam: float, read: float, *, gain: float = 2.0, offset: float = 100.0
) -> LikelihoodResult:
    """Judge one approximation at one light level and read noise.

    Parameters
    ----------
    approximation : str
        Name in :data:`APPROXIMATIONS`.
    lam : float
        Expected photoelectrons λ.
    read : float
        Read-noise standard deviation σ, electrons (> 0).
    gain : float, default 2.0
        Camera gain, ADU per electron.
    offset : float, default 100.0
        Camera offset, ADU.

    Returns
    -------
    LikelihoodResult
        The log-likelihood error and the per-parameter bias and efficiency.
    """
    k, step = _grid(lam, read)
    truth = {"lam": lam, "gain": gain, "offset": offset, "read": read}
    y = gain * k + offset  # observations on a fine grid; weights below are their probabilities

    def exact(k: Tensor, lam: Tensor, var: Tensor) -> Tensor:
        return ops.poisson_gaussian_log_prob(k, lam, var, terms=256)

    approx = APPROXIMATIONS[approximation]
    reference, scores_exact = _terms(exact, y, truth)
    values, _scores = _terms(approx, y, truth)
    weights = torch.exp(reference + math.log(gain)) * step  # exact probability of each cell
    weights = weights / weights.sum()
    finite = torch.isfinite(values)
    ll_error = float((weights * torch.where(finite, (values - reference).abs(), 0.0)).sum())
    bias, bias_se, efficiency, crlb_ratio = {}, {}, {}, {}
    for name in PARAMETERS:

        def scores_at(value: float, name: str = name) -> Tensor:
            return _terms(approx, y, {**truth, name: value})[1][name]

        def expected(value: float) -> float:
            return float((weights * scores_at(value)).sum())

        fisher = float((weights * scores_exact[name] ** 2).sum())
        scale = 1.0 / math.sqrt(fisher)  # one-pixel standard error
        lower = 0.0 if name != "offset" else -math.inf
        root = _root(expected, truth[name], scale, lower, floor=1e-9 * math.sqrt(fisher))
        bias[name] = root - truth[name]
        bias_se[name] = bias[name] / scale
        if not math.isfinite(root) or root <= lower:
            efficiency[name] = crlb_ratio[name] = math.nan  # no interior estimate
            continue
        h = min(1e-3 * scale, 0.5 * (root - lower))
        curve = -(expected(root + h) - expected(root - h)) / (2.0 * h)
        at_root = scores_at(root)
        spread = float((weights * (at_root - (weights * at_root).sum()) ** 2).sum())
        if curve <= 1e-6 * fisher or spread <= 0:
            efficiency[name] = crlb_ratio[name] = math.nan  # flat: no curvature to speak of
            continue
        efficiency[name] = curve * curve / (fisher * spread)
        crlb_ratio[name] = fisher / curve
    return LikelihoodResult(
        approximation, lam, read, ll_error, bias, bias_se, efficiency, crlb_ratio
    )


def run(
    lams: Sequence[float] = (0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0),
    reads: Sequence[float] = (0.8, 1.6, 3.0),
    approximations: Sequence[str] = tuple(APPROXIMATIONS),
) -> list[LikelihoodResult]:
    """Sweep light levels and read noises.

    Parameters
    ----------
    lams : sequence of float
        Expected photoelectrons (the plan's 0.1–100 range).
    reads : sequence of float
        Read noises, electrons rms (typical sCMOS pixels and a noisy one).
    approximations : sequence of str
        Names in :data:`APPROXIMATIONS`.

    Returns
    -------
    list of LikelihoodResult
        One result per approximation, light level and read noise.
    """
    return [evaluate(a, lam, r) for a in approximations for r in reads for lam in lams]


def speed(
    approximation: str,
    *,
    pixels: int = 1 << 20,
    device: str | None = None,
    repeats: int = 20,
) -> float:
    """Time a forward and backward pass of an approximation, milliseconds per call.

    Parameters
    ----------
    approximation : str
        Name in :data:`APPROXIMATIONS`.
    pixels : int, default 1048576
        Pixels per call.
    device : str, optional
        Device; CUDA when available.
    repeats : int, default 20
        Timed calls (after two warm-up calls); the median is returned.

    Returns
    -------
    float
        Median milliseconds per forward + backward call, float32.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    # deterministic, well-spread test values (low-discrepancy sequences, no RNG)
    index = torch.arange(pixels, dtype=torch.float64)
    lam = (10.0 * ((index * 0.6180339887) % 1.0)).float().to(device).requires_grad_()
    jitter = 6.0 * ((index * 0.7548776662) % 1.0) - 3.0
    k = (lam.detach() + jitter.float().to(device)).contiguous()
    var = torch.full((pixels,), 2.5, device=device)
    fn = APPROXIMATIONS[approximation]
    times = []
    for i in range(repeats + 2):
        if device == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        fn(k, lam, var).sum().backward()
        if device == "cuda":
            torch.cuda.synchronize()
        if i >= 2:
            times.append(1e3 * (time.perf_counter() - start))
    return sorted(times)[len(times) // 2]
