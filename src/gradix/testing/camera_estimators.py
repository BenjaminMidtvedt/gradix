"""The camera estimator suite (§11.6): bias of Poisson gradient estimators on several losses.

For a Poisson draw ``y ~ Poisson(λ)`` and a loss ``L(y)``, the true derivative
``d/dλ E[L(y)]`` is computed exactly from the pmf; a pathwise estimator's mean
``E[L'(y)·dy/dλ]`` is estimated from keyed draws. Plain straight-through gives exactly zero on
variance-sensitive losses, so the suite flags it; scaled straight-through is exact on losses up
to quadratic in counts.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Callable, Mapping

import torch
from torch import Tensor

from gradix._core import keys as _keys
from gradix.ops import detect as ops

__all__ = ["LOSSES", "EstimatorResult", "run", "true_derivative"]


def _quadratic(y: Tensor, lam: Tensor) -> Tensor:
    return (y - lam.detach()) ** 2


LOSSES: dict[str, Callable[[Tensor, Tensor], Tensor]] = {
    "mean": lambda y, lam: y,
    "quadratic": _quadratic,
    # Draws are non-negative in value; no clamp (its gradient vanishes at y = 0).
    "anscombe": lambda y, lam: 2.0 * torch.sqrt(y + 0.375),
    "log1p": lambda y, lam: torch.log1p(y),
}
"""Losses of the suite: mean- and variance-sensitive ones, and two common transforms."""

ESTIMATORS: dict[str, Callable[[Tensor, Tensor], Tensor]] = {
    "scaled_st": ops.scaled_st,
    "straight_through": ops.straight_through,
}
"""Estimators under test."""


def true_derivative(loss: str, lam: float, *, h: float = 1e-4) -> float:
    """Return ``d/dλ E[L(y)]`` with the loss's λ-argument held fixed, from the Poisson pmf.

    Parameters
    ----------
    loss : str
        A key of :data:`LOSSES`.
    lam : float
        The rate.
    h : float, default 1e-4
        Central-difference step in λ (the expectation is a smooth function of λ).

    Returns
    -------
    float
        The derivative.
    """
    fn = LOSSES[loss]
    held = torch.tensor(lam, dtype=torch.float64)

    def expectation(rate: float) -> float:
        n_max = int(rate + 20.0 * math.sqrt(rate + 1.0) + 40.0)
        k = torch.arange(n_max + 1, dtype=torch.float64)
        if rate == 0.0:
            pmf = (k == 0).to(torch.float64)
        else:
            pmf = torch.exp(k * math.log(rate) - rate - torch.lgamma(k + 1.0))
        return float((pmf * fn(k, held)).sum())

    lo = max(lam - h, 0.0)
    return (expectation(lam + h) - expectation(lo)) / (lam + h - lo)


@dataclasses.dataclass(frozen=True)
class EstimatorResult:
    """One cell of the suite.

    Parameters
    ----------
    estimator : str
        Estimator name.
    loss : str
        Loss name.
    lam : float
        The rate.
    estimate : float
        Mean pathwise gradient.
    stderr : float
        Standard error of the estimate.
    truth : float
        The exact derivative.
    """

    estimator: str
    loss: str
    lam: float
    estimate: float
    stderr: float
    truth: float

    @property
    def bias(self) -> float:
        """The estimate minus the truth.

        Returns
        -------
        float
            The bias.
        """
        return self.estimate - self.truth

    def consistent(self, sigmas: float = 5.0, floor: float = 1e-3) -> bool:
        """Return whether the estimate agrees with the truth within its standard error.

        Parameters
        ----------
        sigmas : float, default 5.0
            Allowed number of standard errors.
        floor : float, default 1e-3
            Absolute tolerance floor.

        Returns
        -------
        bool
            Whether ``|bias| <= sigmas·stderr + floor``.
        """
        return abs(self.bias) <= sigmas * self.stderr + floor


def run(
    lams: tuple[float, ...] = (0.0, 0.2, 1.0, 3.0),
    *,
    draws: int = 200_000,
    estimators: Mapping[str, Callable[[Tensor, Tensor], Tensor]] | None = None,
    losses: tuple[str, ...] = tuple(LOSSES),
    device: torch.device | str = "cpu",
    seed: int = 0,
) -> list[EstimatorResult]:
    """Run the suite.

    Parameters
    ----------
    lams : tuple of float, default (0.0, 0.2, 1.0, 3.0)
        Rates.
    draws : int, default 200000
        Keyed Poisson draws per rate.
    estimators : Mapping[str, callable], optional
        Estimators by name; :data:`ESTIMATORS` by default.
    losses : tuple of str, default all of :data:`LOSSES`
        Losses to evaluate.
    device : torch.device or str, default "cpu"
        Device of the draws.
    seed : int, default 0
        Image key of the draws.

    Returns
    -------
    list of EstimatorResult
        One result per (estimator, loss, rate).
    """
    estimators = dict(estimators or ESTIMATORS)
    key = torch.tensor([seed], dtype=torch.int64, device=device)
    results: list[EstimatorResult] = []
    for lam in lams:
        rate = torch.full((1, draws), lam, dtype=torch.float64, device=device, requires_grad=True)
        counts = _keys.poisson(key, "estimators", rate)
        for est_name, est in estimators.items():
            y = est(rate, counts)
            for loss in losses:
                (g,) = torch.autograd.grad(LOSSES[loss](y, rate).sum(), rate, retain_graph=True)
                g = g.flatten()
                results.append(
                    EstimatorResult(
                        est_name,
                        loss,
                        lam,
                        float(g.mean()),
                        float(g.std() / math.sqrt(draws)),
                        true_derivative(loss, lam),
                    )
                )
    return results
