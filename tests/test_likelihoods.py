"""The sCMOS likelihood benchmark (Q6, §11.6): the three approximations on the exact model."""

import math

import pytest
import torch

from gradix.ops.detect import poisson_gaussian_log_prob
from gradix.testing import likelihoods


def test_the_truncated_convolution_is_the_exact_density():
    for lam, read in ((0.0, 1.5), (0.3, 0.8), (100.0, 5.0)):
        spread = math.sqrt(lam + read * read)
        k = torch.linspace(lam - 12 * spread - 5, lam + 12 * spread + 5, 8001, dtype=torch.float64)
        density = poisson_gaussian_log_prob(k, torch.tensor(lam), read**2).exp()
        assert float(density.sum() * (k[1] - k[0])) == pytest.approx(1.0, abs=1e-9)
    out = poisson_gaussian_log_prob(torch.tensor([1.0]), torch.tensor([-1.0]), 1.0)
    assert out.item() == -math.inf


def test_the_benchmark_ranks_the_approximations():
    exact = likelihoods.evaluate("convolution", 0.3, 1.6)
    assert exact.ll_error < 1e-12
    assert all(abs(e - 1.0) < 1e-4 for e in exact.efficiency.values())  # finite differences
    gaussian = likelihoods.evaluate("gaussian", 1.0, 1.6)
    assert all(abs(b) < 1e-9 for b in gaussian.bias_se.values())  # moments match: unbiased
    assert gaussian.efficiency["lam"] > 0.95
    shifted = likelihoods.evaluate("shifted_poisson", 0.1, 0.8)
    # biased where k + σ² < 0 is common: gain and read noise by more than half a standard error
    assert abs(shifted.bias_se["gain"]) > 0.5 and abs(shifted.bias_se["read"]) > 0.5


def test_the_convolution_keeps_outliers_and_the_rate_derivative_at_zero():
    def brute(k, lam, var):
        n = torch.arange(0, 2000, dtype=torch.float64)
        log_pois = n * math.log(lam) - lam - torch.lgamma(n + 1)
        gauss = -0.5 * ((k - n) ** 2 / var + math.log(2 * math.pi * var))
        return float(torch.logsumexp(log_pois + gauss, 0))

    for k, lam in ((200.0, 5.0), (-30.0, 50.0)):
        got = poisson_gaussian_log_prob(
            torch.tensor(k, dtype=torch.float64), torch.tensor(lam), 2.56
        )
        assert float(got) == pytest.approx(brute(k, lam, 2.56), rel=1e-8)
    lam = torch.zeros(2, dtype=torch.float64, requires_grad=True)
    poisson_gaussian_log_prob(torch.tensor([5.0, 0.0]), lam, 2.56).sum().backward()
    expected = [math.exp((2 * k - 1) / (2 * 2.56)) - 1 for k in (5.0, 0.0)]
    assert lam.grad is not None and torch.allclose(
        lam.grad, torch.tensor(expected, dtype=torch.float64)
    )
    counts = poisson_gaussian_log_prob(torch.tensor([3, 4]), torch.tensor([2.5, 2.5]), 2.56)
    assert torch.allclose(
        counts, poisson_gaussian_log_prob(torch.tensor([3.0, 4.0]), torch.tensor([2.5, 2.5]), 2.56)
    )
