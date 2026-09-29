"""Lorenz–Mie coefficients and amplitudes against independent references (§11.2)."""

import numpy as np
import pytest
import torch
from numpy.polynomial import legendre
from scipy import special

from gradix.special import mie


def reference(x, m, n):
    """Bohren–Huffman eq. 4.53 with scipy's spherical Bessel functions (no recurrences)."""
    orders = np.arange(1, n + 1)
    mx = m * x
    jx, djx = special.spherical_jn(orders, x), special.spherical_jn(orders, x, derivative=True)
    yx, dyx = special.spherical_yn(orders, x), special.spherical_yn(orders, x, derivative=True)
    jm, djm = special.spherical_jn(orders, mx), special.spherical_jn(orders, mx, derivative=True)
    psi_x, dpsi_x = x * jx, jx + x * djx
    psi_m, dpsi_m = mx * jm, jm + mx * djm
    h, dh = jx + 1j * yx, djx + 1j * dyx
    xi, dxi = x * h, h + x * dh
    a = (m * psi_m * dpsi_x - psi_x * dpsi_m) / (m * psi_m * dxi - xi * dpsi_m)
    b = (psi_m * dpsi_x - m * psi_x * dpsi_m) / (psi_m * dxi - m * xi * dpsi_m)
    return a, b


CASES = [
    (x, m)
    for x in (0.05, 0.3, 1.0, 2.5, 5.213, 12.0)
    for m in (1.2, 1.5, 1.59 + 0.01j, 0.9, 1.4 + 0.6j)
]


@pytest.mark.parametrize(("x", "m"), CASES)
def test_coefficients_match_the_bessel_function_reference(x, m):
    n = mie.terms(x)
    a, b = mie.coefficients(torch.tensor(x, dtype=torch.float64), m, n)
    ref_a, ref_b = reference(x, m, n)
    np.testing.assert_allclose(a.numpy(), ref_a, rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(b.numpy(), ref_b, rtol=1e-9, atol=1e-12)


def test_the_bohren_huffman_example():
    # BHMIE's test case: m = 1.55, λ = 0.6328 µm, r = 0.525 µm in vacuum (x = 5.2128)
    x = torch.tensor(2 * np.pi * 0.525 / 0.6328, dtype=torch.float64)
    q_ext, q_sca, q_back = mie.efficiencies(x, *mie.coefficients(x, 1.55, mie.terms(float(x))))
    assert float(q_ext) == pytest.approx(3.10543, rel=1e-5)
    assert float(q_sca) == pytest.approx(3.10543, rel=1e-5)
    assert float(q_back) == pytest.approx(2.92534, rel=1e-5)


def test_angular_functions_are_legendre_derivatives():
    mu = torch.linspace(-1.0, 1.0, 41, dtype=torch.float64)
    pi, tau = mie.angular(mu, 6)
    for n in range(1, 7):
        coef = np.zeros(n + 1)
        coef[n] = 1.0
        dp = legendre.legder(coef)  # π_n = dP_n/dμ
        ddp = legendre.legder(coef, 2)
        m = mu.numpy()
        np.testing.assert_allclose(pi[:, n - 1].numpy(), legendre.legval(m, dp), atol=1e-12)
        tau_ref = m * legendre.legval(m, dp) - (1 - m * m) * legendre.legval(m, ddp)
        np.testing.assert_allclose(tau[:, n - 1].numpy(), tau_ref, atol=1e-11)


def test_amplitudes_obey_the_optical_theorem_and_the_symmetries():
    x = torch.tensor(3.0, dtype=torch.float64)
    a, b = mie.coefficients(x, 1.59 + 0.02j, mie.terms(3.0))
    ends = torch.tensor([1.0, -1.0], dtype=torch.float64)
    s1, s2 = mie.amplitudes(a, b, *mie.angular(ends, a.shape[-1]))
    q_ext, _, q_back = mie.efficiencies(x, a, b)
    assert torch.isclose(s1[0], s2[0])  # forward
    assert torch.isclose(s1[1], -s2[1])  # backward
    assert float(4.0 / x**2 * s1[0].real) == pytest.approx(float(q_ext), rel=1e-12)
    assert float(4.0 * s1[1].abs() ** 2 / x**2) == pytest.approx(float(q_back), rel=1e-12)


def test_amplitudes_continue_to_complex_angles():
    # π_n and τ_n are polynomials in cos Θ: a complex argument is their analytic continuation
    mu = torch.tensor([1.2 + 0.0j, 0.3 + 0.8j], dtype=torch.complex128)
    pi, tau = mie.angular(mu, 4)
    m = mu.numpy()
    assert np.allclose(pi[:, 1].numpy(), 3 * m) and np.allclose(tau[:, 1].numpy(), 6 * m * m - 3)


def test_coefficients_are_differentiable_in_both_modes():
    x = torch.tensor([0.4, 2.0], dtype=torch.float64, requires_grad=True)
    m = torch.tensor([1.45 + 0.01j, 1.59 + 0.0j], dtype=torch.complex128, requires_grad=True)

    def s_forward(x, m):
        a, b = mie.coefficients(x, m, 6, start=40)
        return torch.view_as_real(mie.amplitudes(a, b, *mie.angular(x.new_tensor(0.3), 6))[0])

    assert torch.autograd.gradcheck(s_forward, (x, m), check_forward_ad=True)
