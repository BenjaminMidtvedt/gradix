"""Lorenz–Mie scattering by a homogeneous sphere (Bohren–Huffman conventions), in fp64 torch.

The coefficients a_n and b_n come from the logarithmic derivative D_n(mx), by downward
recurrence, and the Riccati–Bessel functions ψ_n(x) and ξ_n(x), by upward recurrences (as in
BHMIE and Wiscombe's MIEV0), with :func:`terms` orders. The angular functions π_n and τ_n are
polynomials in cos Θ, so the amplitudes S₁ and S₂ continue analytically to complex angles
(evanescent directions). Everything is plain torch: gradients and forward-mode derivatives
reach the size parameter and the complex relative index.

References
----------
C. F. Bohren and D. R. Huffman, *Absorption and Scattering of Light by Small Particles*
(Wiley, 1983), ch. 4 and appendix A. W. J. Wiscombe, "Improved Mie scattering algorithms",
Appl. Opt. 19, 1505–1509 (1980), doi:10.1364/AO.19.001505.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

__all__ = ["amplitudes", "angular", "coefficients", "efficiencies", "terms"]

_SMALL = 0.1
"""Below this size parameter ψ₁ uses its series (the direct form cancels)."""


def terms(x: float) -> int:
    """Return the number of Mie orders for a size parameter (Wiscombe's criterion).

    Parameters
    ----------
    x : float
        Largest size parameter k·r, with k the wavenumber in the host medium.

    Returns
    -------
    int
        ``int(x + 4.05·x^⅓ + 2) + 1``, at least 1.

    Examples
    --------
    >>> terms(1.0)
    8
    """
    x = max(float(x), 0.0)
    return max(int(x + 4.05 * x ** (1.0 / 3.0) + 2.0) + 1, 1)


def coefficients(
    x: Tensor, m: Tensor | complex, n: int, *, start: int | None = None
) -> tuple[Tensor, Tensor]:
    """Return the Mie coefficients a_n and b_n for n = 1 … ``n``.

    Parameters
    ----------
    x : Tensor
        Size parameters k·r (real, k the wavenumber in the host), any shape.
    m : Tensor or complex
        Relative refractive indices n_sphere/n_host (complex for absorbing spheres),
        broadcastable against ``x``.
    n : int
        Number of orders (:func:`terms`).
    start : int, optional
        Order at which the downward recurrence of D_n(mx) starts; by default
        ``max(n, ⌈max|mx|⌉) + 16`` from the values (a host synchronisation).

    Returns
    -------
    a : Tensor
        Electric coefficients, complex128 ``[..., n]``.
    b : Tensor
        Magnetic coefficients, complex128 ``[..., n]``.

    Notes
    -----
    With D_n(z) = ψ_n'(z)/ψ_n(z) (Bohren–Huffman eq. 4.88),
    ``a_n = ((D_n/m + n/x)ψ_n − ψ_{n−1}) / ((D_n/m + n/x)ξ_n − ξ_{n−1})`` and
    ``b_n = ((m·D_n + n/x)ψ_n − ψ_{n−1}) / ((m·D_n + n/x)ξ_n − ξ_{n−1})``, with ξ_n = ψ_n − iχ_n.
    The upward recurrence of ψ_n loses accuracy for orders beyond x, where the coefficients are
    already negligible; size parameters below about 0.01 need the Rayleigh limit instead.

    Examples
    --------
    >>> import torch
    >>> a, b = coefficients(torch.tensor(1.0), 1.5, terms(1.0))
    >>> a.shape
    torch.Size([8])
    """
    x = torch.as_tensor(x, dtype=torch.float64)  # Python numbers stay exact
    m = torch.as_tensor(m, dtype=torch.complex128, device=x.device)
    x, m = torch.broadcast_tensors(x, m)
    tiny = 1e-8  # a sphere of zero radius scatters nothing; keep its gradients finite
    x = torch.where(x > tiny, x, torch.full_like(x, tiny))
    mx = m * x
    if start is None:
        start = max(n, math.ceil(float(mx.abs().max()))) + 16 if mx.numel() else n + 16
    start = max(start, n + 1)
    d = torch.zeros_like(mx)  # D_start ≈ 0: the downward recurrence forgets it
    logderiv: dict[int, Tensor] = {}
    for k in range(start, 0, -1):
        kz = k / mx
        d = kz - 1.0 / (d + kz)  # D_{k-1}
        if 1 <= k - 1 <= n:
            logderiv[k - 1] = d
    sin, cos = torch.sin(x), torch.cos(x)
    series = x * x / 3.0 * (1.0 - x * x / 10.0 * (1.0 - x * x / 28.0))  # ψ₁ for small x
    psi_prev, psi = sin, torch.where(x < _SMALL, series, sin / x - cos)
    chi_prev, chi = cos, cos / x + sin
    a_list: list[Tensor] = []
    b_list: list[Tensor] = []
    for k in range(1, n + 1):
        xi, xi_prev = torch.complex(psi, -chi), torch.complex(psi_prev, -chi_prev)
        dk = logderiv[k]
        ta = dk / m + k / x
        tb = m * dk + k / x
        a_list.append((ta * psi - psi_prev) / (ta * xi - xi_prev))
        b_list.append((tb * psi - psi_prev) / (tb * xi - xi_prev))
        psi_prev, psi = psi, (2 * k + 1) / x * psi - psi_prev
        chi_prev, chi = chi, (2 * k + 1) / x * chi - chi_prev
    return torch.stack(a_list, -1), torch.stack(b_list, -1)


def angular(mu: Tensor, n: int) -> tuple[Tensor, Tensor]:
    """Return the angular functions π_n(cos Θ) and τ_n(cos Θ) for n = 1 … ``n``.

    Parameters
    ----------
    mu : Tensor
        The cosine of the scattering angle Θ, real or complex (analytic continuation), any
        shape.
    n : int
        Number of orders.

    Returns
    -------
    pi : Tensor
        ``[..., n]``, the dtype of ``mu``.
    tau : Tensor
        ``[..., n]``, the dtype of ``mu``.

    Notes
    -----
    π_1 = 1, π_{k+1} = ((2k+1)μπ_k − (k+1)π_{k−1})/k and τ_k = kμπ_k − (k+1)π_{k−1}
    (Bohren–Huffman eqs. 4.47).

    Examples
    --------
    >>> import torch
    >>> pi, tau = angular(torch.tensor([1.0]), 3)
    >>> pi.tolist(), tau.tolist()  # n(n+1)/2 forward
    ([[1.0, 3.0, 6.0]], [[1.0, 3.0, 6.0]])
    """
    pi_prev = torch.zeros_like(mu)
    pi = torch.ones_like(mu)
    pis: list[Tensor] = []
    taus: list[Tensor] = []
    for k in range(1, n + 1):
        pis.append(pi)
        taus.append(k * mu * pi - (k + 1) * pi_prev)
        pi_prev, pi = pi, ((2 * k + 1) * mu * pi - (k + 1) * pi_prev) / k
    return torch.stack(pis, -1), torch.stack(taus, -1)


def amplitudes(a: Tensor, b: Tensor, pi: Tensor, tau: Tensor) -> tuple[Tensor, Tensor]:
    """Return the amplitude functions S₁ and S₂ from coefficients and angular functions.

    Parameters
    ----------
    a : Tensor
        Electric coefficients ``[..., n]``.
    b : Tensor
        Magnetic coefficients ``[..., n]``.
    pi : Tensor
        The angular functions π_n at the scattering angles, ``[..., n]``, broadcastable against
        the coefficients.
    tau : Tensor
        The angular functions τ_n at the scattering angles, ``[..., n]``.

    Returns
    -------
    s1 : Tensor
        ``Σ (2n+1)/(n(n+1))·(a_n π_n + b_n τ_n)``, complex, the broadcast shape without n.
    s2 : Tensor
        ``Σ (2n+1)/(n(n+1))·(a_n τ_n + b_n π_n)``.

    Examples
    --------
    >>> import torch
    >>> a, b = coefficients(torch.tensor(0.5), 1.33, terms(0.5))
    >>> pi, tau = angular(torch.tensor(1.0, dtype=torch.float64), a.shape[-1])
    >>> s1, s2 = amplitudes(a, b, pi, tau)
    >>> bool(torch.isclose(s1, s2))  # S₁ = S₂ in the forward direction
    True
    """
    n = torch.arange(1, a.shape[-1] + 1, dtype=torch.float64, device=a.device)
    c = (2.0 * n + 1.0) / (n * (n + 1.0))
    s1 = (c * (a * pi + b * tau)).sum(-1)
    s2 = (c * (a * tau + b * pi)).sum(-1)
    return s1, s2


def efficiencies(x: Tensor, a: Tensor, b: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Return the extinction, scattering and backscattering efficiencies.

    Parameters
    ----------
    x : Tensor
        Size parameters, ``[...]``.
    a : Tensor
        Electric coefficients ``[..., n]``.
    b : Tensor
        Magnetic coefficients ``[..., n]``.

    Returns
    -------
    q_ext : Tensor
        ``(2/x²) Σ (2n+1) Re(a_n + b_n)``.
    q_sca : Tensor
        ``(2/x²) Σ (2n+1)(|a_n|² + |b_n|²)``.
    q_back : Tensor
        ``(1/x²) |Σ (2n+1)(−1)^n (a_n − b_n)|²``; cross sections are Q·πr².

    Examples
    --------
    >>> import torch
    >>> x = torch.tensor(1.0, dtype=torch.float64)
    >>> q_ext, q_sca, _ = efficiencies(x, *coefficients(x, 1.5, terms(1.0)))
    >>> bool(torch.isclose(q_ext, q_sca))  # no absorption
    True
    """
    x = torch.as_tensor(x, dtype=torch.float64, device=a.device)
    n = torch.arange(1, a.shape[-1] + 1, dtype=torch.float64, device=a.device)
    weight = 2.0 * n + 1.0
    q_ext = 2.0 / x**2 * (weight * (a + b).real).sum(-1)
    q_sca = 2.0 / x**2 * (weight * (a.abs() ** 2 + b.abs() ** 2)).sum(-1)
    sign = torch.where(n.remainder(2) == 0, 1.0, -1.0)
    q_back = (weight * sign * (a - b)).sum(-1).abs() ** 2 / x**2
    return q_ext, q_sca, q_back
