"""Zernike polynomials: OSA/ANSI indexing, RMS normalisation, float64.

``Z_n^m(ρ, φ) = N_n^m · R_n^|m|(ρ) · (cos mφ for m ≥ 0, sin |m|φ for m < 0)`` with
``N_n^m = √(2(n + 1)/(1 + δ_m0))``, so every polynomial has unit RMS over the unit disc.
The radial polynomials are evaluated in float64 through the Jacobi form
``R_n^m(ρ) = (−1)^k·ρ^m·P_k^{(m,0)}(1 − 2ρ²)``, ``k = (n − m)/2``, and its three-term
recurrence, which stays within 2·10⁻¹⁴ of exact rational arithmetic up to n = 60 (the explicit
sum loses all digits by n ≈ 40 in float64 and n ≈ 20 in float32).
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

from gradix.conventions import ansi_to_nm

__all__ = ["radial", "zernike", "zernike_ansi"]


def radial(n: int, m: int, rho: Tensor) -> Tensor:
    """Return the radial polynomial ``R_n^|m|(ρ)``.

    Parameters
    ----------
    n : int
        Radial order.
    m : int
        Azimuthal frequency; its sign is ignored.
    rho : Tensor
        Normalised radius in [0, 1].

    Returns
    -------
    Tensor
        ``R_n^|m|(ρ)``, same shape and dtype as ``rho`` (computed in float64).

    Raises
    ------
    ValueError
        If ``(n, m)`` is not a Zernike index: ``n ≥ 0``, ``|m| ≤ n``, ``n − |m|`` even.
    """
    _check_index(n, m)
    m = abs(m)
    r = rho.to(torch.float64)
    x = 1.0 - 2.0 * r * r
    k_max = (n - m) // 2
    p_prev, p = torch.ones_like(x), torch.ones_like(x)
    if k_max >= 1:
        p = (m + 1) + (m + 2) * (x - 1.0) / 2.0
    for k in range(2, k_max + 1):
        a1 = 2 * k * (k + m) * (2 * k + m - 2)
        a2 = (2 * k + m - 1) * m * m
        a3 = (2 * k + m - 1) * (2 * k + m) * (2 * k + m - 2)
        a4 = 2 * (k + m - 1) * (k - 1) * (2 * k + m)
        p_prev, p = p, ((a2 + a3 * x) * p - a4 * p_prev) / a1
    out = (-1.0 if k_max % 2 else 1.0) * r**m * p
    return out.to(rho.dtype)


def _check_index(n: int, m: int) -> None:
    """Raise unless ``(n, m)`` indexes a Zernike polynomial."""
    if n < 0 or abs(m) > n or (n - abs(m)) % 2:
        msg = f"(n, m) = ({n}, {m}) is not a Zernike index: need n >= 0, |m| <= n, n - |m| even"
        raise ValueError(msg)


def zernike(n: int, m: int, rho: Tensor, phi: Tensor) -> Tensor:
    """Return the RMS-normalised Zernike polynomial ``Z_n^m(ρ, φ)``.

    Parameters
    ----------
    n : int
        Radial order.
    m : int
        Azimuthal frequency (negative for sine terms).
    rho : Tensor
        Normalised radius in [0, 1].
    phi : Tensor
        Azimuth from +x toward +y, rad.

    Returns
    -------
    Tensor
        The polynomial; values outside the unit disc are not masked.

    Raises
    ------
    ValueError
        If ``(n, m)`` is not a Zernike index.
    """
    _check_index(n, m)
    norm = math.sqrt(2.0 * (n + 1) / (2.0 if m == 0 else 1.0))
    r = radial(n, m, rho)
    if m > 0:
        return norm * r * torch.cos(m * phi)
    if m < 0:
        return norm * r * torch.sin(-m * phi)
    return norm * r


def zernike_ansi(j: int, rho: Tensor, phi: Tensor) -> Tensor:
    """Return the RMS-normalised Zernike polynomial with OSA/ANSI index ``j``.

    Parameters
    ----------
    j : int
        ANSI index (0 = piston, 4 = defocus).
    rho : Tensor
        Normalised radius in [0, 1].
    phi : Tensor
        Azimuth, rad.

    Returns
    -------
    Tensor
        The polynomial.
    """
    n, m = ansi_to_nm(j)
    return zernike(n, m, rho, phi)
