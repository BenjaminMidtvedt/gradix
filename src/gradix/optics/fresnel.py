"""Fresnel coefficients of planar interfaces in wave-vector form (§4.1, §5.12).

Every coefficient is written with the axial wavenumbers ``k_z = 2π·√(n² − |u|²)/λ`` on the
principal branch (Im k_z ≥ 0), so it holds for evanescent waves too: beyond the critical
angle the transmitted wave decays and |r| = 1. The s coefficients relate the transverse
electric fields; the p ones use the convention in which r_p equals r_s at normal incidence.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

from gradix.conventions import safe_sqrt

__all__ = ["axial", "reflection", "transmission"]


def axial(n: Tensor, u2: Tensor, wavelength: Tensor) -> Tensor:
    """Return the axial wavenumber ``k_z = 2π·√(n² − |u|²)/λ`` on the principal branch.

    Parameters
    ----------
    n : Tensor
        Refractive index (real or complex), broadcastable.
    u2 : Tensor
        ``|u|²`` of the directions, NA units squared, broadcastable.
    wavelength : Tensor
        Vacuum wavelength, µm, broadcastable.

    Returns
    -------
    Tensor
        Complex ``k_z`` in rad/µm, the broadcast shape; imaginary (decaying) where |u| > n.

    Examples
    --------
    >>> import torch
    >>> kz = axial(torch.tensor(1.0), torch.tensor(4.0), torch.tensor(1.0))
    >>> bool(kz.real == 0 and kz.imag > 0)
    True
    """
    square = torch.as_tensor(n) ** 2 - u2
    if not square.is_complex():
        square = square.to(torch.complex128 if square.dtype == torch.float64 else torch.complex64)
    return 2.0 * math.pi * safe_sqrt(square) / wavelength


def transmission(
    kz1: Tensor, kz2: Tensor, n1: Tensor, n2: Tensor, polarization: str = "s"
) -> Tensor:
    """Return the field transmission coefficient from medium 1 into medium 2.

    Parameters
    ----------
    kz1 : Tensor
        Axial wavenumber in medium 1 (complex).
    kz2 : Tensor
        Axial wavenumber in medium 2 (complex).
    n1 : Tensor
        Index of medium 1.
    n2 : Tensor
        Index of medium 2.
    polarization : {"s", "p"}, default "s"
        ``t_s = 2k_z1/(k_z1 + k_z2)`` or ``t_p = 2n₁n₂k_z1/(n₂²k_z1 + n₁²k_z2)``.

    Returns
    -------
    Tensor
        Complex coefficients, the broadcast shape.

    Examples
    --------
    >>> import torch
    >>> kz = torch.tensor([1.5 + 0j, 1.0 + 0j])
    >>> round(float(transmission(kz[0], kz[1], torch.tensor(1.5), torch.tensor(1.0)).real), 6)
    1.2
    """
    if polarization == "s":
        return 2.0 * kz1 / (kz1 + kz2)
    return 2.0 * n1 * n2 * kz1 / (n2 * n2 * kz1 + n1 * n1 * kz2)


def reflection(kz1: Tensor, kz2: Tensor, n1: Tensor, n2: Tensor, polarization: str = "s") -> Tensor:
    """Return the field reflection coefficient of light in medium 1 meeting medium 2.

    Parameters
    ----------
    kz1 : Tensor
        Axial wavenumber in medium 1 (complex).
    kz2 : Tensor
        Axial wavenumber in medium 2 (complex).
    n1 : Tensor
        Index of medium 1.
    n2 : Tensor
        Index of medium 2.
    polarization : {"s", "p"}, default "s"
        ``r_s = (k_z1 − k_z2)/(k_z1 + k_z2)`` or ``r_p = (n₂²k_z1 − n₁²k_z2)/(n₂²k_z1 + n₁²k_z2)``
        with the sign that makes r_p = r_s at normal incidence.

    Returns
    -------
    Tensor
        Complex coefficients, the broadcast shape.

    Examples
    --------
    >>> import torch
    >>> k0 = 2 * 3.141592653589793 / 0.532
    >>> r = reflection(torch.tensor(1.52 * k0 + 0j), torch.tensor(1.33 * k0 + 0j),
    ...                torch.tensor(1.52), torch.tensor(1.33))
    >>> round(float(r.real), 4)  # (1.52 − 1.33)/(1.52 + 1.33)
    0.0667
    """
    if polarization == "s":
        return (kz1 - kz2) / (kz1 + kz2)
    return -(n2 * n2 * kz1 - n1 * n1 * kz2) / (n2 * n2 * kz1 + n1 * n1 * kz2)
