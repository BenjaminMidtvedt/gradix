"""Fourier helpers: the matrix Fourier transform (MFT) at arbitrary frequencies and positions.

The MFT evaluates ``X(f) = Σ_r x(r)·exp(−2πi·f·r)`` (the ``torch.fft`` sign and ``"backward"``
normalisation) on any separable set of output frequencies with two matmuls. Phase matrices are
built in float64 and cast (§4.1), and the matmuls run at IEEE precision (TF32 would cost
three orders of magnitude, §8.1).
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

from gradix._core.precision import ieee_matmul, unit_phasor

__all__ = ["dft_matrix", "mft2"]


def dft_matrix(
    freq: Tensor, coords: Tensor, *, sign: int = -1, dtype: torch.dtype = torch.complex64
) -> Tensor:
    """Return the 1-D Fourier matrix ``exp(sign·2πi·f_k·r_j)``.

    Parameters
    ----------
    freq : Tensor
        Output frequencies in cycles/µm, ``[K]``.
    coords : Tensor
        Sample positions in µm, ``[J]``.
    sign : {-1, 1}, default -1
        −1 for the forward transform (``torch.fft.fft`` sign), +1 for the inverse.
    dtype : torch.dtype, default torch.complex64
        Complex dtype of the result.

    Returns
    -------
    Tensor
        ``[K, J]`` complex matrix, built from a float64 phase.
    """
    phase = (
        (sign * 2.0 * math.pi) * freq.to(torch.float64)[:, None] * coords.to(torch.float64)[None, :]
    )
    return unit_phasor(phase, dtype=dtype)


def mft2(
    x: Tensor,
    coords: tuple[Tensor, Tensor],
    freq: tuple[Tensor, Tensor],
    *,
    sign: int = -1,
) -> Tensor:
    """Evaluate the 2-D Fourier sum of samples at arbitrary separable frequencies.

    Parameters
    ----------
    x : Tensor
        Complex (or real) samples ``[..., Y, X]``.
    coords : tuple of Tensor
        ``(y, x)`` sample positions in µm, ``[Y]`` and ``[X]``.
    freq : tuple of Tensor
        ``(fy, fx)`` output frequencies in cycles/µm, ``[Ky]`` and ``[Kx]``.
    sign : {-1, 1}, default -1
        Exponent sign: −1 matches ``torch.fft.fft2``, +1 the unnormalised inverse.

    Returns
    -------
    Tensor
        ``[..., Ky, Kx]``: ``Σ x[y, x]·exp(sign·2πi(fy·y + fx·x))``. With integer coordinates
        ``0 … N−1`` and ``fftfreq`` frequencies it equals ``torch.fft.fft2(x)``.
    """
    cdtype = torch.complex128 if x.dtype in (torch.float64, torch.complex128) else torch.complex64
    ey = dft_matrix(freq[0], coords[0], sign=sign, dtype=cdtype).to(x.device)
    ex = dft_matrix(freq[1], coords[1], sign=sign, dtype=cdtype).to(x.device)
    with ieee_matmul():
        return ey @ x.to(cdtype) @ ex.transpose(0, 1)
