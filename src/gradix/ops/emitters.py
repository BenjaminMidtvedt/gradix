"""Emitter kernels: pixel-integrated Gaussian sprites (the ``emit.gaussian`` tier).

A sprite is an isotropic 2-D Gaussian integrated exactly over each pixel by erf differences, so
pixel integration agrees with the pupil tiers' MTF integration (the consistency contract, §5.9).
The image is separable, ``I = Gyᵀ·diag(w)·Gx``, and rendered as one batched matmul at IEEE
precision.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

from gradix._core.precision import ieee_matmul

__all__ = ["erf_profile", "gaussian_sprites"]

_INV_SQRT2 = 1.0 / math.sqrt(2.0)


def erf_profile(centre: Tensor, sigma: Tensor, edges: Tensor) -> Tensor:
    """Integrate unit-mass 1-D Gaussians over pixel intervals.

    Parameters
    ----------
    centre : Tensor
        Centres, µm, shape ``[G, N]``.
    sigma : Tensor
        Standard deviations, µm, broadcastable to ``[G, N]``.
    edges : Tensor
        Pixel edges, µm, shape ``[G|1, P + 1]``.

    Returns
    -------
    Tensor
        ``[G, N, P]``: the mass of each Gaussian inside each pixel. Tails are differences of
        ``erfc(|t|)``, not of ``erf``, so far pixels keep their relative precision (a 10⁶-photon
        sprite in float32 is accurate far beyond 5σ instead of losing its tail to cancellation).
    """
    scale = _INV_SQRT2 / sigma
    t = (edges[:, None, :] - centre[..., None]) * scale[..., None]
    right = t >= 0
    # erfc(|t|) with the exact derivative at t = 0 (both branches are smooth there)
    c = torch.where(right, torch.special.erfc(t), torch.special.erfc(-t))
    c_lo, c_hi = c[..., :-1], c[..., 1:]
    r_lo, r_hi = right[..., :-1], right[..., 1:]
    # erf(t) = 1 − c on the right, c − 1 on the left
    mass = torch.where(r_lo, c_lo - c_hi, torch.where(r_hi, 2.0 - c_hi - c_lo, c_hi - c_lo))
    return 0.5 * mass


def gaussian_sprites(
    position: Tensor,
    weight: Tensor,
    sigma: Tensor,
    pitch: Tensor | float,
    shape: tuple[int, int],
    origin: tuple[float, float] = (0.0, 0.0),
) -> Tensor:
    """Render pixel-integrated isotropic Gaussians onto a pixel grid.

    Parameters
    ----------
    position : Tensor
        ``[G, N, 2]`` centres ``(x, y)`` in µm; x runs along columns.
    weight : Tensor
        ``[G, N]`` photons of each sprite (its integral over the plane).
    sigma : Tensor
        Standard deviations in µm, broadcastable to ``[G, N]``.
    pitch : Tensor or float
        Pixel pitch in µm: a float, or a tensor broadcastable to ``[G]``.
    shape : tuple of int
        ``(H, W)`` pixels.
    origin : tuple of float, default (0.0, 0.0)
        ``(x, y)`` of the grid's top-left corner, µm.

    Returns
    -------
    Tensor
        ``[G, H, W]`` photons per pixel.
    """
    height, width = shape
    dtype = position.dtype
    device = position.device
    p = torch.as_tensor(pitch, dtype=dtype, device=device).reshape(-1, 1)
    ex = origin[0] + torch.arange(width + 1, dtype=dtype, device=device) * p
    ey = origin[1] + torch.arange(height + 1, dtype=dtype, device=device) * p
    gx = erf_profile(position[..., 0], sigma, ex)  # [G, N, W]
    gy = erf_profile(position[..., 1], sigma, ey)  # [G, N, H]
    with ieee_matmul():
        return torch.bmm((gy * weight[..., None]).transpose(1, 2), gx)
