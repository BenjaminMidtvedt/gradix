"""Band-limited angular-spectrum propagation of sampled fields (L0; §4.1 conventions)."""

from __future__ import annotations

import math

import torch
from torch import Tensor

__all__ = ["angular_spectrum"]


def angular_spectrum(
    field: Tensor,
    pitch: float | Tensor,
    wavelength: float | Tensor,
    n: float | Tensor,
    distance: float | Tensor,
    *,
    band: float | Tensor | None = None,
    pad: int = 0,
) -> Tensor:
    """Propagate sampled fields by a distance along +z with the angular-spectrum method.

    Each spatial frequency f⊥ is multiplied by ``exp(i·(k_z − k)·Δz)`` with
    ``k_z = 2π·√((n/λ)² − |f⊥|²)``: the on-axis carrier ``exp(i·k·Δz)`` is removed, so a plane
    wave at normal incidence is unchanged. Frequencies beyond ``band`` (a numerical aperture,
    default ``n``) are removed with a soft edge one frequency cell wide, which keeps backward
    propagation (Δz < 0) from amplifying evanescent components.

    Parameters
    ----------
    field : Tensor
        Complex fields ``[..., H, W]`` on a grid of spacing ``pitch``.
    pitch : float or Tensor
        Sample spacing, µm (a number, or broadcastable against the leading axes).
    wavelength : float or Tensor
        Vacuum wavelength, µm.
    n : float or Tensor
        Refractive index of the medium.
    distance : float or Tensor
        Propagation distance along +z, µm; per image as ``[B]`` (broadcast against the leading
        axes from the left).
    band : float or Tensor, optional
        Largest direction |u| = λ|f⊥| kept, in NA units; None keeps the propagating band ``n``.
    pad : int, default 0
        Zero padding on every side before the FFT (removed afterwards), against periodic wrap.

    Returns
    -------
    Tensor
        The propagated fields, same shape and dtype.

    Examples
    --------
    >>> import torch
    >>> flat = torch.ones(1, 8, 8, dtype=torch.complex64)
    >>> bool(torch.allclose(angular_spectrum(flat, 0.1, 0.5, 1.33, 3.0), flat))
    True
    """
    if pad:
        field = torch.nn.functional.pad(field, (pad, pad, pad, pad))
    ny, nx = field.shape[-2:]
    real = torch.float64 if field.dtype == torch.complex128 else torch.float32
    device = field.device

    def as_tensor(value: float | Tensor) -> Tensor:
        t = torch.as_tensor(value, dtype=real, device=device)
        return t.reshape(*t.shape, *([1] * (field.ndim - t.ndim))) if t.ndim else t

    p, wl, index = as_tensor(pitch), as_tensor(wavelength), as_tensor(n)
    dz = torch.as_tensor(distance, dtype=real, device=device)
    dz = dz.reshape(*dz.shape, *([1] * (field.ndim - dz.ndim))) if dz.ndim else dz
    fy = torch.fft.fftfreq(ny, dtype=real, device=device)[:, None] / p
    fx = torch.fft.fftfreq(nx, dtype=real, device=device)[None, :] / p
    u2 = (fx * fx + fy * fy) * wl * wl  # |u|² in NA units
    limit = index if band is None else as_tensor(band)
    du = wl / (p * max(ny, nx))  # one frequency cell, in NA units
    edge = torch.clamp((limit - torch.sqrt(u2)) / du + 0.5, 0.0, 1.0)
    keep = edge * edge * (3.0 - 2.0 * edge)
    root = torch.sqrt(torch.clamp(index * index - u2, min=0.0))
    # (k_z − k)·Δz in the cancellation-free form −2π|u|²/(λ(√(n² − |u|²) + n))·Δz
    phase = (-2.0 * math.pi) * u2 / (wl * (root + index)) * dz
    transfer = keep * torch.polar(torch.ones_like(phase), phase)
    out = torch.fft.ifft2(torch.fft.fft2(field) * transfer.to(field.dtype))
    if pad:
        out = out[..., pad:-pad, pad:-pad]
    return out
