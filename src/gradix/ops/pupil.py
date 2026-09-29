"""Pupil kernels: pupil grids, soft apertures, apodisation, defocus, the ROI MFT and the pixel MTF.

Pupils live in direction space ``u = λ₀·f⊥`` (NA units), which is independent of wavelength:
one grid serves every wavelength bin, only the frequency ``f = u/λ`` changes (§4.3). Phases are
kept carrier-free and matmuls run at IEEE precision (§4.1).
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

from gradix._core.precision import ieee_matmul

__all__ = [
    "defocus_phase",
    "mft_roi",
    "obliquity",
    "pixel_mtf",
    "pupil_axis",
    "soft_aperture",
]


def pupil_axis(
    n: int,
    u_max: float,
    *,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str | None = None,
) -> tuple[Tensor, float]:
    """Return the cell centres of one axis of an ``n × n`` pupil grid over ``[-u_max, u_max]``.

    Parameters
    ----------
    n : int
        Cells per axis.
    u_max : float
        Half-width of the grid, NA units.
    dtype : torch.dtype, default torch.float64
        Dtype of the result.
    device : torch.device or str, optional
        Device of the result.

    Returns
    -------
    tuple of (Tensor, float)
        Cell centres ``[n]`` in NA units and the cell width ``du``.
    """
    du = 2.0 * u_max / n
    k = torch.arange(n, dtype=torch.float64, device=device)
    return (-u_max + (k + 0.5) * du).to(dtype), du


def soft_aperture(radius: Tensor, na: Tensor, du: float) -> Tensor:
    """Return the amplitude of a circular aperture with a soft edge one cell wide, centred on NA.

    The *power* transmission is a smoothstep across the edge cell, which preserves the
    aperture's area (and so the collected power, §4.1) to second order in ``du``; the amplitude
    is its square root. Near either end of the edge the smoothstep is quadratic, so the amplitude
    stays differentiable in NA (exact almost everywhere).

    Parameters
    ----------
    radius : Tensor
        ``|u|`` of the pupil samples, NA units.
    na : Tensor
        Numerical aperture, broadcastable to ``radius``.
    du : float
        Cell width, NA units.

    Returns
    -------
    Tensor
        Amplitude transmission in [0, 1].
    """
    t = torch.clamp((na - radius) / du + 0.5, 0.0, 1.0)
    power = t * t * (3.0 - 2.0 * t)
    safe = torch.where(power > 0, power, torch.ones_like(power))
    return torch.where(power > 0, torch.sqrt(safe), torch.zeros_like(power))


def obliquity(u2: Tensor, n: Tensor) -> Tensor:
    """Return ``cosθ = √(1 − |u|²/n²)`` with a safe square root (finite gradients at grazing).

    Parameters
    ----------
    u2 : Tensor
        ``|u|²`` of the pupil samples, NA units squared.
    n : Tensor
        Refractive index of the medium, broadcastable to ``u2``.

    Returns
    -------
    Tensor
        ``cosθ``, at least ``1e-6``.
    """
    c2 = 1.0 - u2 / (n * n)
    safe = torch.where(c2 > 1e-12, c2, torch.ones_like(c2))
    return torch.where(c2 > 1e-12, torch.sqrt(safe), torch.full_like(c2, 1e-6))


def defocus_phase(u2: Tensor, n: Tensor, wavelength: Tensor, dz: Tensor) -> Tensor:
    """Return the phase of propagating by ``dz`` along +z, relative to the on-axis carrier, in rad.

    That is ``(k_z − k·n)·dz``; an emitter at z imaged in focus at ``focus`` propagates by
    ``dz = focus − z`` (§4.1). The cancellation-free form
    ``k_z − k·n = −2π·|u|²/(λ·(√(n² − |u|²) + n))`` is used. The removed carrier
    ``exp(2πi·n·dz/λ)`` is one global phase per emitter, so intensities do not change; what
    remains is bounded by ``2π·|dz|·NA²/(λ·n)``, small enough for the compute dtype.

    Parameters
    ----------
    u2 : Tensor
        ``|u|²`` of the pupil samples, NA units squared.
    n : Tensor
        Refractive index of the medium.
    wavelength : Tensor
        Vacuum wavelength, µm.
    dz : Tensor
        Propagation distance along +z, µm.

    Returns
    -------
    Tensor
        The phase in rad, broadcast over the inputs; propagating samples only (``|u| < n``).
    """
    kz2 = n * n - u2
    safe = torch.where(kz2 > 0, kz2, torch.ones_like(kz2))
    root = torch.where(kz2 > 0, torch.sqrt(safe), torch.zeros_like(kz2))
    return (-2.0 * math.pi) * u2 / (wavelength * (root + n)) * dz


def mft_roi(
    pupil: Tensor, u: Tensor, du: float, wavelength: Tensor, dx: Tensor, dy: Tensor
) -> Tensor:
    """Evaluate the fields of many pupils at their own ROI sample positions.

    ``E(r) = Σ_u P(u)·exp(+2πi·u·r/λ)·(du/λ)²``: the inverse Fourier sum of the pupil,
    evaluated at positions ``r`` relative to each emitter, so sub-pixel placement is exact.

    Parameters
    ----------
    pupil : Tensor
        Complex pupils ``[G, Np, Np]`` (rows along u_y).
    u : Tensor
        Pupil cell centres ``[Np]``, NA units.
    du : float
        Pupil cell width, NA units.
    wavelength : Tensor
        Vacuum wavelength of each pupil ``[G]``, µm.
    dx : Tensor
        ROI sample x positions relative to each emitter ``[G, Rx]``, µm.
    dy : Tensor
        ROI sample y positions relative to each emitter ``[G, Ry]``, µm.

    Returns
    -------
    Tensor
        Complex fields ``[G, Ry, Rx]`` in √(photons/µm²).
    """
    real = torch.float64 if pupil.dtype == torch.complex128 else torch.float32
    wl = wavelength.to(real)[:, None, None]
    uu = u.to(real)[None, None, :]
    # |u·r/λ| stays below a few hundred cycles on an ROI, so the compute dtype is accurate enough
    phase_y = 2.0 * math.pi * dy.to(real)[:, :, None] * uu / wl
    phase_x = 2.0 * math.pi * dx.to(real)[:, :, None] * uu / wl
    ey = torch.polar(torch.ones_like(phase_y), phase_y)
    ex = torch.polar(torch.ones_like(phase_x), phase_x)
    scale = ((du / wavelength) ** 2).to(pupil.dtype)[:, None, None]
    with ieee_matmul():
        return (ey @ pupil @ ex.transpose(1, 2)) * scale


def pixel_mtf(intensity: Tensor, oversample: int) -> Tensor:
    """Integrate an intensity over camera pixels by the pixel transfer function (§4.3).

    The grid samples the camera pixel pitch ``p`` ``oversample`` times (spacing ``p/s``), so the
    transfer function ``sinc(f_x·p)·sinc(f_y·p)`` depends only on s: ``sinc(k·s/n)`` in FFT index
    units. Each output sample is the pixel-area average around it; exact for intensities
    band-limited below the grid's Nyquist frequency, up to periodic wrap.

    Parameters
    ----------
    intensity : Tensor
        Real samples ``[..., Y, X]``.
    oversample : int
        Samples per camera pixel pitch, s.

    Returns
    -------
    Tensor
        The pixel-averaged intensity, same shape and units per sample.
    """
    ny, nx = intensity.shape[-2:]
    fy = torch.fft.fftfreq(ny, dtype=intensity.dtype, device=intensity.device) * oversample
    fx = torch.fft.rfftfreq(nx, dtype=intensity.dtype, device=intensity.device) * oversample
    mtf = torch.sinc(fy)[:, None] * torch.sinc(fx)[None, :]
    return torch.fft.irfft2(torch.fft.rfft2(intensity) * mtf, s=(ny, nx))
