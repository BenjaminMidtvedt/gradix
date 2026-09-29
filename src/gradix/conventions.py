"""Conventions: frame, Fourier signs, propagation, radiometry and Zernike indexing (§4.1).

This module is the single source of the conventions every element follows. It is frozen at M0
and pinned by ``tests/test_conventions.py``.

- **Units.** Lengths and vacuum wavelengths in µm; spatial frequency f in cycles/µm (the
  ``fftfreq`` convention); directions ``u = λ₀·f⊥ = n·sinθ·(cosφ, sinφ)``.
- **Frame.** Right-handed. x runs along columns and y along rows; arrays are ``[..., Y, X]``.
  The lateral origin is the top-left corner of the field of view; pixel ``(i, j)`` has its centre
  at ``((j + ½)·p, (i + ½)·p)``. +z points toward the detection objective, and defocus is
  ``Δ = z − focus`` (Δ > 0: between the focal plane and the objective).
- **Fourier.** Time dependence ``e^{−iωt}``; plane waves ``e^{+ik·r}``; ``torch.fft`` with
  ``norm="backward"``. Propagating by +Δz multiplies an angular spectrum by ``e^{+i·k_z·Δz}``,
  ``k_z = 2π·√((n/λ₀)² − |f⊥|²)`` on the branch ``Im k_z ≥ 0`` (evanescent waves decay), in the
  ``H = exp(Δz·H_exp)`` form so gradients reach Δz.
- **Radiometry.** The PSF of an isotropic emitter integrates to its collection efficiency,
  ``(1 − cosθ_max)/2`` in an index-matched medium.
- **Aberrations** are OPDs in µm; Zernike polynomials use OSA/ANSI indices with RMS
  normalisation (:mod:`gradix.special.zernike`).
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

__all__ = [
    "FFT_NORM",
    "GAUSSIAN_SIGMA_FACTOR",
    "PLANE_WAVE_SIGN",
    "TIME_SIGN",
    "angular_frequencies",
    "ansi_to_nm",
    "collection_efficiency",
    "defocus",
    "delta_k2",
    "depth_of_field",
    "direction",
    "fourier_shift",
    "from_angular_spectrum",
    "gaussian_sigma",
    "kz",
    "nm_to_ansi",
    "noll_to_ansi",
    "noll_to_nm",
    "pixel_centers",
    "propagator",
    "safe_sqrt",
    "sin_theta_max",
    "to_angular_spectrum",
]

FFT_NORM = "backward"
"""The ``norm`` argument of every ``torch.fft`` call."""
TIME_SIGN = -1
"""Time dependence ``exp(TIME_SIGN·iωt)``."""
PLANE_WAVE_SIGN = +1
"""Plane waves are ``exp(PLANE_WAVE_SIGN·i·k·r)``."""
GAUSSIAN_SIGMA_FACTOR = 0.22
"""σ = 0.22·λ/NA: the L2-optimal Gaussian approximation of the in-focus scalar PSF (§5.1)."""


def pixel_centers(
    n: int,
    spacing: float | Tensor,
    origin: float | Tensor = 0.0,
    *,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str | None = None,
) -> Tensor:
    """Return the centres of ``n`` pixels along one axis: ``origin + (j + ½)·spacing``.

    Parameters
    ----------
    n : int
        Number of pixels.
    spacing : float or Tensor
        Pixel pitch in µm; a tensor broadcasts against the result (its trailing dim is added).
    origin : float or Tensor, default 0.0
        Position of the first pixel's leading edge, µm.
    dtype : torch.dtype, default torch.float32
        Dtype of the result.
    device : torch.device or str, optional
        Device of the result.

    Returns
    -------
    Tensor
        Shape ``[..., n]`` in µm.
    """
    j = torch.arange(n, dtype=dtype, device=device) + 0.5
    if isinstance(spacing, Tensor) or isinstance(origin, Tensor):
        sp = torch.as_tensor(spacing, dtype=dtype, device=device)[..., None]
        og = torch.as_tensor(origin, dtype=dtype, device=device)[..., None]
        return og + j * sp
    return origin + j * spacing


def angular_frequencies(
    shape: tuple[int, int],
    spacing: float,
    *,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str | None = None,
) -> tuple[Tensor, Tensor]:
    """Return the FFT frequencies of a grid in cycles/µm, as ``[Ky, 1]`` and ``[1, Kx]``.

    Parameters
    ----------
    shape : tuple of int
        ``(Y, X)``.
    spacing : float
        Sample spacing, µm.
    dtype : torch.dtype, default torch.float64
        Dtype of the result.
    device : torch.device or str, optional
        Device of the result.

    Returns
    -------
    tuple of Tensor
        ``(fy, fx)`` with shapes ``[Y, 1]`` and ``[1, X]``.
    """
    fy = torch.fft.fftfreq(shape[0], d=spacing, dtype=dtype, device=device)[:, None]
    fx = torch.fft.fftfreq(shape[1], d=spacing, dtype=dtype, device=device)[None, :]
    return fy, fx


def safe_sqrt(x: Tensor) -> Tensor:
    """Return ``√x`` with a finite (zero) gradient where ``x = 0`` (the double-where form).

    Complex inputs take the principal root, so ``Im √x ≥ 0`` for ``x`` on the negative real axis
    with ``Im x = +0``; real inputs must be non-negative.

    Parameters
    ----------
    x : Tensor
        Real (non-negative) or complex values.

    Returns
    -------
    Tensor
        ``√x``, with zero gradient at ``x = 0`` instead of an infinite or NaN one.
    """
    zero = x == 0
    safe = torch.where(zero, torch.ones_like(x), x)
    return torch.where(zero, torch.zeros_like(x), torch.sqrt(safe))


def kz(n: Tensor | complex, wavelength: Tensor | float, fx: Tensor, fy: Tensor) -> Tensor:
    """Return the axial wavenumber ``k_z = 2π·√((n/λ₀)² − f²)`` in rad/µm, with ``Im k_z ≥ 0``.

    Parameters
    ----------
    n : Tensor or complex
        Refractive index of the medium.
    wavelength : Tensor or float
        Vacuum wavelength, µm.
    fx : Tensor
        Frequencies along x, cycles/µm.
    fy : Tensor
        Frequencies along y, cycles/µm.

    Returns
    -------
    Tensor
        Complex128 k_z, broadcast over the inputs. Evanescent components have ``Re k_z = 0``
        and ``Im k_z > 0``, so ``exp(i·k_z·Δz)`` decays for Δz > 0. The gradient is finite at
        the cut-off ``|f| = n/λ₀`` (:func:`safe_sqrt`).
    """
    k2 = (torch.as_tensor(n) / torch.as_tensor(wavelength)) ** 2 - (fx * fx + fy * fy)
    return 2.0 * math.pi * safe_sqrt(torch.as_tensor(k2).to(torch.complex128))


def delta_k2(shape: tuple[int, int], spacing: float) -> float:
    """Return the area ``Δk_x·Δk_y`` of one angular-spectrum bin of an FFT grid, (rad/µm)².

    Parameters
    ----------
    shape : tuple of int
        ``(Y, X)``.
    spacing : float
        Sample spacing, µm.

    Returns
    -------
    float
        ``(2π/(X·d))·(2π/(Y·d))``.
    """
    return (2.0 * math.pi / (shape[1] * spacing)) * (2.0 * math.pi / (shape[0] * spacing))


def to_angular_spectrum(field: Tensor, spacing: float) -> Tensor:
    """Return the angular spectrum ``A(k⊥)`` of a sampled field, in the continuous convention.

    ``E(r⊥) = ∫ A(k⊥)·e^{i k⊥·r⊥} d²k⊥`` (§4.1), so ``A(k⊥) = (2π)⁻² ∫ E·e^{−i k⊥·r⊥} d²r⊥ ≈
    d²/(2π)²·FFT(E)``, on the frequencies of :func:`angular_frequencies` (times 2π).

    Parameters
    ----------
    field : Tensor
        Complex or real samples ``[..., Y, X]`` on pixel centres ``(i − (N−1)/2)·d`` or on
        ``i·d``; the spectrum's phase refers to the sample with index 0.
    spacing : float
        Sample spacing d, µm.

    Returns
    -------
    Tensor
        ``A`` on the FFT frequency grid, ``[..., Y, X]``, in field units·µm².
    """
    scale = spacing * spacing / (2.0 * math.pi) ** 2
    return torch.fft.fft2(field, norm=FFT_NORM) * scale


def from_angular_spectrum(spectrum: Tensor, spacing: float) -> Tensor:
    """Return the field of an angular spectrum: the discrete ``Σ A(k⊥)·e^{i k⊥·r⊥}·Δk²``.

    The inverse of :func:`to_angular_spectrum`; a direct-summation test pins the factors.

    Parameters
    ----------
    spectrum : Tensor
        ``A`` on the FFT frequency grid, ``[..., Y, X]``, field units·µm².
    spacing : float
        Sample spacing d of the output, µm.

    Returns
    -------
    Tensor
        Complex field samples ``[..., Y, X]``.
    """
    y, x = spectrum.shape[-2:]
    scale = delta_k2((y, x), spacing) * (y * x)
    return torch.fft.ifft2(spectrum, norm=FFT_NORM) * scale


def propagator(kz_: Tensor, dz: Tensor | float) -> Tensor:
    """Return the angular-spectrum propagator ``H = exp(Δz·H_exp)`` with ``H_exp = i·k_z``.

    Parameters
    ----------
    kz_ : Tensor
        Complex k_z from :func:`kz`, rad/µm.
    dz : Tensor or float
        Propagation distance along +z, µm.

    Returns
    -------
    Tensor
        Complex transfer function; gradients reach ``dz``.
    """
    return torch.exp(torch.as_tensor(dz) * (1j * kz_))


def fourier_shift(x: Tensor, shift: tuple[float, float]) -> Tensor:
    """Shift a sampled function by ``(dx, dy)`` samples through its spectrum.

    ``y(r) = x(r − d)`` is ``Y(f) = X(f)·exp(−2πi·f·d)`` under the ``e^{+ik·r}`` convention.
    For integer shifts this equals ``torch.roll(x, (dy, dx), (-2, -1))``.

    Parameters
    ----------
    x : Tensor
        ``[..., Y, X]`` samples.
    shift : tuple of float
        ``(dx, dy)`` in samples (x along columns).

    Returns
    -------
    Tensor
        The shifted samples, complex.
    """
    dx, dy = shift
    ny, nx = x.shape[-2:]
    fy = torch.fft.fftfreq(ny, dtype=torch.float64, device=x.device)[:, None]
    fx = torch.fft.fftfreq(nx, dtype=torch.float64, device=x.device)[None, :]
    ramp = torch.exp(-2j * math.pi * (fx * dx + fy * dy))
    spectrum = torch.fft.fft2(x.to(torch.complex128), norm=FFT_NORM)
    return torch.fft.ifft2(spectrum * ramp, norm=FFT_NORM)


def defocus(z: Tensor | float, focus: Tensor | float) -> Tensor | float:
    """Return the defocus ``Δ = z − focus``; Δ > 0 lies between the focal plane and the objective.

    Parameters
    ----------
    z : Tensor or float
        Axial position, µm.
    focus : Tensor or float
        Focal-plane position, µm.

    Returns
    -------
    Tensor or float
        Δ in µm.
    """
    return z - focus


def direction(theta: Tensor | float, phi: Tensor | float, n: Tensor | float = 1.0) -> Tensor:
    """Return the direction ``u = n·sinθ·(cosφ, sinφ)`` of a plane wave.

    Parameters
    ----------
    theta : Tensor or float
        Polar angle from +z, rad.
    phi : Tensor or float
        Azimuth from +x toward +y, rad.
    n : Tensor or float, default 1.0
        Refractive index of the medium.

    Returns
    -------
    Tensor
        ``[..., 2]`` in NA units.
    """
    theta_t = torch.as_tensor(theta, dtype=torch.float64)
    phi_t = torch.as_tensor(phi, dtype=torch.float64)
    s = torch.as_tensor(n, dtype=torch.float64) * torch.sin(theta_t)
    return torch.stack([s * torch.cos(phi_t), s * torch.sin(phi_t)], dim=-1)


def sin_theta_max(na: Tensor | float, n: Tensor | float) -> Tensor:
    """Return ``sinθ_max = min(NA/n, 1)``.

    Parameters
    ----------
    na : Tensor or float
        Numerical aperture.
    n : Tensor or float
        Refractive index of the (index-matched) medium.

    Returns
    -------
    Tensor
        The sine of the collection half-angle.
    """
    return torch.clamp(torch.as_tensor(na) / torch.as_tensor(n), max=1.0)


def collection_efficiency(na: Tensor | float, n: Tensor | float) -> Tensor:
    """Return the collection efficiency ``(1 − cosθ_max)/2`` of an isotropic emitter (§4.1).

    Parameters
    ----------
    na : Tensor or float
        Numerical aperture.
    n : Tensor or float
        Refractive index of the index-matched medium.

    Returns
    -------
    Tensor
        Fraction of the emitted photons collected; ≈0.31 for NA 1.4 in oil (n = 1.518).
    """
    s = sin_theta_max(na, n)
    return 0.5 * (1.0 - torch.sqrt(torch.clamp(1.0 - s * s, min=0.0)))


def gaussian_sigma(wavelength: Tensor | float, na: Tensor | float) -> Tensor:
    """Return the in-focus width ``σ = 0.22·λ/NA`` of the Gaussian tier (µm).

    Parameters
    ----------
    wavelength : Tensor or float
        Vacuum wavelength, µm.
    na : Tensor or float
        Numerical aperture.

    Returns
    -------
    Tensor
        The width σ in µm.
    """
    return GAUSSIAN_SIGMA_FACTOR * torch.as_tensor(wavelength) / torch.as_tensor(na)


def depth_of_field(wavelength: Tensor | float, na: Tensor | float, n: Tensor | float) -> Tensor:
    """Return the depth of field ``n·λ₀/NA²`` (µm).

    Parameters
    ----------
    wavelength : Tensor or float
        Vacuum wavelength, µm.
    na : Tensor or float
        Numerical aperture.
    n : Tensor or float
        Refractive index of the medium.

    Returns
    -------
    Tensor
        The depth of field, µm.
    """
    na_t = torch.as_tensor(na)
    return torch.as_tensor(n) * torch.as_tensor(wavelength) / (na_t * na_t)


def ansi_to_nm(j: int) -> tuple[int, int]:
    """Convert an OSA/ANSI Zernike index to ``(n, m)``.

    Parameters
    ----------
    j : int
        ANSI index, from 0 (piston).

    Returns
    -------
    tuple of int
        Radial order n and azimuthal frequency m (negative for sine terms).
    """
    if j < 0:
        msg = f"ANSI indices start at 0, got {j}"
        raise ValueError(msg)
    n = math.ceil((-3 + math.sqrt(9 + 8 * j)) / 2)
    m = 2 * j - n * (n + 2)
    return n, m


def nm_to_ansi(n: int, m: int) -> int:
    """Convert ``(n, m)`` to the OSA/ANSI index ``(n(n + 2) + m)/2``.

    Parameters
    ----------
    n : int
        Radial order.
    m : int
        Azimuthal frequency, ``|m| ≤ n`` with ``n − m`` even.

    Returns
    -------
    int
        The ANSI index.
    """
    if abs(m) > n or (n - m) % 2:
        msg = f"invalid Zernike indices (n, m) = ({n}, {m})"
        raise ValueError(msg)
    return (n * (n + 2) + m) // 2


def noll_to_nm(j: int) -> tuple[int, int]:
    """Convert a Noll Zernike index to ``(n, m)``.

    Parameters
    ----------
    j : int
        Noll index, from 1 (piston).

    Returns
    -------
    tuple of int
        Radial order n and azimuthal frequency m (negative for sine terms).
    """
    if j < 1:
        msg = f"Noll indices start at 1, got {j}"
        raise ValueError(msg)
    n = 0
    j1 = j - 1
    while j1 > n:
        n += 1
        j1 -= n
    m = (-1) ** j * ((n % 2) + 2 * ((j1 + ((n + 1) % 2)) // 2))
    return n, m


def noll_to_ansi(j: int) -> int:
    """Convert a Noll index to the OSA/ANSI index.

    Parameters
    ----------
    j : int
        Noll index, from 1.

    Returns
    -------
    int
        The ANSI index.
    """
    return nm_to_ansi(*noll_to_nm(j))
