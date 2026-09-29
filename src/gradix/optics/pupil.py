"""Pupil modifiers (``gx.pupil``): complex multipliers of the objective's pupil (§5.3, §7.5).

A modifier maps pupil coordinates to a complex multiplier; an imaging element fuses every
modifier of ``Objective.pupil`` into one pupil. Aberrations are optical path differences in µm,
so they are correct at every wavelength (§4.1). Plugins subclass :class:`PupilModifier` and
register with ``gx.register.pupil_modifier``.
"""

from __future__ import annotations

import dataclasses
import math

import torch
from torch import Tensor

from gradix import register
from gradix.schema.base import DataObject
from gradix.schema.fields import field, knob
from gradix.schema.layout import canonical
from gradix.special.zernike import zernike_ansi

__all__ = ["Filter", "PixelPupil", "PupilContext", "PupilModifier", "Zernike"]


@dataclasses.dataclass(frozen=True)
class PupilContext:
    """What a modifier may read besides the pupil coordinates.

    Parameters
    ----------
    na : Tensor
        Numerical aperture, ``[B|1, 1, 1, 1]`` (broadcasts against ``[B, S·L, Ky, Kx]``).
    n : Tensor
        Refractive index of the medium the pupil is defined in (the immersion of a layered
        medium), ``[B|1, 1, 1, 1]``, or ``[B|1, S·L, 1, 1]`` when it is dispersive.
    """

    na: Tensor
    n: Tensor


@dataclasses.dataclass(frozen=True, kw_only=True, eq=False)
class PupilModifier(DataObject):
    """Base of pupil modifiers: ``modifier(fx, fy, wl, ctx)`` returns a complex multiplier."""

    def __call__(self, fx: Tensor, fy: Tensor, wl: Tensor, ctx: PupilContext) -> Tensor:
        """Evaluate the multiplier.

        Parameters
        ----------
        fx : Tensor
            Pupil frequencies along x, cycles/µm, ``[B|1, S·L, Ky, Kx]`` (one row per species
            and wavelength bin of the emission table).
        fy : Tensor
            Pupil frequencies along y, cycles/µm, same layout.
        wl : Tensor
            Vacuum wavelengths, µm, ``[B|1, S·L, 1, 1]``.
        ctx : PupilContext
            NA and index.

        Returns
        -------
        Tensor
            Complex multiplier broadcastable to ``[B|1, S·L, Ky, Kx]`` (a ``[Ky, Kx]`` map is
            fine).
        """
        raise NotImplementedError(type(self).__name__)


@register.pupil_modifier("zernike")
@dataclasses.dataclass(frozen=True, kw_only=True, eq=False)
class Zernike(PupilModifier):
    """Aberrations as OSA/ANSI Zernike polynomials of the pupil, in µm of optical path.

    ``OPD(ρ, φ) = Σ_j c_j·Z_j(ρ, φ)`` with ρ = |u|/NA; the phase is ``2π·OPD/λ``.

    Parameters
    ----------
    coeffs : Tensor
        OPD coefficients in µm (RMS-normalised polynomials): ``[J]`` or per image ``[B, J]``.
    indices : tuple of int
        ANSI index of each coefficient.

    Examples
    --------
    >>> import torch
    >>> Zernike(coeffs=torch.tensor([0.05]), indices=(12,)).indices
    (12,)
    """

    coeffs: Tensor = field(quantity="opd", role="image", event=(-1,), doc="OPD coefficients")
    indices: tuple[int, ...] = knob(doc="ANSI indices")

    def __post_init__(self) -> None:
        super().__post_init__()
        count = self.coeffs.shape[-1]
        if count != len(self.indices):
            msg = f"Zernike has {count} coefficients but {len(self.indices)} indices"
            raise ValueError(msg)

    def __call__(self, fx: Tensor, fy: Tensor, wl: Tensor, ctx: PupilContext) -> Tensor:
        """Evaluate ``exp(2πi·OPD/λ)``.

        Parameters
        ----------
        fx : Tensor
            Pupil frequencies along x, cycles/µm.
        fy : Tensor
            Pupil frequencies along y, cycles/µm.
        wl : Tensor
            Vacuum wavelengths, µm.
        ctx : PupilContext
            NA and index.

        Returns
        -------
        Tensor
            Complex multiplier ``[B|1, L, Ky, Kx]``.
        """
        u_x = (fx * wl).to(torch.float64)
        u_y = (fy * wl).to(torch.float64)
        rho = torch.sqrt(u_x * u_x + u_y * u_y) / ctx.na.to(torch.float64)
        phi = torch.atan2(u_y, u_x)
        c = canonical(self.coeffs, self.schema()["coeffs"], dtype=torch.float64, device=fx.device)
        opd = torch.zeros_like(rho)
        for j, index in enumerate(self.indices):
            opd = opd + c[:, j].reshape(-1, 1, 1, 1) * zernike_ansi(index, rho, phi)
        phase = 2.0 * math.pi * opd / wl.to(torch.float64)
        return torch.polar(torch.ones_like(phase), phase)


@register.pupil_modifier("pixel_pupil")
@dataclasses.dataclass(frozen=True, kw_only=True, eq=False)
class PixelPupil(PupilModifier):
    """A pupil given as maps over the pupil disc: an OPD map (µm) and an optional amplitude map.

    The maps span ``u ∈ [−NA, NA]²`` on their own square grid and are resampled bilinearly onto
    the pupil samples, so they stay differentiable (the pixel pupil of example §9(b)).

    Parameters
    ----------
    opd : Tensor
        OPD map in µm, ``[Q, Q]`` or per image ``[B, Q, Q]``.
    amplitude : Tensor, optional
        Amplitude map, same layout; None is 1.
    """

    opd: Tensor = field(quantity="opd", role="image", event=(-1, -1), doc="OPD map over the pupil")
    amplitude: Tensor | None = field(
        quantity="dimensionless",
        role="image",
        event=(-1, -1),
        default=None,
        doc="amplitude map",
    )

    def _sample(self, name: str, ux: Tensor, uy: Tensor, na: Tensor) -> Tensor:
        """Resample a map onto the pupil samples, ``[B, Ky, Kx]``.

        Bilinear interpolation written with gathers (``grid_sample`` has no forward-mode AD):
        the map's corners sit at ``u = ±NA`` and values outside repeat the border.
        """
        maps = canonical(getattr(self, name), self.schema()[name], device=ux.device)  # [B|1,Q,Q]
        maps = maps.to(ux.dtype)
        q_rows, q_cols = maps.shape[-2:]
        scale = na.reshape(-1, 1, 1)
        col = torch.clamp((ux / scale + 1.0) * 0.5 * (q_cols - 1), 0.0, q_cols - 1.0)
        row = torch.clamp((uy / scale + 1.0) * 0.5 * (q_rows - 1), 0.0, q_rows - 1.0)
        batch = max(maps.shape[0], col.shape[0])
        col, row = col.expand(batch, *col.shape[1:]), row.expand(batch, *row.shape[1:])
        maps = maps.expand(batch, q_rows, q_cols)
        c0 = torch.clamp(col.detach().floor(), 0, q_cols - 2).to(torch.int64)
        r0 = torch.clamp(row.detach().floor(), 0, q_rows - 2).to(torch.int64)
        fc, fr = col - c0.to(col.dtype), row - r0.to(row.dtype)
        flat = maps.reshape(batch, -1)

        def at(r: Tensor, c: Tensor) -> Tensor:
            index = (r * q_cols + c).reshape(batch, -1)
            return torch.gather(flat, 1, index).reshape(r.shape)

        top = at(r0, c0) * (1.0 - fc) + at(r0, c0 + 1) * fc
        bottom = at(r0 + 1, c0) * (1.0 - fc) + at(r0 + 1, c0 + 1) * fc
        return top * (1.0 - fr) + bottom * fr

    def __call__(self, fx: Tensor, fy: Tensor, wl: Tensor, ctx: PupilContext) -> Tensor:
        """Evaluate ``amplitude·exp(2πi·OPD/λ)`` at the pupil samples.

        Parameters
        ----------
        fx : Tensor
            Pupil frequencies along x, cycles/µm.
        fy : Tensor
            Pupil frequencies along y, cycles/µm.
        wl : Tensor
            Vacuum wavelengths, µm.
        ctx : PupilContext
            NA and index.

        Returns
        -------
        Tensor
            Complex multiplier ``[B|1, L, Ky, Kx]``.
        """
        # direction space is wavelength independent: the first bin's u serves every bin
        ux = (fx * wl)[:, 0]
        uy = (fy * wl)[:, 0]
        na = ctx.na.reshape(-1)
        opd = self._sample("opd", ux, uy, na)[:, None]  # [B, 1, Ky, Kx]
        phase = 2.0 * math.pi * opd / wl
        if self.amplitude is None:
            amp = torch.ones_like(phase)
        else:
            amp = self._sample("amplitude", ux, uy, na)[:, None].expand_as(phase)
        return torch.polar(amp, phase)


@register.pupil_modifier("filter")
@dataclasses.dataclass(frozen=True, kw_only=True, eq=False)
class Filter(PupilModifier):
    """A central pupil filter: an amplitude and a phase inside a disc about the axis.

    In iSCAT it attenuates (and may phase-shift) the reflected reference, which passes the pupil
    at its centre, to raise the contrast; scattered light inside the disc is filtered with it.
    As a background is evaluated at its own direction, the filter acts on it exactly (§4.4).

    Parameters
    ----------
    radius : Tensor or float
        Disc radius in NA units, or per image ``[B]``.
    transmission : Tensor or float, default 1.0
        Amplitude transmission inside the disc (its square is the power transmission).
    phase : Tensor or float, default 0.0
        Phase inside the disc, rad.
    edge : float, default 0.01
        Width of the soft edge, NA units.

    Examples
    --------
    >>> import torch
    >>> Filter(radius=0.1, transmission=0.1).transmission
    0.1
    """

    radius: Tensor | float = field(
        quantity="dimensionless", role="image", constraint="positive", doc="disc radius, NA units"
    )
    transmission: Tensor | float = field(
        quantity="dimensionless",
        role="image",
        constraint="nonnegative",
        default=1.0,
        doc="amplitude transmission inside the disc",
    )
    phase: Tensor | float = field(quantity="angle", role="image", default=0.0, doc="phase")
    edge: float = knob(default=0.01, doc="soft edge width, NA units")

    def __call__(self, fx: Tensor, fy: Tensor, wl: Tensor, ctx: PupilContext) -> Tensor:
        """Evaluate ``1 + s(u)·(t·e^{iφ} − 1)``, s a smoothstep that is 1 inside the disc.

        Parameters
        ----------
        fx : Tensor
            Pupil frequencies along x, cycles/µm.
        fy : Tensor
            Pupil frequencies along y, cycles/µm.
        wl : Tensor
            Vacuum wavelengths, µm.
        ctx : PupilContext
            NA and index.

        Returns
        -------
        Tensor
            Complex multiplier ``[B|1, L, Ky, Kx]``.
        """
        spec = self.schema()
        real = torch.float64 if fx.dtype == torch.float64 else torch.float32
        radius = canonical(self.radius, spec["radius"], dtype=real, device=fx.device)
        t = canonical(self.transmission, spec["transmission"], dtype=real, device=fx.device)
        phi = canonical(self.phase, spec["phase"], dtype=real, device=fx.device)
        u = torch.sqrt((fx * wl) ** 2 + (fy * wl) ** 2)
        s = torch.clamp((radius.reshape(-1, 1, 1, 1) - u) / self.edge + 0.5, 0.0, 1.0)
        s = s * s * (3.0 - 2.0 * s)
        inner = torch.polar(t.reshape(-1, 1, 1, 1), phi.reshape(-1, 1, 1, 1))
        return 1.0 + s * (inner - 1.0)
