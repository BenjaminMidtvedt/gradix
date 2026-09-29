"""Analytic excitation sources (``gx.light``; §4.5): uniform, evanescent, SIM beams, light sheet.

Every source takes an ``irradiance`` in photons·µm⁻² per exposure at the sample and produces
an analytic field (a :class:`~gradix._core.carriers.Background`): :class:`~gradix.PlaneWaves`,
or a :class:`~gradix.GaussianSheet`. An excitation element (``gx.excite.Linear``) evaluates it
exactly at each emitter.
"""

from __future__ import annotations

import dataclasses
import math
from typing import ClassVar, TypeVar

import torch
from torch import Tensor

from gradix import register
from gradix._core.carriers import GaussianSheet, PlaneWaves
from gradix._core.contract import Capabilities, Element, Slot, Static
from gradix._core.precision import result_dtype
from gradix.conventions import safe_sqrt
from gradix.schema.base import iter_leaves
from gradix.schema.fields import field, knob
from gradix.schema.layout import canonical, common_device

__all__ = ["Evanescent", "SIMBeams", "Sheet", "Uniform"]

_W = TypeVar("_W", PlaneWaves, GaussianSheet)


class _Source(Element[_W]):
    """Shared plumbing of analytic sources: dtype, device and the eager call."""

    slot: ClassVar[Slot] = Slot.SOURCE
    caps: ClassVar[Capabilities] = Capabilities(
        accepts=frozenset(), produces=PlaneWaves, polarization=frozenset({1})
    )

    def _setup(self) -> tuple[torch.dtype, torch.device | None]:
        leaves = [v for _, _, v in iter_leaves(self)]
        return result_dtype(*leaves), common_device(*leaves)

    def _value(self, name: str, dtype: torch.dtype, device: torch.device | None) -> Tensor:
        return canonical(getattr(self, name), self.schema()[name], dtype=dtype, device=device)

    def __call__(self, *inputs: object, grid: object = None, static: Static | None = None) -> _W:
        """Build the waves eagerly; sources take no carriers.

        Parameters
        ----------
        *inputs : object
            None.
        grid : object, optional
            Unused; sources are analytic.
        static : Static, optional
            Unused.

        Returns
        -------
        PlaneWaves
            The waves.
        """
        return self.forward(static=static or Static())


def _waves(amplitude: Tensor, u: Tensor, wavelength: Tensor, travel: int) -> PlaneWaves:
    """Pack ``amplitude [B|1, J]`` and ``u [B|1, J, 2]`` into a one-mode, scalar PlaneWaves."""
    cdtype = (
        torch.complex128
        if amplitude.dtype in (torch.float64, torch.complex128)
        else (torch.complex64)
    )
    amp = amplitude.to(cdtype)[:, None, None, :, None, None]  # [B|1, 1, M=1, J, L=1, P=1]
    return PlaneWaves(
        amplitude=amp,
        u=u[:, None, None],  # [B|1, 1, 1, J, 2]
        wavelengths=wavelength[:, None],
        z0=0.0,
        travel=travel,
    )


@register.element("source.uniform")
@dataclasses.dataclass(frozen=True, kw_only=True, eq=False)
class Uniform(_Source[PlaneWaves]):
    """Uniform excitation: one plane wave at normal incidence.

    Parameters
    ----------
    irradiance : Tensor or float, default 1.0
        Irradiance at the sample, photons/µm² per exposure, or per image ``[B]``.
    wavelength : Tensor or float, default 0.488
        Vacuum wavelength, µm.

    Examples
    --------
    >>> float(Uniform(irradiance=4.0)().amplitude.abs() ** 2)
    4.0
    """

    irradiance: Tensor | float = field(
        quantity="irradiance", role="image", constraint="nonnegative", default=1.0, doc="irradiance"
    )
    wavelength: Tensor | float = field(
        quantity="wavelength",
        role="image",
        shape_affecting=True,
        constraint="positive",
        default=0.488,
        doc="vacuum wavelength",
    )

    def forward(self, *inputs: object, static: Static) -> PlaneWaves:
        """Build the plane wave.

        Parameters
        ----------
        *inputs : object
            None.
        static : Static
            The (empty) configuration.

        Returns
        -------
        PlaneWaves
            One wave with ``|amplitude|² = irradiance``.
        """
        dtype, device = self._setup()
        irr = self._value("irradiance", dtype, device)
        wl = self._value("wavelength", dtype, device)
        amp = safe_sqrt(irr)[:, None]
        u = torch.zeros(amp.shape[0], 1, 2, dtype=dtype, device=device)
        return _waves(amp, u, wl, travel=-1)


@register.element("source.evanescent")
@dataclasses.dataclass(frozen=True, kw_only=True, eq=False)
class Evanescent(_Source[PlaneWaves]):
    """TIRF excitation: a wave totally reflected at the coverslip, evanescent in the sample.

    The wave travels from the coverslip (z = 0) into the sample (z < 0) with
    ``u = n_glass·sin(angle)`` along x; beyond the critical angle (``u > n_sample``) its
    irradiance decays as ``exp(2·Im k_z·z)``, with the penetration depth
    ``d = λ/(4π·√(u² − n_sample²))`` for the irradiance. The irradiance is referenced at the
    interface (the transmitted evanescent irradiance at z = 0).

    Parameters
    ----------
    angle : Tensor or float
        Angle of incidence in the coverslip, rad, or per image ``[B]``.
    n_glass : Tensor or float, default 1.518
        Index of the coverslip.
    irradiance : Tensor or float, default 1.0
        Irradiance at the interface, photons/µm² per exposure, or per image ``[B]``.
    wavelength : Tensor or float, default 0.488
        Vacuum wavelength, µm.

    Examples
    --------
    >>> import math
    >>> Evanescent(angle=math.radians(70.0)).n_glass
    1.518
    """

    angle: Tensor | float = field(quantity="angle", role="image", doc="angle of incidence")
    n_glass: Tensor | float = field(
        quantity="index", role="image", constraint="positive", default=1.518, doc="coverslip"
    )
    irradiance: Tensor | float = field(
        quantity="irradiance", role="image", constraint="nonnegative", default=1.0, doc="irradiance"
    )
    wavelength: Tensor | float = field(
        quantity="wavelength",
        role="image",
        shape_affecting=True,
        constraint="positive",
        default=0.488,
        doc="vacuum wavelength",
    )

    def forward(self, *inputs: object, static: Static) -> PlaneWaves:
        """Build the evanescent wave.

        Parameters
        ----------
        *inputs : object
            None.
        static : Static
            The (empty) configuration.

        Returns
        -------
        PlaneWaves
            One wave travelling toward −z with ``|u| = n_glass·sin(angle)``.
        """
        dtype, device = self._setup()
        irr = self._value("irradiance", dtype, device)
        wl = self._value("wavelength", dtype, device)
        ux = self._value("n_glass", dtype, device) * torch.sin(self._value("angle", dtype, device))
        batch = max(irr.shape[0], wl.shape[0], ux.shape[0])
        amp = safe_sqrt(irr).expand(batch)[:, None]
        u = torch.stack([ux.expand(batch), torch.zeros(batch, dtype=dtype, device=device)], -1)
        return _waves(amp, u[:, None, :], wl.expand(batch), travel=-1)


@register.element("source.sim_beams")
@dataclasses.dataclass(frozen=True, kw_only=True, eq=False)
class SIMBeams(_Source[PlaneWaves]):
    """Two-beam structured illumination: ``I(r) = I₀·(1 + m·cos(2π·k·r + φ))``.

    Two coherent plane waves at ``±u`` along the pattern direction interfere into lateral
    fringes of the given period; ``modulation`` sets their relative amplitudes.

    Parameters
    ----------
    period : Tensor or float
        Fringe period, µm.
    angle : Tensor or float, default 0.0
        Direction of the fringe wave vector from +x, rad (or per image ``[B]``).
    phase : Tensor or float, default 0.0
        Pattern phase φ, rad (or per image ``[B]``).
    modulation : Tensor or float, default 1.0
        Modulation depth m in [0, 1].
    irradiance : Tensor or float, default 1.0
        Mean irradiance I₀, photons/µm² per exposure.
    wavelength : Tensor or float, default 0.488
        Vacuum wavelength, µm.

    Examples
    --------
    >>> SIMBeams(period=0.3).modulation
    1.0
    """

    period: Tensor | float = field(
        quantity="length", role="image", constraint="positive", doc="fringe period"
    )
    angle: Tensor | float = field(quantity="angle", role="image", default=0.0, doc="direction")
    phase: Tensor | float = field(quantity="angle", role="image", default=0.0, doc="phase")
    modulation: Tensor | float = field(
        quantity="dimensionless",
        role="image",
        constraint="unit_interval",
        default=1.0,
        doc="modulation depth",
    )
    irradiance: Tensor | float = field(
        quantity="irradiance", role="image", constraint="nonnegative", default=1.0, doc="irradiance"
    )
    wavelength: Tensor | float = field(
        quantity="wavelength",
        role="image",
        shape_affecting=True,
        constraint="positive",
        default=0.488,
        doc="vacuum wavelength",
    )

    def forward(self, *inputs: object, static: Static) -> PlaneWaves:
        """Build the two coherent beams.

        Parameters
        ----------
        *inputs : object
            None.
        static : Static
            The (empty) configuration.

        Returns
        -------
        PlaneWaves
            One mode with J = 2 coherent waves.
        """
        dtype, device = self._setup()
        values = {
            name: self._value(name, dtype, device)
            for name in ("period", "angle", "phase", "modulation", "irradiance", "wavelength")
        }
        batch = max(v.shape[0] for v in values.values())
        v = {k: t.expand(batch) for k, t in values.items()}
        # fringe spatial frequency 1/period = 2·|u|/λ, so |u| = λ/(2·period)
        radius = v["wavelength"] / (2.0 * v["period"])
        ux, uy = radius * torch.cos(v["angle"]), radius * torch.sin(v["angle"])
        u = torch.stack([torch.stack([ux, uy], -1), torch.stack([-ux, -uy], -1)], 1)  # [B, 2, 2]
        # amplitudes a, b with a² + b² = I₀ and 2ab = m·I₀
        m = v["modulation"]
        a = safe_sqrt(v["irradiance"] * (1.0 + safe_sqrt(1.0 - m * m)) / 2.0)
        b = safe_sqrt(v["irradiance"] * (1.0 - safe_sqrt(1.0 - m * m)) / 2.0)
        half = torch.polar(torch.ones_like(v["phase"]), v["phase"] / 2.0)
        amp = torch.stack([a * half, b * half.conj()], 1)  # [B, 2]
        return _waves(amp, u, v["wavelength"], travel=-1)


@register.element("source.gaussian_sheet")
@dataclasses.dataclass(frozen=True, kw_only=True, eq=False)
class Sheet(_Source[GaussianSheet]):
    """A Gaussian light sheet (minimum fidelity, §4.5): uniform across, Gaussian in z.

    Parameters
    ----------
    irradiance : Tensor or float, default 1.0
        Peak irradiance at the waist, photons/µm² per exposure.
    waist : Tensor or float, default 1.0
        Waist 1/e² half-width in z, µm.
    center : Tensor or float, default 0.0
        Height of the sheet's plane, µm.
    focus : Tensor or float, default 0.0
        Position of the waist along ``axis``, µm (field-of-view coordinates).
    wavelength : Tensor or float, default 0.488
        Vacuum wavelength, µm.
    axis : {"x", "y"}, default "x"
        Direction of travel.

    Examples
    --------
    >>> sheet = Sheet(waist=1.5, center=-2.0)()
    >>> float(sheet.waist.reshape(()))
    1.5
    """

    caps: ClassVar[Capabilities] = Capabilities(
        accepts=frozenset(), produces=GaussianSheet, polarization=frozenset({1})
    )

    irradiance: Tensor | float = field(
        quantity="irradiance", role="image", constraint="nonnegative", default=1.0, doc="peak"
    )
    waist: Tensor | float = field(
        quantity="length", role="image", constraint="positive", default=1.0, doc="waist"
    )
    center: Tensor | float = field(quantity="length", role="image", default=0.0, doc="plane")
    focus: Tensor | float = field(quantity="length", role="image", default=0.0, doc="waist at")
    wavelength: Tensor | float = field(
        quantity="wavelength",
        role="image",
        constraint="positive",
        default=0.488,
        shape_affecting=True,
        doc="vacuum wavelength",
    )
    axis: str = knob(default="x", choices=("x", "y"), doc="direction of travel")

    def forward(self, *inputs: object, static: Static) -> GaussianSheet:
        """Build the sheet.

        Parameters
        ----------
        *inputs : object
            None.
        static : Static
            The (empty) configuration.

        Returns
        -------
        GaussianSheet
            The analytic sheet.
        """
        dtype, device = self._setup()
        values = {n: self._value(n, dtype, device) for n in ("irradiance", "waist", "center")}
        return GaussianSheet(
            irradiance=values["irradiance"],
            waist=values["waist"],
            center=values["center"],
            focus=self._value("focus", dtype, device),
            wavelengths=self._value("wavelength", dtype, device)[:, None],
            axis=self.axis,
        )


def penetration_depth(angle: float, n_glass: float, n_sample: float, wavelength: float) -> float:
    """Return the irradiance penetration depth of a TIRF wave, µm.

    Parameters
    ----------
    angle : float
        Angle of incidence in the coverslip, rad.
    n_glass : float
        Index of the coverslip.
    n_sample : float
        Index of the sample.
    wavelength : float
        Vacuum wavelength, µm.

    Returns
    -------
    float
        ``λ/(4π·√((n_glass·sin θ)² − n_sample²))``; infinite below the critical angle.
    """
    excess = (n_glass * math.sin(angle)) ** 2 - n_sample**2
    return math.inf if excess <= 0 else wavelength / (4.0 * math.pi * math.sqrt(excess))
