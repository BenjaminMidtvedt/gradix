"""The plane-wave source element (``source.plane_waves``)."""

from __future__ import annotations

import dataclasses
from typing import ClassVar

import torch
from torch import Tensor

from gradix import register
from gradix._core.carriers import PlaneWaves
from gradix._core.contract import Capabilities, Element, Slot, Static
from gradix._core.precision import result_dtype
from gradix.conventions import safe_sqrt
from gradix.light.polarization import Polarization
from gradix.schema.base import iter_leaves
from gradix.schema.fields import child, field, knob
from gradix.schema.layout import canonical, common_device

__all__ = ["PlaneWave"]


@register.element("source.plane_waves")
@dataclasses.dataclass(frozen=True, eq=False)
class PlaneWave(Element[PlaneWaves]):
    """A plane-wave source: analytic illumination at one direction.

    Parameters
    ----------
    wavelength : Tensor or float
        Vacuum wavelength in µm, or a per-image ``[B]`` tensor.
    direction : Tensor, optional
        Direction ``u = n·sinθ·(cosφ, sinφ)`` in NA units: ``[2]`` or per image ``[B, 2]``;
        None is normal incidence. ``|u| > n`` is evanescent (TIRF).
    irradiance : Tensor or float, default 1.0
        Irradiance at the sample in photons/µm² per exposure, or per image ``[B]``.
    polarization : Polarization, optional
        Polarisation state; None is scalar light (P = 1).
    travel : {1, -1}, default 1
        +1 travels toward the objective (transmitted illumination, defined in the sample
        medium); −1 away from it (epi illumination through the objective, defined in the
        coverslip when the medium is layered, where the coverslip reflects the iSCAT
        reference).

    Examples
    --------
    >>> from gradix.units import nm
    >>> waves = PlaneWave(532 * nm, irradiance=1e5)()
    >>> tuple(waves.amplitude.shape)
    (1, 1, 1, 1, 1, 1)
    """

    slot: ClassVar[Slot] = Slot.SOURCE
    caps: ClassVar[Capabilities] = Capabilities(
        accepts=frozenset(), produces=PlaneWaves, polarization=frozenset({1, 2})
    )
    schema_version: ClassVar[int] = 2
    """2: the travel direction (epi illumination, M3a)."""

    wavelength: Tensor | float = field(
        quantity="wavelength",
        role="image",
        shape_affecting=True,
        constraint="positive",
        doc="vacuum wavelength",
    )
    _: dataclasses.KW_ONLY
    direction: Tensor | None = field(
        quantity="dimensionless",
        role="image",
        event=(2,),
        components=("x", "y"),
        default=None,
        doc="direction in NA units",
    )
    irradiance: Tensor | float = field(
        quantity="irradiance",
        role="image",
        constraint="nonnegative",
        default=1.0,
        doc="photons per µm² per exposure",
    )
    polarization: Polarization | None = child(default=None, doc="polarisation state")
    travel: int = knob(default=1, choices=(1, -1), doc="+1 toward the objective, -1 away")

    def forward(self, *inputs: object, static: Static) -> PlaneWaves:
        """Build the plane waves.

        Parameters
        ----------
        *inputs : object
            None; sources take no carriers.
        static : Static
            The (empty) configuration.

        Returns
        -------
        PlaneWaves
            ``amplitude [B|1, 1, M, 1, 1, P]`` with ``|amplitude|² = irradiance``,
            ``u [B|1, 1, M, 1, 2]`` and ``wavelengths [B|1, 1]``.
        """
        leaves = [v for _, _, v in iter_leaves(self)]
        dtype = result_dtype(*leaves)
        device = common_device(*leaves)
        spec = self.schema()
        wl = canonical(self.wavelength, spec["wavelength"], dtype=dtype, device=device)[:, None]
        irr = canonical(self.irradiance, spec["irradiance"], dtype=dtype, device=device)
        if self.direction is None:
            u = torch.zeros(1, 2, dtype=dtype, device=device)
        else:
            u = canonical(self.direction, spec["direction"], dtype=dtype, device=device)
        cdtype = torch.complex128 if dtype == torch.float64 else torch.complex64
        if self.polarization is None:
            jones = torch.ones(1, 1, 1, dtype=cdtype, device=device)  # [1, M=1, P=1]
        else:
            jones = self.polarization.modes(dtype=cdtype, device=device)  # [B|1, M, 2]
        amp = safe_sqrt(irr).to(cdtype)[:, None, None] * jones  # [B|1, M, P]
        amplitude = amp[:, None, :, None, None, :]  # [B|1, 1, M, 1, 1, P]
        modes = jones.shape[1]
        direction = u[:, None, None, None, :].expand(u.shape[0], 1, modes, 1, 2)
        return PlaneWaves(
            amplitude=amplitude, u=direction, wavelengths=wl, z0=0.0, travel=self.travel
        )

    def __call__(
        self, *inputs: object, grid: object = None, static: Static | None = None
    ) -> PlaneWaves:
        """Build the plane waves eagerly; sources take no carriers.

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
