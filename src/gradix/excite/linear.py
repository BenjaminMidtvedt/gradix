"""Linear transduction (``gx.excite.Linear``; ``transduce.linear``): excitation → emission (§5.1).

For point emitters the excitation is evaluated exactly at each emitter (the analytic plane
waves, including evanescent decay and SIM fringes), and each emitter's photon count is scaled by
the local excitation irradiance relative to a reference irradiance: an emitter's ``photons`` is
what it emits under ``reference`` photons/µm², so doubling the source's irradiance doubles the
emission. Optional saturation follows ``I/(1 + I/I_sat)``, normalised to be linear at low
irradiance.
"""

from __future__ import annotations

import dataclasses
from typing import ClassVar

import torch
from torch import Tensor

from gradix import register
from gradix._core.carriers import (
    Background,
    EmitterDensity,
    EmitterSet,
    GaussianSheet,
    PlaneWaves,
)
from gradix._core.contract import Capabilities, Element, Slot, Static
from gradix._core.errors import StructureError
from gradix.objects.environment import Medium
from gradix.schema.fields import field
from gradix.schema.layout import canonical

__all__ = ["Linear", "excitation_irradiance"]


def excitation_irradiance(waves: Background, position: Tensor, medium: Medium) -> Tensor:
    """Return the excitation irradiance at emitter positions, photons/µm² per exposure.

    Parameters
    ----------
    waves : Background
        The excitation: plane waves, a light sheet, or any analytic field.
    position : Tensor
        Emitter positions ``[B|1, A|1, N, 3]``, µm.
    medium : Medium
        The medium the waves travel in (its sample index sets evanescent decay).

    Returns
    -------
    Tensor
        ``Σ_M |Σ_J E|²`` per emitter, ``[B, A, N]``: the source's irradiance for a single
        propagating plane wave.
    """
    wavelengths = waves.wavelengths.to(device=position.device, dtype=position.dtype)
    n = medium.index(dtype=position.dtype, device=position.device, wavelength=wavelengths)
    n = n.real if n.is_complex() else n  # [B|1, L] for a dispersive sample, else [B|1, 1]
    field_at = waves.at(position, n)  # [B, A, M, L, P, N]
    return (field_at.real**2 + field_at.imag**2).sum((2, 3, 4)).to(position.dtype)


def _voxel_centres(density: EmitterDensity) -> Tensor:
    """Return the voxel centres of a density's grid as points, ``[1, 1, Z·Y·X, 3]`` µm."""
    grid = density.grid
    dtype, device = density.data.dtype, density.data.device
    z = grid.z(dtype=dtype, device=device)
    y = grid.xy.y(dtype=dtype, device=device)
    x = grid.xy.x(dtype=dtype, device=device)
    zz, yy, xx = torch.meshgrid(z, y, x, indexing="ij")
    return torch.stack([xx, yy, zz], -1).reshape(1, 1, -1, 3)


@register.element("transduce.linear")
@dataclasses.dataclass(frozen=True, kw_only=True, eq=False)
class Linear(Element[EmitterSet | EmitterDensity]):
    """Linear (optionally saturating) excitation of point emitters.

    Parameters
    ----------
    reference : Tensor or float, default 1.0
        Irradiance, photons/µm² per exposure, under which an emitter emits its ``photons``.
    saturation : Tensor or float, optional
        Saturation irradiance, photons/µm² per exposure; None is linear.

    Examples
    --------
    >>> Linear().saturation is None
    True
    """

    slot: ClassVar[Slot] = Slot.TRANSDUCE
    caps: ClassVar[Capabilities] = Capabilities(
        accepts=frozenset({EmitterSet, EmitterDensity, PlaneWaves, GaussianSheet}),
        produces=(EmitterSet, EmitterDensity),
        reads={"position": "exact", "photons": "exact", "light": "exact", "environment": "exact"},
    )

    reference: Tensor | float = field(
        quantity="irradiance",
        role="image",
        constraint="positive",
        default=1.0,
        doc="reference irradiance",
    )
    saturation: Tensor | float | None = field(
        quantity="irradiance",
        role="image",
        constraint="positive",
        default=None,
        doc="saturation irradiance",
    )

    def forward(self, *inputs: object, static: Static) -> EmitterSet | EmitterDensity:
        """Scale each emitter's photons by its local excitation.

        Parameters
        ----------
        *inputs : object
            The :class:`~gradix.EmitterSet`, the :class:`~gradix.PlaneWaves`, then the medium.
        static : Static
            The (empty) configuration.

        Returns
        -------
        EmitterSet
            The emitters with ``photons`` scaled by ``I(r)/reference``.
        """
        if len(inputs) != 3:
            raise StructureError("Linear takes (EmitterSet, PlaneWaves, medium)")
        emitters, waves, medium = inputs
        emitting = isinstance(emitters, (EmitterSet, EmitterDensity))
        if not emitting or not isinstance(waves, Background):
            raise StructureError(
                "Linear takes emitters (points or a density) and an analytic field"
            )
        if not isinstance(medium, Medium):
            raise StructureError("Linear needs a medium", fix="pass environment=gx.env....")
        if isinstance(emitters, EmitterDensity):
            # the product grid: the excitation at every voxel centre scales its photons
            points = _voxel_centres(emitters)  # [1, 1, Z·Y·X, 3]
            relative = self._relative(excitation_irradiance(waves, points, medium))
            data = emitters.data  # [B|1, A|1, S, Z, Y, X]
            scale = relative.reshape(*relative.shape[:2], 1, *data.shape[-3:])
            return emitters.replace(data=data * scale.to(data.dtype))
        relative = self._relative(excitation_irradiance(waves, emitters.position, medium))
        return emitters.replace(photons=emitters.photons * relative)

    def _relative(self, irradiance: Tensor) -> Tensor:
        """Return the excitation in reference units, saturated when a saturation is given."""
        dtype, device = irradiance.dtype, irradiance.device
        spec = self.schema()
        reference = canonical(self.reference, spec["reference"], dtype=dtype, device=device)
        relative = irradiance / reference.reshape(-1, 1, 1)
        if self.saturation is not None:
            sat = canonical(self.saturation, spec["saturation"], dtype=dtype, device=device)
            ratio = reference.reshape(-1, 1, 1) / sat.reshape(-1, 1, 1)
            relative = relative / (1.0 + relative * ratio)  # I/(1 + I/I_sat), in reference units
        return relative

    def __call__(
        self, *inputs: object, grid: object = None, static: Static | None = None
    ) -> EmitterSet | EmitterDensity:
        """Scale photons eagerly (no configuration is needed).

        Parameters
        ----------
        *inputs : object
            The emitter set, the plane waves and the medium.
        grid : object, optional
            Unused.
        static : Static, optional
            Unused.

        Returns
        -------
        EmitterSet
            The excited emitters.
        """
        return self.forward(*inputs, static=static or Static())
