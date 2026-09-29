"""Emitter dipole models (``gx.dipole``) and fluorescent labelings of solids (§4.6)."""

from __future__ import annotations

import dataclasses
from typing import ClassVar

from torch import Tensor

from gradix import register
from gradix._core.errors import StructureError
from gradix.objects.spectrum import Spectrum
from gradix.schema.base import DataObject, Node
from gradix.schema.fields import child, field, knob

__all__ = ["Isotropic", "Labeling"]


@dataclasses.dataclass(frozen=True, kw_only=True, eq=False)
class Isotropic(DataObject):
    """A freely rotating (isotropic) emitter dipole: the default of :class:`gradix.Emitters`.

    Fixed and wobbling dipoles, which need the vectorial PSF, arrive with M4b.
    """

    registry_name: ClassVar[str | None] = "dipole.isotropic"
    """Stable name for signatures and saved inputs."""


@register.data_object("labeling")
@dataclasses.dataclass(frozen=True, kw_only=True, eq=False)
class Labeling(DataObject):
    """A fluorescent label on a population of solids: a density, or photons per object (§4.6).

    One geometry drives both refractive-index contrast and label density, so brightfield, phase
    and fluorescence renders of the same objects stay consistent. Give exactly one of
    ``density`` (per µm³ of the objects' volume), ``surface_density`` (per µm² of their surface)
    or ``photons`` (per object, spread over the volume or surface as ``where`` says). All are
    photons per exposure; photophysics (blinking, bleaching) comes from the caller as per-frame
    values ``[B, T, N]``.

    Parameters
    ----------
    emission : Spectrum
        Emission spectrum.
    density : Tensor or float, optional
        ``[B, T|1, N]`` photons per µm³ of volume.
    surface_density : Tensor or float, optional
        ``[B, T|1, N]`` photons per µm² of surface.
    photons : Tensor or float, optional
        ``[B, T|1, N]`` photons per object.
    where : {"volume", "surface"}, default "volume"
        Where ``photons`` spread.
    dipole : Node, default gx.dipole.Isotropic()
        Dipole model of the fluorophores.

    Examples
    --------
    >>> import gradix as gx
    >>> membrane = Labeling(emission=gx.Spectrum.line(0.6), surface_density=50.0)
    >>> membrane.on_surface
    True
    """

    emission: Spectrum = child(doc="emission spectrum")
    density: Tensor | float | None = field(
        quantity="density",
        role="object",
        constraint="nonnegative",
        default=None,
        doc="photons per µm³ of volume per exposure",
    )
    surface_density: Tensor | float | None = field(
        quantity="surface_density",
        role="object",
        constraint="nonnegative",
        default=None,
        doc="photons per µm² of surface per exposure",
    )
    photons: Tensor | float | None = field(
        quantity="photons",
        role="object",
        constraint="nonnegative",
        default=None,
        doc="photons per object per exposure",
    )
    where: str = knob(
        default="volume", choices=("volume", "surface"), doc="where photons per object spread"
    )
    dipole: Node = child(default_factory=Isotropic, doc="dipole model")

    def __post_init__(self) -> None:
        names = ("density", "surface_density", "photons")
        given = [n for n in names if getattr(self, n) is not None]
        if len(given) != 1:
            msg = f"a labeling takes exactly one of {', '.join(names)}, got {given}"
            raise StructureError(msg, fix="density per µm³, surface_density per µm², or photons")
        if self.density is not None and self.where == "surface":
            msg = "density is per µm³ of volume; a surface label takes surface_density"
            raise StructureError(msg, fix="pass surface_density=… (photons per µm²)")
        super().__post_init__()

    @property
    def on_surface(self) -> bool:
        """Whether the label lies on the objects' surface rather than in their volume.

        Returns
        -------
        bool
            True for ``surface_density`` or for ``photons`` with ``where="surface"``.
        """
        return self.surface_density is not None or (
            self.photons is not None and self.where == "surface"
        )
