"""Object sets: populations of objects as structs of arrays over ``[B, T|1, N]`` (§4.6)."""

from __future__ import annotations

import dataclasses
from typing import ClassVar

from torch import Tensor

from gradix import register
from gradix.objects.labeling import Isotropic
from gradix.objects.spectrum import Spectrum
from gradix.schema.base import DataObject, Node
from gradix.schema.fields import child, field

__all__ = ["Emitters", "ObjectSet", "Spheres"]


@dataclasses.dataclass(frozen=True, kw_only=True, eq=False)
class ObjectSet(DataObject):
    """Base of object sets: one population of one primitive kind.

    Per-object fields take ``[B, N]`` or ``[B, T, N]`` (plus their event shape); use ``[1, N]``
    for a single image and ``[B, 1]`` for one value per image. Python numbers are shared.

    Parameters
    ----------
    position : Tensor
        ``[B, T|1, N, 3]`` µm, world frame (§4.1).
    rotation : Tensor, optional
        ``[B, T|1, N, 4]`` unit quaternions; None is the identity.
    presence : Tensor, optional
        ``[B, T|1, N]`` in [0, 1]; weights each object linearly. None means all present.
    material : Tensor or complex, optional
        Refractive index per object ``[B, 1, N]`` (possibly complex); material models from M3.
    labeling : Node, optional
        Fluorescent labeling of the objects (M1).
    parent : Node, optional
        Parent population link for hierarchies (M4).
    id : Tensor, optional
        ``[B, 1, N]`` int64 persistent identity, for tracking labels.
    """

    kind: ClassVar[str] = "object"
    """The population kind the planner routes on, such as ``"point"`` or ``"sphere"``."""
    contrast: ClassVar[str] = "coherent"
    """``"emission"`` for emitters, ``"coherent"`` for scatterers."""

    position: Tensor | float = field(
        quantity="length",
        role="object",
        event=(3,),
        components=("x", "y", "z"),
        shape_affecting=True,
        doc="object positions",
    )
    rotation: Tensor | None = field(
        quantity="dimensionless", role="object", event=(4,), default=None, doc="unit quaternions"
    )
    presence: Tensor | float | None = field(
        quantity="dimensionless",
        role="object",
        constraint="unit_interval",
        default=None,
        doc="presence weight of each object",
    )
    material: Tensor | complex | None = field(
        quantity="index", role="object", dtype="number", default=None, doc="refractive index"
    )
    labeling: Node | None = child(default=None, doc="fluorescent labeling (M1)")
    parent: Node | None = child(default=None, doc="parent population link (M4)")
    id: Tensor | None = field(
        quantity="index_map", role="object", dtype="integer", default=None, doc="identity"
    )


@register.object_set("emitters")
@dataclasses.dataclass(frozen=True, kw_only=True, eq=False)
class Emitters(ObjectSet):
    """Point emitters, never voxelised.

    Parameters
    ----------
    position : Tensor
        ``[B, T|1, N, 3]`` µm.
    rotation : Tensor, optional
        ``[B, T|1, N, 4]`` unit quaternions; unused by isotropic emitters.
    presence : Tensor, optional
        ``[B, T|1, N]`` in [0, 1]; None means all present.
    material : Tensor, optional
        Unused by emitters.
    labeling : Node, optional
        Unused by emitters.
    parent : Node, optional
        Parent population link (M4).
    id : Tensor, optional
        ``[B, 1, N]`` int64 persistent identity.
    photons : Tensor or float
        ``[B, T|1, N]`` expected emitted photons per exposure (before collection).
    emission : Spectrum
        Emission spectrum.
    dipole : Node, default gx.dipole.Isotropic()
        Dipole model.

    Examples
    --------
    >>> import torch
    >>> from gradix.units import nm
    >>> beads = Emitters(
    ...     position=torch.tensor([[[6.4, 6.4, 0.0]]]),
    ...     photons=torch.tensor([[2000.0]]),
    ...     emission=Spectrum.line(600 * nm),
    ... )
    >>> beads.kind
    'point'
    """

    kind: ClassVar[str] = "point"
    contrast: ClassVar[str] = "emission"

    photons: Tensor | float = field(
        quantity="photons",
        role="object",
        constraint="nonnegative",
        doc="expected emitted photons per exposure",
    )
    emission: Spectrum = child(doc="emission spectrum")
    dipole: Node = child(default_factory=Isotropic, doc="dipole model")


@register.object_set("spheres")
@dataclasses.dataclass(frozen=True, kw_only=True, eq=False)
class Spheres(ObjectSet):
    """Homogeneous spheres (coherent rendering with Mie, dipole or Born from M3).

    Parameters
    ----------
    position : Tensor
        ``[B, T|1, N, 3]`` µm, sphere centres.
    rotation : Tensor, optional
        Unused by spheres.
    presence : Tensor, optional
        ``[B, T|1, N]`` in [0, 1]; None means all present.
    material : Tensor or complex, optional
        Refractive index per sphere ``[B, 1, N]``, possibly complex.
    labeling : Node, optional
        Fluorescent labeling (M1).
    parent : Node, optional
        Parent population link (M4).
    id : Tensor, optional
        ``[B, 1, N]`` int64 persistent identity.
    radius : Tensor or float
        ``[B, T|1, N]`` µm.
    """

    kind: ClassVar[str] = "sphere"

    radius: Tensor | float = field(
        quantity="length",
        role="object",
        shape_affecting=True,
        constraint="positive",
        doc="sphere radius",
    )
