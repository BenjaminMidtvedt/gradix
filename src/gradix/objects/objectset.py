"""Object sets: populations of objects as structs of arrays over ``[B, T|1, N]`` (§4.6)."""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Mapping
from typing import ClassVar

import torch
from torch import Tensor

from gradix import register
from gradix.objects.labeling import Isotropic
from gradix.objects.spectrum import Spectrum
from gradix.schema.base import DataObject, Node
from gradix.schema.fields import child, field

__all__ = ["Emitters", "ObjectSet", "Solid", "Spheres", "norm"]

_SQRT2 = math.sqrt(2.0)
_SQRT2PI = math.sqrt(2.0 * math.pi)


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
        ``[B, T|1, N, 4]`` unit quaternions ``(w, x, y, z)`` that turn each object's local frame
        into the world frame; None is the identity.
    presence : Tensor, optional
        ``[B, T|1, N]`` in [0, 1]; weights each object linearly. None means all present.
    material : Tensor or complex, optional
        Refractive index per object ``[B, 1, N]`` (possibly complex); material models from M3.
    labeling : Node, optional
        Fluorescent labeling of the objects (:class:`gradix.Labeling`).
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
        quantity="dimensionless",
        role="object",
        event=(4,),
        default=None,
        doc="unit quaternions (w, x, y, z), local to world",
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
    labeling: Node | None = child(default=None, doc="fluorescent labeling")
    parent: Node | None = child(default=None, doc="parent population link (M4)")
    id: Tensor | None = field(
        quantity="index_map", role="object", dtype="integer", default=None, doc="identity"
    )


def norm(x: Tensor, eps: float | Tensor = 1e-24) -> Tensor:
    """Return ``|x|`` over the last axis with a finite gradient at 0.

    Parameters
    ----------
    x : Tensor
        Vectors ``[..., D]``.
    eps : float or Tensor, default 1e-24
        Added under the square root (in squared units of ``x``; broadcastable).

    Returns
    -------
    Tensor
        ``[...]`` lengths.
    """
    return torch.sqrt((x * x).sum(-1) + eps)


@dataclasses.dataclass(frozen=True, kw_only=True, eq=False)
class Solid(ObjectSet):
    """Base of solid object sets: objects with a shape, rasterised onto grids on demand (§4.6).

    A solid's shape is given in its local frame: ``position`` is the object's centre, and
    ``rotation`` turns the local frame into the world frame. Each kind supplies a signed distance
    (:meth:`sdf`, exact or first order; negative inside) and a bounding radius (:meth:`reach`).
    :meth:`occupancy` and :meth:`surface` are the shape's indicator and surface density blurred
    by a Gaussian of width σ, which the raster lowering samples on the grid its consumer asks for
    (ADR-41). The methods take the kind's :attr:`shape_fields` as one mapping, by field name,
    broadcast against the points' leading axes (vector fields keep their last axis): shape field
    names never share a namespace with the methods' own arguments (ADR-42).

    Parameters
    ----------
    position : Tensor
        ``[B, T|1, N, 3]`` µm, object centres.
    rotation : Tensor, optional
        ``[B, T|1, N, 4]`` unit quaternions ``(w, x, y, z)``; None is the identity.
    presence : Tensor, optional
        ``[B, T|1, N]`` in [0, 1]; None means all present.
    material : Tensor or complex, optional
        Refractive index per object.
    labeling : Node, optional
        Fluorescent labeling (:class:`gradix.Labeling`).
    parent : Node, optional
        Parent population link (M4).
    id : Tensor, optional
        ``[B, 1, N]`` int64 persistent identity.
    """

    kind: ClassVar[str] = "solid"
    shape_fields: ClassVar[tuple[str, ...]] = ()
    """The per-object fields that set the shape, in the order the capability methods take them."""
    has_surface: ClassVar[bool] = True
    """Whether the kind has a surface (a label on it, a mask edge); a Gaussian blob has none."""

    def sdf(self, x: Tensor, shape: Mapping[str, Tensor]) -> Tensor:
        """Return the signed distance of local points to the surface, µm (negative inside).

        Parameters
        ----------
        x : Tensor
            Local coordinates ``[..., 3]``, µm.
        shape : Mapping[str, Tensor]
            The shape fields, broadcastable against ``x[..., 0]``.

        Returns
        -------
        Tensor
            ``[...]`` signed distances.

        Raises
        ------
        NotImplementedError
            For kinds without a signed distance.
        """
        raise NotImplementedError(f"{type(self).__name__} has no signed distance")

    def occupancy(self, x: Tensor, blur: float, shape: Mapping[str, Tensor]) -> Tensor:
        """Return the indicator blurred by a Gaussian of width σ (the first-order SDF ramp).

        Parameters
        ----------
        x : Tensor
            Local coordinates ``[..., 3]``, µm.
        blur : float
            Width of the Gaussian blur, µm.
        shape : Mapping[str, Tensor]
            The shape fields.

        Returns
        -------
        Tensor
            ``[...]`` occupancy in [0, 1]: ``½·erfc(d / (√2·σ))`` with ``d`` the signed distance.
        """
        return 0.5 * torch.erfc(self.sdf(x, shape) / (_SQRT2 * blur))

    def surface(self, x: Tensor, blur: float, shape: Mapping[str, Tensor]) -> Tensor:
        """Return the surface blurred by a Gaussian of width σ: a density integrating to the area.

        Parameters
        ----------
        x : Tensor
            Local coordinates ``[..., 3]``, µm.
        blur : float
            Width of the Gaussian blur, µm.
        shape : Mapping[str, Tensor]
            The shape fields.

        Returns
        -------
        Tensor
            ``[...]`` in 1/µm: a Gaussian of the signed distance (exact for exact distances).

        Raises
        ------
        NotImplementedError
            For kinds without a surface.
        """
        if not self.has_surface:
            raise NotImplementedError(f"{type(self).__name__} has no surface")
        d = self.sdf(x, shape)
        return torch.exp(-0.5 * (d / blur) ** 2) / (_SQRT2PI * blur)

    def reach(self, shape: Mapping[str, Tensor]) -> Tensor:
        """Return each object's bounding radius: the farthest its surface gets from its centre.

        Parameters
        ----------
        shape : Mapping[str, Tensor]
            The shape fields.

        Returns
        -------
        Tensor
            Bounding radii, µm.

        Raises
        ------
        NotImplementedError
            In the base class.
        """
        raise NotImplementedError(f"{type(self).__name__} has no bounding radius")


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
class Spheres(Solid):
    """Homogeneous spheres: rasterised with the closed-form blurred ball; Mie and friends from M3.

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
    shape_fields: ClassVar[tuple[str, ...]] = ("radius",)

    radius: Tensor | float = field(
        quantity="length",
        role="object",
        shape_affecting=True,
        constraint="positive",
        doc="sphere radius",
    )

    def sdf(self, x: Tensor, shape: Mapping[str, Tensor]) -> Tensor:
        """Return the exact signed distance ``|x| − R``, µm.

        Parameters
        ----------
        x : Tensor
            Local coordinates ``[..., 3]``, µm.
        shape : Mapping[str, Tensor]
            ``radius``.

        Returns
        -------
        Tensor
            ``[...]`` signed distances.
        """
        return norm(x) - shape["radius"]

    def occupancy(self, x: Tensor, blur: float, shape: Mapping[str, Tensor]) -> Tensor:
        """Return the ball convolved with a Gaussian of width σ, in closed form.

        Parameters
        ----------
        x : Tensor
            Local coordinates ``[..., 3]``, µm.
        blur : float
            Width of the Gaussian blur, µm.
        shape : Mapping[str, Tensor]
            ``radius``.

        Returns
        -------
        Tensor
            ``[...]`` occupancy; it integrates to 4πR³/3 for any σ, so volume gradients are exact.
        """
        radius = shape["radius"]
        r = norm(x).clamp_min(1e-6 * blur)
        mean = 0.5 * (
            torch.erf((radius - r) / (_SQRT2 * blur)) + torch.erf((radius + r) / (_SQRT2 * blur))
        )
        # σ/(r√2π)·[e^{−(R+r)²/2σ²} − e^{−(R−r)²/2σ²}], without cancellation near r = 0
        tail = torch.exp(-0.5 * ((radius - r) / blur) ** 2) * torch.expm1(
            -2.0 * radius * r / blur**2
        )
        return mean + blur / (r * _SQRT2PI) * tail

    def surface(self, x: Tensor, blur: float, shape: Mapping[str, Tensor]) -> Tensor:
        """Return the sphere's surface convolved with a Gaussian of width σ, in closed form.

        Parameters
        ----------
        x : Tensor
            Local coordinates ``[..., 3]``, µm.
        blur : float
            Width of the Gaussian blur, µm.
        shape : Mapping[str, Tensor]
            ``radius``.

        Returns
        -------
        Tensor
            ``[...]`` in 1/µm; it integrates to 4πR².
        """
        radius = shape["radius"]
        r = norm(x).clamp_min(1e-6 * blur)
        # R/(rσ√2π)·[e^{−(r−R)²/2σ²} − e^{−(r+R)²/2σ²}], without cancellation near r = 0
        shell = torch.exp(-0.5 * ((r - radius) / blur) ** 2) * -torch.expm1(
            -2.0 * radius * r / blur**2
        )
        return radius / (r * blur * _SQRT2PI) * shell

    def reach(self, shape: Mapping[str, Tensor]) -> Tensor:
        """Return the bounding radius: the radius.

        Parameters
        ----------
        shape : Mapping[str, Tensor]
            ``radius``.

        Returns
        -------
        Tensor
            Radii, µm.
        """
        return shape["radius"]
