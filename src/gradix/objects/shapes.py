"""Solid shapes beyond spheres: ellipsoids, capsules, cylinders, boxes and Gaussian blobs (§4.6).

Each kind's shape lives in its local frame, centred on ``position`` and turned into the world
frame by ``rotation`` (quaternions ``(w, x, y, z)``). Solids of revolution (capsules, cylinders)
have their axis along local z. Signed distances are exact except for ellipsoids, whose distance is
first order (exact on the surface); the raster lowering samples their band-limited occupancy or
surface on the grid a consumer asks for (ADR-41).
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from typing import ClassVar

import torch
from torch import Tensor

from gradix import register
from gradix.objects.objectset import Solid, norm
from gradix.schema.fields import field

__all__ = ["Boxes", "Capsules", "Cylinders", "Ellipsoids", "Gaussians"]


@register.object_set("ellipsoids")
@dataclasses.dataclass(frozen=True, kw_only=True, eq=False)
class Ellipsoids(Solid):
    """Ellipsoids with semi-axes along their local x, y and z.

    Parameters
    ----------
    position : Tensor
        ``[B, T|1, N, 3]`` µm, centres.
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
    semi_axes : Tensor
        ``[B, T|1, N, 3]`` µm.

    Examples
    --------
    >>> import torch
    >>> axes = torch.tensor([[[1.0, 0.5, 0.5]]])  # [B, N, 3] µm
    >>> cells = Ellipsoids(position=torch.zeros(1, 1, 3), semi_axes=axes)
    >>> cells.kind
    'ellipsoid'
    """

    kind: ClassVar[str] = "ellipsoid"
    shape_fields: ClassVar[tuple[str, ...]] = ("semi_axes",)

    semi_axes: Tensor = field(
        quantity="length",
        role="object",
        event=(3,),
        components=("x", "y", "z"),
        shape_affecting=True,
        constraint="positive",
        doc="semi-axes along local x, y, z",
    )

    def sdf(self, x: Tensor, shape: Mapping[str, Tensor]) -> Tensor:
        """Return the first-order signed distance ``k₀(k₀ − 1)/k₁``, µm.

        Parameters
        ----------
        x : Tensor
            Local coordinates ``[..., 3]``, µm.
        shape : Mapping[str, Tensor]
            ``semi_axes`` ``[..., 3]``.

        Returns
        -------
        Tensor
            ``[...]``: exact on the surface (with a unit gradient there), first order off it;
            ``k₀ = |x/a|`` and ``k₁ = |x/a²|``.
        """
        axes = shape["semi_axes"]
        shortest = axes.min(dim=-1).values
        k0 = norm(x / axes, 1e-12)
        k1 = norm(x / (axes * axes), 1e-12 / shortest**2)
        return k0 * (k0 - 1.0) / k1

    def reach(self, shape: Mapping[str, Tensor]) -> Tensor:
        """Return the bounding radius: the longest semi-axis.

        Parameters
        ----------
        shape : Mapping[str, Tensor]
            ``semi_axes`` ``[..., 3]``.

        Returns
        -------
        Tensor
            Bounding radii, µm.
        """
        return shape["semi_axes"].max(dim=-1).values


@register.object_set("capsules")
@dataclasses.dataclass(frozen=True, kw_only=True, eq=False)
class Capsules(Solid):
    """Capsules (spherocylinders) along their local z: rod-shaped bacteria, for example.

    Parameters
    ----------
    position : Tensor
        ``[B, T|1, N, 3]`` µm, centres.
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
    length : Tensor or float
        ``[B, T|1, N]`` µm, tip to tip (at least ``2·radius``).
    radius : Tensor or float
        ``[B, T|1, N]`` µm.
    """

    kind: ClassVar[str] = "capsule"
    shape_fields: ClassVar[tuple[str, ...]] = ("length", "radius")

    length: Tensor | float = field(
        quantity="length",
        role="object",
        shape_affecting=True,
        constraint="positive",
        doc="length tip to tip",
    )
    radius: Tensor | float = field(
        quantity="length",
        role="object",
        shape_affecting=True,
        constraint="positive",
        doc="radius",
    )

    def sdf(self, x: Tensor, shape: Mapping[str, Tensor]) -> Tensor:
        """Return the exact signed distance: to the axis segment, minus the radius, µm.

        Parameters
        ----------
        x : Tensor
            Local coordinates ``[..., 3]``, µm.
        shape : Mapping[str, Tensor]
            ``length`` and ``radius``.

        Returns
        -------
        Tensor
            ``[...]`` signed distances.
        """
        radius = shape["radius"]
        half = torch.clamp(0.5 * shape["length"] - radius, min=0.0)
        z = torch.maximum(torch.minimum(x[..., 2], half), -half)
        axis = torch.stack([torch.zeros_like(z), torch.zeros_like(z), z], dim=-1)
        return norm(x - axis) - radius

    def reach(self, shape: Mapping[str, Tensor]) -> Tensor:
        """Return the bounding radius: half the length.

        Parameters
        ----------
        shape : Mapping[str, Tensor]
            ``length`` and ``radius``.

        Returns
        -------
        Tensor
            Bounding radii, µm.
        """
        return torch.maximum(0.5 * shape["length"], shape["radius"])


@register.object_set("cylinders")
@dataclasses.dataclass(frozen=True, kw_only=True, eq=False)
class Cylinders(Solid):
    """Flat-capped cylinders along their local z.

    Parameters
    ----------
    position : Tensor
        ``[B, T|1, N, 3]`` µm, centres.
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
    length : Tensor or float
        ``[B, T|1, N]`` µm, end to end.
    radius : Tensor or float
        ``[B, T|1, N]`` µm.
    """

    kind: ClassVar[str] = "cylinder"
    shape_fields: ClassVar[tuple[str, ...]] = ("length", "radius")

    length: Tensor | float = field(
        quantity="length",
        role="object",
        shape_affecting=True,
        constraint="positive",
        doc="length end to end",
    )
    radius: Tensor | float = field(
        quantity="length",
        role="object",
        shape_affecting=True,
        constraint="positive",
        doc="radius",
    )

    def sdf(self, x: Tensor, shape: Mapping[str, Tensor]) -> Tensor:
        """Return the exact signed distance of a capped cylinder, µm.

        Parameters
        ----------
        x : Tensor
            Local coordinates ``[..., 3]``, µm.
        shape : Mapping[str, Tensor]
            ``length`` and ``radius``.

        Returns
        -------
        Tensor
            ``[...]`` signed distances.
        """
        radial = norm(x[..., :2]) - shape["radius"]
        axial = x[..., 2].abs() - 0.5 * shape["length"]
        outside = torch.sqrt(radial.clamp(min=0.0) ** 2 + axial.clamp(min=0.0) ** 2 + 1e-24)
        inside = torch.maximum(radial, axial).clamp(max=0.0)
        return torch.where((radial > 0) | (axial > 0), outside, inside)

    def reach(self, shape: Mapping[str, Tensor]) -> Tensor:
        """Return the bounding radius: from the centre to a rim.

        Parameters
        ----------
        shape : Mapping[str, Tensor]
            ``length`` and ``radius``.

        Returns
        -------
        Tensor
            Bounding radii, µm.
        """
        return torch.sqrt((0.5 * shape["length"]) ** 2 + shape["radius"] ** 2)


@register.object_set("boxes")
@dataclasses.dataclass(frozen=True, kw_only=True, eq=False)
class Boxes(Solid):
    """Rectangular boxes with edges along their local axes.

    Parameters
    ----------
    position : Tensor
        ``[B, T|1, N, 3]`` µm, centres.
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
    size : Tensor
        ``[B, T|1, N, 3]`` µm, edge lengths along local x, y, z.
    """

    kind: ClassVar[str] = "box"
    shape_fields: ClassVar[tuple[str, ...]] = ("size",)

    size: Tensor = field(
        quantity="length",
        role="object",
        event=(3,),
        components=("x", "y", "z"),
        shape_affecting=True,
        constraint="positive",
        doc="edge lengths along local x, y, z",
    )

    def sdf(self, x: Tensor, shape: Mapping[str, Tensor]) -> Tensor:
        """Return the exact signed distance of a box, µm.

        Parameters
        ----------
        x : Tensor
            Local coordinates ``[..., 3]``, µm.
        shape : Mapping[str, Tensor]
            ``size`` ``[..., 3]``.

        Returns
        -------
        Tensor
            ``[...]`` signed distances.
        """
        q = x.abs() - 0.5 * shape["size"]
        outside = norm(q.clamp(min=0.0))
        inside = q.max(dim=-1).values.clamp(max=0.0)
        return torch.where((q > 0).any(dim=-1), outside, inside)

    def reach(self, shape: Mapping[str, Tensor]) -> Tensor:
        """Return the bounding radius: half the diagonal.

        Parameters
        ----------
        shape : Mapping[str, Tensor]
            ``size`` ``[..., 3]``.

        Returns
        -------
        Tensor
            Bounding radii, µm.
        """
        return 0.5 * norm(shape["size"])


@register.object_set("gaussians")
@dataclasses.dataclass(frozen=True, kw_only=True, eq=False)
class Gaussians(Solid):
    """Anisotropic Gaussian blobs, which add where they overlap; they have no surface.

    The occupancy is ``exp(−½·|x/σ|²)`` (1 at the centre), so a labeling ``density`` is the
    peak density, and a blob holds ``(2π)^{3/2}·σx·σy·σz`` µm³ of it.

    Parameters
    ----------
    position : Tensor
        ``[B, T|1, N, 3]`` µm, centres.
    rotation : Tensor, optional
        ``[B, T|1, N, 4]`` unit quaternions ``(w, x, y, z)``; None is the identity.
    presence : Tensor, optional
        ``[B, T|1, N]`` in [0, 1]; None means all present.
    material : Tensor or complex, optional
        Refractive index contrast per object.
    labeling : Node, optional
        Fluorescent labeling (:class:`gradix.Labeling`; volume labels only).
    parent : Node, optional
        Parent population link (M4).
    id : Tensor, optional
        ``[B, 1, N]`` int64 persistent identity.
    sigma : Tensor
        ``[B, T|1, N, 3]`` µm, standard deviations along local x, y, z.
    """

    kind: ClassVar[str] = "gaussian"
    shape_fields: ClassVar[tuple[str, ...]] = ("sigma",)
    has_surface: ClassVar[bool] = False

    sigma: Tensor = field(
        quantity="length",
        role="object",
        event=(3,),
        components=("x", "y", "z"),
        shape_affecting=True,
        constraint="positive",
        doc="standard deviations along local x, y, z",
    )

    def occupancy(self, x: Tensor, blur: float, shape: Mapping[str, Tensor]) -> Tensor:
        """Return the blob convolved with a Gaussian of width σ (exactly: variances add).

        Parameters
        ----------
        x : Tensor
            Local coordinates ``[..., 3]``, µm.
        blur : float
            Width of the Gaussian blur, µm.
        shape : Mapping[str, Tensor]
            ``sigma`` ``[..., 3]``, the blob's own widths.

        Returns
        -------
        Tensor
            ``[...]``; it integrates to ``(2π)^{3/2}·σx·σy·σz`` for any blur.
        """
        own = shape["sigma"]
        wide = torch.sqrt(own * own + blur**2)
        peak = (own / wide).prod(dim=-1)
        return peak * torch.exp(-0.5 * ((x / wide) ** 2).sum(dim=-1))

    def reach(self, shape: Mapping[str, Tensor]) -> Tensor:
        """Return the bounding radius: four standard deviations of the widest axis.

        Parameters
        ----------
        shape : Mapping[str, Tensor]
            ``sigma`` ``[..., 3]``.

        Returns
        -------
        Tensor
            Bounding radii, µm.
        """
        return 4.0 * shape["sigma"].max(dim=-1).values
