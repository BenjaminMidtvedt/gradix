"""Labels on the camera grid (``gx.labels``; §6.6): heatmaps, tables, masks, distances."""

from __future__ import annotations

import dataclasses
from typing import ClassVar

import torch
from torch import Tensor

from gradix import register
from gradix._core.carriers import EmitterSet
from gradix._core.errors import StructureError
from gradix.coords import to_pixels
from gradix.detect.camera import Camera
from gradix.labels.positions import Label
from gradix.objects.objectset import ObjectSet
from gradix.optics.objective import Objective
from gradix.schema.fields import knob
from gradix.schema.layout import canonical

__all__ = ["DistanceMap", "EmitterTable", "Heatmap", "InstanceMask", "SemanticMask"]


def _pixels(objects: ObjectSet, camera: Camera, objective: Objective) -> tuple[Tensor, Tensor]:
    """Positions in pixel coordinates ``[B|1, T|1, N, 3]`` and presence ``[B|1, T|1, N]``."""
    schema = objects.schema()
    pos = canonical(objects.position, schema["position"])
    px = to_pixels(pos, camera, objective)
    if objects.presence is None:
        presence = torch.ones(px.shape[:-1], dtype=px.dtype, device=px.device)
    else:
        presence = canonical(objects.presence, schema["presence"], dtype=px.dtype)
        presence = presence.to(px.device)
    return px, presence


def _squeeze_time(t: Tensor) -> Tensor:
    return t[:, 0] if t.shape[1] == 1 else t


@register.label("heatmap")
@dataclasses.dataclass(frozen=True, eq=False)
class Heatmap(Label):
    """A Gaussian heatmap of a population's positions on the camera grid.

    Each present object adds a unit-peak (or, with ``normalize``, unit-sum) Gaussian of width
    ``sigma_px`` pixels at its sub-pixel position; absent objects (presence 0) add nothing.

    Parameters
    ----------
    population : str
        Name of the population.
    sigma_px : float, default 1.0
        Gaussian standard deviation, pixels.
    normalize : bool, default False
        Unit-sum blobs instead of unit peaks.

    Examples
    --------
    >>> Heatmap("beads", sigma_px=1.5).sigma_px
    1.5
    """

    population: str = knob(doc="population name")
    _: dataclasses.KW_ONLY
    sigma_px: float = knob(default=1.0, doc="Gaussian width, pixels")
    normalize: bool = knob(default=False, doc="unit-sum blobs")

    def gradient_quality(self, path: str) -> str:
        """Return the gradient quality of the heatmap with respect to a field.

        Parameters
        ----------
        path : str
            Logical field path.

        Returns
        -------
        str
            ``"exact"`` for the population's positions and presence, the pixel pitch and the
            magnification; ``"zero"`` otherwise.
        """
        if path in (f"{self.population}.position", f"{self.population}.presence"):
            return "exact"
        return "exact" if path in ("camera.pixel_size", "objective.magnification") else "zero"

    def render(
        self,
        objects: ObjectSet,
        camera: Camera,
        objective: Objective,
        *,
        emitters: EmitterSet | None = None,
    ) -> dict[str, Tensor]:
        """Render the heatmap.

        Parameters
        ----------
        objects : ObjectSet
            The population.
        camera : Camera
            The camera.
        objective : Objective
            The objective.
        emitters : EmitterSet, optional
            Unused: the heatmap marks positions and presence.

        Returns
        -------
        dict of str to Tensor
            ``""``: the heatmap ``[B, (T,) H, W]``.
        """
        px, presence = _pixels(objects, camera, objective)
        height, width = camera.shape
        rows = torch.arange(height, dtype=px.dtype, device=px.device)
        cols = torch.arange(width, dtype=px.dtype, device=px.device)
        s2 = 2.0 * self.sigma_px * self.sigma_px
        gx_ = torch.exp(-((cols - px[..., 0:1]) ** 2) / s2)  # [B, T, N, W]
        gy_ = torch.exp(-((rows - px[..., 1:2]) ** 2) / s2)  # [B, T, N, H]
        if self.normalize:
            scale = 1.0 / (torch.pi * s2)
            gx_ = gx_ * scale
        heat = torch.einsum("btnh,btnw->bthw", gy_ * presence[..., None], gx_)
        return {"": _squeeze_time(heat)}


@register.label("emitter_table")
@dataclasses.dataclass(frozen=True, eq=False)
class EmitterTable(Label):
    """A padded table of a population's emitters in the image frame.

    Columns: x and y (pixels, integer values at pixel centres), z (µm), photons and presence;
    padded slots keep presence 0.

    Parameters
    ----------
    population : str
        Name of the population.

    Examples
    --------
    >>> EmitterTable("beads").population
    'beads'
    """

    rendered: ClassVar[frozenset[str]] = frozenset({"photons"})

    population: str = knob(doc="population name")

    def gradient_quality(self, path: str) -> str:
        """Return the gradient quality of the table with respect to a field.

        Parameters
        ----------
        path : str
            Logical field path.

        Returns
        -------
        str
            ``"exact"`` for the population's positions, photons and presence, the pixel pitch
            and the magnification; ``"zero"`` otherwise.
        """
        # with excitation the photons column is the rendered photons: the gradient table adds the
        # routes of the light, excitation and medium fields through them (``rendered``)
        mine = {f"{self.population}.{f}" for f in ("position", "photons", "presence")}
        if path in mine or path in ("camera.pixel_size", "objective.magnification"):
            return "exact"
        return "zero"

    def render(
        self,
        objects: ObjectSet,
        camera: Camera,
        objective: Objective,
        *,
        emitters: EmitterSet | None = None,
    ) -> dict[str, Tensor]:
        """Render the table.

        Parameters
        ----------
        objects : ObjectSet
            The population.
        camera : Camera
            The camera.
        objective : Objective
            The objective.
        emitters : EmitterSet, optional
            The population as rendered; its photons (after excitation, per frame) fill the
            photons column. Without it the population's own photons are used.

        Returns
        -------
        dict of str to Tensor
            ``""``: ``[B, (T,) N, 5]`` columns (x_px, y_px, z_µm, photons, presence);
            ``"in_fov"``: flags ``[B, (T,) N]``.
        """
        px, presence = _pixels(objects, camera, objective)
        photons = getattr(objects, "photons", None)
        if emitters is not None:
            photons_t = emitters.photons.to(px.dtype)  # rendered: [B|1, A|1, N]
        elif photons is None:
            photons_t = torch.ones_like(presence)
        else:
            photons_t = canonical(photons, objects.schema()["photons"], dtype=px.dtype)
            photons_t = photons_t.to(px.device)
        shape = torch.broadcast_shapes(px.shape[:-1], presence.shape, photons_t.shape)
        table = torch.stack(
            [
                px[..., 0].expand(shape),
                px[..., 1].expand(shape),
                px[..., 2].expand(shape),
                photons_t.expand(shape),
                presence.expand(shape),
            ],
            -1,
        )
        height, width = camera.shape
        x, y = table[..., 0], table[..., 1]
        in_fov = (x >= -0.5) & (x < width - 0.5) & (y >= -0.5) & (y < height - 0.5)
        return {"": _squeeze_time(table), "in_fov": _squeeze_time(in_fov)}


def _disks(
    objects: ObjectSet, camera: Camera, objective: Objective
) -> tuple[Tensor, Tensor, Tensor]:
    """Return signed distances of pixel centres to each object's projected disk, in pixels.

    Returns ``distance [B, T, N, H, W]`` (negative inside), ``presence [B, T, N]`` and
    ``z [B, T, N]`` (µm).
    """
    radius_value = getattr(objects, "radius", None)
    if radius_value is None:
        name = type(objects).__name__
        raise StructureError(
            f"{name} has no radius, so it has no mask",
            fix="masks need objects with an extent, such as gx.Spheres",
        )
    px, presence = _pixels(objects, camera, objective)
    radius = canonical(radius_value, objects.schema()["radius"], dtype=px.dtype)
    size = canonical(camera.pixel_size, camera.schema()["pixel_size"], dtype=px.dtype)
    mag = canonical(objective.magnification, objective.schema()["magnification"], dtype=px.dtype)
    pitch = (size / mag).to(px.device).reshape(-1, 1, 1)
    radius_px = radius.to(px.device) / pitch
    height, width = camera.shape
    rows = torch.arange(height, dtype=px.dtype, device=px.device)
    cols = torch.arange(width, dtype=px.dtype, device=px.device)
    dx = cols - px[..., 0:1]  # [B, T, N, W]
    dy = rows - px[..., 1:2]  # [B, T, N, H]
    centre = torch.sqrt(dy[..., :, None] ** 2 + dx[..., None, :] ** 2)
    return centre - radius_px[..., None, None], presence, px[..., 2]


@register.label("instance_mask")
@dataclasses.dataclass(frozen=True, eq=False)
class InstanceMask(Label):
    """An instance mask of a population's objects on the camera grid.

    Pixel values are 0 for background and ``k + 1`` for object slot k; where projected disks
    overlap, the object nearest the objective (largest z) wins. Absent objects are skipped.

    Parameters
    ----------
    population : str
        Name of the population (objects with a ``radius``, such as spheres).

    Examples
    --------
    >>> InstanceMask("cells").population
    'cells'
    """

    population: str = knob(doc="population name")

    def gradient_quality(self, path: str) -> str:
        """Return the gradient quality of the mask: integer masks have no gradient.

        Parameters
        ----------
        path : str
            Logical field path.

        Returns
        -------
        str
            ``"zero"``.
        """
        return "zero"

    def render(
        self,
        objects: ObjectSet,
        camera: Camera,
        objective: Objective,
        *,
        emitters: EmitterSet | None = None,
    ) -> dict[str, Tensor]:
        """Render the instance mask.

        Parameters
        ----------
        objects : ObjectSet
            The population.
        camera : Camera
            The camera.
        objective : Objective
            The objective.
        emitters : EmitterSet, optional
            Unused.

        Returns
        -------
        dict of str to Tensor
            ``""``: int64 ``[B, (T,) H, W]``.
        """
        distance, presence, z = _disks(objects, camera, objective)
        if distance.shape[2] == 0:  # no objects: all background
            lead = torch.broadcast_shapes(distance.shape[:2], presence.shape[:2])
            empty = distance.new_zeros(*lead, *distance.shape[3:], dtype=torch.int64)
            return {"": _squeeze_time(empty)}
        inside = (distance <= 0) & (presence[..., None, None] > 0)
        depth = torch.where(inside, z[..., None, None].expand_as(distance), -torch.inf)
        best = depth.argmax(dim=2)  # the object nearest the objective
        covered = inside.any(dim=2)
        mask = torch.where(covered, best + 1, torch.zeros_like(best))
        return {"": _squeeze_time(mask)}


@register.label("semantic_mask")
@dataclasses.dataclass(frozen=True, eq=False)
class SemanticMask(Label):
    """A semantic (occupancy) mask of a population on the camera grid.

    Parameters
    ----------
    population : str
        Name of the population (objects with a ``radius``).
    soft : float, optional
        Edge width in pixels of a soft (sigmoid) occupancy, differentiable in positions and
        radii; None gives a hard 0/1 mask.

    Examples
    --------
    >>> SemanticMask("cells", soft=0.5).soft
    0.5
    """

    population: str = knob(doc="population name")
    _: dataclasses.KW_ONLY
    soft: float | None = knob(default=None, doc="soft edge width, pixels")

    def gradient_quality(self, path: str) -> str:
        """Return the gradient quality: exact for soft masks, zero for hard ones.

        Parameters
        ----------
        path : str
            Logical field path.

        Returns
        -------
        str
            The quality.
        """
        if self.soft is None:
            return "zero"
        fields = {f"{self.population}.{f}" for f in ("position", "radius", "presence")}
        return "exact" if path in fields else "zero"

    def render(
        self,
        objects: ObjectSet,
        camera: Camera,
        objective: Objective,
        *,
        emitters: EmitterSet | None = None,
    ) -> dict[str, Tensor]:
        """Render the occupancy.

        Parameters
        ----------
        objects : ObjectSet
            The population.
        camera : Camera
            The camera.
        objective : Objective
            The objective.
        emitters : EmitterSet, optional
            Unused.

        Returns
        -------
        dict of str to Tensor
            ``""``: ``[B, (T,) H, W]`` in [0, 1].
        """
        distance, presence, _z = _disks(objects, camera, objective)
        if self.soft is None:
            occupancy = (distance <= 0).to(distance.dtype)
        else:
            occupancy = torch.sigmoid(-distance / self.soft)
        weighted = occupancy * presence[..., None, None]
        union = 1.0 - torch.prod(1.0 - weighted, dim=2)
        return {"": _squeeze_time(union)}


@register.label("distance_map")
@dataclasses.dataclass(frozen=True, eq=False)
class DistanceMap(Label):
    """The signed distance of every pixel centre to the nearest object boundary, in pixels.

    Negative inside an object, positive outside; absent objects are ignored.

    Parameters
    ----------
    population : str
        Name of the population (objects with a ``radius``).

    Examples
    --------
    >>> DistanceMap("cells").population
    'cells'
    """

    population: str = knob(doc="population name")

    def gradient_quality(self, path: str) -> str:
        """Return the gradient quality: exact almost everywhere for positions and radii.

        Parameters
        ----------
        path : str
            Logical field path.

        Returns
        -------
        str
            The quality.
        """
        fields = {f"{self.population}.{f}" for f in ("position", "radius")}
        if path in fields or path in ("camera.pixel_size", "objective.magnification"):
            return "exact-a.e."
        return "zero"

    def render(
        self,
        objects: ObjectSet,
        camera: Camera,
        objective: Objective,
        *,
        emitters: EmitterSet | None = None,
    ) -> dict[str, Tensor]:
        """Render the distance map.

        Parameters
        ----------
        objects : ObjectSet
            The population.
        camera : Camera
            The camera.
        objective : Objective
            The objective.
        emitters : EmitterSet, optional
            Unused.

        Returns
        -------
        dict of str to Tensor
            ``""``: ``[B, (T,) H, W]`` pixels (inf where no object is present).
        """
        distance, presence, _z = _disks(objects, camera, objective)
        if distance.shape[2] == 0:  # no objects: infinitely far from every boundary
            lead = torch.broadcast_shapes(distance.shape[:2], presence.shape[:2])
            far = distance.new_full((*lead, *distance.shape[3:]), torch.inf)
            return {"": _squeeze_time(far)}
        present = presence[..., None, None] > 0
        masked = torch.where(present, distance, torch.full_like(distance, torch.inf))
        return {"": _squeeze_time(masked.min(dim=2).values)}
