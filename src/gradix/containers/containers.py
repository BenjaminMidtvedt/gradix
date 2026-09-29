"""L4 containers: :class:`Sample` (populations and environment) and :class:`Microscope` (optics)."""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from typing import ClassVar

from torch import Tensor

from gradix._core.contract import Element
from gradix._core.errors import StructureError
from gradix.detect.camera import Camera, check_pixel_map
from gradix.objects.environment import Medium
from gradix.objects.objectset import ObjectSet
from gradix.objects.volumes import Voxels
from gradix.optics.objective import Objective
from gradix.schema.base import Node
from gradix.schema.fields import child, field

__all__ = ["Microscope", "Sample"]


@dataclasses.dataclass(frozen=True, eq=False)
class Sample(Node):
    """The resolved sample: named populations of object sets, in an environment.

    Parameters
    ----------
    populations : Mapping[str, ObjectSet]
        Populations by name.
    environment : Medium
        The medium around the sample.

    Examples
    --------
    >>> import torch, gradix as gx
    >>> from gradix.units import nm
    >>> beads = gx.Emitters(
    ...     position=torch.zeros(1, 4, 3), photons=500.0, emission=gx.Spectrum.line(600 * nm)
    ... )
    >>> sorted(Sample({"beads": beads}, environment=gx.env.Homogeneous(1.33)).parts())
    ['beads', 'environment']
    """

    registry_name: ClassVar[str | None] = "sample"
    """Stable name for signatures and saved inputs."""

    populations: Mapping[str, ObjectSet | Voxels] = child(
        container="mapping", doc="populations by name"
    )
    _: dataclasses.KW_ONLY
    environment: Medium = child(doc="the medium")

    def __post_init__(self) -> None:
        super().__post_init__()
        for name, pop in self.populations.items():
            if not isinstance(pop, (ObjectSet, Voxels)):
                raise StructureError(
                    f"population {name!r} is a {type(pop).__name__}, not an object set or volume"
                )

    def parts(self) -> dict[str, Node]:
        """Return the named parts whose fields have envelope and binding paths.

        Returns
        -------
        dict of str to Node
            Populations and ``environment``.
        """
        return {**self.populations, "environment": self.environment}


@dataclasses.dataclass(frozen=True, kw_only=True, eq=False)
class Microscope(Node):
    """The optics: light, objective, camera, acquisition and background.

    Parameters
    ----------
    objective : Objective
        The objective.
    camera : Camera
        The camera.
    light : Element, optional
        The source element (drives coherent and excitation paths from M1).
    acquisition : Node, optional
        Acquisition axes (M1).
    background : Tensor or float, optional
        Background photons per pixel per exposure.

    Examples
    --------
    >>> import gradix as gx
    >>> from gradix.units import um
    >>> scope = Microscope(
    ...     objective=gx.Objective(NA=1.4, magnification=100),
    ...     camera=gx.Camera(pixel_size=6.5 * um, shape=(64, 64)),
    ... )
    >>> sorted(scope.parts())
    ['camera', 'objective']
    """

    registry_name: ClassVar[str | None] = "microscope"
    """Stable name for signatures and saved inputs."""

    objective: Objective = child(doc="the objective")
    camera: Camera = child(doc="the camera")
    light: Element | None = child(default=None, doc="source element")
    acquisition: Node | None = child(default=None, doc="acquisition axes")
    background: Tensor | float | None = field(
        quantity="photons", role="pixels", constraint="nonnegative", default=None, doc="background"
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        if isinstance(self.background, Tensor):
            spec = type(self).schema()["background"]
            check_pixel_map(self.background, spec, self.camera.shape, where="background")

    def parts(self) -> dict[str, Node]:
        """Return the named parts whose fields have envelope and binding paths.

        Returns
        -------
        dict of str to Node
            ``objective``, ``camera`` and, when present, ``light`` and ``acquisition``.
        """
        out: dict[str, Node] = {"objective": self.objective, "camera": self.camera}
        if self.light is not None:
            out["light"] = self.light
        if self.acquisition is not None:
            out["acquisition"] = self.acquisition
        return out
