"""Geometry-derived labels (``gx.labels``; §6.6): positions in the image frame, in-FOV flags."""

from __future__ import annotations

import dataclasses
from typing import ClassVar

from torch import Tensor

from gradix import register
from gradix._core.carriers import EmitterSet
from gradix.coords import to_pixels
from gradix.detect.camera import Camera
from gradix.objects.objectset import ObjectSet
from gradix.optics.objective import Objective
from gradix.schema.base import Node
from gradix.schema.fields import knob
from gradix.schema.layout import canonical

__all__ = ["Label", "Positions", "render"]


@dataclasses.dataclass(frozen=True, kw_only=True, eq=False)
class Label(Node):
    """Base of label renderers: requested as outputs, rendered in the image's frame.

    A label that reads the population *as rendered* (the ``emitters`` passed to
    :meth:`render`: photons after excitation, for example) names those fields in
    :attr:`rendered`, so the gradient table routes the excitation's fields to it.
    """

    rendered: ClassVar[frozenset[str]] = frozenset()
    """Fields of the rendered EmitterSet the label reads (``"photons"``)."""

    def gradient_quality(self, path: str) -> str:
        """Return the gradient quality of the label with respect to a field.

        Parameters
        ----------
        path : str
            Logical field path, such as ``"beads.position"`` or ``"camera.pixel_size"``.

        Returns
        -------
        str
            ``"exact"`` by default, so a label that does not declare its dependencies never
            causes a zero-path refusal; built-in labels declare them.
        """
        return "exact"

    def render(
        self,
        objects: ObjectSet,
        camera: Camera,
        objective: Objective,
        *,
        emitters: EmitterSet | None = None,
    ) -> dict[str, Tensor]:
        """Render the label.

        Parameters
        ----------
        objects : ObjectSet
            The labelled population.
        camera : Camera
            The camera (grid and pitch).
        objective : Objective
            The objective (magnification).
        emitters : EmitterSet, optional
            The population as rendered (lowered, and excited when the Chain has excitation),
            ``[B|1, A|1, N, …]``; given by Chains and Pipelines for emitter populations.

        Returns
        -------
        dict of str to Tensor
            Label tensors by suffix; ``""`` is the main tensor.
        """
        raise NotImplementedError(type(self).__name__)


@register.label("positions")
@dataclasses.dataclass(frozen=True, eq=False)
class Positions(Label):
    """Object positions in the image frame, with in-FOV flags.

    The output ``name`` holds ``[B, N, D]`` positions (``[B, T, N, D]`` with frames) and
    ``name + ".in_fov"`` the matching boolean flags. In pixels, integer coordinates are pixel
    centres (:mod:`gradix.coords`); z stays in µm.

    Parameters
    ----------
    population : str
        Name of the population in the Chain or Sample.
    unit : {"px", "um"}, default "px"
        Unit of x and y.
    dims : {"xy", "xyz"}, default "xy"
        Components returned.

    Examples
    --------
    >>> Positions("beads", unit="um").dims
    'xy'
    """

    population: str = knob(doc="population name")
    _: dataclasses.KW_ONLY
    unit: str = knob(default="px", choices=("px", "um"), doc="unit of x and y")
    dims: str = knob(default="xy", choices=("xy", "xyz"), doc="components")

    def gradient_quality(self, path: str) -> str:
        """Return the gradient quality of the positions with respect to a field.

        Parameters
        ----------
        path : str
            Logical field path.

        Returns
        -------
        str
            ``"exact"`` for the population's positions and, in pixels, for the pixel pitch and
            magnification; ``"zero"`` otherwise.
        """
        if path == f"{self.population}.position":
            return "exact"
        if self.unit == "px" and path in ("camera.pixel_size", "objective.magnification"):
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
        """Render the positions of a population.

        Parameters
        ----------
        objects : ObjectSet
            The population.
        camera : Camera
            The camera.
        objective : Objective
            The objective.
        emitters : EmitterSet, optional
            Unused: positions come from the population.

        Returns
        -------
        dict of str to Tensor
            ``""``: positions ``[B, (T,) N, D]``; ``"in_fov"``: flags ``[B, (T,) N]``, true when
            the position lies on the camera's pixel area.
        """
        spec = objects.schema()["position"]
        pos = canonical(objects.position, spec)  # [B|1, T|1, N, 3]
        px = to_pixels(pos, camera, objective)
        height, width = camera.shape
        x, y = px[..., 0], px[..., 1]
        in_fov = (x >= -0.5) & (x < width - 0.5) & (y >= -0.5) & (y < height - 0.5)
        out = px if self.unit == "px" else pos
        out = out[..., : len(self.dims)]
        if out.shape[1] == 1:
            out, in_fov = out[:, 0], in_fov[:, 0]
        return {"": out.clone(), "in_fov": in_fov}


def render(
    label: Label, objects: ObjectSet, camera: Camera, objective: Objective
) -> dict[str, Tensor]:
    """Render a label outside a Pipeline (``gx.labels.render``).

    Parameters
    ----------
    label : Label
        The label renderer, such as :class:`Positions`.
    objects : ObjectSet
        The population.
    camera : Camera
        The camera.
    objective : Objective
        The objective.

    Returns
    -------
    dict of str to Tensor
        Label tensors by suffix; ``""`` is the main tensor.
    """
    return label.render(objects, camera, objective)
