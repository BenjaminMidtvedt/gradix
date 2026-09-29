"""The widefield fluorescence preset."""

from __future__ import annotations

from torch import Tensor

from gradix import register
from gradix._core.contract import Element
from gradix.containers.containers import Microscope
from gradix.detect.camera import Camera
from gradix.optics.objective import Objective

__all__ = ["Widefield"]


@register.preset("widefield")
def Widefield(
    *,
    objective: Objective,
    camera: Camera,
    light: Element | None = None,
    background: Tensor | float | None = None,
) -> Microscope:
    """Build a widefield fluorescence microscope.

    Parameters
    ----------
    objective : Objective
        The objective.
    camera : Camera
        The camera.
    light : Element, optional
        Excitation source (used from M1 with ``gx.excite``).
    background : Tensor or float, optional
        Background photons per pixel per exposure.

    Returns
    -------
    Microscope
        The microscope.
    """
    return Microscope(objective=objective, camera=camera, light=light, background=background)
