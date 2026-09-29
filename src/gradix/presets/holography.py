"""Interferometric presets (§5.12): in-line holography now; off-axis, iSCAT and QWLSI follow."""

from __future__ import annotations

from torch import Tensor

from gradix import register
from gradix._core.carriers import PlaneWaves
from gradix._core.contract import Element
from gradix._core.errors import StructureError
from gradix.containers.containers import Microscope
from gradix.detect.camera import Camera
from gradix.optics.objective import Objective

__all__ = ["InlineHolography"]


@register.preset("inline_holography")
def InlineHolography(
    *,
    light: Element,
    objective: Objective,
    camera: Camera,
    background: Tensor | float | None = None,
) -> Microscope:
    """Build an in-line (Gabor) holographic microscope: the unscattered light is the reference.

    With ``gx.plan``, spheres in the Sample render as Mie scatterers (``spheres="mie"`` at
    ``standard`` and above; ``"auto"`` at ``draft`` takes Mie-dressed dipoles for small ones),
    imaged by ``gx.imaging.Coherent`` with explicit interference.

    Parameters
    ----------
    light : Element
        A coherent plane-wave source, such as ``gx.light.PlaneWave(0.532, irradiance=…)``.
    objective : Objective
        The objective.
    camera : Camera
        The camera.
    background : Tensor or float, optional
        Incoherent background photons per pixel per exposure.

    Returns
    -------
    Microscope
        The microscope.

    Raises
    ------
    StructureError
        If ``light`` does not produce plane waves.

    Examples
    --------
    >>> import gradix as gx
    >>> scope = InlineHolography(
    ...     light=gx.light.PlaneWave(0.532),
    ...     objective=gx.Objective(NA=0.8, magnification=60),
    ...     camera=gx.Camera(pixel_size=6.5, shape=(32, 32)),
    ... )
    >>> type(scope.light).__name__
    'PlaneWave'
    """
    produced = light.caps.produces
    produced = produced if isinstance(produced, tuple) else (produced,)
    if not any(issubclass(p, PlaneWaves) for p in produced):
        raise StructureError(
            f"in-line holography needs coherent plane-wave light, not {type(light).__name__}",
            fix="light=gx.light.PlaneWave(wavelength, irradiance=...)",
        )
    return Microscope(objective=objective, camera=camera, light=light, background=background)
