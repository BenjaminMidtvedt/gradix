"""The TIRF and SIM presets (§7.2): widefield optics with analytic excitation."""

from __future__ import annotations

from torch import Tensor

from gradix import register
from gradix.containers.containers import Microscope
from gradix.detect.camera import Camera
from gradix.light.sources import Evanescent, SIMBeams
from gradix.optics.objective import Objective

__all__ = ["SIM2D", "TIRF"]


@register.preset("tirf")
def TIRF(
    *,
    objective: Objective,
    camera: Camera,
    angle: Tensor | float,
    n_glass: Tensor | float = 1.518,
    irradiance: Tensor | float = 1.0,
    wavelength: Tensor | float = 0.488,
    background: Tensor | float | None = None,
) -> Microscope:
    """Build a TIRF microscope: evanescent excitation from the coverslip.

    The plan adds a linear excitation element: an emitter's ``photons`` is its emission under
    1 photon/µm² (``gx.excite.Linear(reference=1)``), scaled by the local evanescent irradiance,
    which decays with depth (penetration depth ``gx.light.penetration_depth``).

    Parameters
    ----------
    objective : Objective
        The objective.
    camera : Camera
        The camera.
    angle : Tensor or float
        Angle of incidence in the coverslip, rad (beyond the critical angle for TIRF).
    n_glass : Tensor or float, default 1.518
        Index of the coverslip.
    irradiance : Tensor or float, default 1.0
        Irradiance at the interface, photons/µm² per exposure.
    wavelength : Tensor or float, default 0.488
        Excitation wavelength, µm.
    background : Tensor or float, optional
        Background photons per pixel per exposure.

    Returns
    -------
    Microscope
        The microscope.
    """
    light = Evanescent(angle=angle, n_glass=n_glass, irradiance=irradiance, wavelength=wavelength)
    return Microscope(objective=objective, camera=camera, light=light, background=background)


@register.preset("sim2d")
def SIM2D(
    *,
    objective: Objective,
    camera: Camera,
    period: Tensor | float,
    angle: Tensor | float = 0.0,
    phase: Tensor | float = 0.0,
    modulation: Tensor | float = 1.0,
    irradiance: Tensor | float = 1.0,
    wavelength: Tensor | float = 0.488,
    background: Tensor | float | None = None,
) -> Microscope:
    """Build a two-beam SIM microscope (one pattern per image; stack angles and phases along B).

    Parameters
    ----------
    objective : Objective
        The objective.
    camera : Camera
        The camera.
    period : Tensor or float
        Fringe period, µm.
    angle : Tensor or float, default 0.0
        Pattern direction, rad (per image ``[B]`` for a SIM series).
    phase : Tensor or float, default 0.0
        Pattern phase, rad (per image ``[B]``).
    modulation : Tensor or float, default 1.0
        Modulation depth.
    irradiance : Tensor or float, default 1.0
        Mean irradiance, photons/µm² per exposure.
    wavelength : Tensor or float, default 0.488
        Excitation wavelength, µm.
    background : Tensor or float, optional
        Background photons per pixel per exposure.

    Returns
    -------
    Microscope
        The microscope.
    """
    light = SIMBeams(
        period=period,
        angle=angle,
        phase=phase,
        modulation=modulation,
        irradiance=irradiance,
        wavelength=wavelength,
    )
    return Microscope(objective=objective, camera=camera, light=light, background=background)
