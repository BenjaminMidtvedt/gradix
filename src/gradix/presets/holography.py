"""Interferometric presets (§5.12): in-line and off-axis holography, iSCAT and QWLSI."""

from __future__ import annotations

from torch import Tensor

from gradix import register
from gradix._core.carriers import PlaneWaves
from gradix._core.contract import Element
from gradix._core.errors import StructureError
from gradix.containers.containers import Microscope
from gradix.detect.camera import Camera
from gradix.light.reference import ReferenceBeam
from gradix.optics.grating import Grating
from gradix.optics.objective import Objective
from gradix.optics.pupil import Filter

__all__ = ["ISCAT", "QWLSI", "InlineHolography", "OffAxisHolography"]


def _check_plane_waves(light: Element, what: str) -> None:
    produced = light.caps.produces
    produced = produced if isinstance(produced, tuple) else (produced,)
    if not any(issubclass(p, PlaneWaves) for p in produced):
        raise StructureError(
            f"{what} needs coherent plane-wave light, not {type(light).__name__}",
            fix="light=gx.light.PlaneWave(wavelength, irradiance=...)",
        )


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
    _check_plane_waves(light, "in-line holography")
    return Microscope(objective=objective, camera=camera, light=light, background=background)


@register.preset("off_axis_holography")
def OffAxisHolography(
    *,
    light: Element,
    objective: Objective,
    camera: Camera,
    angle: Tensor,
    reference_irradiance: Tensor | float | None = None,
    reference_phase: Tensor | float = 0.0,
    background: Tensor | float | None = None,
) -> Microscope:
    """Build an off-axis digital holographic microscope: a tilted reference on the camera.

    The reference bypasses the sample and the objective (a Mach–Zehnder arm) and meets the
    image at the image-side tilt ``angle``: object-space direction u_ref = Mag·sin θ. The
    sideband separates from the autocorrelation when |u_ref| ≥ 3·NA, and the camera samples it
    when |u_ref| + NA stays below λ/(2p) per axis (p the object-space pixel); the imaging
    element warns otherwise. ``gx.recon.OffAxis.from_chain`` reconstructs the field.

    Parameters
    ----------
    light : Element
        A coherent plane-wave source, such as ``gx.light.PlaneWave(0.532, irradiance=…)``.
    objective : Objective
        The objective.
    camera : Camera
        The camera.
    angle : Tensor
        Image-side tilt ``(θx, θy)`` of the reference, rad: ``[2]`` or per image ``[B, 2]``.
    reference_irradiance : Tensor or float, optional
        The reference's irradiance, referred to the object plane; None matches the
        illumination's, which maximises the fringe contrast of the unscattered light.
    reference_phase : Tensor or float, default 0.0
        Its phase at the camera's origin, rad (per frame for phase stepping).
    background : Tensor or float, optional
        Incoherent background photons per pixel per exposure.

    Returns
    -------
    Microscope
        The microscope, with the reference under the name ``"reference"``.

    Raises
    ------
    StructureError
        If ``light`` does not produce plane waves.

    Examples
    --------
    >>> import torch, gradix as gx
    >>> scope = OffAxisHolography(
    ...     light=gx.light.PlaneWave(0.532, irradiance=1e4),
    ...     objective=gx.Objective(NA=0.5, magnification=60),
    ...     camera=gx.Camera(pixel_size=6.5, shape=(64, 64)),
    ...     angle=torch.tensor([0.02, 0.02]),
    ... )
    >>> sorted(scope.references)
    ['reference']
    """
    _check_plane_waves(light, "off-axis holography")
    irradiance = reference_irradiance
    if irradiance is None:
        irradiance = getattr(light, "irradiance", 1.0)
    reference = ReferenceBeam(irradiance, angle=angle, phase=reference_phase)
    return Microscope(
        objective=objective,
        camera=camera,
        light=light,
        background=background,
        references={"reference": reference},
    )


@register.preset("iscat")
def ISCAT(
    *,
    light: Element,
    objective: Objective,
    camera: Camera,
    attenuation: Tensor | float | None = None,
    filter_radius: Tensor | float | None = None,
    filter_phase: Tensor | float = 0.0,
    background: Tensor | float | None = None,
) -> Microscope:
    """Build an interferometric scattering (iSCAT) microscope: epi light, the coverslip's echo.

    The illumination comes through the objective (``travel=-1``) and is defined in the
    coverslip. Its reflection at the sample interface of the Sample's
    :class:`~gradix.env.LayeredMedium` is the reference, and the spheres' backscatter is
    collected through the stack, supercritical angles included (§4.1, §5.12). An optional
    central pupil filter attenuates the reference to raise the contrast.

    Parameters
    ----------
    light : Element
        A plane wave travelling away from the objective, such as
        ``gx.light.PlaneWave(0.532, irradiance=…, travel=-1)``.
    objective : Objective
        The objective (oil immersion, typically NA ≈ 1.4).
    camera : Camera
        The camera.
    attenuation : Tensor or float, optional
        Amplitude transmission of a central pupil filter on the reference; None is no filter.
    filter_radius : Tensor or float, optional
        The filter's radius in NA units; by default 5 % of the NA.
    filter_phase : Tensor or float, default 0.0
        The filter's phase, rad.
    background : Tensor or float, optional
        Incoherent background photons per pixel per exposure.

    Returns
    -------
    Microscope
        The microscope; the Sample carries the layered medium.

    Raises
    ------
    StructureError
        If ``light`` does not produce plane waves travelling away from the objective.

    Examples
    --------
    >>> import gradix as gx
    >>> scope = ISCAT(
    ...     light=gx.light.PlaneWave(0.532, irradiance=1e4, travel=-1),
    ...     objective=gx.Objective(NA=1.4, magnification=100),
    ...     camera=gx.Camera(pixel_size=6.5, shape=(64, 64)),
    ...     attenuation=0.1,
    ... )
    >>> type(scope.objective.pupil[0]).__name__
    'Filter'
    """
    _check_plane_waves(light, "iSCAT")
    if getattr(light, "travel", 1) != -1:
        raise StructureError(
            "iSCAT illuminates through the objective: the light must travel away from it",
            fix="light=gx.light.PlaneWave(wavelength, irradiance=..., travel=-1)",
        )
    if attenuation is not None:
        na = objective.NA
        radius = filter_radius if filter_radius is not None else 0.05 * na
        stop = Filter(radius=radius, transmission=attenuation, phase=filter_phase)
        objective = objective.replace(pupil=(*objective.pupil, stop))
    return Microscope(objective=objective, camera=camera, light=light, background=background)


@register.preset("qwlsi")
def QWLSI(
    *,
    light: Element,
    objective: Objective,
    camera: Camera,
    distance: Tensor | float,
    period: Tensor | float | None = None,
    kind: str = "hartmann",
    rotation: Tensor | float = 0.0,
    background: Tensor | float | None = None,
) -> Microscope:
    """Build a quadriwave lateral shearing interferometer: a grating just before the camera.

    Transmitted plane-wave light images the sample onto the camera through a 2-D grating a
    distance d before it (``gx.optics.Grating``, ADR-45): the camera records the fringes of
    four sheared copies of the image field, whose phases hold the wavefront's gradients over
    the shear λd/Λ (§5.12). ``gx.recon.QWLSI.from_chain`` recovers the field.

    Parameters
    ----------
    light : Element
        Plane-wave light, such as ``gx.light.PlaneWave(0.532, irradiance=...)``.
    objective : Objective
        The objective.
    camera : Camera
        The camera, in focus.
    distance : Tensor or float
        Grating-to-camera distance d, µm.
    period : Tensor or float, optional
        Fringe period Λ on the camera, µm; by default four camera pixels.
    kind : {"hartmann", "checkerboard"}, default "hartmann"
        The grating's mask.
    rotation : Tensor or float, default 0.0
        Rotation of the grating's axes, rad.
    background : Tensor or float, optional
        Incoherent background photons per pixel per exposure.

    Returns
    -------
    Microscope
        The microscope, with the grating in ``detection_optics["grating"]``.

    Examples
    --------
    >>> import gradix as gx
    >>> scope = QWLSI(
    ...     light=gx.light.PlaneWave(0.532, irradiance=1e4),
    ...     objective=gx.Objective(NA=0.8, magnification=100),
    ...     camera=gx.Camera(pixel_size=6.5, shape=(64, 64)),
    ...     distance=500.0,
    ... )
    >>> float(scope.detection_optics["grating"].period)
    26.0
    """
    _check_plane_waves(light, "QWLSI")
    if period is None:
        pixel = camera.pixel_size
        period = 4.0 * (pixel if isinstance(pixel, (int, float)) else float(pixel))
    grating = Grating(period=period, distance=distance, rotation=rotation, kind=kind)
    return Microscope(
        objective=objective,
        camera=camera,
        light=light,
        background=background,
        detection_optics={"grating": grating},
    )
