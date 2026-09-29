"""Reference beams, mutually coherent with the illumination (``source.reference``; §4.5, §5.12)."""

from __future__ import annotations

import dataclasses
from typing import ClassVar

import torch
from torch import Tensor

from gradix import register
from gradix._core.carriers import PlaneWaves
from gradix._core.contract import Capabilities, Element, Slot, Static
from gradix._core.errors import StructureError
from gradix.conventions import safe_sqrt
from gradix.schema.fields import field, knob
from gradix.schema.layout import canonical

__all__ = ["ReferenceBeam"]


@register.element("source.reference")
@dataclasses.dataclass(frozen=True, eq=False)
class ReferenceBeam(Element[PlaneWaves]):
    """A reference beam from the illumination's laser, joining the light at the camera.

    With ``path="image"`` (off-axis digital holography, a Mach–Zehnder arm) the reference
    bypasses the sample and the objective and meets the image field as a tilted plane wave
    (§4.5, construction 4): no aperture, no apodisation, and a direction on the image side,
    where an image-side tilt θ is the object-space direction u = Mag·sin θ, free of the |u| ≤ n
    bound. It shares the illumination's wavelengths and is coherent with it.

    Parameters
    ----------
    irradiance : Tensor or float, default 1.0
        Irradiance at the camera in photons/µm² per exposure, referred to the object plane as
        the illumination's is: equal irradiances give equal intensities. Per image ``[B]``.
    angle : Tensor, optional
        Image-side tilt ``(θx, θy)`` in rad: ``[2]`` or per image ``[B, 2]``; None is normal
        incidence (a phase-shifting in-line reference).
    phase : Tensor or float, default 0.0
        Phase at the camera's origin, rad: a number, per image ``[B]``, or per image and frame
        ``[B, T]`` (phase-shifting sequences on the acquisition axis).
    path : {"image"}, default "image"
        Where the reference joins the light; sample-side references arrive with iSCAT.

    Examples
    --------
    >>> import torch, gradix as gx
    >>> reference = ReferenceBeam(1e4, angle=torch.tensor([0.02, 0.0]))
    >>> waves = reference(gx.light.PlaneWave(0.532)())
    >>> tuple(waves.u.shape), float(waves.amplitude.abs().reshape(()))
    ((1, 1, 1, 1, 2), 100.0)
    """

    slot: ClassVar[Slot] = Slot.SOURCE
    caps: ClassVar[Capabilities] = Capabilities(
        accepts=frozenset({PlaneWaves}), produces=PlaneWaves
    )

    irradiance: Tensor | float = field(
        quantity="irradiance",
        role="image",
        constraint="nonnegative",
        default=1.0,
        doc="photons per µm² at the camera, referred to the object plane",
    )
    _: dataclasses.KW_ONLY
    angle: Tensor | None = field(
        quantity="angle",
        role="image",
        event=(2,),
        components=("x", "y"),
        default=None,
        doc="image-side tilt",
    )
    phase: Tensor | float = field(
        quantity="angle", role="setting", default=0.0, doc="phase at the camera's origin"
    )
    path: str = knob(default="image", choices=("image",), doc="where the reference joins")

    def forward(self, *inputs: object, static: Static) -> PlaneWaves:
        """Build the reference waves at the illumination's wavelengths.

        Parameters
        ----------
        *inputs : object
            The illumination's :class:`~gradix.PlaneWaves` (for its wavelengths).
        static : Static
            The (empty) configuration.

        Returns
        -------
        PlaneWaves
            ``amplitude [B|1, A|1, 1, 1, L, 1]`` with ``|amplitude|² = irradiance`` and the
            phase, and ``u [B|1, 1, 1, 1, 2]`` holding the image-side direction sines
            ``sin θ`` (the imaging element scales them by its magnification).
        """
        if not inputs or not isinstance(inputs[0], PlaneWaves):
            raise StructureError("ReferenceBeam takes the illumination's PlaneWaves")
        light = inputs[0]
        spec = self.schema()
        dtype, device = light.wavelengths.dtype, light.wavelengths.device
        irr = canonical(self.irradiance, spec["irradiance"], dtype=dtype, device=device)
        phase = canonical(self.phase, spec["phase"], dtype=dtype, device=device)  # [B|1, A|1]
        amp = safe_sqrt(irr).reshape(-1, 1) * torch.ones_like(phase)
        amplitude = torch.polar(amp, phase)[:, :, None, None, None, None]  # [B, A, 1, 1, 1, 1]
        bins = light.wavelengths.shape[1]
        amplitude = amplitude.expand(*amplitude.shape[:4], bins, 1)
        if self.angle is None:
            u = torch.zeros(1, 1, 1, 1, 2, dtype=dtype, device=device)
        else:
            angle = canonical(self.angle, spec["angle"], dtype=dtype, device=device)  # [B|1, 2]
            u = torch.sin(angle)[:, None, None, None, :]
        return PlaneWaves(amplitude=amplitude, u=u, wavelengths=light.wavelengths, z0=0.0, travel=1)

    def __call__(
        self, *inputs: object, grid: object = None, static: Static | None = None
    ) -> PlaneWaves:
        """Build the reference waves eagerly.

        Parameters
        ----------
        *inputs : object
            The illumination's :class:`~gradix.PlaneWaves`.
        grid : object, optional
            Unused; references are analytic.
        static : Static, optional
            Unused.

        Returns
        -------
        PlaneWaves
            The reference waves.
        """
        return self.forward(*inputs, static=static or Static())
