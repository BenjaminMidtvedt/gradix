"""Scalar coherent pupil imaging (``gx.imaging.Coherent``; M1 coherent smoke thread, §5.3).

The unscattered light stays analytic: every incident plane wave inside the aperture reaches the
image plane as an exact plane wave (with its pupil transmission and its phase at the focal
plane). Sparse producers (:class:`~gradix.ObjectSpectra`) are evaluated on the pupil support and
transformed to the camera samples by a matrix Fourier transform. Detection is explicit
interference: per incoherent mode and wavelength bin, ``|E_background + Σ E_scattered|²``,
integrated over pixels by the pixel MTF (§4.3). Index-matched homogeneous media, scalar light
(P = 1), magnification maps object space onto the camera.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Mapping
from typing import ClassVar

import torch
from torch import Tensor

from gradix import register
from gradix._core.carriers import Irradiance, ObjectSpectra, PlaneWaves
from gradix._core.contract import (
    Capabilities,
    Description,
    Element,
    Slot,
    Static,
    Violation,
    report,
)
from gradix._core.envelope import Envelope
from gradix._core.errors import StructureError
from gradix._core.grid import Grid2D
from gradix._core.rules import detection_spacing
from gradix.conventions import safe_sqrt
from gradix.detect.camera import Camera
from gradix.imaging._shared import ratio_per_image
from gradix.objects.environment import Medium
from gradix.ops import pupil as ops
from gradix.optics.objective import Objective
from gradix.schema.base import Node
from gradix.schema.fields import child, knob
from gradix.schema.layout import canonical

__all__ = ["Coherent", "CoherentStatic"]


@dataclasses.dataclass(frozen=True)
class CoherentStatic(Static):
    """Static configuration of :class:`Coherent`.

    Parameters
    ----------
    decisions : tuple of Decision, default ()
        Sizing decisions.
    grid : Grid2D, optional
        The camera grid in object space.
    oversample : int, default 1
        Samples per camera pixel before the pixel MTF.
    pupil_samples : int, default 64
        Pupil cells per axis.
    na_max : float, default 1.0
        Largest NA of the envelope: sets the pupil extent (a static, so calls never read NA
        values back to the host).
    """

    grid: Grid2D | None = None
    oversample: int = 1
    pupil_samples: int = 64
    na_max: float = 1.0


@register.element("coherent.pupil")
@dataclasses.dataclass(frozen=True, eq=False)
class Coherent(Element[Irradiance]):
    """Scalar coherent imaging of plane waves and sparse scatterers, with explicit interference.

    Parameters
    ----------
    objective : Objective
        The objective (NA, magnification, focus).
    camera : Camera
        The camera.
    oversample : int or "auto", default "auto"
        Samples per camera pixel; ``"auto"`` meets λ_min/(4·NA).
    pupil_samples : int, default 64
        Pupil cells per axis for the scattered spectra.

    Examples
    --------
    >>> import gradix as gx
    >>> camera = gx.Camera(pixel_size=6.5, shape=(32, 32))
    >>> Coherent(gx.Objective(NA=0.8, magnification=50), camera).pupil_samples
    64
    """

    slot: ClassVar[Slot] = Slot.OBJECTIVE
    fidelity_knobs: ClassVar[Mapping[str, str | tuple[str, Mapping[str, str]]]] = {
        "oversample": "oversample"
    }
    caps: ClassVar[Capabilities] = Capabilities(
        accepts=frozenset({ObjectSpectra, PlaneWaves}),
        produces=Irradiance,
        grad_quality={
            "camera": "zero",
            "camera.pixel_size": "exact",
            "objective.NA": "exact-a.e.",
            "objective.pupil": "zero",  # the smoke thread ignores pupil modifiers
        },
    )

    objective: Objective = child(doc="the objective")
    camera: Camera = child(doc="the camera")
    _: dataclasses.KW_ONLY
    oversample: int | str = knob(default="auto", doc="samples per camera pixel")
    pupil_samples: int = knob(default=64, doc="pupil cells per axis")

    def configure(self, desc: Description, envelope: Envelope) -> CoherentStatic:
        """Choose the sampling of the camera grid.

        Parameters
        ----------
        desc : Description
            Static description.
        envelope : Envelope
            The envelope (NA, magnification, pixel size, wavelengths).

        Returns
        -------
        CoherentStatic
            The configuration.
        """
        # the coarsest pitch and the largest NA of any image set the sampling
        pitch = float(self._number("pixel_size", self.camera, envelope, "camera", pick="hi")) / (
            float(self._number("magnification", self.objective, envelope, "objective"))
        )
        na = float(self._number("NA", self.objective, envelope, "objective", pick="hi"))
        wl = envelope.range("light.wavelengths") or (0.5, 0.5)
        _spacing, s, decision = detection_spacing(pitch, wl[0], na, oversample=self.oversample)
        grid = Grid2D(tuple(self.camera.shape), pitch, (0.0, 0.0))
        return CoherentStatic(
            decisions=(decision,),
            grid=grid,
            oversample=s,
            pupil_samples=self.pupil_samples,
            na_max=na,
        )

    def validity(self, desc: Description, envelope: Envelope) -> list[Violation]:
        """Report findings: the scalar smoke element needs NA below the medium's index.

        Parameters
        ----------
        desc : Description
            Static description.
        envelope : Envelope
            The envelope.

        Returns
        -------
        list of Violation
            Findings.
        """
        medium = desc.nodes.get(desc.part("environment")) or desc.nodes.get("environment")
        if not isinstance(medium, Medium):
            return []
        ratio = ratio_per_image(self.objective, medium)  # NA/n per image, largest
        if ratio is None or ratio < 1.0:
            return []
        return [
            Violation(
                "error",
                "NA reaches the medium's index; the coherent smoke element images homogeneous "
                "media only",
                entry=f"{desc.part('objective')}.NA",
                value=ratio,
                limit=1.0,
                element=desc.path or "coherent",
                fix="lower the NA below the medium's index",
            )
        ]

    def _number(
        self, name: str, node: Node, envelope: Envelope, part: str, pick: str = "lo"
    ) -> float:
        value = getattr(node, name)
        if isinstance(value, (int, float)):
            return float(value)
        interval = envelope.range(f"{part}.{name}")
        if interval is None:  # per-image values of an eager call: the extreme one
            values = torch.as_tensor(value).detach().reshape(-1)
            return float(values.max() if pick == "hi" else values.min())
        return interval[1] if pick == "hi" else interval[0]

    def eager_static(self, *inputs: object, envelope: Envelope | None = None) -> Static:
        """Configure for an eager call from the element's own optics and the waves' wavelengths.

        Parameters
        ----------
        *inputs : object
            The contributions, the plane waves and the medium.
        envelope : Envelope, optional
            Unused: the configuration reads the element's values directly.

        Returns
        -------
        Static
            The configuration.
        """
        waves = inputs[1] if len(inputs) > 1 and isinstance(inputs[1], PlaneWaves) else None
        wl = float(waves.wavelengths.detach().min()) if waves is not None else 0.5
        entries = {"light.wavelengths": (wl, wl)}
        medium = inputs[2] if len(inputs) > 2 and isinstance(inputs[2], Node) else None
        desc = Description(nodes={"environment": medium} if medium is not None else {})
        report(self.validity(desc, Envelope(entries)), "warn")  # errors raise
        return self.configure(desc, Envelope(entries))

    def forward(self, *inputs: object, static: Static) -> Irradiance:
        """Image the background plane waves and the scattered spectra onto the camera.

        Parameters
        ----------
        *inputs : object
            The contributions (an :class:`~gradix.ObjectSpectra` or a tuple of them), the
            incident :class:`~gradix.PlaneWaves`, then the medium.
        static : Static
            A :class:`CoherentStatic`.

        Returns
        -------
        Irradiance
            ``[B, A, 1, H, W]`` photons per pixel.
        """
        if len(inputs) != 3:
            raise StructureError("Coherent takes (contributions, PlaneWaves, medium)")
        contributions, waves, medium = inputs
        if isinstance(contributions, ObjectSpectra):
            contributions = (contributions,)
        if not isinstance(contributions, tuple) or not isinstance(waves, PlaneWaves):
            raise StructureError("Coherent takes a tuple of ObjectSpectra and PlaneWaves")
        if not isinstance(medium, Medium) or medium.layered:
            raise StructureError("the smoke-thread Coherent element needs a homogeneous medium")
        if not isinstance(static, CoherentStatic) or static.grid is None:
            raise StructureError("Coherent needs a CoherentStatic")
        ospec, cspec = self.objective.schema(), self.camera.schema()
        # the scene's precision and device (waves from scalar source parameters are float32 and
        # live on the CPU)
        dtype = waves.wavelengths.dtype
        device = waves.wavelengths.device
        if contributions:
            dtype = torch.promote_types(dtype, contributions[0].position.dtype)
            device = contributions[0].position.device
        cdtype = torch.complex128 if dtype == torch.float64 else torch.complex64

        def image(node: Node, spec: dict, name: str) -> Tensor:
            return canonical(getattr(node, name), spec[name], dtype=dtype, device=device)

        # background layout [B, A, M, J, L]
        na = image(self.objective, ospec, "NA").reshape(-1, 1, 1, 1, 1)
        focus = image(self.objective, ospec, "focus")  # [B|1, A|1] (a setting)
        pitch = image(self.camera, cspec, "pixel_size") / image(
            self.objective, ospec, "magnification"
        )
        index = medium.index(dtype=dtype, device=device)
        n_image = (index.real if index.is_complex() else index).reshape(-1)  # [B|1]
        n = n_image.reshape(-1, 1, 1, 1, 1)
        lam = waves.wavelengths.to(device=device, dtype=dtype)  # [B|1, L]
        wl = lam[:, None, None, None, :]  # [B|1, 1, 1, 1, L]
        s = static.oversample
        height, width = static.grid.shape
        # camera samples (object space): s per pixel, the first at each pixel centre
        p = pitch.reshape(-1, 1)
        ky = torch.arange(height * s, device=device, dtype=dtype)
        kx = torch.arange(width * s, device=device, dtype=dtype)
        ys = (ky // s + 0.5) * p + (ky % s) * p / s  # [B|1, Hs]
        xs = (kx // s + 0.5) * p + (kx % s) * p / s  # [B|1, Ws]
        # background: each plane wave with its pupil transmission and focal-plane phase
        u = waves.u.to(device=device, dtype=dtype)  # [B|1, A|1, M, J, 2]
        radius = safe_sqrt((u * u).sum(-1))[..., None]  # [B|1, A|1, M, J, 1]; finite grad at 0
        du = 2.0 * static.na_max * 1.03125 / static.pupil_samples
        # only waves travelling toward the objective and propagating (|u| < n) reach the camera;
        # an aplanatic objective maps a wave of irradiance I at angle θ to I·cosθ on the camera
        cos_b = torch.sqrt(torch.clamp(1.0 - radius * radius / (n * n), min=0.0))
        transmission = ops.soft_aperture(radius, na, du) * torch.sqrt(cos_b)
        transmission = torch.where(radius < n, transmission, torch.zeros_like(transmission))
        if waves.travel != 1:  # epi illumination travels away from the objective
            transmission = torch.zeros_like(transmission)
        kz = 2.0 * math.pi * torch.sqrt(torch.clamp(n * n - radius * radius, min=0.0)) / wl
        z0 = torch.as_tensor(waves.z0, dtype=dtype, device=device)
        focus5 = focus.reshape(focus.shape[0], focus.shape[1], 1, 1, 1)  # [B|1, A|1, 1, 1, 1]
        phase_z = kz * waves.travel * (focus5 - z0)  # [B, A, M, J, L]
        amp = (
            waves.amplitude[..., 0].to(device=device, dtype=cdtype)
            * transmission
            * torch.polar(torch.ones_like(phase_z), phase_z)
        )  # [B, A, M, J, L]
        k0 = (2.0 * math.pi / wl)[..., None]  # [B|1, 1, 1, 1, L, 1]
        ux, uy = u[..., 0][..., None, None], u[..., 1][..., None, None]  # [B, A, M, J, 1, 1]
        phx = k0 * ux * xs.reshape(xs.shape[0], 1, 1, 1, 1, -1)
        phy = k0 * uy * ys.reshape(ys.shape[0], 1, 1, 1, 1, -1)
        wave_x = torch.polar(torch.ones_like(phx), phx)  # [B, A, M, J, L, Ws]
        wave_y = torch.polar(torch.ones_like(phy), phy)  # [B, A, M, J, L, Hs]
        field = torch.einsum("bamjl,bamjly,bamjlx->bamlyx", amp, wave_y, wave_x)
        for contribution in contributions:
            if contribution.position.shape[2] == 0:
                continue  # no scatterers: the background alone
            field = field + self._scattered(
                contribution, waves, n_image, na, focus, lam, xs, ys, static.na_max
            )
        intensity = (field.real**2 + field.imag**2).sum((2, 3))  # modes and bins: [B, A, Hs, Ws]
        pixels = ops.pixel_mtf(intensity, s)[..., ::s, ::s] * (p.reshape(-1, 1, 1, 1) ** 2)
        return Irradiance(data=pixels[:, :, None], grid=static.grid)

    def _scattered(
        self,
        spectra: ObjectSpectra,
        waves: PlaneWaves,
        n: Tensor,
        na: Tensor,
        focus: Tensor,
        lam: Tensor,
        xs: Tensor,
        ys: Tensor,
        na_max: float,
    ) -> Tensor:
        """Return one producer's scattered field at the camera samples, ``[B, A, M, L, Hs, Ws]``.

        ``n`` and ``na`` are per image ``[B|1]`` (any trailing ones), ``focus`` ``[B|1, A|1]``
        and ``lam`` ``[B|1, L]``; the pupil work runs in the layout ``[B, A, M, L, N, P, P]``.
        """
        if spectra.evaluator != "dipole":
            raise StructureError(f"Coherent cannot evaluate {spectra.evaluator!r} spectra yet")
        f = getattr(spectra.params, "amplitude", None)
        if not isinstance(f, Tensor):
            raise StructureError("dipole spectra need params.amplitude")
        dtype = xs.dtype
        pos = spectra.position.to(dtype)  # [B|1, A|1, N, 3]
        n_image = n.reshape(-1)
        incident = waves.at(pos, n_image[:, None])  # [B, A, M, L, P, N]
        incident = incident[..., 0, :]  # scalar P = 1: [B, A, M, L, N]
        if spectra.presence is not None:
            incident = incident * spectra.presence.to(dtype)[:, :, None, None, :]
        strength = incident * f.permute(0, 1, 3, 2)[:, :, None]  # f·E_inc: [B, A, M, L, N]
        samples = self.pupil_samples
        u, du = ops.pupil_axis(samples, na_max * 1.03125, dtype=dtype, device=xs.device)
        uy, ux = torch.meshgrid(u, u, indexing="ij")
        u2 = ux * ux + uy * uy
        seven = (-1, 1, 1, 1, 1, 1, 1)
        aperture = ops.soft_aperture(torch.sqrt(u2), na.reshape(seven), du)  # [B|1, …, P, P]
        n7 = n_image.reshape(seven)
        wl7 = lam.reshape(lam.shape[0], 1, 1, lam.shape[1], 1, 1, 1)  # [B|1, 1, 1, L, 1, 1, 1]
        inside = u2 < n7 * n7  # the far field holds propagating waves only
        cos = torch.sqrt(torch.clamp(1.0 - u2 / (n7 * n7), min=1e-12))
        kz = 2.0 * math.pi * n7 * cos / wl7
        focus7 = focus.reshape(focus.shape[0], focus.shape[1], 1, 1, 1, 1, 1)
        z7 = pos[..., 2].reshape(pos.shape[0], pos.shape[1], 1, 1, pos.shape[2], 1, 1)
        phase = kz * (focus7 - z7)  # [B, A, 1, L, N, P, P]
        # pupil in f-measure: (2π)²·A(k⊥) = 2πi·f·E_inc/k_z, propagated to the focal plane, with
        # the aplanatic √cosθ that conserves the collected power (§5.3)
        weight = torch.where(inside, aperture * torch.sqrt(cos) / kz, torch.zeros_like(kz))
        pupil = (2j * math.pi) * weight * torch.polar(torch.ones_like(phase), phase)
        pupil = pupil * strength[..., None, None]  # [B, A, M, L, N, P, P]
        batch = pupil.shape[:5]
        flat = pupil.reshape(-1, samples, samples)
        g = flat.shape[0]
        wl_g = lam.reshape(lam.shape[0], 1, 1, -1, 1).expand(*batch).reshape(-1)
        dx = xs.reshape(xs.shape[0], 1, 1, 1, 1, -1) - pos[..., 0][:, :, None, None, :, None]
        dy = ys.reshape(ys.shape[0], 1, 1, 1, 1, -1) - pos[..., 1][:, :, None, None, :, None]
        dx = dx.expand(*batch, dx.shape[-1]).reshape(g, -1)
        dy = dy.expand(*batch, dy.shape[-1]).reshape(g, -1)
        field = ops.mft_roi(flat, u, du, wl_g, dx, dy).reshape(*batch, dy.shape[-1], dx.shape[-1])
        return field.sum(4)  # superpose the objects: [B, A, M, L, Hs, Ws]
