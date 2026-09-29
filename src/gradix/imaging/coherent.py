"""Scalar coherent pupil imaging (``gx.imaging.Coherent``; §5.3, §5.12).

The unscattered light stays analytic: every incident plane wave inside the aperture reaches the
image plane as an exact plane wave, with its pupil transmission, the objective's pupil
modifiers evaluated at its direction, and its phase at the focal plane. Sparse producers
(:class:`~gradix.ObjectSpectra`) are evaluated on the pupil samples: for Mie spectra, S₁ and S₂
at the scattering angle between every incident wave and every pupil direction, reduced to the
co-polarised S∥ = S₂cos²φ + S₁sin²φ (P = 1, §4.1), converted to angular spectra and transformed
to the camera samples by a matrix Fourier transform. Image-side references (off-axis
holography, §4.5) join after the pupil as tilted plane waves. Detection is explicit
interference, term by term: |E_b|² + 2Re(E_b*·E_s) + |E_s|² and the reference's terms, per
incoherent mode and wavelength bin, integrated over pixels by the pixel MTF (§4.3).

In a :class:`~gradix.env.LayeredMedium` the layered collection rule applies (§4.1): scattered
spectra propagate in the sample to the coverslip only (supercritical components decay), cross
the stack with t_s (S₁) and t_p (S₂), and are apodised by the flux in the immersion,
C = t·√(Re k_z,i/k_s). Transmitted illumination (travel +1) is defined in the sample and
crosses the stack the same way. Epi illumination (travel −1) is defined in the coverslip: its
reflection at the sample interface is the analytic reference of iSCAT, and its transmission
into the sample illuminates the scatterers. Magnification maps object space onto the camera.
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
from gradix._core.rules import Decision, detection_spacing
from gradix.conventions import safe_sqrt
from gradix.detect.camera import Camera
from gradix.imaging._shared import fused_modifiers, ratio_per_image
from gradix.light.reference import ReferenceBeam
from gradix.objects.environment import LayeredMedium, Medium
from gradix.objects.objectset import ObjectSet
from gradix.ops import pupil as ops
from gradix.optics import fresnel
from gradix.optics.objective import Objective
from gradix.optics.pupil import PupilContext, PupilModifier
from gradix.schema.base import Node
from gradix.schema.fields import child, knob
from gradix.schema.layout import canonical
from gradix.special import mie

__all__ = ["Coherent", "CoherentStatic"]

_PAD = 1.03125
"""The pupil grid extends this far beyond NA, so the soft aperture edge fits inside it."""


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


@dataclasses.dataclass(frozen=True)
class _Optics:
    """Per-call optics on the scene's device and precision."""

    dtype: torch.dtype
    cdtype: torch.dtype
    device: torch.device
    na: Tensor  # [B|1]
    magnification: Tensor  # [B|1]
    n: Tensor  # [B|1], the real index of the sample (or homogeneous) medium
    n_g: Tensor  # [B|1], the coverslip (n in a homogeneous medium)
    n_i: Tensor  # [B|1], the immersion (n in a homogeneous medium)
    layered: bool
    focus: Tensor  # [B|1, A|1]
    pitch: Tensor  # [B|1], object-space camera pitch
    lam: Tensor  # [B|1, L]
    xs: Tensor  # [B|1, Ws], sample positions in object space
    ys: Tensor  # [B|1, Hs]


def _flux(kz: Tensor) -> Tensor:
    """Return √(Re k_z) with finite forward-mode tangents where the wave is evanescent."""
    re = kz.real
    positive = re > 0
    return torch.where(positive, torch.sqrt(torch.where(positive, re, 1.0)), torch.zeros_like(re))


@register.element("coherent.pupil")
@dataclasses.dataclass(frozen=True, eq=False)
class Coherent(Element[Irradiance]):
    """Scalar coherent imaging of plane waves and sparse scatterers, with explicit interference.

    Parameters
    ----------
    objective : Objective
        The objective (NA, magnification, focus, pupil modifiers).
    camera : Camera
        The camera.
    oversample : int or "auto", default "auto"
        Samples per camera pixel; ``"auto"`` meets λ_min/(4·NA).
    pupil_samples : int or "auto", default "auto"
        Pupil cells per axis for the scattered spectra; ``"auto"`` is 64, or 128 for
        supercritical collection (a layered medium with NA above the sample's index).

    Examples
    --------
    >>> import gradix as gx
    >>> camera = gx.Camera(pixel_size=6.5, shape=(32, 32))
    >>> Coherent(gx.Objective(NA=0.8, magnification=50), camera).pupil_samples
    'auto'
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
        },
    )

    schema_version: ClassVar[int] = 2
    """2: layered media and ``pupil_samples="auto"`` (M3a)."""

    objective: Objective = child(doc="the objective")
    camera: Camera = child(doc="the camera")
    _: dataclasses.KW_ONLY
    oversample: int | str = knob(default="auto", doc="samples per camera pixel")
    pupil_samples: int | str = knob(default="auto", doc="pupil cells per axis")

    # ---- static -----------------------------------------------------------------------------

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
        wl = envelope.range("light.wavelength") or envelope.range("light.wavelengths")
        wl = wl or (0.5, 0.5)
        u_ref = self._reference_u(desc, envelope)
        _spacing, s, decision = detection_spacing(
            pitch, wl[0], na, reference_u=u_ref, oversample=self.oversample
        )
        grid = Grid2D(tuple(self.camera.shape), pitch, (0.0, 0.0))
        medium = desc.nodes.get(desc.part("environment")) or desc.nodes.get("environment")
        if self.pupil_samples == "auto":
            supercritical = False
            if isinstance(medium, LayeredMedium):
                n_s = float(medium.index(dtype=torch.float64).detach().reshape(-1).min())
                supercritical = na > n_s
            samples = 128 if supercritical else 64
            rule = "128: supercritical collection" if supercritical else "64"
        else:
            samples, rule = int(self.pupil_samples), "pinned"
        return CoherentStatic(
            decisions=(decision, Decision("pupil.samples", samples, rule, {"na": na})),
            grid=grid,
            oversample=s,
            pupil_samples=samples,
            na_max=na,
        )

    def validity(self, desc: Description, envelope: Envelope) -> list[Violation]:
        """Report findings: scalar imaging here needs NA below the medium's index.

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
        found = self._reference_validity(desc, envelope)
        medium = desc.nodes.get(desc.part("environment")) or desc.nodes.get("environment")
        if not isinstance(medium, Medium):
            return found
        if isinstance(medium, LayeredMedium):
            return found + self._layered_validity(desc, envelope, medium)
        found += self._epi_validity(desc)
        ratio = ratio_per_image(self.objective, medium)  # NA/n per image, largest
        if ratio is None or ratio < 1.0:
            return found
        return [
            *found,
            Violation(
                "error",
                "NA reaches the medium's index; coherent imaging of homogeneous media needs "
                "NA below it",
                entry=f"{desc.part('objective')}.NA",
                value=ratio,
                limit=1.0,
                element=desc.path or "coherent",
                fix="lower the NA below the medium's index",
            ),
        ]

    @staticmethod
    def _epi_validity(desc: Description) -> list[Violation]:
        """Report epi light in a homogeneous medium: iSCAT-lite, or no reference at all."""
        if not any(getattr(node, "travel", 1) == -1 for node in desc.nodes.values()):
            return []
        where = desc.path or "coherent"
        if any(isinstance(node, ReferenceBeam) for node in desc.nodes.values()):
            return [
                Violation(
                    "info",
                    "iSCAT-lite: epi light in a homogeneous medium against a constant reference; "
                    "this approximation tier omits the Fresnel transmission, r_s/r_p(θ) and the "
                    "index change at the coverslip",
                    element=where,
                    fix="a gx.env.LayeredMedium reflects the reference at the coverslip (§5.12)",
                )
            ]
        return [
            Violation(
                "warn",
                "epi light in a homogeneous medium: nothing reflects a reference, so the camera "
                "records the backscatter alone",
                element=where,
                fix="give the Sample a gx.env.LayeredMedium, whose coverslip reflects the iSCAT "
                "reference, or add an on-axis gx.light.ReferenceBeam (iSCAT-lite)",
            )
        ]

    def _layered_validity(
        self, desc: Description, envelope: Envelope, medium: LayeredMedium
    ) -> list[Violation]:
        """NA below the immersion's index; scatterers in the sample (z ≤ 0), near the interface."""
        where = desc.path or "coherent"
        found: list[Violation] = []
        wl = envelope.range("light.wavelength") or envelope.range("light.wavelengths")
        reach = wl[1] if wl is not None else 0.7  # "near": within a vacuum wavelength
        ratio = ratio_per_image(self.objective, medium)  # NA over the immersion index
        if ratio is not None and ratio >= 1.0:
            found.append(
                Violation(
                    "error",
                    "NA reaches the immersion's index",
                    entry=f"{desc.part('objective')}.NA",
                    value=ratio,
                    limit=1.0,
                    element=where,
                    fix="lower the NA below the immersion index",
                )
            )
        for name, node in desc.nodes.items():
            if not isinstance(node, ObjectSet):
                continue
            z = envelope.range(f"{name}.position.z")
            if z is None:
                values = torch.as_tensor(node.position).detach()
                z = (0.0, float(values[..., 2].max())) if values.numel() else None
            if z is not None and z[1] > 0.0:
                found.append(
                    Violation(
                        "error",
                        f"scatterers {name!r} reach z > 0: in a layered medium the sample lies "
                        "at z < 0, below the coverslip (§4.1)",
                        entry=f"{name}.position.z",
                        value=z[1],
                        limit=0.0,
                        element=where,
                        fix="place the scatterers at z ≤ 0 (gx.coords.depth(d) gives z = −d)",
                    )
                )
            elif z is not None and z[1] > -reach:
                found.append(
                    Violation(
                        "info",
                        f"scatterers {name!r} come within λ of the coverslip: the light they "
                        "exchange with the interface by multiple reflection is not modelled "
                        "(cap-32)",
                        entry=f"{name}.position.z",
                        value=z[1],
                        limit=-reach,
                        element=where,
                    )
                )
        return found

    def _reference_u(self, desc: Description, envelope: Envelope) -> float | None:
        """Return the largest object-space direction |u_ref| = Mag·|sin θ| of the references."""
        references = {k: v for k, v in desc.nodes.items() if isinstance(v, ReferenceBeam)}
        if not references:
            return None
        mag = float(self._number("magnification", self.objective, envelope, "objective", "hi"))
        largest = 0.0
        for name, reference in references.items():
            if reference.angle is None:
                continue
            x, y = envelope.range(f"{name}.angle.x"), envelope.range(f"{name}.angle.y")
            if x is not None and y is not None:
                sx = max(abs(math.sin(x[0])), abs(math.sin(x[1])))
                sy = max(abs(math.sin(y[0])), abs(math.sin(y[1])))
            else:  # an eager call reads the values
                angle = torch.as_tensor(reference.angle).detach().reshape(-1, 2).abs()
                sx, sy = (math.sin(float(v)) for v in angle.max(0).values)
            largest = max(largest, mag * math.hypot(sx, sy))
        return largest

    def _reference_validity(self, desc: Description, envelope: Envelope) -> list[Violation]:
        """Warn when an off-axis sideband overlaps the autocorrelation or aliases on the camera."""
        u_ref = self._reference_u(desc, envelope)
        if not u_ref:
            return []
        na = float(self._number("NA", self.objective, envelope, "objective", pick="hi"))
        pitch = float(self._number("pixel_size", self.camera, envelope, "camera", pick="hi")) / (
            float(self._number("magnification", self.objective, envelope, "objective"))
        )
        wl = envelope.range("light.wavelength") or envelope.range("light.wavelengths")
        wl_min = wl[0] if wl is not None else 0.5
        where = desc.path or "coherent"
        found: list[Violation] = []
        if u_ref < 3.0 * na:
            found.append(
                Violation(
                    "warn",
                    "the off-axis sideband overlaps the autocorrelation term: the reference "
                    f"direction |u_ref| = {u_ref:.3g} is below 3·NA = {3 * na:.3g}",
                    value=u_ref,
                    limit=3.0 * na,
                    element=where,
                    fix="tilt the reference more (u_ref = Mag·sin θ)",
                )
            )
        nyquist = wl_min / (2.0 * pitch)  # the camera's band edge, NA units
        if u_ref + na > nyquist * math.sqrt(2.0):
            found.append(
                Violation(
                    "warn",
                    "the camera undersamples the off-axis sideband: |u_ref| + NA = "
                    f"{u_ref + na:.3g} exceeds the pixels' band edge λ/(2p) = {nyquist:.3g} "
                    "along the diagonal",
                    value=u_ref + na,
                    limit=nyquist * math.sqrt(2.0),
                    element=where,
                    fix="smaller object-space pixels (more magnification) or a smaller tilt",
                )
            )
        return found

    def memory(self, desc: Description, static: Static) -> int | None:
        """Estimate the per-image peak bytes: the pupil work and one field per sphere.

        Parameters
        ----------
        desc : Description
            Static description (slot capacities, frames, spectral bins).
        static : Static
            A :class:`CoherentStatic`.

        Returns
        -------
        int or None
            Bytes per image (complex64 working set with autograd), or None without a grid.
        """
        if not isinstance(static, CoherentStatic) or static.grid is None:
            return None
        spheres = sum(desc.slots.values()) or 1  # every scatterer population's slots
        pupil = static.pupil_samples
        height, width = (static.oversample * n for n in static.grid.shape)
        per_sphere = 128 * pupil * pupil + 24 * height * width + 16 * pupil * max(height, width)
        return int(desc.frames * desc.bins * (spheres * per_sphere + 32 * height * width))

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
        entries: dict[str, tuple[float, float]] = {"light.wavelength": (wl, wl)}
        medium = inputs[2] if len(inputs) > 2 and isinstance(inputs[2], Node) else None
        nodes: dict[str, Node] = {"environment": medium} if medium is not None else {}
        references = inputs[3] if len(inputs) > 3 else ()
        for i, reference in enumerate(self._as_references(references)):
            # the waves carry sin θ; the static sizing reads it back as an angle range
            u = reference.u.detach().to(torch.float64).reshape(-1, 2).abs().max(0).values
            ax, ay = (math.asin(min(float(v), 1.0)) for v in u)
            nodes[f"reference{i}"] = ReferenceBeam(angle=torch.tensor([ax, ay]))
        desc = Description(nodes=nodes)
        report(self.validity(desc, Envelope(entries)), "warn")  # errors raise
        return self.configure(desc, Envelope(entries))

    @staticmethod
    def _as_references(references: object) -> tuple[PlaneWaves, ...]:
        if references is None:
            return ()
        if isinstance(references, PlaneWaves):
            return (references,)
        if isinstance(references, tuple) and all(isinstance(r, PlaneWaves) for r in references):
            return references
        raise StructureError("references must be PlaneWaves or a tuple of them")

    # ---- run time ---------------------------------------------------------------------------

    def forward(self, *inputs: object, static: Static) -> Irradiance:
        """Image the background plane waves and the scattered spectra onto the camera.

        Parameters
        ----------
        *inputs : object
            The contributions (an :class:`~gradix.ObjectSpectra` or a tuple of them), the
            incident :class:`~gradix.PlaneWaves`, the medium, and optionally the image-side
            references (:class:`~gradix.PlaneWaves` from :class:`~gradix.light.ReferenceBeam`,
            one or a tuple).
        static : Static
            A :class:`CoherentStatic`.

        Returns
        -------
        Irradiance
            ``[B, A, 1, H, W]`` photons per pixel.
        """
        background, scattered, optics, static = self._fields(inputs[:3], static)
        references = self._as_references(inputs[3] if len(inputs) > 3 else ())
        # explicit interference, term by term: weak scatterers keep their contrast (§4.4)
        intensity = background.real**2 + background.imag**2
        if scattered is not None:
            intensity = intensity + 2.0 * (background.conj() * scattered).real
            intensity = intensity + scattered.real**2 + scattered.imag**2
        if references:
            if background.shape[2] != 1:
                msg = "an image-side reference needs one illumination mode (M = 1)"
                raise StructureError(msg, fix="use a single coherent plane wave")
            reference = self._references(references, optics)  # [B, A, 1, L, Hs, Ws]
            obj = background if scattered is None else background + scattered
            intensity = intensity + reference.real**2 + reference.imag**2
            intensity = intensity + 2.0 * (reference.conj() * obj).real
        intensity = intensity.sum((2, 3))  # modes and bins: [B, A, Hs, Ws]
        s = static.oversample
        pixels = ops.pixel_mtf(intensity, s)[..., ::s, ::s]
        pixels = pixels * optics.pitch.reshape(-1, 1, 1, 1) ** 2
        assert static.grid is not None
        return Irradiance(data=pixels[:, :, None], grid=static.grid)

    def image_field(self, *inputs: object, static: Static) -> Tensor:
        """Return the complex image field on the camera's samples, background included.

        Parameters
        ----------
        *inputs : object
            As for :meth:`forward`: the contributions, the plane waves and the medium.
        static : Static
            A :class:`CoherentStatic`.

        Returns
        -------
        Tensor
            Complex ``[B, A, M, L, H·s, W·s]`` in √(photons/µm²), s the oversampling: the field
            whose squared magnitude, summed over modes and bins, is the image before the
            pixel MTF.
        """
        background, scattered, _optics, _static = self._fields(inputs[:3], static)
        return background if scattered is None else background + scattered

    def _fields(
        self, inputs: tuple[object, ...], static: Static
    ) -> tuple[Tensor, Tensor | None, _Optics, CoherentStatic]:
        if len(inputs) != 3:
            raise StructureError("Coherent takes (contributions, PlaneWaves, medium)")
        contributions, waves, medium = inputs
        if isinstance(contributions, ObjectSpectra):
            contributions = (contributions,)
        if not isinstance(contributions, tuple) or not isinstance(waves, PlaneWaves):
            raise StructureError("Coherent takes a tuple of ObjectSpectra and PlaneWaves")
        if not isinstance(medium, Medium):
            raise StructureError("Coherent needs a medium", fix="gx.env.Homogeneous(n)")
        if medium.layered and not isinstance(medium, LayeredMedium):
            raise StructureError(f"Coherent cannot image a {type(medium).__name__}")
        if not isinstance(static, CoherentStatic) or static.grid is None:
            raise StructureError("Coherent needs a CoherentStatic")
        optics = self._optics(contributions, waves, medium, static)
        background = self._background(waves, medium, optics, static)
        scattered: Tensor | None = None
        for contribution in contributions:
            if not isinstance(contribution, ObjectSpectra):
                raise StructureError(f"Coherent cannot image a {type(contribution).__name__}")
            if contribution.position.shape[2] == 0:
                continue  # no scatterers: the background alone
            field = self._scattered(contribution, waves, medium, optics, static)
            scattered = field if scattered is None else scattered + field
        return background, scattered, optics, static

    def _optics(
        self,
        contributions: tuple[object, ...],
        waves: PlaneWaves,
        medium: Medium,
        static: CoherentStatic,
    ) -> _Optics:
        ospec, cspec = self.objective.schema(), self.camera.schema()
        # the scene's precision and device (waves from scalar source parameters are float32 and
        # live on the CPU)
        dtype = waves.wavelengths.dtype
        device = waves.wavelengths.device
        for c in contributions:
            if isinstance(c, ObjectSpectra):
                dtype = torch.promote_types(dtype, c.position.dtype)
                device = c.position.device
                break
        cdtype = torch.complex128 if dtype == torch.float64 else torch.complex64

        def image(node: Node, spec: dict, name: str) -> Tensor:
            return canonical(getattr(node, name), spec[name], dtype=dtype, device=device)

        na = image(self.objective, ospec, "NA").reshape(-1)
        focus = image(self.objective, ospec, "focus")  # [B|1, A|1] (a setting)
        magnification = image(self.objective, ospec, "magnification").reshape(-1)
        pitch = (image(self.camera, cspec, "pixel_size").reshape(-1) / magnification).reshape(-1)
        index = medium.index(dtype=dtype, device=device)
        n = (index.real if index.is_complex() else index).reshape(-1)  # [B|1]
        n_g = n_i = n
        layered = isinstance(medium, LayeredMedium)
        if isinstance(medium, LayeredMedium):
            glass = medium.coverslip_index(dtype=dtype, device=device)
            oil = medium.immersion_index(dtype=dtype, device=device)
            n_g = (glass.real if glass.is_complex() else glass).reshape(-1)
            n_i = (oil.real if oil.is_complex() else oil).reshape(-1)
        lam = waves.wavelengths.to(device=device, dtype=dtype)  # [B|1, L]
        s = static.oversample
        assert static.grid is not None
        height, width = static.grid.shape
        # camera samples (object space): s per pixel, the first at each pixel centre
        p = pitch.reshape(-1, 1)
        ky = torch.arange(height * s, device=device, dtype=dtype)
        kx = torch.arange(width * s, device=device, dtype=dtype)
        ys = (ky // s + 0.5) * p + (ky % s) * p / s  # [B|1, Hs]
        xs = (kx // s + 0.5) * p + (kx % s) * p / s  # [B|1, Ws]
        return _Optics(
            dtype,
            cdtype,
            device,
            na,
            magnification,
            n,
            n_g,
            n_i,
            layered,
            focus,
            pitch,
            lam,
            xs,
            ys,
        )

    def _modifiers_at(self, u: Tensor, medium: Medium, o: _Optics) -> Tensor | None:
        """Evaluate the pupil modifiers at plane-wave directions ``u [B|1, A|1, M, J, 2]``.

        Returns ``[B, A, M, J, L]`` complex multipliers, or None without modifiers.
        """
        modifiers = [m for m in self.objective.pupil if isinstance(m, PupilModifier)]
        if len(modifiers) != len(self.objective.pupil):
            raise StructureError("objective.pupil holds something that is not a gx.PupilModifier")
        if not modifiers:
            return None
        b, a, m, j, _ = u.shape
        q = u.to(torch.float64).reshape(b, 1, 1, a * m * j, 2)
        wl = o.lam.to(torch.float64)[:, :, None, None]  # [B|1, L, 1, 1]
        fx, fy = q[..., 0] / wl, q[..., 1] / wl  # [B, L, 1, Q]
        ctx = PupilContext(  # the pupil is defined in the immersion
            na=o.na.to(torch.float64).reshape(-1, 1, 1, 1),
            n=o.n_i.to(torch.float64).reshape(-1, 1, 1, 1),
        )
        total = modifiers[0](fx, fy, wl, ctx)
        for modifier in modifiers[1:]:
            total = total * modifier(fx, fy, wl, ctx)
        total = torch.broadcast_to(total, torch.broadcast_shapes(total.shape, fx.shape))
        bins = total.shape[1]
        total = total[:, :, 0, :].reshape(total.shape[0], bins, a, m, j)
        return total.permute(0, 2, 3, 4, 1).to(o.cdtype)  # [B, A, M, J, L]

    def _background(
        self, waves: PlaneWaves, medium: Medium, o: _Optics, static: CoherentStatic
    ) -> Tensor:
        """Return the background plane waves on the camera samples, ``[B, A, M, L, Hs, Ws]``."""
        if o.layered:
            assert isinstance(medium, LayeredMedium)
            return self._background_layered(waves, medium, o, static)
        dtype, device = o.dtype, o.device
        n = o.n.reshape(-1, 1, 1, 1, 1)
        na = o.na.reshape(-1, 1, 1, 1, 1)
        wl = o.lam[:, None, None, None, :]  # [B|1, 1, 1, 1, L]
        u = waves.u.to(device=device, dtype=dtype)  # [B|1, A|1, M, J, 2]
        radius = safe_sqrt((u * u).sum(-1))[..., None]  # [B|1, A|1, M, J, 1]; finite grad at 0
        du = 2.0 * static.na_max * _PAD / static.pupil_samples
        # only waves travelling toward the objective and propagating (|u| < n) reach the camera;
        # an aplanatic objective maps a wave of irradiance I at angle θ to I·cosθ on the camera
        cos_b = torch.sqrt(torch.clamp(1.0 - radius * radius / (n * n), min=0.0))
        transmission = ops.soft_aperture(radius, na, du) * torch.sqrt(cos_b)
        transmission = torch.where(radius < n, transmission, torch.zeros_like(transmission))
        if waves.travel != 1:  # epi illumination travels away from the objective
            transmission = torch.zeros_like(transmission)
        kz = 2.0 * math.pi * torch.sqrt(torch.clamp(n * n - radius * radius, min=0.0)) / wl
        z0 = torch.as_tensor(waves.z0, dtype=dtype, device=device)
        focus5 = o.focus.reshape(o.focus.shape[0], o.focus.shape[1], 1, 1, 1)  # [B|1, A|1, 1, 1, 1]
        phase_z = kz * waves.travel * (focus5 - z0)  # [B, A, M, J, L]
        amp = (
            waves.amplitude[..., 0].to(device=device, dtype=o.cdtype)
            * transmission
            * torch.polar(torch.ones_like(phase_z), phase_z)
        )  # [B, A, M, J, L]
        modifiers = self._modifiers_at(u, medium, o)
        if modifiers is not None:
            amp = amp * modifiers
        return self._on_camera(amp, u, o)

    def _background_layered(
        self, waves: PlaneWaves, medium: LayeredMedium, o: _Optics, static: CoherentStatic
    ) -> Tensor:
        """Return the background through a layered medium: transmitted light or the epi echo."""
        dtype, device = o.dtype, o.device
        u = waves.u.to(device=device, dtype=dtype)  # [B|1, A|1, M, J, 2]
        radius = safe_sqrt((u * u).sum(-1))[..., None]  # [B|1, A|1, M, J, 1]
        u2 = radius * radius
        five = (-1, 1, 1, 1, 1)
        wl = o.lam[:, None, None, None, :]  # [B|1, 1, 1, 1, L]
        n_s, n_g, n_i = (t.reshape(five) for t in (o.n, o.n_g, o.n_i))
        kzs, kzg, kzi = (fresnel.axial(n, u2, wl) for n in (n_s, n_g, n_i))
        du = 2.0 * static.na_max * _PAD / static.pupil_samples
        aperture = ops.soft_aperture(radius, o.na.reshape(five), du)
        focus5 = o.focus.reshape(o.focus.shape[0], o.focus.shape[1], 1, 1, 1)
        phase = kzi * focus5  # the interface (z = 0) to the focal plane, in the immersion
        mismatch = medium.mismatch_phase(u2, wl)
        if mismatch is not None:
            phase = phase + mismatch
        flux = _flux(kzi)
        t_gi = fresnel.transmission(kzg, kzi, n_g, n_i)
        if waves.travel == 1:  # transmitted light, defined in the sample at z0
            z0 = torch.as_tensor(waves.z0, dtype=dtype, device=device)
            t_sg = fresnel.transmission(kzs, kzg, n_s, n_g)
            k_s = 2.0 * math.pi * n_s / wl
            factor = t_sg * t_gi * flux / torch.sqrt(k_s) * torch.exp(1j * kzs * (0.0 - z0))
        else:  # epi light, defined in the coverslip at the interface: its reflection
            r = fresnel.reflection(kzg, kzs, n_g, n_s)
            k_g = 2.0 * math.pi * n_g / wl
            factor = r * t_gi * flux / torch.sqrt(k_g)
        amp = waves.amplitude[..., 0].to(device=device, dtype=o.cdtype) * (
            aperture * factor * torch.exp(1j * phase)
        ).to(o.cdtype)  # [B, A, M, J, L]
        modifiers = self._modifiers_at(u, medium, o)
        if modifiers is not None:
            amp = amp * modifiers
        return self._on_camera(amp, u, o)

    def _incident(self, waves: PlaneWaves, pos: Tensor, o: _Optics) -> Tensor:
        """Return each incident wave at the spheres, in the sample: ``[B, A, M, J, L, N]``.

        Epi light in a layered medium is transmitted into the sample first (t_s·√(n_s/n_g):
        field amplitudes normalised to the irradiance in each medium).
        """
        if o.layered and waves.travel != 1:
            u = waves.u.to(device=o.device, dtype=o.dtype)
            u2 = (u * u).sum(-1)[..., None]  # [B|1, A|1, M, J, 1]
            five = (-1, 1, 1, 1, 1)
            wl = o.lam[:, None, None, None, :]
            n_s, n_g = o.n.reshape(five), o.n_g.reshape(five)
            kzs, kzg = fresnel.axial(n_s, u2, wl), fresnel.axial(n_g, u2, wl)
            t = fresnel.transmission(kzg, kzs, n_g, n_s) * torch.sqrt(n_s / n_g)
            amplitude = waves.amplitude * t.to(waves.amplitude.dtype)[..., None]
            waves = waves.replace(amplitude=amplitude, z0=0.0)
        return waves.at_waves(pos, o.n.reshape(-1, 1))[..., 0, :]

    def _collection(self, u2: Tensor, medium: Medium, o: _Optics) -> tuple[Tensor, Tensor, Tensor]:
        """Return the s and p collection weights C/k_z,s and the pupil phase's axial wavenumbers.

        ``u2 [K]`` are the pupil samples. Returns ``c_s``, ``c_p`` as ``[B|1, 1, 1, 1, L, 1, K]``
        (the f-measure's 1/k_z,s folded in, finite at the critical angle) and the complex
        ``(k_z,s, k_z,i)`` as ``[B|1, 1, 1, L, 1, K]``.
        """
        seven = (-1, 1, 1, 1, 1, 1, 1)
        wl = o.lam.reshape(o.lam.shape[0], 1, 1, 1, o.lam.shape[1], 1, 1)
        n_s, n_g, n_i = (t.to(torch.float64).reshape(seven) for t in (o.n, o.n_g, o.n_i))
        wl = wl.to(torch.float64)
        kzs, kzg, kzi = (fresnel.axial(n, u2, wl) for n in (n_s, n_g, n_i))
        k_s = 2.0 * math.pi * n_s / wl
        if o.layered:
            # t/k_z,s through the sample–coverslip interface, finite where k_z,s → 0
            s_over = 2.0 / (kzs + kzg)
            p_over = 2.0 * n_s * n_g / (n_g * n_g * kzs + n_s * n_s * kzg)
            flux = _flux(kzi) / torch.sqrt(k_s)
            c_s = s_over * fresnel.transmission(kzg, kzi, n_g, n_i, "s") * flux
            c_p = p_over * fresnel.transmission(kzg, kzi, n_g, n_i, "p") * flux
        else:
            # homogeneous: only propagating waves reach the far field; √cosθ/k_z = 1/√(k·k_z)
            inside = u2 < n_s * n_s
            real_kz = torch.where(inside, kzs.real, torch.ones_like(kzs.real))
            weight = torch.where(inside, 1.0 / torch.sqrt(k_s * real_kz), torch.zeros_like(real_kz))
            c_s = c_p = weight.to(kzs.dtype)
        kz_pair = torch.stack([kzs[:, :, :, 0], kzi[:, :, :, 0]])  # drop J: [2, B, A, M, L, N, K]
        return c_s, c_p, kz_pair

    def _on_camera(self, amp: Tensor, u: Tensor, o: _Optics) -> Tensor:
        """Sum plane waves on the camera samples: ``[B, A, M, L, Hs, Ws]``.

        ``amp [B, A, M, J, L]`` are the waves' amplitudes and ``u [B, A, M, J, 2]`` their
        object-space directions.
        """
        wl = o.lam[:, None, None, None, :]  # [B|1, 1, 1, 1, L]
        k0 = (2.0 * math.pi / wl)[..., None]  # [B|1, 1, 1, 1, L, 1]
        ux, uy = u[..., 0][..., None, None], u[..., 1][..., None, None]  # [B, A, M, J, 1, 1]
        phx = k0 * ux * o.xs.reshape(o.xs.shape[0], 1, 1, 1, 1, -1)
        phy = k0 * uy * o.ys.reshape(o.ys.shape[0], 1, 1, 1, 1, -1)
        wave_x = torch.polar(torch.ones_like(phx), phx)  # [B, A, M, J, L, Ws]
        wave_y = torch.polar(torch.ones_like(phy), phy)  # [B, A, M, J, L, Hs]
        return torch.einsum("bamjl,bamjly,bamjlx->bamlyx", amp, wave_y, wave_x)

    def _references(self, references: tuple[PlaneWaves, ...], o: _Optics) -> Tensor:
        """Return the image-side references on the camera samples, ``[B, A, 1, L, Hs, Ws]``.

        Each carries ``sin θ`` on the image side; the object-space direction is Mag·sin θ, and
        the wave reaches the camera without aperture or apodisation (§4.5, construction 4).
        """
        total: Tensor | None = None
        for reference in references:
            u = reference.u.to(device=o.device, dtype=o.dtype)
            u = u * o.magnification.reshape(-1, 1, 1, 1, 1)  # object-space directions
            amp = reference.amplitude[..., 0].to(device=o.device, dtype=o.cdtype)
            field = self._on_camera(amp, u, o)
            total = field if total is None else total + field
        assert total is not None
        return total

    def _scattered(
        self,
        spectra: ObjectSpectra,
        waves: PlaneWaves,
        medium: Medium,
        o: _Optics,
        static: CoherentStatic,
    ) -> Tensor:
        """Return one producer's scattered field on the camera samples, ``[B, A, M, L, Hs, Ws]``.

        The pupil work runs in the layout ``[B, A, M, L, N, P, P]``; the angular part (π_n, τ_n)
        is computed once on the pupil grid and contracted with every sphere's coefficients by
        a batched matrix product.
        """
        if spectra.evaluator != "mie":
            raise StructureError(f"Coherent cannot evaluate {spectra.evaluator!r} spectra")
        params = spectra.params
        a_n, b_n = getattr(params, "a", None), getattr(params, "b", None)
        if not isinstance(a_n, Tensor) or not isinstance(b_n, Tensor):
            raise StructureError("Mie spectra need params.a and params.b")
        dtype, cdtype, device = o.dtype, o.cdtype, o.device
        pos = spectra.position.to(device=device, dtype=dtype)  # [B|1, A|1, N, 3]
        count = pos.shape[2]
        samples = static.pupil_samples
        u, du = ops.pupil_axis(samples, static.na_max * _PAD, dtype=torch.float64, device=device)
        uy, ux = torch.meshgrid(u, u, indexing="ij")
        ux, uy = ux.reshape(-1), uy.reshape(-1)  # [K = P·P]
        u2 = ux * ux + uy * uy
        # ---- geometry in fp64, layout [B|1, A|1, M, J, L|1, K] ----
        n64 = o.n.to(torch.float64).reshape(-1, 1, 1, 1, 1, 1)
        u_in = waves.u.to(device=device, dtype=torch.float64)  # [B|1, A|1, M, J, 2]
        uix = u_in[..., 0][..., None, None]  # [B|1, A|1, M, J, 1, 1]
        uiy = u_in[..., 1][..., None, None]
        w = safe_sqrt((n64 * n64 - u2).to(torch.complex128))  # pupil directions, toward +z
        w_in = safe_sqrt((n64 * n64 - uix * uix - uiy * uiy).to(torch.complex128))
        mu = (ux * uix + uy * uiy + waves.travel * w * w_in) / (n64 * n64)  # cos Θ
        dx, dy = ux - uix, uy - uiy  # azimuth of the scattering plane about the incidence
        r2 = dx * dx + dy * dy
        safe_r2 = torch.where(r2 > 0, r2, torch.ones_like(r2))
        cos2 = torch.where(r2 > 0, dx * dx / safe_r2, torch.ones_like(r2))  # x analyser
        terms = a_n.shape[-1]
        pi, tau = mie.angular(mu, terms)  # [B|1, A|1, M, J, 1, K, n]
        # ---- S₁, S₂ per sphere: [B, A, M, J, L, N, K] by matrix products over the orders ----
        order = torch.arange(1, terms + 1, dtype=torch.float64, device=device)
        c = ((2.0 * order + 1.0) / (order * (order + 1.0))).to(cdtype)
        a8 = (a_n.to(device=device, dtype=cdtype) * c).permute(0, 1, 3, 2, 4)  # [B, A, L, N, n]
        b8 = (b_n.to(device=device, dtype=cdtype) * c).permute(0, 1, 3, 2, 4)
        a8, b8 = a8[:, :, None, None], b8[:, :, None, None]  # [B|1, A|1, 1, 1, L, N, n]
        pi_t = pi.to(cdtype).transpose(-1, -2)  # [B|1, A|1, M, J, 1, n, K]
        tau_t = tau.to(cdtype).transpose(-1, -2)
        s1 = a8 @ pi_t + b8 @ tau_t  # [B, A, M, J, L, N, K]
        s2 = a8 @ tau_t + b8 @ pi_t
        cos2 = cos2.to(dtype)[..., None, :]  # [B|1, A|1, M, J, 1, 1, K]
        c_s, c_p, kz_pair = self._collection(u2, medium, o)  # [B|1, 1, 1, 1, L, 1, K]
        s_par = c_p.to(cdtype) * s2 * cos2 + c_s.to(cdtype) * s1 * (1.0 - cos2)
        k_m = 2.0 * math.pi * o.n.reshape(-1, 1) / o.lam  # [B|1, L]: per-image index or bins
        k_m = k_m.reshape(k_m.shape[0], 1, 1, 1, k_m.shape[1], 1, 1)  # [B|1, 1, 1, 1, L, 1, 1]
        f = (1j * s_par) / k_m  # scattering amplitude f = iS/k, µm
        # ---- incident waves at the spheres, each on its own: [B, A, M, J, L, N, 1] ----
        incident = self._incident(waves, pos, o).to(cdtype)
        if spectra.presence is not None:
            presence = spectra.presence.to(device=device, dtype=dtype)  # [B|1, A|1, N]
            incident = incident * presence[:, :, None, None, None, :]
        strength = (f * incident[..., None]).sum(3)  # coherent sum over J: [B, A, M, L, N, K]
        strength = strength.reshape(*strength.shape[:-1], samples, samples)
        # ---- pupil weights, layout [B, A, 1, L, N, P, P] ----
        seven = (-1, 1, 1, 1, 1, 1, 1)
        up = u.to(dtype)
        uyp, uxp = torch.meshgrid(up, up, indexing="ij")
        aperture = ops.soft_aperture(torch.sqrt(uxp * uxp + uyp * uyp), o.na.reshape(seven), du)
        # the axial wavenumbers on the pupil grid: [B|1, 1, 1, L, 1, P, P]
        kzs, kzi = (k.reshape(*k.shape[:-1], samples, samples) for k in kz_pair)
        focus7 = o.focus.reshape(o.focus.shape[0], o.focus.shape[1], 1, 1, 1, 1, 1)
        z7 = pos[..., 2].to(torch.float64).reshape(pos.shape[0], pos.shape[1], 1, 1, count, 1, 1)
        if o.layered:
            # to the interface in the sample (evanescent parts decay), then to the focal plane
            # in the immersion, with the coverslip's Gibson–Lanni mismatch
            depth = torch.clamp(-z7, min=0.0)
            phase = kzs * depth + kzi * focus7.to(torch.float64)
            wl64 = o.lam.to(torch.float64).reshape(o.lam.shape[0], 1, 1, o.lam.shape[1], 1, 1, 1)
            u2_grid = (uxp * uxp + uyp * uyp).to(torch.float64)
            assert isinstance(medium, LayeredMedium)
            mismatch = medium.mismatch_phase(u2_grid, wl64)
            if mismatch is not None:
                phase = phase + mismatch
        else:
            phase = kzs * (focus7.to(torch.float64) - z7)  # [B, A, 1, L, N, P, P]
        # pupil in f-measure: (2π)²·A(k⊥) = 2πi·f·E_inc/k_z with the collection folded into
        # c_s and c_p above; here the aperture and the propagation to the focal plane
        pupil = (2j * math.pi) * aperture * torch.exp(1j * phase).to(cdtype)
        table = o.lam.to(torch.float64)[:, None, :]  # [B|1, 1, L]
        modifiers = fused_modifiers(self.objective, table, u, o.na, medium)
        if modifiers is not None:  # [B|1, 1, L, P, P]
            pupil = pupil * modifiers.to(cdtype)[:, :, None, :, None]
        pupil = pupil.to(cdtype) * strength  # [B, A, M, L, N, P, P]
        batch = pupil.shape[:5]
        flat = pupil.reshape(-1, samples, samples)
        g = flat.shape[0]
        wl_g = o.lam.reshape(o.lam.shape[0], 1, 1, -1, 1).expand(*batch).reshape(-1)
        dx_s = o.xs.reshape(o.xs.shape[0], 1, 1, 1, 1, -1) - pos[..., 0][:, :, None, None, :, None]
        dy_s = o.ys.reshape(o.ys.shape[0], 1, 1, 1, 1, -1) - pos[..., 1][:, :, None, None, :, None]
        dx_s = dx_s.expand(*batch, dx_s.shape[-1]).reshape(g, -1)
        dy_s = dy_s.expand(*batch, dy_s.shape[-1]).reshape(g, -1)
        field = ops.mft_roi(flat, up, du, wl_g, dx_s, dy_s)
        field = field.reshape(*batch, dy_s.shape[-1], dx_s.shape[-1])
        return field.sum(4)  # superpose the spheres: [B, A, M, L, Hs, Ws]
