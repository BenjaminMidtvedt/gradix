"""The pupil-PSF emission element for point emitters (``emit.pupil_mft``; the ``standard`` tier).

Each emitter's scalar pupil (soft aperture, collection apodisation, defocus to the focal plane,
the objective's pupil modifiers) is transformed onto a small region of interest around it by a
matrix Fourier transform, squared, integrated over camera pixels by the pixel MTF and added to
the image (§5.1, §5.3). Homogeneous, index-matched media use the ``cos^-½θ`` collection
apodisation; a :class:`~gradix.env.LayeredMedium` uses the layered collection rule (§4.1).
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Mapping
from typing import ClassVar

import torch
from torch import Tensor

from gradix import register
from gradix._core.carriers import EmitterSet, Irradiance
from gradix._core.contract import (
    Capabilities,
    Description,
    Edge,
    Element,
    Slot,
    Static,
    Violation,
)
from gradix._core.envelope import LINEAR_BUCKET, Envelope
from gradix._core.errors import StructureError
from gradix._core.grid import Grid2D, PupilGrid, nice_size
from gradix._core.precision import ieee_matmul
from gradix._core.rules import Decision, detection_spacing
from gradix.detect.camera import Camera
from gradix.imaging._shared import MARGIN, fused_modifiers, material_range, ratio_per_image
from gradix.imaging._shared import observed as observed_of
from gradix.imaging._shared import reach as reach_of
from gradix.imaging._shared import value as value_of
from gradix.lower.emitters import emission_rows
from gradix.objects.environment import Medium
from gradix.ops import pupil as ops
from gradix.optics.objective import Objective
from gradix.schema.base import Node
from gradix.schema.fields import child, knob
from gradix.schema.layout import canonical

__all__ = ["PointPSF", "PointPSFStatic", "fused_modifiers", "ratio_per_image"]


@dataclasses.dataclass(frozen=True)
class PointPSFStatic(Static):
    """Static configuration of :class:`PointPSF`.

    Parameters
    ----------
    decisions : tuple of Decision, default ()
        Sizing decisions with provenance.
    grid : Grid2D, optional
        The camera grid in object space (nominal spacing).
    oversample : int, default 1
        Samples per camera pixel on the ROI, s.
    roi : int, default 16
        ROI size in camera pixels, R.
    pupil : PupilGrid, optional
        The pupil grid in NA units.
    deterministic : bool, default False
        Accumulate ROIs slot by slot in a fixed order instead of with one atomic ``index_add``.
    method : {"roi", "global"}, default "roi"
        Sparse ROIs, or every emitter evaluated on the whole frame.
    half : float, default 0.0
        PSF half-width, µm, that the pupil period keeps free of aliases on the global path.
    """

    grid: Grid2D | None = None
    oversample: int = 1
    roi: int = 16
    pupil: PupilGrid | None = None
    deterministic: bool = False
    method: str = "roi"
    half: float = 0.0


@register.element("emit.pupil_mft")
@dataclasses.dataclass(frozen=True, eq=False)
class PointPSF(Element[Irradiance]):
    """Scalar pupil PSFs of point emitters, rendered on regions of interest (the sparse path).

    For each emitter and wavelength bin the pupil is ``c·A(u)·cos^-½θ·exp(i·φ(u))·M(u)`` on a
    pupil grid in NA units: A is a soft aperture, ``cos^-½θ`` the collection apodisation of an
    isotropic emitter (§4.1), φ the defocus phase that carries the emitter's field to the focal
    plane (propagation by ``focus − z``), M the objective's pupil modifiers, and
    ``c² = photons/(4π(n/λ)²)`` so the PSF integrates to the collection efficiency. A matrix
    Fourier transform evaluates the field at the ROI samples around the emitter (exact sub-pixel
    placement), the intensity is integrated over camera pixels by the pixel MTF, and the ROIs
    are added into the image. Each ROI is rescaled to the pupil's Parseval power, so the total
    photons equal every other tier's; the PSF tail beyond the ROI (about 3 % at the default ROI)
    is thereby folded into the ROI, and the core reads that much high. A larger ``roi`` shrinks
    the bias at quadratic cost.

    Parameters
    ----------
    objective : Objective
        The objective (NA, magnification, focus, pupil modifiers).
    camera : Camera
        The camera whose pixel grid the PSFs are integrated on.
    oversample : int or "auto", default "auto"
        ROI samples per camera pixel; ``"auto"`` meets λ_min/(4·NA) (§4.3).
    roi : int or "auto", default "auto"
        ROI size in camera pixels; ``"auto"`` applies the stationary-phase rule (§4.3).
    pupil_samples : int or "auto", default "auto"
        Pupil grid cells per axis; ``"auto"`` is 64 or more (128 for supercritical collection),
        enough for the ROI's period.
    psf : {"auto", "scalar", "vectorial"}, default "auto"
        The PSF model. ``"auto"`` means vectorial above ``vectorial_above``; until a vectorial
        implementation is registered it falls back to scalar with a warning (§5.3).
    vectorial_above : float, default 0.7
        NA/n above which ``"auto"`` asks for the vectorial model (n: the medium's index, or the
        immersion index of a layered medium).
    method : {"roi", "global"}, default "roi"
        Sparse ROIs, or every emitter evaluated on the whole frame (``emit.global_spectrum``,
        the fallback when the envelope makes the ROI larger than the camera). Nothing is
        renormalised (power leaving the frame is lost), and the result equals the discretised
        pupil's PSF on the camera up to the pixel MTF's wrap over the 16-pixel margin (~1e-6
        rel-L2 for sources inside). The pupil grid grows with the emitters' reach, and the soft
        aperture's one-cell edge biases the PSF by O(du) (~1 % at 64 cells, 0.4 % at 256).

    Examples
    --------
    >>> import gradix as gx
    >>> camera = gx.Camera(pixel_size=6.5, shape=(32, 32))
    >>> psf = PointPSF(gx.Objective(NA=1.2, magnification=60), camera)
    >>> psf.roi
    'auto'
    """

    slot: ClassVar[Slot] = Slot.EMIT
    fidelity_knobs: ClassVar[Mapping[str, str | tuple[str, Mapping[str, str]]]] = {
        "psf": "psf",
        "oversample": "oversample",
        "roi": "roi",
        "emitter_path": ("method", {"sparse": "roi", "global": "global"}),
    }
    caps: ClassVar[Capabilities] = Capabilities(
        accepts=frozenset({EmitterSet}),
        produces=Irradiance,
        grad_quality={
            "camera": "zero",
            "camera.pixel_size": "exact",
            "objective.NA": "exact-a.e.",
        },
        reads={
            "position": "exact",
            "photons": "exact",
            "presence": "exact",
            "emission": "exact",
            "environment": "exact",
        },
        approximates=(
            Edge(
                "emit.global_spectrum",
                regime="every PSF inside its ROI (the stationary-phase rule)",
                tolerance="ladders/roi_global@v1.0",
            ),
        ),
    )

    objective: Objective = child(doc="the objective")
    camera: Camera = child(doc="the camera")
    _: dataclasses.KW_ONLY
    oversample: int | str = knob(default="auto", doc="ROI samples per camera pixel")
    roi: int | str = knob(default="auto", doc="ROI size in camera pixels")
    pupil_samples: int | str = knob(default="auto", doc="pupil cells per axis")
    psf: str = knob(default="auto", choices=("auto", "scalar", "vectorial"), doc="PSF model")
    vectorial_above: float = knob(default=0.7, doc="NA/n threshold of psf='auto'")
    method: str = knob(default="roi", choices=("roi", "global"), doc="ROI or global spectrum")

    def _optics(self, desc: Description, envelope: Envelope) -> dict[str, float]:
        obj, cam, env = desc.part("objective"), desc.part("camera"), desc.part("environment")
        values = {
            "pitch": value_of(envelope, f"{cam}.pixel_size", self.camera.pixel_size, "mid"),
            "mag": value_of(envelope, f"{obj}.magnification", self.objective.magnification, "mid"),
            "na": value_of(envelope, f"{obj}.NA", self.objective.NA, "hi"),
        }
        missing = [k for k, v in values.items() if v is None]
        if missing:
            raise StructureError(f"the envelope lacks {missing}", fix="derive it from the inputs")
        # per-image optics: the finest object-space pitch sets ROI sizes in pixels, the coarsest
        # the oversampling and the global frame's extent (the inputs' actual values)
        size_lo = observed_of(envelope, f"{cam}.pixel_size", self.camera.pixel_size, "lo")
        size_hi = observed_of(envelope, f"{cam}.pixel_size", self.camera.pixel_size, "hi")
        mag_lo = observed_of(envelope, f"{obj}.magnification", self.objective.magnification, "lo")
        mag_hi = observed_of(envelope, f"{obj}.magnification", self.objective.magnification, "hi")
        pitch_mid = float(values["pitch"] or 1.0) / float(values["mag"] or 1.0)
        pitch_lo = (size_lo / mag_hi) if size_lo and mag_hi else pitch_mid
        pitch_hi = (size_hi / mag_lo) if size_hi and mag_lo else pitch_mid
        medium = desc.nodes.get(env) or desc.nodes.get("environment")
        lo, hi, dz = math.inf, 0.0, 0.0
        # the envelope reads the bound Chain: a focus stack's planes are in objective.focus
        focus = envelope.range(f"{obj}.focus") or (0.0, 0.0)
        for pop in desc.populations:
            wl = envelope.range(f"{pop}.emission.wavelengths") or envelope.range(
                f"{pop}.wavelengths"
            )
            if wl is not None:
                lo, hi = min(lo, wl[0]), max(hi, wl[1])
            z = envelope.range(f"{pop}.position.z")
            if z is not None:
                dz = max(dz, abs(z[1] - focus[0]), abs(z[0] - focus[1]))
        if hi == 0.0:
            raise StructureError("the envelope has no emission wavelengths")
        band = (lo, hi)  # dispersive materials are read over the emission band
        n = (
            envelope.range(f"{env}.n")
            or envelope.range(f"{env}.immersion.n")
            or material_range(medium, "immersion", band)
            or (1.0, 1.0)
        )
        out = {k: float(v) for k, v in values.items() if v is not None}
        # the inputs' actual NA and index (before headroom): validity errors are about these
        n_seen = (
            envelope.observed(f"{env}.n")
            or envelope.observed(f"{env}.immersion.n")
            or material_range(medium, "immersion", band)
            or n
        )
        na_seen = observed_of(envelope, f"{obj}.NA", self.objective.NA, "hi") or out["na"]
        z_seen = max(
            (
                s[1]
                for p in desc.populations
                if (s := envelope.observed(f"{p}.position.z") or envelope.range(f"{p}.position.z"))
            ),
            default=-math.inf,
        )
        sample = envelope.range(f"{env}.sample.n") or material_range(medium, "sample", band) or n
        na_lo = value_of(envelope, f"{obj}.NA", self.objective.NA, "lo") or out["na"]
        # the pupil and ROI rules cannot use more aperture than the medium transmits
        na_eff = min(out["na"], n[1])
        sin_max = min(na_eff / n[0], 0.999)
        tan_max = sin_max / math.sqrt(1.0 - sin_max * sin_max)
        half = dz * tan_max + 2.0 * hi / max(na_lo, 1e-6)
        environment = desc.nodes.get("environment")
        if getattr(environment, "layered", False):
            # the depth propagation in the sample spreads rays by h·u/√(n_s² − u²) (§4.3); the
            # supercritical ring decays, so u is capped at 0.9·n_s
            depth = max(
                (-r[0] for p in desc.populations if (r := envelope.range(f"{p}.position.z"))),
                default=0.0,
            )
            u = min(na_eff, 0.9 * sample[0])
            half += max(depth, 0.0) * u / math.sqrt(sample[0] ** 2 - u * u)
        roi_px = 2.0 * half / pitch_lo + 4.0
        out["half"] = half
        out["pitch_lo"], out["pitch_hi"] = pitch_lo, pitch_hi
        return {
            **out,
            "na_lo": na_lo,
            "na_seen": na_seen,
            "n_seen": n_seen[0],
            "z_seen": z_seen,
            "na_eff": na_eff,
            "n": n[0],
            "n_hi": n[1],
            "n_sample": sample[0],
            "wl_min": lo,
            "wl_max": hi,
            "dz": dz,
            "tan_max": tan_max,
            "roi_px": roi_px,
        }

    def validity(self, desc: Description, envelope: Envelope) -> list[Violation]:
        """Report findings: an NA at or above the medium's index needs a layered medium.

        Parameters
        ----------
        desc : Description
            Static description of the element's context.
        envelope : Envelope
            The envelope.

        Returns
        -------
        list of Violation
            Findings.
        """
        optics = self._optics(desc, envelope)
        layered = getattr(desc.nodes.get("environment"), "layered", False)
        where = desc.path or "psf"
        if self.psf == "vectorial":
            return [
                Violation(
                    "error",
                    "vectorial PSFs arrive with M4b",
                    element=where,
                    fix="use psf='scalar' or 'auto'",
                )
            ]
        found: list[Violation] = []
        medium = desc.nodes.get(desc.part("environment")) or desc.nodes.get("environment")
        seen = ratio_per_image(self.objective, medium)  # the inputs' actual NA/n, per image
        ratio = seen if seen is not None else optics["na_seen"] / optics["n_seen"]
        if self.psf == "auto" and ratio > self.vectorial_above:
            found.append(  # an info line, not a warning, until a vectorial model is registered
                Violation(
                    "info",
                    "psf='auto' wants the vectorial model here but falls back to scalar: the "
                    f"scalar peak is 2.3–35 % high at NA/n > {self.vectorial_above} (§5.3)",
                    entry=f"{desc.part('objective')}.NA",
                    value=ratio,
                    limit=self.vectorial_above,
                    element=where,
                    fix="pass psf='scalar' to accept the scalar model",
                )
            )
        if not layered and ratio >= 1.0:
            return [
                Violation(
                    "error",
                    "NA reaches the medium's index; supercritical collection needs a layered "
                    "medium",
                    entry=f"{desc.part('objective')}.NA",
                    value=ratio,
                    limit=1.0,
                    element=where,
                    fix="declare environment=gx.env.LayeredMedium(...)",
                )
            ]
        if not layered and optics["na"] >= optics["n"]:
            found.append(
                Violation(
                    "warn",
                    "the envelope allows NA at or above the medium's index; the pupil is cut "
                    "at |u| = n",
                    entry=f"{desc.part('objective')}.NA",
                    value=optics["na"],
                    limit=optics["n"],
                    element=where,
                    fix="tighten the NA or index envelope entries",
                )
            )
        z = [envelope.range(f"{pop}.position.z") for pop in desc.populations]
        above = [r for r in z if r is not None]
        if layered and optics["z_seen"] > 0:
            found.append(
                Violation(
                    "error",
                    "emitters above the coverslip (z > 0) in a layered medium: the sample lies "
                    "at z < 0",
                    element=where,
                    fix="place emitters at z <= 0 (gx.coords.depth(d) gives z for a depth d)",
                )
            )
        elif layered and any(r[1] > LINEAR_BUCKET["length"] for r in above):
            # beyond one envelope bucket: some emitters may lie above the interface
            found.append(
                Violation(
                    "warn",
                    "the envelope reaches z > 0 in a layered medium; emitters there are placed "
                    "at the interface",
                    element=where,
                    fix="place emitters at z <= 0",
                )
            )
        if self.method == "roi" and optics["roi_px"] > max(self.camera.shape):
            found.append(
                Violation(
                    "info",
                    f"the ROI ({optics['roi_px']:.0f} px) exceeds the camera; method='global' "
                    "evaluates the frame instead, exactly and more cheaply",
                    element=where,
                )
            )
        return found

    def configure(self, desc: Description, envelope: Envelope) -> PointPSFStatic:
        """Size the ROI, the oversampling and the pupil grid from the envelope.

        Parameters
        ----------
        desc : Description
            Static description of the element's context.
        envelope : Envelope
            The envelope.

        Returns
        -------
        PointPSFStatic
            The configuration, with the decisions that set it.
        """
        o = self._optics(desc, envelope)
        pitch, fine_pitch = o["pitch_hi"], o["pitch_lo"]  # coarsest and finest image
        _spacing, s, d_spacing = detection_spacing(
            pitch, o["wl_min"], o["na_eff"], oversample=self.oversample
        )
        # diffraction from the smallest NA (the widest PSF), defocus from the largest, plus the
        # depth spread in a layered medium
        half = o["half"]
        if self.roi == "auto":
            roi = nice_size(2 * math.ceil(math.ceil(2.0 * half / fine_pitch + 4.0) / 2))
            roi += roi % 2
            roi_rule = "R·p ≥ 2(|Δz|·tanθ_max + 2λ_max/NA) + 4p, even and 2·3·5·7-smooth"
        else:
            roi, roi_rule = int(self.roi), "pinned"
        u_max = o["na_eff"] * 1.03125
        if self.method == "global":
            # the pupil's period λ/du must exceed twice every emitter-to-sample offset (plus
            # PSF half-widths): each emitter's aliases then lie farther from the frame than the
            # emitter itself, so they add less than its own tail (emitters may lie outside)
            reach = reach_of(desc, envelope, self.camera.shape, pitch)
            needed = math.ceil(2.0 * u_max * (2.0 * reach + 2.0 * half) / o["wl_min"])
        else:
            needed = math.ceil(4.0 * u_max * roi * pitch / o["wl_min"])  # period ≥ 2·ROI extent
        if self.pupil_samples == "auto":
            # the supercritical ring (n_sample < |u| ≤ NA) has a kink at |u| = n_sample: a finer
            # grid keeps the collected power within 1e-3 (§4.1 layered tests)
            layered = bool(getattr(desc.nodes.get("environment"), "layered", False))
            floor = 128 if layered and o["na"] > o["n_sample"] else 64
            samples = max(floor, nice_size(needed))
            extent = "2·frame + PSF" if self.method == "global" else "2·R·p"
            pupil_rule = f"max({floor}, period λ_min/du ≥ {extent})"
        else:
            samples, pupil_rule = int(self.pupil_samples), "pinned"
        grid = Grid2D(tuple(self.camera.shape), pitch, (0.0, 0.0))
        decisions = (
            d_spacing,
            Decision(
                "roi", roi, roi_rule, {"|Δz|max": o["dz"], "λ_max": o["wl_max"], "NA": o["na"]}
            ),
            Decision("pupil.samples", samples, pupil_rule, {"u_max": u_max, "λ_min": o["wl_min"]}),
        )
        if self.method == "global":
            rule = "pinned: every emitter on the whole frame, no ROI"
            decisions = (*decisions, Decision("method", "global", rule, {"roi_px": o["roi_px"]}))
        return PointPSFStatic(
            decisions=decisions,
            grid=grid,
            oversample=s,
            roi=roi,
            pupil=PupilGrid((samples, samples), u_max),
            deterministic=desc.deterministic,
            method=self.method,
            half=half,
        )

    def memory(self, desc: Description, static: Static) -> int | None:
        """Estimate the per-image peak bytes: per-emitter pupils, ROI fields and their gradients.

        Parameters
        ----------
        desc : Description
            Static description (slot capacities, frames, spectral bins).
        static : Static
            A :class:`PointPSFStatic`.

        Returns
        -------
        int or None
            Bytes per image (complex64 working set with autograd), or None without a pupil grid.
        """
        if not isinstance(static, PointPSFStatic) or static.pupil is None:
            return None
        slots = sum(desc.slots.get(p, 1) for p in desc.populations) or 1
        pupil = static.pupil.shape[0]
        if static.method == "global" and static.grid is not None:
            height, width = (static.oversample * (n + 2 * MARGIN) for n in static.grid.shape)
            per_emitter = 48 * pupil * pupil + 40 * height * width + 16 * pupil * max(height, width)
            return int(desc.frames * slots * desc.bins * per_emitter)
        samples = static.roi * static.oversample
        per_emitter = 48 * pupil * pupil + 40 * samples * samples + 16 * pupil * samples
        return int(desc.frames * slots * desc.bins * per_emitter)

    def eager_parts(self, *inputs: object) -> dict[str, Node]:
        """Return the parts an eager call derives its envelope from.

        Parameters
        ----------
        *inputs : object
            The emitter set and the medium.

        Returns
        -------
        dict of str to Node
            ``objective``, ``camera``, ``emitters`` and ``environment``.
        """
        parts: dict[str, Node] = {"objective": self.objective, "camera": self.camera}
        if inputs and isinstance(inputs[0], EmitterSet):
            parts["emitters"] = inputs[0]
        if len(inputs) > 1 and isinstance(inputs[1], Node):
            parts["environment"] = inputs[1]
        return parts

    def eager_description(self, *inputs: object) -> Description:
        """Return the description of an eager call: one population named ``"emitters"``.

        Parameters
        ----------
        *inputs : object
            The emitter set and the medium.

        Returns
        -------
        Description
            The description.
        """
        nodes: dict[str, Node] = {}
        if len(inputs) > 1 and isinstance(inputs[1], Node):
            nodes["environment"] = inputs[1]
        return Description(populations=("emitters",), nodes=nodes)

    def _modifiers(
        self, emitters: EmitterSet, u: Tensor, na: Tensor, medium: Medium
    ) -> Tensor | None:
        """Fuse the pupil modifiers per (image, species, bin), gathered per emitter.

        Returns ``[B|1, 1, N|1, L, Np, Np]`` complex multipliers, or None without modifiers.
        The context's index is the medium the pupil is defined in (the immersion of a layered
        medium), per image and bin.
        """
        total = fused_modifiers(self.objective, emitters.wavelengths, u, na, medium)
        if total is None:
            return None
        if emitters.species is None:
            return total[:, :1, None]  # [B|1, 1, 1, L, Np, Np]
        index = emitters.species[:, 0, :]  # [B|1, N]
        batch = max(total.shape[0], index.shape[0])
        total = total.expand(batch, *total.shape[1:])
        index = index.expand(batch, index.shape[1])
        rows = torch.arange(batch, device=index.device)[:, None]
        return total[rows, index][:, None]  # [B, 1, N, L, Np, Np]

    def forward(self, *inputs: object, static: Static) -> Irradiance:
        """Render the emitters' pupil PSFs.

        Parameters
        ----------
        *inputs : object
            The :class:`~gradix.EmitterSet`, then the medium.
        static : Static
            A :class:`PointPSFStatic`.

        Returns
        -------
        Irradiance
            ``[B, A|1, 1, H, W]`` photons per pixel on the camera grid.
        """
        if len(inputs) != 2:
            raise StructureError("PointPSF takes (EmitterSet, medium)")
        emitters, medium = inputs[0], inputs[1]
        if not isinstance(emitters, EmitterSet):
            raise StructureError(f"PointPSF takes an EmitterSet, got {type(emitters).__name__}")
        if not isinstance(medium, Medium):
            msg = "PointPSF needs a medium"
            raise StructureError(msg, fix="pass environment=gx.env.Homogeneous(n)")
        if not isinstance(static, PointPSFStatic) or static.grid is None or static.pupil is None:
            raise StructureError("PointPSF needs a PointPSFStatic")
        pos = emitters.position
        dtype, device = pos.dtype, pos.device
        ospec, cspec = self.objective.schema(), self.camera.schema()

        def per_image(node: Node, spec: dict, name: str) -> Tensor:
            return canonical(getattr(node, name), spec[name], dtype=dtype, device=device)

        six = (-1, 1, 1, 1, 1, 1)
        na = per_image(self.objective, ospec, "NA")
        focus = per_image(self.objective, ospec, "focus")  # [B|1, A|1] (a setting)
        size = per_image(self.camera, cspec, "pixel_size")
        pitch = size / per_image(self.objective, ospec, "magnification")
        wl, w = emission_rows(emitters)  # [B|1, 1, N|1, L]
        photons = (
            emitters.photons if emitters.presence is None else emitters.photons * emitters.presence
        )
        s, roi = static.oversample, static.roi
        samples = static.pupil.shape[0]
        u, du = ops.pupil_axis(samples, static.pupil.u_max, dtype=dtype, device=device)
        uy, ux = torch.meshgrid(u, u, indexing="ij")
        u2 = ux * ux + uy * uy
        wl6 = wl[..., None, None]
        aperture = ops.soft_aperture(torch.sqrt(u2), na.reshape(six), du)  # [B|1, 1, 1, 1, P, P]
        if medium.layered:
            # propagation to the interface, transmission and flux come from the medium (per
            # emitter depth); one refocusing term in the immersion carries the field from the
            # interface (z = 0) to the nominal focal plane at z = focus (§4.1)
            amplitude, n = medium.pupil_amplitude(u2, wl6, pos[..., 2][..., None, None, None])
            base = aperture * amplitude  # [B, A, N, L, P, P], normalised per emitted photon
            travel = focus[:, :, None, None, None, None]  # interface → focal plane
            measure = (du / wl) ** 2  # Σ|P|²·(du/λ)² is the collected fraction
        else:
            index = medium.index(dtype=dtype, device=device)
            n = (index.real if index.is_complex() else index).reshape(-1)
            # a homogeneous medium radiates only |u| < n into the far field: zero beyond it (the
            # soft aperture's edge cell may reach past n when NA is close to n)
            inside = u2 < n.reshape(six) ** 2
            obliquity = ops.obliquity(u2, n.reshape(six))
            base = torch.where(
                inside, aperture / torch.sqrt(obliquity), torch.zeros_like(obliquity)
            )
            # the pupil carries each emitter's field to the focal plane: propagate by focus − z
            travel = (focus[:, :, None] - pos[..., 2])[..., None, None, None]
            measure = du * du / (4.0 * math.pi * n.reshape(-1, 1, 1, 1) ** 2)  # |c|²·(du/λ)²
        mods = self._modifiers(emitters, u.to(torch.float64), na, medium)
        if mods is not None:
            base = base * mods.to(torch.complex128 if dtype == torch.float64 else torch.complex64)
        p3 = pitch.reshape(-1, 1, 1)
        x_e, y_e = pos[..., 0], pos[..., 1]
        if static.method == "global":
            # every emitter is evaluated on the frame (and a margin), from its first pixel
            ix0 = torch.full(x_e.shape, -MARGIN, dtype=torch.int64, device=device)
            iy0 = ix0
        else:
            # ROI placement (integer, no gradient): the ROI centre, ix0 + (R − 1)/2 in pixel
            # coordinates, lies within half a pixel of the emitter, so truncated tails balance
            # on average (no centroid bias)
            ix0 = torch.round(x_e.detach() / p3.detach() - roi / 2).to(torch.int64)
            iy0 = torch.round(y_e.detach() / p3.detach() - roi / 2).to(torch.int64)
        # the ROI's first sample relative to the emitter enters as a pupil phase, so the
        # sampling matrices below are shared by every emitter (large single GEMMs)
        ox = (p3 * (ix0.to(dtype) + 0.5) - x_e)[..., None, None, None]
        oy = (p3 * (iy0.to(dtype) + 0.5) - y_e)[..., None, None, None]
        k_wave = 2.0 * math.pi / wl6  # [B|1, 1, N|1, L, 1, 1]
        n6 = n if n.ndim == 6 else n.reshape(six)
        defocus = ops.defocus_phase(u2, n6, wl6, torch.ones((), dtype=dtype))
        # the lateral phase is separable: a row and a column vector per emitter (small)
        across = k_wave * ox * u  # [B, A, N, L, 1, P]
        down = (k_wave * oy * u[:, None]).expand(*across.shape[:-2], samples, 1)
        phase = torch.addcmul(across + down, defocus, travel)  # [B, A, N, L, P, P]
        if base.is_complex():  # modifiers or a layered medium: multiply (|·| and arg are
            # not differentiable where the pupil vanishes)
            unit = torch.ones((), dtype=phase.dtype, device=device).expand_as(phase)
            field = base * torch.polar(unit, phase)
        else:
            field = torch.polar(base.to(phase.dtype).expand_as(phase), phase)
        # sampling matrices exp(2πi·k·(p/s)·u/λ) for ROI sample k: [B|1, 1, N|1, L, R·s, P]
        step = (pitch / s).reshape(six)

        def sampling(count: int) -> Tensor:
            k = torch.arange(count, device=device, dtype=dtype)
            sample_phase = k_wave * step * k[:, None] * u[None, :]
            return torch.polar(torch.ones_like(sample_phase), sample_phase)  # small

        if static.method == "global":
            height, width = static.grid.shape
            # an emitter's alias sits one period λ/du away: it is kept only while its aliases lie
            # farther from the frame than it does (far < (period + span)/2 on both axes), which
            # every emitter inside the envelope satisfies; beyond, it is dropped (far outside)
            period = wl * samples / (2.0 * static.pupil.u_max)  # [B|1, 1, N|1, L]
            first, last = -MARGIN * p3, (width + MARGIN) * p3
            top, bottom = -MARGIN * p3, (height + MARGIN) * p3
            far_x = torch.maximum((last - x_e).abs(), (x_e - first).abs())[..., None]
            far_y = torch.maximum((bottom - y_e).abs(), (y_e - top).abs())[..., None]
            keep = (far_x < 0.5 * (period + (last - first)[..., None])) & (
                far_y < 0.5 * (period + (bottom - top)[..., None])
            )
            photons = torch.where(keep.all(-1), photons, torch.zeros_like(photons))
            rows_y = sampling((height + 2 * MARGIN) * s)
            rows_x = sampling((width + 2 * MARGIN) * s)
            e = _sample(rows_y, field, rows_x)  # [B, A, N, L, Hs, Ws]
            intensity = torch.view_as_real(e).square().sum(-1)
            # photons per sample from the discrete Parseval relation: Σ_u|P|² = (du·Δx/λ)²·Σ_x|e|²
            absolute = photons[..., None] * w * measure * (du * step[..., 0, 0] / wl) ** 2
            fine = (intensity * absolute[..., None, None]).sum((2, 3))  # [B, A, Hs, Ws]
            pixels = (s * s) * ops.pixel_mtf(fine, s)[..., ::s, ::s]
            frame = pixels[..., MARGIN : MARGIN + height, MARGIN : MARGIN + width]
            return Irradiance(data=frame[:, :, None], grid=static.grid, acq=emitters.acq)
        # collection efficiency of each (emitter, bin) from the discrete pupil (Parseval): the
        # defocus and placement phases have unit modulus, so they do not change it
        eta = (base.real**2 + base.imag**2 if base.is_complex() else base * base).sum((-2, -1))
        eta = eta * measure
        rows = sampling(roi * s)
        e = _sample(rows, field)  # [B, A, N, L, R·s, R·s]
        intensity = torch.view_as_real(e).square().sum(-1)
        captured = intensity.sum((-2, -1))
        ok = captured > 1e-30
        # every (emitter, bin) delivers photons·w·η: the ROI is scaled to the pupil's power, so
        # the radiometry holds even when a PSF tail leaves the ROI (identical at every tier)
        target = photons[..., None] * w * eta
        scale = torch.where(ok, target / torch.where(ok, captured, torch.ones_like(captured)), 0.0)
        fine = (intensity * scale[..., None, None]).sum(3)  # [B, A, N, R·s, R·s]
        if fine.shape[2] == 0:  # no emitters: an empty frame (FFTs of nothing are errors)
            empty = fine.new_zeros(fine.shape[0], fine.shape[1], *static.grid.shape)
            return Irradiance(data=empty[:, :, None], grid=static.grid, acq=emitters.acq)
        pixels = (s * s) * ops.pixel_mtf(fine, s)[..., ::s, ::s]  # [B, A, N, R, R]
        data = _accumulate(pixels, ix0, iy0, static.grid.shape, static.deterministic)
        return Irradiance(data=data[:, :, None], grid=static.grid, acq=emitters.acq)


def _sample(rows: Tensor, field: Tensor, cols: Tensor | None = None) -> Tensor:
    """Evaluate ``rows @ field @ colsᵀ``: pupils on their samples (``[..., Y, X]``).

    ``cols`` defaults to ``rows`` (square ROIs). When the sampling matrices are shared by every
    emitter (one pitch, one spectrum) the products are single large GEMMs; otherwise they
    broadcast.
    """
    cols = rows if cols is None else cols
    with ieee_matmul():
        if math.prod(rows.shape[:-2]) == 1 and math.prod(cols.shape[:-2]) == 1:
            left = rows.reshape(rows.shape[-2:])
            right = cols.reshape(cols.shape[-2:])
            half = torch.matmul(field, right.transpose(0, 1))  # [..., P, X]
            return torch.matmul(left, half)  # [..., Y, X]
        half = torch.matmul(field, cols.transpose(-1, -2))
        return torch.matmul(rows, half)


def _accumulate(
    rois: Tensor, ix0: Tensor, iy0: Tensor, shape: tuple[int, int], deterministic: bool
) -> Tensor:
    """Add ``[B, A, N, R, R]`` ROIs with top-left corners ``(iy0, ix0)`` into ``[B, A, H, W]``."""
    batch, frames, count, roi, _ = rois.shape
    height, width = shape
    ix0 = ix0.expand(batch, frames, count)
    iy0 = iy0.expand(batch, frames, count)
    r = torch.arange(roi, device=rois.device)
    rows = iy0[..., None] + r
    cols = ix0[..., None] + r
    inside_rows = (rows >= 0) & (rows < height)
    inside_cols = (cols >= 0) & (cols < width)
    valid = inside_rows[..., :, None] & inside_cols[..., None, :]
    flat = rows.clamp(0, height - 1)[..., :, None] * width + cols.clamp(0, width - 1)[..., None, :]
    image = torch.arange(batch * frames, device=rois.device).reshape(batch, frames, 1, 1, 1)
    idx = flat + image * (height * width)
    vals = torch.where(valid, rois, torch.zeros_like(rois))
    out = torch.zeros(batch * frames * height * width, dtype=rois.dtype, device=rois.device)
    if deterministic:
        # slot by slot: indices within one slot are unique, so the summation order is fixed
        flat_idx, flat_vals = idx.movedim(2, 0), vals.movedim(2, 0)
        for n in range(count):
            out = out.index_add(0, flat_idx[n].reshape(-1), flat_vals[n].reshape(-1))
    else:
        out = out.index_add(0, idx.reshape(-1), vals.reshape(-1))
    return out.reshape(batch, frames, height, width)
