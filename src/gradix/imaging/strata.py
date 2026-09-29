"""Dense emission by depth strata (``emit.strata_otf``; the dense fluorescence path, §5.3).

An emitter density on a stack of planes is imaged plane by plane: each plane is convolved with
the PSF of a point on that plane, and the planes are summed. The PSF of each plane (and species
and wavelength bin) is the pupil PSF of :class:`~gradix.imaging.PointPSF`, evaluated by the same
matrix Fourier transform on the same pupil grid, so a density voxel renders exactly like a point
emitter at its centre. The convolution is linear, not circular (the FFT grid holds the density
and the camera region), and nothing is truncated or renormalised. The pixel MTF is applied on
the camera's fine grid over the frame and a 16-pixel margin, as on the global sparse path.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Mapping
from typing import ClassVar

import torch
from torch import Tensor

from gradix import register
from gradix._core.carriers import EmitterDensity, Irradiance
from gradix._core.contract import (
    Capabilities,
    DensityRequest,
    Description,
    Element,
    Slot,
    Static,
    Violation,
)
from gradix._core.envelope import Envelope
from gradix._core.errors import StructureError
from gradix._core.grid import Grid2D, PupilGrid, VolumeGrid, nice_size
from gradix._core.precision import ieee_matmul
from gradix._core.rules import Decision, detection_spacing
from gradix.detect.camera import Camera
from gradix.imaging._shared import MARGIN, fused_modifiers, material_range, ratio_per_image
from gradix.imaging._shared import value as value_of
from gradix.objects.environment import Medium
from gradix.objects.objectset import Solid
from gradix.objects.turbid import TurbidSlab
from gradix.objects.volumes import Voxels
from gradix.ops import pupil as ops
from gradix.optics.objective import Objective
from gradix.schema.base import Node
from gradix.schema.fields import child, knob
from gradix.schema.layout import canonical

__all__ = ["Strata", "StrataStatic"]


@dataclasses.dataclass(frozen=True)
class StrataStatic(Static):
    """Static configuration of :class:`Strata`.

    Parameters
    ----------
    decisions : tuple of Decision, default ()
        Sizing decisions with provenance.
    grid : Grid2D, optional
        The camera grid in object space.
    oversample : int, default 1
        Fine samples per camera pixel, s; densities must be sampled at the pitch ``p/s``.
    pupil : PupilGrid, optional
        The pupil grid in NA units.
    volume : VolumeGrid, optional
        The grid labelled solids are rasterised onto (ADR-41); None without solids.
    reach : tuple of (str, float), default ()
        Each solid population's largest bounding radius, µm.
    blur : float, default 0.0
        The raster prefilter σ_r, µm.
    deterministic : bool, default False
        Rasterise with a fixed summation order.
    """

    grid: Grid2D | None = None
    oversample: int = 1
    pupil: PupilGrid | None = None
    volume: VolumeGrid | None = None
    reach: tuple[tuple[str, float], ...] = ()
    blur: float = 0.0
    deterministic: bool = False


def _offset_span(grid: VolumeGrid, frame: tuple[tuple[float, float], ...]) -> float:
    """Return the largest voxel-to-sample offset between a density and the frame, µm."""
    spans = []
    for axis, (lo, hi) in enumerate(frame):
        first = grid.xy.origin[axis] + 0.5 * grid.xy.spacing
        last = first + (grid.xy.shape[1 - axis] - 1) * grid.xy.spacing
        spans.append(max(hi - first, last - lo))
    return max(spans)


def _pitch(objective: Objective, camera: Camera, like: Tensor) -> Tensor:
    """Return each image's object-space pitch, µm, ``[B|1]``."""
    size = canonical(camera.pixel_size, camera.schema()["pixel_size"], dtype=like.dtype)
    mag = canonical(objective.magnification, objective.schema()["magnification"], dtype=like.dtype)
    return (size.to(like.device) / mag.to(like.device)).reshape(-1)


def _alignment(offset: float, what: str) -> int:
    """Return an offset in fine samples, which must be an integer."""
    index = round(offset)
    if abs(offset - index) > 1e-6:
        msg = f"the density's {what} voxel centres fall between the camera's fine samples"
        fix = "place voxel centres on the fine grid: origin = (i + 0.5)·p + j·p/s"
        raise StructureError(msg, fix=fix)
    return int(index)


def _axial_limit(wavelength: float, na: float, n: float) -> float:
    """Return the plane spacing that resolves the axial band: λ/(2(n − √(n² − NA²))), µm."""
    na_s = min(na, n)
    return wavelength / (2.0 * (n - math.sqrt(max(n * n - na_s * na_s, 0.0))))


def _solid_reach(envelope: Envelope, name: str, node: Solid) -> float:
    """Return a solid population's largest bounding radius from its envelope, µm."""
    spec = type(node).schema()
    args: dict[str, Tensor] = {}
    for field in node.shape_fields:
        paths = [f"{name}.{field}.{c}" for c in spec[field].components or ()] or [f"{name}.{field}"]
        spans = [envelope.range(p) for p in paths]
        if any(span is None for span in spans):  # no entry: the values themselves
            value = canonical(getattr(node, field), spec[field], dtype=torch.float64).detach()
            args[field] = value.reshape(-1, *value.shape[3:]).amax(dim=0)
        else:
            highs = [span[1] for span in spans if span is not None]
            args[field] = torch.tensor(highs if spec[field].components else highs[0])
    return float(node.reach(args).max())


def _window(data: Tensor, size: tuple[int, int], start: tuple[int, int]) -> Tensor:
    """Place ``[..., Y, X]`` samples on one period: sample j at ``start + j``; outside dropped."""
    parts: list[tuple[int, int, int]] = []  # (first kept sample, count, its index) per axis
    for n, m, a in zip(data.shape[-2:], size, start, strict=True):
        lo, hi = min(max(0, -a), n), max(min(n, m - a), 0)
        parts.append((lo, hi - lo, a + lo) if hi > lo else (0, 0, m))
    (ly, cy, iy), (lx, cx, ix) = parts
    kept = data.narrow(-2, ly, cy).narrow(-1, lx, cx)  # empty slices keep the graph connected
    return torch.nn.functional.pad(kept, (ix, size[1] - ix - cx, iy, size[0] - iy - cy))


@register.element("emit.strata_otf")
@dataclasses.dataclass(frozen=True, eq=False)
class Strata(Element[Irradiance]):
    """Dense fluorescence: an emitter density imaged plane by plane with pupil PSFs.

    The density's lateral samples must lie on the camera's fine grid (spacing ``p/s``, centres
    at ``(i + ½)·p + j·p/s``), with ``s`` this element's oversampling; its planes may lie at any
    heights. The pitch is structure: images rendered with another pixel size or magnification
    than the grid's come out NaN. Every plane uses the exact pupil PSF of a point at that
    height (homogeneous or layered media, pupil modifiers), so strata and sparse points agree
    on-plane.

    Parameters
    ----------
    objective : Objective
        The objective (NA, magnification, focus, pupil modifiers).
    camera : Camera
        The camera.
    oversample : int or "auto", default "auto"
        Fine samples per camera pixel without densities; ``"auto"`` meets λ_min/(4·NA) (§4.3).
        A density's lateral spacing sets it otherwise (``p/s``).
    pupil_samples : int or "auto", default "auto"
        Pupil grid cells per axis; ``"auto"`` holds every density-to-camera offset in one period.
    turbid : TurbidSlab, optional
        A phenomenological turbid slab (``gx.env.TurbidSlab``): each plane's PSF is attenuated
        and blurred with depth, and the scattered light spread into a diffuse halo (which wraps
        over the FFT grid).
    boundary : {"linear", "periodic"}, default "linear"
        ``"linear"`` convolves exactly: every voxel reaches every pixel, on an FFT grid as large
        as the density plus the frame and its margin (about 4.8× the frame's area for a density
        that fills it). ``"periodic"`` convolves circularly over one period of the frame and its
        margin: voxels outside that window are dropped, each PSF is cut at half the period, and
        light from near one edge wraps to the opposite one. The margin absorbs most of the wrap:
        at equal ``pupil_samples``, densities near focus and inside the frame agree with
        ``"linear"`` to about 1e-6, while defocused planes whose PSFs outgrow the margin wrap
        visibly. With ``pupil_samples="auto"`` the periodic kernels get a smaller pupil (they
        span one period, not every offset), which moves the PSF by up to about 2e-3 (the
        pupil's O(du) convergence).
    raster_sigma : float, default 0.3
        Width of the raster prefilter of labelled solids, in lateral voxel spacings (σ_r = c·h).
    plane_spacing : float or "auto", default "auto"
        Plane spacing of rasterised solids, µm; ``"auto"`` resolves the axial band,
        λ_min/(2(n − √(n² − NA²))).

    Examples
    --------
    >>> import gradix as gx
    >>> camera = gx.Camera(pixel_size=6.5, shape=(32, 32))
    >>> Strata(gx.Objective(NA=0.8, magnification=50), camera).oversample
    'auto'
    """

    slot: ClassVar[Slot] = Slot.EMIT
    fidelity_knobs: ClassVar[Mapping[str, str | tuple[str, Mapping[str, str]]]] = {
        "oversample": "oversample",
        "raster_sigma": "raster_sigma",
        "dense_boundary": ("boundary", {"linear": "linear", "periodic": "periodic"}),
    }
    schema_version: ClassVar[int] = 4
    caps: ClassVar[Capabilities] = Capabilities(
        accepts=frozenset({EmitterDensity}),
        produces=Irradiance,
        grad_quality={
            "camera": "zero",  # the fine grid is static: densities are sampled on it
            "objective.magnification": "zero",
            "objective.NA": "exact-a.e.",
        },
        reads={"values": "exact", "photons": "exact", "emission": "exact", "environment": "exact"},
    )

    objective: Objective = child(doc="the objective")
    camera: Camera = child(doc="the camera")
    _: dataclasses.KW_ONLY
    oversample: int | str = knob(default="auto", doc="fine samples per camera pixel")
    pupil_samples: int | str = knob(default="auto", doc="pupil cells per axis")
    turbid: TurbidSlab | None = child(default=None, doc="turbid slab")
    boundary: str = knob(
        default="linear",
        choices=("linear", "periodic"),
        doc="exact linear convolution or a circular one over the frame and margin",
    )
    raster_sigma: float = knob(default=0.3, doc="raster prefilter width, in lateral voxel spacings")
    plane_spacing: float | str = knob(default="auto", doc="plane spacing of rasterised solids, µm")

    def _bands(self, desc: Description, envelope: Envelope) -> list[tuple[float, float]]:
        out = []
        for p in (*desc.populations, "density"):
            band = envelope.range(f"{p}.emission.wavelengths") or envelope.range(f"{p}.wavelengths")
            if band is not None:
                out.append(band)
        return out

    def input_quality(self, name: str, output: str = "expected") -> str:
        """Return the gradient quality with respect to a population field.

        Densities' values and every geometry and label field of a solid (positions, rotations,
        sizes, label densities and photons) reach the image through the differentiable raster
        lowering; materials, identities and parent links emit nothing.

        Parameters
        ----------
        name : str
            Field path within the population, or ``"environment"``.
        output : {"expected", "image"}, default "expected"
            The output kind the element's result feeds.

        Returns
        -------
        str
            ``"zero"`` for materials, identities and parent links; ``"exact"`` otherwise.
        """
        if name.split(".")[0] in ("material", "id", "parent"):
            return "zero"
        return "exact"

    def _solids(self, desc: Description) -> dict[str, Solid]:
        return {n: v for n in desc.populations if isinstance(v := desc.nodes.get(n), Solid)}

    def _grids(self, desc: Description) -> list[Voxels | EmitterDensity]:
        found: list[Voxels | EmitterDensity] = []
        for name in (*desc.populations, "density"):
            node = desc.nodes.get(name)
            if isinstance(node, (Voxels, EmitterDensity)):
                found.append(node)
        return found

    def configure(self, desc: Description, envelope: Envelope) -> StrataStatic:
        """Choose the fine sampling and the pupil grid.

        Parameters
        ----------
        desc : Description
            Static description; its nodes hold the density populations (their grids).
        envelope : Envelope
            The envelope (NA, pitch, magnification, emission wavelengths, index).

        Returns
        -------
        StrataStatic
            The configuration.
        """
        obj, env = desc.part("objective"), desc.part("environment")
        # densities are sampled on the fine grid: the object-space pitch must be exact and
        # shared by every image (pixel size and magnification may differ in step)
        like = torch.zeros((), dtype=torch.float64)
        pitches = _pitch(self.objective, self.camera, like).detach()
        pitch = float(pitches.max())
        if float(pitches.min()) < pitch * (1.0 - 1e-9):
            msg = f"Strata needs one object-space pitch for every image, got {pitches.tolist()}"
            fix = "render images with different pitches in separate batches"
            raise StructureError(msg, fix=fix)
        na = value_of(envelope, f"{obj}.NA", self.objective.NA, "hi")
        if na is None:
            raise StructureError("the envelope lacks the NA")
        bands = self._bands(desc, envelope)
        grids = self._grids(desc)
        wl_min = min((b[0] for b in bands), default=0.5)
        band = (wl_min, max((b[1] for b in bands), default=wl_min))
        medium = desc.nodes.get(env) or desc.nodes.get("environment")
        n = (
            envelope.range(f"{env}.n")
            or envelope.range(f"{env}.immersion.n")
            or material_range(medium, "immersion", band)
            or (1.0, 1.0)
        )
        na_eff = min(na, n[1])
        # the densities' lateral sampling sets the fine grid: s = camera pitch / voxel spacing
        spacings = {(g.grid() if isinstance(g, Voxels) else g.grid).xy.spacing for g in grids}
        if len(spacings) > 1:
            raise StructureError("density populations must share one lateral spacing")
        if spacings:
            dx = spacings.pop()
            s = round(pitch / dx)
            if s < 1 or abs(s * dx - pitch) > 1e-6 * pitch:
                msg = f"the densities' spacing {dx:g} µm does not divide the pitch {pitch:g} µm"
                fix = f"sample the density at the camera pitch / an integer (e.g. {pitch:g} µm)"
                raise StructureError(msg, fix=fix)
            # the densities' sampling wins over the oversample knob (a preset may set one)
            rule = "pitch / voxel spacing"
            if isinstance(self.oversample, int) and self.oversample != s:
                rule += f" (oversample={self.oversample} ignored)"
            d_spacing = Decision("detection.oversample", s, rule, {"dx": dx})
        else:
            _spacing, s, d_spacing = detection_spacing(
                pitch, wl_min, na_eff, oversample=self.oversample
            )
        u_max = na_eff * 1.03125
        height, width = self.camera.shape
        solids = self._solids(desc)
        volume, reach, blur = None, (), 0.0
        if solids:
            blur = self.raster_sigma * pitch / s
            reach = tuple((k, _solid_reach(envelope, k, v)) for k, v in sorted(solids.items()))
            if grids:  # volumes fix the grid; solids are rasterised onto it
                volume = grids[0].grid() if isinstance(grids[0], Voxels) else grids[0].grid
            else:
                volume = self._volume(envelope, env, medium, reach, blur, pitch, s, band, na_eff)
        # the pupil's period λ/du must exceed every voxel-to-sample offset plus a PSF half-width
        frame = ((-MARGIN * pitch, (width + MARGIN) * pitch),
                 (-MARGIN * pitch, (height + MARGIN) * pitch))  # fmt: skip
        extent = max(frame[0][1] - frame[0][0], frame[1][1] - frame[1][0])
        focus = envelope.range(f"{obj}.focus") or (0.0, 0.0)
        dz = 0.0
        planned = [g.grid() if isinstance(g, Voxels) else g.grid for g in grids]
        for grid in [*planned, *([volume] if volume is not None else [])]:
            extent = max(extent, _offset_span(grid, frame))
            z_lo, z_hi = grid.z0, grid.z0 + (grid.nz - 1) * grid.dz
            dz = max(dz, abs(z_hi - focus[0]), abs(focus[1] - z_lo))
        if self.boundary == "periodic":  # the kernels span one period of the frame and margin
            extent = 0.5 * max(nice_size((m + 2 * MARGIN) * s) for m in (height, width)) * pitch / s
        sin_max = min(na_eff / max(n[0], 1e-6), 0.999)
        half = dz * sin_max / math.sqrt(1.0 - sin_max**2) + 2.0 * band[1] / max(na_eff, 1e-6)
        if self.pupil_samples == "auto":
            # twice every offset: aliases lie farther from the frame than their voxels
            needed = math.ceil(2.0 * u_max * (2.0 * extent + 2.0 * half) / wl_min)
            samples, rule = max(64, nice_size(needed)), "max(64, period λ_min/du ≥ offsets + PSF)"
        else:
            samples, rule = int(self.pupil_samples), "pinned"
        decisions = (
            d_spacing,
            Decision("pupil.samples", samples, rule, {"u_max": u_max, "extent": extent}),
        )
        if volume is not None:
            decisions = (
                *decisions,
                Decision(
                    "raster.planes",
                    volume.nz,
                    "solids' z range ± reach, spacing "
                    + ("λ_min/(2(n − √(n² − NA²)))" if self.plane_spacing == "auto" else "pinned"),
                    {"dz": volume.dz, "blur": blur},
                ),
            )
        return StrataStatic(
            decisions=decisions,
            grid=Grid2D((height, width), pitch, (0.0, 0.0)),
            oversample=s,
            pupil=PupilGrid((samples, samples), u_max),
            volume=volume,
            reach=reach,
            blur=blur,
            deterministic=desc.deterministic,
        )

    def _volume(
        self,
        envelope: Envelope,
        env: str,
        medium: object,
        reach: tuple[tuple[str, float], ...],
        blur: float,
        pitch: float,
        s: int,
        band: tuple[float, float],
        na: float,
    ) -> VolumeGrid:
        """Return the grid solids are rasterised onto: the fine frame, planes over their z range."""
        height, width = self.camera.shape
        dx = pitch / s
        # laterally the fine grid over the frame and its margin, voxel centres on fine samples
        corner = (0.5 - MARGIN) * pitch - 0.5 * dx
        xy = Grid2D(((height + 2 * MARGIN) * s, (width + 2 * MARGIN) * s), dx, (corner, corner))
        z_lo, z_hi = math.inf, -math.inf
        for name, r in reach:
            span = envelope.range(f"{name}.position.z") or (0.0, 0.0)
            z_lo, z_hi = min(z_lo, span[0] - r - 3.0 * blur), max(z_hi, span[1] + r + 3.0 * blur)
        if getattr(medium, "layered", False):
            z_hi = min(z_hi, 0.0)  # the sample lies below the coverslip
            z_lo = min(z_lo, z_hi)
        n = envelope.range(f"{env}.n") or envelope.range(f"{env}.sample.n") or (1.33, 1.33)
        if self.plane_spacing == "auto":
            dz = _axial_limit(band[0], na, n[0])
        else:
            dz = float(self.plane_spacing)
        nz = max(1, math.ceil((z_hi - z_lo) / dz - 1e-9) + 1)
        z0 = 0.5 * (z_lo + z_hi) - 0.5 * (nz - 1) * dz
        return VolumeGrid(xy=xy, z0=z0, dz=dz, nz=nz)

    def density_request(self, static: Static) -> DensityRequest | None:
        """Return the grid labelled solids are rasterised onto (ADR-41).

        Parameters
        ----------
        static : Static
            A :class:`StrataStatic`.

        Returns
        -------
        DensityRequest or None
            The volume grid, raster blur and reaches; None when no solids were planned for.
        """
        if not isinstance(static, StrataStatic) or static.volume is None:
            return None
        return DensityRequest(
            grid=static.volume,
            blur=static.blur,
            reach=dict(static.reach),
            deterministic=static.deterministic,
        )

    def validity(self, desc: Description, envelope: Envelope) -> list[Violation]:
        """Report findings: NA beyond a homogeneous medium's index; planes too far apart (§5.8).

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
        obj, env = desc.part("objective"), desc.part("environment")
        na = value_of(envelope, f"{obj}.NA", self.objective.NA, "hi")
        n = envelope.range(f"{env}.n") or envelope.range(f"{env}.sample.n") or (1.33, 1.33)
        wl = min((b[0] for b in self._bands(desc, envelope)), default=0.4)
        found: list[Violation] = []
        if self.boundary == "periodic":
            found.append(
                Violation(
                    "info",
                    "periodic boundary: PSFs are cut at half the frame-and-margin period, and "
                    "light from near one edge wraps to the opposite edge",
                    element=desc.path or "strata",
                    fix='boundary="linear" convolves exactly',
                )
            )
        medium = desc.nodes.get(env) or desc.nodes.get("environment")
        layered = bool(getattr(medium, "layered", False))
        ratio = ratio_per_image(self.objective, medium)  # per image, from the actual values
        if not layered and ratio is not None and ratio >= 1.0:
            found.append(
                Violation(
                    "error",
                    "NA reaches the medium's index; supercritical collection needs a layered "
                    "medium",
                    entry=f"{obj}.NA",
                    value=ratio,
                    limit=1.0,
                    element=desc.path or "strata",
                    fix="declare environment=gx.env.LayeredMedium(...)",
                )
            )
        if self._solids(desc) and self.plane_spacing != "auto" and na is not None:
            limit = _axial_limit(wl, na, n[0])
            if float(self.plane_spacing) > limit:
                found.append(
                    Violation(
                        "warn",
                        "the solids' planes are further apart than the axial resolution",
                        value=float(self.plane_spacing),
                        limit=limit,
                        element=desc.path or "strata",
                        fix='plane_spacing="auto"',
                    )
                )
        for g in self._grids(desc):
            grid = g.grid() if isinstance(g, Voxels) else g.grid
            top = grid.z0 + (grid.nz - 1) * grid.dz
            if layered and top > 0:
                found.append(
                    Violation(
                        "error",
                        "density planes above the coverslip (z > 0) in a layered medium: the "
                        "sample lies at z < 0",
                        value=top,
                        limit=0.0,
                        element=desc.path or "strata",
                        fix="place the density at z <= 0",
                    )
                )
            if na is None or grid.nz < 2:
                continue
            na_s = min(na, n[0])
            limit = wl / (2.0 * (n[0] - math.sqrt(max(n[0] ** 2 - na_s**2, 0.0))))
            if grid.dz > limit:
                found.append(
                    Violation(
                        "warn",
                        "the density's planes are further apart than the axial resolution",
                        value=grid.dz,
                        limit=limit,
                        element=desc.path or "strata",
                        fix="sample the density more finely in z",
                    )
                )
        return found

    def eager_parts(self, *inputs: object) -> dict[str, Node]:
        """Return the parts an eager call derives its envelope from.

        Parameters
        ----------
        *inputs : object
            The emitter density and the medium.

        Returns
        -------
        dict of str to Node
            ``objective``, ``camera``, ``environment`` and ``density``.
        """
        parts: dict[str, Node] = {"objective": self.objective, "camera": self.camera}
        if inputs and isinstance(inputs[0], EmitterDensity):
            parts["density"] = inputs[0]
        if len(inputs) > 1 and isinstance(inputs[1], Node):
            parts["environment"] = inputs[1]
        return parts

    def eager_description(self, *inputs: object) -> Description:
        """Return the description of an eager call: the density as population ``"density"``.

        Parameters
        ----------
        *inputs : object
            The emitter density and the medium.

        Returns
        -------
        Description
            The description.
        """
        nodes: dict[str, Node] = {}
        if inputs and isinstance(inputs[0], EmitterDensity):
            nodes["density"] = inputs[0]
        if len(inputs) > 1 and isinstance(inputs[1], Node):
            nodes["environment"] = inputs[1]
        return Description(populations=("density",), nodes=nodes)

    def memory(self, desc: Description, static: Static) -> int | None:
        """Estimate the per-image peak bytes: per-plane pupils, kernels and spectra.

        Parameters
        ----------
        desc : Description
            Static description (frames, spectral bins, the density grids).
        static : Static
            A :class:`StrataStatic`.

        Returns
        -------
        int or None
            Bytes per image, or None before configuration.
        """
        if not isinstance(static, StrataStatic) or static.pupil is None or static.grid is None:
            return None
        planes, area = 0, 0
        s = static.oversample
        cam = [(n + 2 * MARGIN) * s for n in static.grid.shape]
        grids = [g.grid() if isinstance(g, Voxels) else g.grid for g in self._grids(desc)]
        if static.volume is not None and not grids:
            grids = [static.volume]
        for grid in grids:
            planes += grid.nz
            if self.boundary == "periodic":
                fy, fx = nice_size(cam[0]), nice_size(cam[1])
            else:
                fy, fx = (grid.xy.shape[i] + cam[i] for i in range(2))
            area = max(area, fy * fx)
        pupil = static.pupil.shape[0]
        per_plane = 48 * pupil * pupil + 56 * area
        return int(desc.frames * max(planes, 1) * desc.bins * per_plane)

    def forward(self, *inputs: object, static: Static) -> Irradiance:
        """Image the density onto the camera.

        Parameters
        ----------
        *inputs : object
            The :class:`~gradix.EmitterDensity`, then the medium.
        static : Static
            A :class:`StrataStatic`.

        Returns
        -------
        Irradiance
            ``[B, A|1, 1, H, W]`` photons per pixel on the camera grid.
        """
        if len(inputs) != 2:
            raise StructureError("Strata takes (EmitterDensity, medium)")
        density, medium = inputs
        if not isinstance(density, EmitterDensity):
            raise StructureError(f"Strata takes an EmitterDensity, got {type(density).__name__}")
        if not isinstance(medium, Medium):
            raise StructureError("Strata needs a medium", fix="pass gx.env.Homogeneous(n)")
        if not isinstance(static, StrataStatic) or static.grid is None or static.pupil is None:
            raise StructureError("Strata needs a StrataStatic")
        data = density.data  # [B|1, A|1, S, Z, Y, X]
        s = static.oversample
        pitch = static.grid.spacing
        fine = pitch / s
        grid = density.grid
        dx = grid.xy.spacing
        if abs(dx - fine) > 1e-6 * fine:
            msg = f"the density's lateral spacing {dx:g} µm is not the fine pitch {fine:g} µm"
            fix = f"sample the density at the camera pitch / oversample ({fine:g} µm)"
            raise StructureError(msg, fix=fix)
        yd, xd = grid.xy.shape
        x0 = grid.xy.origin[0] + 0.5 * dx  # first voxel centre
        y0 = grid.xy.origin[1] + 0.5 * dx
        jx0 = _alignment((x0 - 0.5 * pitch) / dx, "x")
        jy0 = _alignment((y0 - 0.5 * pitch) / dx, "y")
        height, width = static.grid.shape
        hc, wc = (height + 2 * MARGIN) * s, (width + 2 * MARGIN) * s
        periodic = self.boundary == "periodic"
        if periodic:
            # one period of the frame and margin: fine index m sits at (m − MARGIN·s)·dx from
            # the first pixel centre, voxel j at MARGIN·s + j0 + j (voxels outside are dropped)
            fy, fx = nice_size(hc), nice_size(wc)
            kernel = self._kernels(density, medium, static, fy, fx, -(fy // 2), -(fx // 2))
            kernel = torch.roll(kernel, shifts=(-(fy // 2), -(fx // 2)), dims=(-2, -1))
            planes = _window(data, (fy, fx), (MARGIN * s + jy0, MARGIN * s + jx0))
        else:
            fy, fx = nice_size(yd + hc - 1), nice_size(xd + wc - 1)
            # kernel sample t sits at offset (d_min + t)·dx from a voxel centre
            dmin_x = -MARGIN * s - jx0 - (xd - 1)
            dmin_y = -MARGIN * s - jy0 - (yd - 1)
            kernel = self._kernels(density, medium, static, fy, fx, dmin_y, dmin_x)
            planes = torch.nn.functional.pad(data, (0, fx - xd, 0, fy - yd))  # not circular
        otf = torch.fft.rfft2(kernel)  # [B, A, S, Z, Fy, Fx/2 + 1]
        if self.turbid is not None:
            z = grid.z(dtype=kernel.dtype, device=kernel.device)
            freq_y = torch.fft.fftfreq(fy, d=dx, dtype=kernel.dtype, device=kernel.device)
            freq_x = torch.fft.rfftfreq(fx, d=dx, dtype=kernel.dtype, device=kernel.device)
            otf = otf * self.turbid.transfer(z, freq_y, freq_x)
        spectrum = (torch.fft.rfft2(planes) * otf).sum((2, 3))  # [B, A, Fy, Fx/2 + 1]
        if periodic:  # the image is periodic: the pixel transfer function folds into the OTF
            freq_y = torch.fft.fftfreq(fy, dtype=kernel.dtype, device=kernel.device) * s
            freq_x = torch.fft.rfftfreq(fx, dtype=kernel.dtype, device=kernel.device) * s
            spectrum = spectrum * (torch.sinc(freq_y)[:, None] * torch.sinc(freq_x)[None, :])
            region = torch.fft.irfft2(spectrum, s=(fy, fx))[..., :hc, :wc]
        else:
            image = torch.fft.irfft2(spectrum, s=(fy, fx))  # [B, A, Fy, Fx]
            region = image[..., yd - 1 : yd - 1 + hc, xd - 1 : xd - 1 + wc]
            region = ops.pixel_mtf(region, s)
        pixels = (s * s) * region[..., ::s, ::s]
        frame = pixels[..., MARGIN : MARGIN + height, MARGIN : MARGIN + width]
        # the grid is static: an image whose pitch differs from it (a call that binds another
        # pixel size or magnification) is marked NaN rather than rendered at the wrong scale
        # (checked on the device, no host synchronisation)
        call = _pitch(self.objective, self.camera, frame)
        wrong = (call - static.grid.spacing).abs() > 1e-6 * static.grid.spacing
        frame = torch.where(wrong.reshape(-1, 1, 1, 1), torch.full_like(frame, math.nan), frame)
        return Irradiance(data=frame[:, :, None], grid=static.grid, acq=density.acq)

    def _kernels(
        self,
        density: EmitterDensity,
        medium: Medium,
        static: StrataStatic,
        fy: int,
        fx: int,
        dmin_y: int,
        dmin_x: int,
    ) -> Tensor:
        """Return each plane's PSF, photons per fine sample per photon, ``[B, A, S, Z, Fy, Fx]``."""
        if static.pupil is None or static.grid is None:  # pragma: no cover - checked by forward
            raise StructureError("Strata needs a StrataStatic")
        data = density.data
        dtype, device = data.dtype, data.device
        ospec = self.objective.schema()
        seven = (-1, 1, 1, 1, 1, 1, 1)

        def per_image(name: str) -> Tensor:
            value = getattr(self.objective, name)
            return canonical(value, ospec[name], dtype=dtype, device=device)

        na = per_image("NA")
        focus = per_image("focus")  # [B|1, A|1]
        table = density.wavelengths.to(device=device, dtype=dtype)  # [B|1, S, L]
        weights = density.weights.to(device=device, dtype=dtype)
        b_t, species, bins = table.shape
        wl7 = table.reshape(b_t, 1, species, 1, bins, 1, 1)
        z = density.grid.z(dtype=dtype, device=device)  # [Z]
        planes = z.shape[0]
        samples, u_max = static.pupil.shape[0], static.pupil.u_max
        u, du = ops.pupil_axis(samples, u_max, dtype=dtype, device=device)
        uy, ux = torch.meshgrid(u, u, indexing="ij")
        u2 = ux * ux + uy * uy
        aperture = ops.soft_aperture(torch.sqrt(u2), na.reshape(seven), du)
        # the index of the medium the pupil is defined in, per image, species and bin
        index = medium.immersion_index(dtype=dtype, device=device, wavelength=table)
        index = index.real if index.is_complex() else index
        if index.ndim == 3:  # dispersive: [B|1, S|1, L|1]
            n7 = index.reshape(index.shape[0], 1, index.shape[1], 1, index.shape[2], 1, 1)
        else:
            n7 = index.reshape(seven)
        if medium.layered:
            # propagation to the interface, transmission and flux from the medium (per plane),
            # then one refocusing term in the immersion to the focal plane (§4.1)
            wl6 = table[:, None, :, None, :].expand(b_t, 1, species, planes, bins)
            wl6 = wl6.reshape(b_t, 1, species * planes, bins, 1, 1)
            z6 = z.repeat(species).reshape(1, 1, species * planes, 1, 1, 1)
            amplitude, _n = medium.pupil_amplitude(u2, wl6, z6)  # [B, 1, S·Z, L, P, P]
            amplitude = amplitude.reshape(amplitude.shape[0], 1, species, planes, bins, *u2.shape)
            base = aperture * amplitude
            travel = focus.reshape(focus.shape[0], focus.shape[1], 1, 1, 1, 1, 1)
            measure = (du / wl7) ** 2
        else:
            inside = u2 < n7**2
            obliquity = ops.obliquity(u2, n7)
            apodised = aperture / torch.sqrt(obliquity)
            base = torch.where(inside, apodised, torch.zeros_like(obliquity))
            travel = focus.reshape(focus.shape[0], focus.shape[1], 1, 1, 1, 1, 1) - z.reshape(
                1, 1, 1, planes, 1, 1, 1
            )  # focus − z: [B|1, A|1, 1, Z, 1, 1, 1]
            measure = du * du / (4.0 * math.pi * n7**2)
        mods = fused_modifiers(self.objective, table, u.to(torch.float64), na, medium)
        if mods is not None:
            cdtype = torch.complex128 if dtype == torch.float64 else torch.complex64
            base = base * mods.to(cdtype).reshape(mods.shape[0], 1, species, 1, bins, *u2.shape)
        defocus = ops.defocus_phase(u2, n7, wl7, torch.ones((), dtype=dtype))
        phase = defocus * travel  # [B, A, S, Z, L, P, P]
        if base.is_complex():
            field = base * torch.polar(torch.ones_like(phase), phase)
        else:
            field = torch.polar(base.expand_as(phase), phase)
        k_wave = 2.0 * math.pi / wl7  # [B|1, 1, S, 1, L, 1, 1]
        step = static.grid.spacing / static.oversample

        def sampling(count: int, first: int) -> Tensor:
            offsets = (first + torch.arange(count, device=device, dtype=dtype)) * step
            ph = k_wave * offsets[:, None] * u[None, :]  # [..., count, P]
            return torch.polar(torch.ones_like(ph), ph)

        rows_y, rows_x = sampling(fy, dmin_y), sampling(fx, dmin_x)
        with ieee_matmul():
            e = torch.matmul(rows_y, torch.matmul(field, rows_x.transpose(-1, -2)))
        intensity = torch.view_as_real(e).square().sum(-1)  # [B, A, S, Z, L, Fy, Fx]
        # photons per fine sample per emitted photon (discrete Parseval, as on the global path)
        scale = weights.reshape(b_t, 1, species, 1, bins, 1, 1) * measure * (du * step / wl7) ** 2
        return (intensity * scale).sum(4)
