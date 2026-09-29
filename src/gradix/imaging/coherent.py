"""Scalar coherent pupil imaging (``gx.imaging.Coherent``; §5.3, §5.12).

The unscattered light stays analytic: every incident plane wave inside the aperture reaches the
image plane as an exact plane wave, with its pupil transmission, the objective's pupil
modifiers evaluated at its direction, and its phase at the focal plane. Sparse producers
(:class:`~gradix.ObjectSpectra`) are evaluated on the pupil samples: for Mie spectra, S₁ and S₂
at the scattering angle between every incident wave and every pupil direction, reduced to the
co-polarised S∥ = S₂cos²φ + S₁sin²φ (P = 1, §4.1), converted to angular spectra and transformed
to the camera samples by a matrix Fourier transform. Detection is explicit interference, term
by term: |E_b|² + 2Re(E_b*·E_s) + |E_s|² per incoherent mode and wavelength bin, integrated over
pixels by the pixel MTF (§4.3). Homogeneous, index-matched media; magnification maps object
space onto the camera.
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
from gradix.imaging._shared import fused_modifiers, ratio_per_image
from gradix.objects.environment import Medium
from gradix.ops import pupil as ops
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
    n: Tensor  # [B|1], the real index of the homogeneous medium
    focus: Tensor  # [B|1, A|1]
    pitch: Tensor  # [B|1], object-space camera pitch
    lam: Tensor  # [B|1, L]
    xs: Tensor  # [B|1, Ws], sample positions in object space
    ys: Tensor  # [B|1, Hs]


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
        },
    )

    objective: Objective = child(doc="the objective")
    camera: Camera = child(doc="the camera")
    _: dataclasses.KW_ONLY
    oversample: int | str = knob(default="auto", doc="samples per camera pixel")
    pupil_samples: int = knob(default=64, doc="pupil cells per axis")

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
        medium = desc.nodes.get(desc.part("environment")) or desc.nodes.get("environment")
        if not isinstance(medium, Medium):
            return []
        ratio = ratio_per_image(self.objective, medium)  # NA/n per image, largest
        if ratio is None or ratio < 1.0:
            return []
        return [
            Violation(
                "error",
                "NA reaches the medium's index; coherent imaging of homogeneous media needs "
                "NA below it",
                entry=f"{desc.part('objective')}.NA",
                value=ratio,
                limit=1.0,
                element=desc.path or "coherent",
                fix="lower the NA below the medium's index",
            )
        ]

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
        entries = {"light.wavelength": (wl, wl)}
        medium = inputs[2] if len(inputs) > 2 and isinstance(inputs[2], Node) else None
        desc = Description(nodes={"environment": medium} if medium is not None else {})
        report(self.validity(desc, Envelope(entries)), "warn")  # errors raise
        return self.configure(desc, Envelope(entries))

    # ---- run time ---------------------------------------------------------------------------

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
        background, scattered, optics, static = self._fields(inputs, static)
        # explicit interference, term by term: weak scatterers keep their contrast (§4.4)
        intensity = background.real**2 + background.imag**2
        if scattered is not None:
            intensity = intensity + 2.0 * (background.conj() * scattered).real
            intensity = intensity + scattered.real**2 + scattered.imag**2
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
        background, scattered, _optics, _static = self._fields(inputs, static)
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
        if not isinstance(medium, Medium) or medium.layered:
            raise StructureError("Coherent images homogeneous media (layered media: M3a phase 3)")
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
        pitch = (
            image(self.camera, cspec, "pixel_size") / image(self.objective, ospec, "magnification")
        ).reshape(-1)
        index = medium.index(dtype=dtype, device=device)
        n = (index.real if index.is_complex() else index).reshape(-1)  # [B|1]
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
        return _Optics(dtype, cdtype, device, na, n, focus, pitch, lam, xs, ys)

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
        ctx = PupilContext(
            na=o.na.to(torch.float64).reshape(-1, 1, 1, 1),
            n=o.n.to(torch.float64).reshape(-1, 1, 1, 1),
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
        k0 = (2.0 * math.pi / wl)[..., None]  # [B|1, 1, 1, 1, L, 1]
        ux, uy = u[..., 0][..., None, None], u[..., 1][..., None, None]  # [B, A, M, J, 1, 1]
        phx = k0 * ux * o.xs.reshape(o.xs.shape[0], 1, 1, 1, 1, -1)
        phy = k0 * uy * o.ys.reshape(o.ys.shape[0], 1, 1, 1, 1, -1)
        wave_x = torch.polar(torch.ones_like(phx), phx)  # [B, A, M, J, L, Ws]
        wave_y = torch.polar(torch.ones_like(phy), phy)  # [B, A, M, J, L, Hs]
        return torch.einsum("bamjl,bamjly,bamjlx->bamlyx", amp, wave_y, wave_x)

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
        s_par = s2 * cos2 + s1 * (1.0 - cos2)
        k_m = 2.0 * math.pi * o.n.reshape(-1, 1) / o.lam  # [B|1, L]: per-image index or bins
        k_m = k_m.reshape(k_m.shape[0], 1, 1, 1, k_m.shape[1], 1, 1)  # [B|1, 1, 1, 1, L, 1, 1]
        f = (1j * s_par) / k_m  # scattering amplitude f = iS/k, µm
        # ---- incident waves at the spheres, each on its own: [B, A, M, J, L, N, 1] ----
        incident = waves.at_waves(pos, o.n.reshape(-1, 1))[..., 0, :].to(cdtype)
        if spectra.presence is not None:
            presence = spectra.presence.to(device=device, dtype=dtype)  # [B|1, A|1, N]
            incident = incident * presence[:, :, None, None, None, :]
        strength = (f * incident[..., None]).sum(3)  # coherent sum over J: [B, A, M, L, N, K]
        strength = strength.reshape(*strength.shape[:-1], samples, samples)
        # ---- pupil weights, layout [B, A, 1, L, N, P, P] ----
        seven = (-1, 1, 1, 1, 1, 1, 1)
        up = u.to(dtype)
        uyp, uxp = torch.meshgrid(up, up, indexing="ij")
        u2p = uxp * uxp + uyp * uyp
        aperture = ops.soft_aperture(torch.sqrt(u2p), o.na.reshape(seven), du)
        n7 = o.n.reshape(seven)
        wl7 = o.lam.reshape(o.lam.shape[0], 1, 1, o.lam.shape[1], 1, 1, 1)
        inside = u2p < n7 * n7  # the far field holds propagating waves only
        cos = torch.sqrt(torch.clamp(1.0 - u2p / (n7 * n7), min=1e-12))
        kz = 2.0 * math.pi * n7 * cos / wl7
        focus7 = o.focus.reshape(o.focus.shape[0], o.focus.shape[1], 1, 1, 1, 1, 1)
        z7 = pos[..., 2].reshape(pos.shape[0], pos.shape[1], 1, 1, count, 1, 1)
        phase = kz * (focus7 - z7)  # [B, A, 1, L, N, P, P]
        # pupil in f-measure: (2π)²·A(k⊥) = 2πi·f·E_inc/k_z, propagated to the focal plane, with
        # the aplanatic √cosθ that conserves the collected power (§5.3)
        weight = torch.where(inside, aperture * torch.sqrt(cos) / kz, torch.zeros_like(kz))
        pupil = (2j * math.pi) * weight * torch.polar(torch.ones_like(phase), phase)
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
