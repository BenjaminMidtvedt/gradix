"""The Gaussian-sprite emission element (``emit.gaussian``; the ``draft`` fidelity tier)."""

from __future__ import annotations

import dataclasses
import math
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
from gradix._core.envelope import Envelope
from gradix._core.errors import StructureError
from gradix._core.grid import Grid2D
from gradix._core.rules import Decision
from gradix.conventions import GAUSSIAN_SIGMA_FACTOR, collection_efficiency
from gradix.detect.camera import Camera
from gradix.lower.emitters import emission_rows
from gradix.objects.environment import Medium
from gradix.ops.emitters import gaussian_sprites
from gradix.optics.objective import Objective
from gradix.schema.base import Node
from gradix.schema.fields import child
from gradix.schema.layout import canonical

__all__ = ["Sprites", "SpritesStatic"]

NA_LIMIT = 0.7
"""Above this NA the Gaussian tier's error grows beyond its tolerance (§5.8)."""


@dataclasses.dataclass(frozen=True)
class SpritesStatic(Static):
    """Static configuration of :class:`Sprites`.

    Parameters
    ----------
    decisions : tuple of Decision, default ()
        Sizing decisions with provenance.
    grid : Grid2D, optional
        The camera grid in object space (nominal spacing; the kernel uses the exact pitch
        tensors, so learnable magnification and pixel size stay differentiable).
    """

    grid: Grid2D | None = None


def _range(envelope: Envelope, path: str) -> tuple[float, float] | None:
    return envelope.range(path)


def _nominal(envelope: Envelope, path: str, value: object) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    interval = envelope.range(path)
    if interval is None:
        return None
    lo, hi = interval
    return math.sqrt(lo * hi) if lo > 0 else 0.5 * (lo + hi)


@register.element("emit.gaussian")
@dataclasses.dataclass(frozen=True, eq=False)
class Sprites(Element[Irradiance]):
    """Pixel-integrated Gaussian sprites for point emitters.

    Each emitter renders as an isotropic Gaussian of width σ₀ = 0.22·λ/NA in focus, widening as
    ``σ(Δ) = σ₀·√(1 + (Δ/z_R)²)`` with ``z_R = 4π·σ₀²·n/λ`` (the Gaussian-beam law) at defocus
    Δ = z − focus, integrated exactly over each pixel. Its integral is the collection efficiency
    ``(1 − cosθ_max)/2`` times the emitted photons (§4.1), the same radiometry as the pupil
    tiers. It approximates the scalar pupil PSF within ≈20 % rel-L2 near focus for NA ≤ 0.7.

    Parameters
    ----------
    objective : Objective
        The objective (NA, magnification, focus).
    camera : Camera
        The camera whose pixel grid the sprites are integrated on.

    Examples
    --------
    >>> import torch, gradix as gx
    >>> from gradix.units import nm, um
    >>> camera = gx.Camera(pixel_size=6.5 * um, shape=(32, 32))
    >>> sprites = Sprites(gx.Objective(NA=0.7, magnification=40), camera)
    >>> beads = gx.Emitters(
    ...     position=torch.tensor([[[2.6, 2.6, 0.0]]]),
    ...     photons=torch.tensor([[1000.0]]),
    ...     emission=gx.Spectrum.line(600 * nm),
    ... )
    >>> irradiance = sprites(gx.lower.emitter_set(beads), gx.env.Homogeneous(1.33))
    >>> tuple(irradiance.data.shape)
    (1, 1, 1, 32, 32)
    """

    slot: ClassVar[Slot] = Slot.EMIT
    caps: ClassVar[Capabilities] = Capabilities(
        accepts=frozenset({EmitterSet}),
        produces=Irradiance,
        grad_quality={"camera": "zero", "camera.pixel_size": "exact", "objective.pupil": "zero"},
        reads={
            "position": "exact",
            "photons": "exact",
            "presence": "exact",
            "emission": "exact",
            "environment": "exact",
        },
        approximates=(
            Edge(
                "emit.pupil_mft",
                regime="isotropic emitters near focus, NA ≤ 0.7, |Δz| ≤ 0.2 µm",
                tolerance="ladders/gaussian_pupil@v1.0",
            ),
        ),
    )

    objective: Objective = child(doc="the objective")
    camera: Camera = child(doc="the camera")

    def validity(self, desc: Description, envelope: Envelope) -> list[Violation]:
        """Report findings: NA above 0.7, and defocus beyond half the depth of field.

        Parameters
        ----------
        desc : Description
            Static description; its populations name the enveloped emitter positions.
        envelope : Envelope
            The envelope.

        Returns
        -------
        list of Violation
            Warnings (§5.8).
        """
        out: list[Violation] = []
        where = desc.path or "sprites"
        objective = desc.part("objective")
        na = _range(envelope, f"{objective}.NA")
        if na is not None and na[1] > NA_LIMIT:
            out.append(
                Violation(
                    "warn",
                    "Gaussian sprites misrepresent the PSF at high NA",
                    entry=f"{objective}.NA",
                    value=na[1],
                    limit=NA_LIMIT,
                    element=where,
                    fix="use the pupil tier (gx.imaging.PointPSF) for NA > 0.7",
                    calibration="literature",
                )
            )
        # the envelope reads the bound Chain: a focus stack's planes are in objective.focus
        focus = _range(envelope, f"{objective}.focus") or (0.0, 0.0)
        n = _range(envelope, f"{desc.part('environment')}.n")
        for pop in desc.populations:
            z = _range(envelope, f"{pop}.position.z")
            wl = _range(envelope, f"{pop}.emission.wavelengths") or _range(
                envelope, f"{pop}.wavelengths"
            )
            if z is None or na is None or n is None or wl is None:
                continue
            defocus = max(abs(z[1] - focus[0]), abs(z[0] - focus[1]))
            half_dof = 0.5 * n[0] * wl[0] / (na[1] * na[1])
            if defocus > half_dof:
                out.append(
                    Violation(
                        "warn",
                        "emitters lie beyond ±DOF/2, where the Gaussian tier is inaccurate",
                        entry=f"{pop}.position.z",
                        value=defocus,
                        limit=half_dof,
                        element=where,
                        fix="use the pupil tier (gx.imaging.PointPSF) for defocused emitters",
                        calibration="literature",
                    )
                )
        return out

    def configure(self, desc: Description, envelope: Envelope) -> SpritesStatic:
        """Place the camera grid in object space.

        Parameters
        ----------
        desc : Description
            Static description of the element's context.
        envelope : Envelope
            The envelope; supplies the nominal pixel pitch and magnification.

        Returns
        -------
        SpritesStatic
            The camera grid, with the decision that set it.
        """
        pitch = _nominal(envelope, f"{desc.part('camera')}.pixel_size", self.camera.pixel_size)
        mag = _nominal(
            envelope, f"{desc.part('objective')}.magnification", self.objective.magnification
        )
        if pitch is None or mag is None:
            msg = "the envelope lacks camera.pixel_size or objective.magnification"
            raise StructureError(msg, fix="derive the envelope from the inputs")
        spacing = pitch / mag
        grid = Grid2D(tuple(self.camera.shape), spacing, (0.0, 0.0))
        decision = Decision(
            "detection.grid",
            (*grid.shape, spacing),
            "camera grid (s = 1): the Gaussian tier integrates each pixel exactly by erf",
            {"camera.pixel_size": pitch, "objective.magnification": mag},
        )
        return SpritesStatic(decisions=(decision,), grid=grid)

    def static_for_grid(self, grid: object) -> SpritesStatic:
        """Return a configuration that renders on a given camera grid.

        Parameters
        ----------
        grid : Grid2D
            The camera grid in object space.

        Returns
        -------
        SpritesStatic
            The configuration.
        """
        if not isinstance(grid, Grid2D):
            raise StructureError(f"Sprites renders on a Grid2D, got {type(grid).__name__}")
        return SpritesStatic(grid=grid)

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
        """Return the description of an eager call.

        Parameters
        ----------
        *inputs : object
            The emitter set and the medium.

        Returns
        -------
        Description
            One population named ``"emitters"``.
        """
        return Description(populations=("emitters",))

    def forward(self, *inputs: object, static: Static) -> Irradiance:
        """Render the emitters.

        Parameters
        ----------
        *inputs : object
            The :class:`~gradix.EmitterSet`, then the medium (:class:`~gradix.env.Homogeneous`).
        static : Static
            A :class:`SpritesStatic`.

        Returns
        -------
        Irradiance
            ``[B, A|1, 1, H, W]`` photons per pixel on the camera grid.
        """
        if len(inputs) != 2:
            raise StructureError("Sprites takes (EmitterSet, medium)")
        emitters, medium = inputs[0], inputs[1]
        if not isinstance(emitters, EmitterSet):
            msg = f"Sprites takes an EmitterSet, got {type(emitters).__name__}"
            raise StructureError(msg, fix="lower emitters with gx.lower.emitter_set")
        if not isinstance(medium, Medium):
            msg = f"Sprites needs a medium (gx.env.*), got {type(medium).__name__}"
            raise StructureError(msg, fix="pass environment=gx.env.Homogeneous(n)")
        if not isinstance(static, SpritesStatic) or static.grid is None:
            raise StructureError("Sprites needs a SpritesStatic with a grid")
        pos = emitters.position
        dtype, device = pos.dtype, pos.device
        ospec, cspec = self.objective.schema(), self.camera.schema()

        def image_value(node: Node, spec: dict, name: str) -> Tensor:
            v = canonical(getattr(node, name), spec[name], dtype=dtype, device=device)
            return v.reshape(-1, 1, 1, 1)  # [B|1, 1, 1, 1] against [B, A, N, L]

        na = image_value(self.objective, ospec, "NA")
        focus = canonical(self.objective.focus, ospec["focus"], dtype=dtype, device=device)
        focus = focus.reshape(focus.shape[0], focus.shape[1], 1, 1)  # [B|1, A|1, 1, 1]
        mag = canonical(
            self.objective.magnification, ospec["magnification"], dtype=dtype, device=device
        )
        size = canonical(self.camera.pixel_size, cspec["pixel_size"], dtype=dtype, device=device)
        wl, w = emission_rows(emitters)  # [B|1, 1, N, L]
        n = medium.index(dtype=dtype, device=device, wavelength=wl)  # at each emission bin
        n = n.real if n.is_complex() else n
        if n.ndim != wl.ndim:
            n = n.reshape(n.shape[0], *([1] * (wl.ndim - 1)))  # [B|1, 1, 1, 1]
        sigma0 = GAUSSIAN_SIGMA_FACTOR * wl / na
        z_r = 4.0 * math.pi * sigma0 * sigma0 * n / wl
        dz = pos[..., 2:3] - focus  # [B|1, A|1, N, 1]
        sigma = sigma0 * torch.sqrt(1.0 + (dz / z_r) ** 2)
        photons = (
            emitters.photons if emitters.presence is None else emitters.photons * emitters.presence
        )
        weight = photons[..., None] * collection_efficiency(na, n) * w  # [B, A, N, L]
        xy = pos[..., None, :2]  # [B|1, A|1, N, 1, 2]
        pitch = (size / mag).reshape(-1, 1, 1, 1)  # [B|1, 1, 1, 1]: per-image optics count too
        shape = torch.broadcast_shapes(weight.shape, sigma.shape, pitch.shape)
        weight, sigma = weight.expand(shape), sigma.expand(shape)
        batch, frames, count, bins = shape
        xy = xy.expand(batch, frames, count, bins, 2)
        pitch = pitch.reshape(-1, 1).expand(batch, frames).reshape(-1)
        data = gaussian_sprites(
            xy.reshape(batch * frames, count * bins, 2),
            weight.reshape(batch * frames, count * bins),
            sigma.reshape(batch * frames, count * bins),
            pitch,
            static.grid.shape,
            static.grid.origin,
        )
        return Irradiance(
            data=data.reshape(batch, frames, 1, *static.grid.shape),
            grid=static.grid,
            acq=emitters.acq,
        )
