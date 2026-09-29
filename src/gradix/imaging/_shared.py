"""Helpers the imaging elements share (PointPSF, Strata and Coherent use the same rules).

Static values from envelopes, the pixel-MTF margin, media and pupil modifiers.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

from gradix._core.contract import Description
from gradix._core.envelope import Envelope
from gradix._core.errors import StructureError
from gradix.objects.environment import Medium
from gradix.objects.materials import Constant, Material
from gradix.optics.objective import Objective
from gradix.optics.pupil import PupilContext, PupilModifier
from gradix.schema.layout import canonical

__all__ = [
    "MARGIN",
    "fused_modifiers",
    "material_range",
    "observed",
    "ratio_per_image",
    "reach",
    "value",
]


MARGIN = 16
"""Camera pixels evaluated beyond each edge on the global and strata paths. The pixel MTF is
applied periodically over the frame and this margin; its wrap changes the frame by ~1e-6
(rel-L2) for sources inside the frame and ~3e-5 for sources just outside (4 px: 5 %)."""


def value(envelope: Envelope, path: str, value: object, pick: str) -> float | None:
    """Return a field's static value: the number itself, or its envelope's lo, hi or middle.

    Parameters
    ----------
    envelope : Envelope
        The envelope.
    path : str
        The field's envelope path.
    value : object
        The field's stored value (a number is its own static value).
    pick : {"lo", "hi", "mid"}
        Which end of the interval (``"mid"``: the geometric mean of a positive interval).

    Returns
    -------
    float or None
        The value, or None when the envelope has no entry.
    """
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    interval = envelope.range(path)
    if interval is None:
        return None
    lo, hi = interval
    if pick == "lo":
        return lo
    if pick == "hi":
        return hi
    return math.sqrt(lo * hi) if lo > 0 else 0.5 * (lo + hi)


def observed(envelope: Envelope, path: str, value: object, pick: str) -> float | None:
    """Return a field's actual value: the number itself, or the inputs' observed lo or hi.

    Parameters
    ----------
    envelope : Envelope
        The envelope (its observed spans, else its intervals).
    path : str
        The field's envelope path.
    value : object
        The field's stored value.
    pick : {"lo", "hi"}
        Which end.

    Returns
    -------
    float or None
        The value, or None when the envelope has no entry.
    """
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    span = envelope.observed(path) or envelope.range(path)
    if span is None:
        return None
    return span[0] if pick == "lo" else span[1]


def ratio_per_image(objective: Objective, medium: object) -> float | None:
    """Return the largest NA/n of any image, from the actual values (per image, not extremes).

    Parameters
    ----------
    objective : Objective
        The objective (its NA, a number or per image).
    medium : object
        The medium; n is its immersion index (the medium itself when homogeneous).

    Returns
    -------
    float or None
        ``max_i NA_i / n_i``, or None without a medium.
    """
    if not isinstance(medium, Medium):
        return None
    na = canonical(objective.NA, objective.schema()["NA"], dtype=torch.float64).reshape(-1)
    n = medium.immersion_index(dtype=torch.float64)
    n = (n.real if n.is_complex() else n).reshape(-1)
    return float((na.detach().cpu() / n.detach().cpu()).max())  # host values: configuration time


def reach(desc: Description, envelope: Envelope, shape: tuple[int, int], pitch: float) -> float:
    """Return the largest emitter-to-sample offset over the frame and its margin, µm.

    Parameters
    ----------
    desc : Description
        Static description (its populations).
    envelope : Envelope
        The envelope (the populations' ``position.x`` and ``position.y``).
    shape : tuple of int
        Camera shape ``(H, W)``.
    pitch : float
        Object-space pitch, µm (the coarsest image's).

    Returns
    -------
    float
        The reach, µm.
    """
    height, width = shape
    lo_x, hi_x = -MARGIN * pitch, (width + MARGIN) * pitch
    lo_y, hi_y = -MARGIN * pitch, (height + MARGIN) * pitch
    reach = max(hi_x - lo_x, hi_y - lo_y)  # emitters inside the frame
    for pop in desc.populations:
        for axis, (lo, hi) in (("x", (lo_x, hi_x)), ("y", (lo_y, hi_y))):
            span = envelope.range(f"{pop}.position.{axis}")
            if span is not None:
                reach = max(reach, hi - span[0], span[1] - lo)
    return reach


def material_range(
    medium: object, name: str, band: tuple[float, float]
) -> tuple[float, float] | None:
    """Return the index range of a medium's dispersive material over a wavelength band.

    Parameters
    ----------
    medium : object
        The medium.
    name : str
        The material's field (``"sample"``, ``"immersion"``, …).
    band : tuple of float
        Vacuum wavelength band, µm.

    Returns
    -------
    tuple of float or None
        ``(n_min, n_max)``, or None for constants (which have envelope entries) and media
        without the field.
    """
    material = getattr(medium, name, None)
    if not isinstance(material, Material) or isinstance(material, Constant):
        return None  # constants have envelope entries
    grid = torch.linspace(band[0], band[1], 16, dtype=torch.float64)[None]
    values = material.index(grid).detach()  # configuration reads values, not gradients
    values = values.real if values.is_complex() else values
    return float(values.min()), float(values.max())


def fused_modifiers(
    objective: Objective, table: Tensor, u: Tensor, na: Tensor, medium: Medium
) -> Tensor | None:
    """Fuse an objective's pupil modifiers per (image, species, bin) on a pupil grid (P7).

    Parameters
    ----------
    objective : Objective
        The objective whose ``pupil`` modifiers are fused.
    table : Tensor
        Emission wavelengths ``[B|1, S, L]``, µm.
    u : Tensor
        Pupil sample coordinates ``[P]``, NA units.
    na : Tensor
        Numerical aperture per image, ``[B|1]`` (any trailing ones).
    medium : Medium
        The medium; its immersion index is the modifiers' ``ctx.n`` per image and bin.

    Returns
    -------
    Tensor or None
        Complex multipliers ``[B|1, S, L, P, P]`` (float64 grid), or None without modifiers.

    Raises
    ------
    StructureError
        If ``objective.pupil`` holds something that is not a :class:`~gradix.PupilModifier`.
    """
    modifiers = [m for m in objective.pupil if isinstance(m, PupilModifier)]
    if len(modifiers) != len(objective.pupil):
        raise StructureError("objective.pupil holds something that is not a gx.PupilModifier")
    if not modifiers:
        return None
    table = table.to(torch.float64)  # [B|1, S, L]
    b, species, bins = table.shape
    wl = table.reshape(b, species * bins, 1, 1)
    uy, ux = torch.meshgrid(u, u, indexing="ij")
    fx, fy = ux[None, None] / wl, uy[None, None] / wl
    n = medium.immersion_index(dtype=torch.float64, device=table.device, wavelength=table)
    n = n.real if n.is_complex() else n
    ctx = PupilContext(
        na=na.to(torch.float64).reshape(-1, 1, 1, 1),
        n=n.reshape(n.shape[0], -1, 1, 1),  # [B|1, S·L or 1, 1, 1]
    )
    shape = torch.broadcast_shapes(fx.shape, wl.shape)  # [B|1, S·L, P, P]
    total = modifiers[0](fx, fy, wl, ctx)
    for m in modifiers[1:]:
        total = total * m(fx, fy, wl, ctx)
    # a modifier may return anything broadcastable (a [P, P] map, a per-image stack, …)
    total = torch.broadcast_to(total, torch.broadcast_shapes(total.shape, shape))
    return total.reshape(total.shape[0], species, bins, *total.shape[-2:])
