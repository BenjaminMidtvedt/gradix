"""Pixel ↔ µm conversion and z helpers (``gx.coords``; §4.1, §6.6).

One pinned pixel convention: integer pixel coordinates are pixel *centres*. Pixel ``(i, j)``
(row i, column j) has its centre at ``((j + ½)·p, (i + ½)·p)`` µm with p the object-space pitch
``pixel_size / magnification``, so ``x_px = x/p − ½`` and ``y_px = y/p − ½``. Components keep
their order (x, y[, z]); z stays in µm.

Examples
--------
>>> import torch, gradix as gx
>>> from gradix.units import um
>>> cam = gx.Camera(pixel_size=6.5 * um, shape=(64, 64))
>>> obj = gx.Objective(NA=0.8, magnification=10)
>>> [round(v, 4) for v in to_pixels(torch.tensor([0.325, 0.975]), cam, obj).tolist()]
[0.0, 1.0]
"""

from __future__ import annotations

import torch
from torch import Tensor

from gradix.detect.camera import Camera
from gradix.optics.objective import Objective
from gradix.schema.layout import canonical

__all__ = ["depth", "from_focus", "from_pixels", "pitch", "shift_focus", "to_pixels"]


def pitch(camera: Camera, objective: Objective, *, like: Tensor) -> Tensor:
    """Return the object-space pixel pitch, broadcastable against coordinates ``like``.

    Parameters
    ----------
    camera : Camera
        The camera (``pixel_size`` in µm).
    objective : Objective
        The objective (``magnification``).
    like : Tensor
        Coordinates ``[B, N, D]`` (or ``[B, T, N, D]``) when the optics are per image; any
        ``[…, D]`` for shared optics.

    Returns
    -------
    Tensor
        ``pixel_size / magnification`` in µm, shaped ``[B|1, 1, …, 1]``.

    Raises
    ------
    ValueError
        If the optics are per image but the coordinates have no ``[B, N, D]`` layout (a 2-d
        ``[N, D]`` array would pair image i's pitch with object i).
    """
    size = canonical(
        camera.pixel_size, camera.schema()["pixel_size"], dtype=like.dtype, device=like.device
    )
    mag = canonical(
        objective.magnification,
        objective.schema()["magnification"],
        dtype=like.dtype,
        device=like.device,
    )
    p = size / mag  # [B|1]
    if p.shape[0] == 1:
        return p.reshape(())
    if like.ndim < 3 or like.shape[0] not in (1, p.shape[0]):
        msg = (
            f"per-image optics ({p.shape[0]} images) need coordinates [B, N, D], "
            f"got {list(like.shape)}"
        )
        raise ValueError(msg)
    return p.reshape(-1, *([1] * (like.ndim - 1)))


def to_pixels(x: Tensor, camera: Camera, objective: Objective) -> Tensor:
    """Convert object-space positions in µm into pixel coordinates.

    Parameters
    ----------
    x : Tensor
        Positions ``[..., 2]`` or ``[..., 3]`` in µm (x, y[, z]); a leading batch axis when the
        optics are per image.
    camera : Camera
        The camera.
    objective : Objective
        The objective.

    Returns
    -------
    Tensor
        ``(x_px, y_px[, z])`` with ``x_px`` a column and ``y_px`` a row coordinate; integer
        values are pixel centres. z is unchanged (µm).
    """
    p = pitch(camera, objective, like=x)
    return torch.cat([x[..., :2] / p - 0.5, x[..., 2:]], dim=-1)


def from_pixels(px: Tensor, camera: Camera, objective: Objective) -> Tensor:
    """Convert pixel coordinates into object-space positions in µm (inverse of :func:`to_pixels`).

    Parameters
    ----------
    px : Tensor
        ``(x_px, y_px[, z])`` coordinates, ``[..., 2]`` or ``[..., 3]``.
    camera : Camera
        The camera.
    objective : Objective
        The objective.

    Returns
    -------
    Tensor
        Positions in µm.
    """
    p = pitch(camera, objective, like=px)
    return torch.cat([(px[..., :2] + 0.5) * p, px[..., 2:]], dim=-1)


def depth(d: Tensor | float) -> Tensor | float:
    """Return the z of a point at depth ``d`` into the sample, from the objective-side interface.

    +z points toward the objective, so a point deeper into the sample has a smaller z.

    Parameters
    ----------
    d : Tensor or float
        Depth in µm (positive into the sample).

    Returns
    -------
    Tensor or float
        ``z = −d``.
    """
    return -d


def from_focus(dz: Tensor | float, focus: Tensor | float) -> Tensor | float:
    """Return the z of a point ``dz`` from the focal plane (+dz toward the objective).

    Parameters
    ----------
    dz : Tensor or float
        Offset from the focal plane, µm.
    focus : Tensor or float
        Axial position of the focal plane, µm.

    Returns
    -------
    Tensor or float
        ``focus + dz``.
    """
    return focus + dz


def shift_focus(objective: Objective, dz: Tensor | float) -> Objective:
    """Return an objective whose focal plane is shifted by ``dz`` (+dz toward the objective).

    Parameters
    ----------
    objective : Objective
        The objective.
    dz : Tensor or float
        Shift in µm.

    Returns
    -------
    Objective
        A copy with ``focus + dz``.
    """
    return objective.replace(focus=objective.focus + dz)
