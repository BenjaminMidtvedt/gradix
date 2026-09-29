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

__all__ = [
    "depth",
    "field_of_view",
    "from_focus",
    "from_pixels",
    "pitch",
    "shift_focus",
    "to_pixels",
]


def pitch(camera: Camera, objective: Objective, *, like: Tensor | None = None) -> Tensor:
    """Return the object-space pixel pitch: a pixel's size in the sample, µm.

    It works as a unit for positions given in pixels: ``20 * pitch(camera, objective)`` is
    twenty pixels in µm.

    Parameters
    ----------
    camera : Camera
        The camera (``pixel_size`` in µm).
    objective : Objective
        The objective (``magnification``).
    like : Tensor, optional
        Coordinates to broadcast against (their dtype and device too): ``[B, N, D]`` (or
        ``[B, T, N, D]``) when the optics are per image, any ``[…, D]`` for shared optics.

    Returns
    -------
    Tensor
        ``pixel_size / magnification`` in µm: a scalar for shared optics; per image
        ``[B, 1, …, 1]`` shaped like ``like``, or ``[B]`` without it.

    Raises
    ------
    ValueError
        If the optics are per image but the coordinates have no ``[B, N, D]`` layout (a 2-d
        ``[N, D]`` array would pair image i's pitch with object i).

    Examples
    --------
    >>> import gradix as gx
    >>> camera = gx.Camera(pixel_size=6.5, shape=(40, 60))
    >>> round(float(20 * pitch(camera, gx.Objective(NA=1.2, magnification=65))), 6)  # 20 px
    2.0
    """
    p = _pitches(camera, objective, like)  # [B|1]
    if p.shape[0] == 1:
        return p.reshape(())
    if like is None:
        return p
    if like.ndim < 3 or like.shape[0] not in (1, p.shape[0]):
        msg = (
            f"per-image optics ({p.shape[0]} images) need coordinates [B, N, D], "
            f"got {list(like.shape)}"
        )
        raise ValueError(msg)
    return p.reshape(-1, *([1] * (like.ndim - 1)))


def field_of_view(camera: Camera, objective: Objective, *, like: Tensor | None = None) -> Tensor:
    """Return the camera frame's extent in the sample, ``(width, height)`` in µm.

    The frame spans ``[0, width) × [0, height)``: pixel ``(0, 0)`` covers ``[0, p)²`` with its
    centre at ``(p/2, p/2)``, ``p`` the object-space pitch. So ``torch.rand(B, N, 2) *
    field_of_view(camera, objective)`` places points anywhere in the image.

    Parameters
    ----------
    camera : Camera
        The camera (``shape`` ``(H, W)`` in pixels, ``pixel_size`` in µm).
    objective : Objective
        The objective (``magnification``).
    like : Tensor, optional
        A tensor whose dtype and device the result takes.

    Returns
    -------
    Tensor
        ``(W·p, H·p)`` in µm: ``[2]`` for shared optics, ``[B, 2]`` per image.

    Examples
    --------
    >>> import gradix as gx
    >>> camera = gx.Camera(pixel_size=6.5, shape=(40, 60))
    >>> fov = field_of_view(camera, gx.Objective(NA=1.2, magnification=65))
    >>> [round(v, 6) for v in fov.tolist()]
    [6.0, 4.0]
    """
    p = _pitches(camera, objective, like)  # [B|1]
    height, width = camera.shape
    extent = p[:, None] * torch.tensor([width, height], dtype=p.dtype, device=p.device)
    return extent[0] if extent.shape[0] == 1 else extent


def _pitches(camera: Camera, objective: Objective, like: Tensor | None) -> Tensor:
    """Return each image's object-space pitch ``[B|1]`` in µm, in ``like``'s dtype and device."""
    dtype = like.dtype if like is not None else torch.get_default_dtype()
    device = like.device if like is not None else None
    size = canonical(camera.pixel_size, camera.schema()["pixel_size"], dtype=dtype, device=device)
    mag = canonical(
        objective.magnification, objective.schema()["magnification"], dtype=dtype, device=device
    )
    return size / mag


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
