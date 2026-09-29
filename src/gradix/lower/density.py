"""The density lowering (§4.6): emitter-density volumes to an :class:`~gradix.EmitterDensity`."""

from __future__ import annotations

from collections.abc import Mapping

import torch

from gradix import register
from gradix._core.axes import AcqIndex
from gradix._core.carriers import EmitterDensity
from gradix._core.errors import StructureError
from gradix._core.precision import result_dtype
from gradix.objects.volumes import Voxels
from gradix.schema.base import iter_leaves
from gradix.schema.layout import canonical, common_device

__all__ = ["emitter_density"]


@register.lowering("voxels->emitter_density")
def emitter_density(
    volumes: Voxels | Mapping[str, Voxels], *, acq: AcqIndex | None = None
) -> EmitterDensity:
    """Lower emitter-density volumes to an :class:`~gradix.EmitterDensity`.

    Populations become species rows in sorted name order; they must share one grid.

    Parameters
    ----------
    volumes : Voxels or Mapping[str, Voxels]
        One density, or densities by name (``quantity="density"``).
    acq : AcqIndex, optional
        The acquisition's index; densities are static over it.

    Returns
    -------
    EmitterDensity
        ``data [B|1, 1, S, Z, Y, X]`` photons per voxel per exposure, the emission table
        ``[B|1, S, L]`` and the grid.

    Raises
    ------
    StructureError
        If a volume is not a density, or the volumes' grids differ.

    Examples
    --------
    >>> import torch, gradix as gx
    >>> cell = gx.Voxels(values=torch.ones(2, 4, 4), spacing=(0.5, 0.1, 0.1),
    ...                  quantity="density", emission=gx.Spectrum.line(0.6))
    >>> emitter_density(cell).data.shape
    torch.Size([1, 1, 1, 2, 4, 4])
    """
    items = [("density", volumes)] if isinstance(volumes, Voxels) else sorted(volumes.items())
    if not items:
        raise StructureError("emitter_density needs at least one volume")
    grids = []
    for name, volume in items:
        if not isinstance(volume, Voxels) or volume.quantity != "density":
            raise StructureError(f"{name!r} is not an emitter density (gx.Voxels, density)")
        grids.append(volume.grid())
    if any(g != grids[0] for g in grids[1:]):
        raise StructureError("density populations must share one grid (shape, spacing, origin)")
    leaves = [v for _, vol in items for _, _, v in iter_leaves(vol)]
    dtype, device = result_dtype(*leaves), common_device(*leaves)
    rows = []
    for _, volume in items:
        values = canonical(volume.values, volume.schema()["values"], dtype=dtype, device=device)
        rows.append(values)  # [B|1, Z, Y, X]
    if len(rows) == 1:  # one population: a view of its values, no copy
        data = rows[0][:, None, None]
    else:
        batch = max(r.shape[0] for r in rows)
        data = torch.stack([r.expand(batch, *r.shape[1:]) for r in rows], 1)[:, None]
    tables = []
    for _, volume in items:
        spectrum = volume.emission
        if spectrum is None:  # pragma: no cover - checked by Voxels
            raise StructureError("an emitter density needs its emission spectrum")
        tables.append(spectrum.arrays(dtype=dtype, device=device))
    bins = max(wl.shape[1] for wl, _ in tables)
    rows_wl, rows_w = [], []
    for wl, w in tables:
        if wl.shape[1] < bins:  # pad bins with zero weight
            pad = bins - wl.shape[1]
            wl = torch.cat([wl, wl[:, :1].expand(wl.shape[0], pad)], 1)
            w = torch.cat([w, torch.zeros(w.shape[0], pad, dtype=w.dtype, device=w.device)], 1)
        rows_wl.append(wl)
        rows_w.append(w)
    table_b = max(t.shape[0] for t in rows_wl)
    wavelengths = torch.stack([t.expand(table_b, bins) for t in rows_wl], 1)
    weights = torch.stack([t.expand(table_b, bins) for t in rows_w], 1)
    return EmitterDensity(
        data=data, wavelengths=wavelengths, weights=weights, grid=grids[0], acq=acq or AcqIndex()
    )
