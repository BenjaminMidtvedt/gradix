"""The density lowering (§4.6, ADR-41): volumes and labelled solids to an EmitterDensity."""

from __future__ import annotations

from collections.abc import Mapping

import torch

from gradix import register
from gradix._core.axes import AcqIndex
from gradix._core.carriers import EmitterDensity
from gradix._core.contract import DensityRequest
from gradix._core.errors import StructureError
from gradix._core.precision import result_dtype
from gradix.lower.raster import label_density
from gradix.objects.labeling import Labeling
from gradix.objects.objectset import Solid
from gradix.objects.volumes import Voxels
from gradix.schema.base import iter_leaves
from gradix.schema.layout import canonical, common_device

__all__ = ["emitter_density"]


@register.lowering("voxels->emitter_density")
def emitter_density(
    volumes: Voxels | Solid | Mapping[str, Voxels | Solid],
    *,
    request: DensityRequest | None = None,
    acq: AcqIndex | None = None,
) -> EmitterDensity:
    """Lower emitter-density volumes and labelled solids to an :class:`~gradix.EmitterDensity`.

    Populations become species rows in sorted name order. Volumes are used on their own grid;
    labelled solids are rasterised onto the grid the consumer asks for (``request``, ADR-41),
    which volumes then have to share.

    Parameters
    ----------
    volumes : Voxels, Solid or Mapping[str, Voxels | Solid]
        One population, or populations by name: volumes with ``quantity="density"`` and solids
        with a :class:`~gradix.Labeling`.
    request : DensityRequest, optional
        The consumer's grid, raster blur and reaches; needed for solids.
    acq : AcqIndex, optional
        The acquisition's index; volumes are static over it, solids may vary per frame.

    Returns
    -------
    EmitterDensity
        ``data [B|1, A|1, S, Z, Y, X]`` photons per voxel per exposure, the emission table
        ``[B|1, S, L]`` and the grid.

    Raises
    ------
    StructureError
        If a volume is not a density or a solid has no labeling, if the volumes' grids differ
        (from each other or from the request), or if solids come without a request.

    Examples
    --------
    >>> import torch, gradix as gx
    >>> cell = gx.Voxels(values=torch.ones(2, 4, 4), spacing=(0.5, 0.1, 0.1),
    ...                  quantity="density", emission=gx.Spectrum.line(0.6))
    >>> emitter_density(cell).data.shape
    torch.Size([1, 1, 1, 2, 4, 4])
    """
    if isinstance(volumes, (Voxels, Solid)):
        items: list[tuple[str, Voxels | Solid]] = [("density", volumes)]
    else:
        items = sorted(volumes.items())
    if not items:
        raise StructureError("emitter_density needs at least one population")
    grids = []
    for name, volume in items:
        if isinstance(volume, Voxels):
            if volume.quantity != "density":
                raise StructureError(f"{name!r} is not an emitter density (gx.Voxels, density)")
            grids.append(volume.grid())
        elif not isinstance(volume, Solid) or not isinstance(volume.labeling, Labeling):
            msg = f"{name!r} is neither an emitter density nor a labelled solid"
            raise StructureError(msg, fix="label solids with labeling=gx.Labeling(...)")
    solids = [name for name, v in items if isinstance(v, Solid)]
    if solids and request is None:
        msg = f"the solids {solids} need a grid to be rasterised onto"
        raise StructureError(msg, fix="render them through an element that asks for one (Strata)")
    grid = request.grid if request is not None and solids else (grids[0] if grids else None)
    if grid is None or any(g != grid for g in grids):
        msg = "density populations must share one grid (shape, spacing, origin)"
        fix = "give volumes the grid of the imaging element, or render them separately"
        raise StructureError(msg, fix=fix)
    leaves = [v for _, vol in items for _, _, v in iter_leaves(vol)]
    dtype, device = result_dtype(*leaves), common_device(*leaves)
    rows, tables = [], []
    for name, volume in items:
        if isinstance(volume, Voxels):
            spec = volume.schema()["values"]
            values = canonical(volume.values, spec, dtype=dtype, device=device)
            rows.append(values[:, None])  # [B|1, 1, Z, Y, X]
            spectrum = volume.emission
        else:
            if request is None:  # pragma: no cover - checked above
                raise StructureError("solids need a grid")
            density = label_density(
                volume,
                request.grid,
                sigma=request.blur,
                reach=request.reach.get(name, 0.0) or _reach(volume),
                deterministic=request.deterministic,
            )
            rows.append(density.to(dtype))  # [B, T, Z, Y, X]
            spectrum = volume.labeling.emission if isinstance(volume.labeling, Labeling) else None
        if spectrum is None:  # pragma: no cover - checked by Voxels and Labeling
            raise StructureError("an emitter density needs its emission spectrum")
        tables.append(spectrum.arrays(dtype=dtype, device=device))
    if len(rows) == 1:  # one population: a view of its values, no copy
        data = rows[0][:, :, None]
    else:
        batch = max(r.shape[0] for r in rows)
        frames = max(r.shape[1] for r in rows)
        data = torch.stack([r.expand(batch, frames, *r.shape[2:]) for r in rows], 2)
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
        data=data, wavelengths=wavelengths, weights=weights, grid=grid, acq=acq or AcqIndex()
    )


def _reach(solid: Solid) -> float:
    """Return a population's largest bounding radius from its values (host read), µm."""
    spec = type(solid).schema()
    shape = {n: canonical(getattr(solid, n), spec[n]).detach() for n in solid.shape_fields}
    return float(solid.reach(shape).max())
