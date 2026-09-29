"""The raster lowering (§4.6, ADR-41): solids onto the grid their consumer asks for.

Every object is evaluated on a patch of voxels around its centre (a static size, from the
largest bounding radius the consumer planned for), with its band-limited occupancy or surface
(:meth:`gradix.Solid.occupancy`, :meth:`~gradix.Solid.surface`) spread over the planes with a
band-limited kernel, then added into the grid. Chunks of objects are recomputed in the backward
pass (a non-reentrant
checkpoint), so memory stays at one chunk's patches, and every tensor the shapes read, including
the parameters of callable geometries, receives its gradient.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor
from torch.utils.checkpoint import checkpoint

from gradix._core.errors import StructureError
from gradix._core.grid import VolumeGrid
from gradix._core.precision import result_dtype
from gradix.objects.labeling import Labeling
from gradix.objects.objectset import Solid
from gradix.schema.base import iter_leaves
from gradix.schema.layout import canonical, common_device

__all__ = ["label_density", "quaternion_matrix", "rasterise"]

_SUBPLANES = 4
"""The most sub-planes per plane spacing the axial kernel is sampled on (planes coarser than the
lateral spacing)."""
_LOBES = 3
"""Lobes of the axial Lanczos kernel. Plane spacings at the axial Nyquist limit (Strata's
``"auto"``) sample the PSF's axial band exactly, so a point between planes is imaged correctly
only when its photons spread over the planes with sinc weights; box or linear weights shift its
defocus (14 % rel-L2 for a bead halfway between Nyquist planes)."""
_SURFACE_BLUR = 0.6
"""The narrowest surface blur, in sample spacings: sampled at spacing h, a Gaussian of width σ
sums to its integral within 2·exp(−2π²σ²/h²), 0.2 % at 0.6·h (34 % at 0.3·h)."""


def quaternion_matrix(q: Tensor) -> Tensor:
    """Return the rotation matrices of quaternions ``(w, x, y, z)`` (normalised first).

    Parameters
    ----------
    q : Tensor
        ``[..., 4]`` quaternions, scalar first.

    Returns
    -------
    Tensor
        ``[..., 3, 3]`` matrices that turn local vectors into world vectors.
    """
    q = q / torch.sqrt((q * q).sum(-1, keepdim=True) + 1e-30)
    w, x, y, z = q.unbind(-1)
    rows = [
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ]
    return torch.stack([torch.stack(row, -1) for row in rows], -2)


def rasterise(
    objects: Solid,
    grid: VolumeGrid,
    *,
    sigma: float,
    reach: float,
    surface: bool = False,
    weight: Tensor | float | None = None,
    normalize: bool = False,
    chunk: int = 16,
    deterministic: bool = False,
) -> Tensor:
    """Sample a population of solids on a grid: occupancy or surface, weighted per object.

    Parameters
    ----------
    objects : Solid
        The population.
    grid : VolumeGrid
        The grid (planes at ``z0 + k·dz``, lateral samples of ``grid.xy``).
    sigma : float
        Width of the band-limiting blur, µm (the raster prefilter σ_r).
    reach : float
        The largest bounding radius of any object, µm: it sizes the patches.
    surface : bool, default False
        Sample the blurred surface (1/µm) instead of the occupancy; its blur is at least 0.6
        sample spacings, so that it sums to the area.
    weight : Tensor or float, optional
        Per-object factors ``[B|1, T|1, N]`` (after presence); 1 by default.
    normalize : bool, default False
        Divide each object's samples by their sum over its patch first (``weight`` is then
        what the whole object adds, even where the grid clips it).
    chunk : int, default 16
        Objects per chunk: the memory (and the backward's recomputation) of one chunk of patches.
    deterministic : bool, default False
        Add objects in a fixed order (one at a time) rather than with atomic additions.

    Returns
    -------
    Tensor
        ``[B, T, Z, Y, X]``: per voxel, the sum over objects of weight × presence × the value
        spread over the planes by a Lanczos-3 kernel ``sinc(u)·sinc(u/3)``, ``u`` the offset in
        plane spacings, normalised so that every photon lands on the planes (its negative lobes
        are harmless to linear consumers such as Strata).

    Raises
    ------
    StructureError
        If the objects have per-object rotations of the wrong shape, or ``reach`` is not
        positive.
    """
    if reach <= 0:
        raise StructureError(f"the reach must be positive, got {reach}")
    schema = type(objects).schema()
    leaves = [v for _, _, v in iter_leaves(objects)]
    dtype, device = result_dtype(*leaves), common_device(*leaves)

    def get(name: str, value: Tensor | float) -> Tensor:
        return canonical(value, schema[name], dtype=dtype, device=device)

    pos = get("position", objects.position)  # [B|1, T|1, N, 3]
    rot = None if objects.rotation is None else get("rotation", objects.rotation)
    presence = None if objects.presence is None else get("presence", objects.presence)
    shape = {name: get(name, getattr(objects, name)) for name in objects.shape_fields}
    w = torch.ones((), dtype=dtype, device=device)
    if weight is not None:
        w = torch.as_tensor(weight, dtype=dtype, device=device)
    if presence is not None:
        w = w * presence
    tensors = [pos, w, *shape.values(), *(() if rot is None else (rot,))]
    batch = max(t.shape[0] for t in tensors if t.ndim >= 3) if pos.ndim else 1
    frames = max(t.shape[1] for t in tensors if t.ndim >= 3)
    count = pos.shape[2]
    nz, (ny, nx) = grid.nz, grid.xy.shape
    out = torch.zeros(batch * frames * nz * ny * nx, dtype=dtype, device=device)
    if count == 0:
        return out.reshape(batch, frames, nz, ny, nx)
    w = w.expand(batch, frames, count)

    dx, dz = grid.xy.spacing, grid.dz
    ox, oy = grid.xy.origin
    margin = reach + 4.0 * sigma
    half_xy, half_z = math.ceil(margin / dx), math.ceil(margin / dz) + _LOBES
    subplanes = min(max(math.ceil(dz / dx - 1e-9), 1), _SUBPLANES)
    if surface:  # a surface blur narrower than the samples does not integrate to the area
        sigma = max(sigma, _SURFACE_BLUR * max(dx, dz / subplanes))
    offsets_xy = torch.arange(-half_xy, half_xy + 1, device=device, dtype=dtype)
    offsets_z = torch.arange(-half_z, half_z + 1, device=device, dtype=dtype)
    # the axial kernel on midpoints of 2·LOBES·subplanes sub-planes (offsets in plane spacings),
    # normalised per sub-plane phase so that each position's weights over the planes sum to 1
    sub = (torch.arange(2 * _LOBES * subplanes, device=device, dtype=dtype) + 0.5) / subplanes
    sub = sub - _LOBES
    lanczos = torch.sinc(sub) * torch.sinc(sub / _LOBES)
    phase = lanczos.reshape(2 * _LOBES, subplanes).sum(dim=0)  # Σ over planes, per phase
    kernel = lanczos / phase.repeat(2 * _LOBES) / subplanes
    image = torch.arange(batch * frames, device=device).reshape(batch, frames, 1, 1, 1, 1)

    def patches(
        pos_c: Tensor, w_c: Tensor, rot_c: Tensor | None, *shape_c: Tensor
    ) -> tuple[Tensor, Tensor]:
        # centre voxel of each object, and the patch's world coordinates
        ci = torch.round((pos_c[..., 0] - ox) / dx - 0.5)
        cj = torch.round((pos_c[..., 1] - oy) / dx - 0.5)
        ck = torch.round((pos_c[..., 2] - grid.z0) / dz)
        ai = ci[..., None] + offsets_xy  # [B, T, n, P]
        aj = cj[..., None] + offsets_xy
        ak = ck[..., None] + offsets_z  # [B, T, n, Pz]
        xw = ox + (ai + 0.5) * dx - pos_c[..., 0, None]
        yw = oy + (aj + 0.5) * dx - pos_c[..., 1, None]
        zw = grid.z0 + (ak[..., None] + sub) * dz - pos_c[..., 2, None, None]  # [B, T, n, Pz, S]
        rel = torch.stack(
            torch.broadcast_tensors(
                xw[..., None, None, None, :],
                yw[..., None, None, :, None],
                zw[..., :, :, None, None],
            ),
            dim=-1,
        )  # [B, T, n, Pz, S, P, P, 3]
        if rot_c is not None:  # world → local: Rᵀ·(x − c)
            matrix = quaternion_matrix(rot_c)[..., None, None, None, None, :, :]
            rel = (matrix.transpose(-1, -2) @ rel[..., None])[..., 0]
        args = {
            name: value.reshape(*value.shape[:3], 1, 1, 1, 1, *value.shape[3:])
            for name, value in zip(objects.shape_fields, shape_c, strict=True)
        }
        sample = objects.surface if surface else objects.occupancy
        vals = (sample(rel, sigma, args) * kernel[:, None, None]).sum(dim=4)  # [B,T,n,Pz,P,P]
        if normalize:
            vals = vals / vals.sum(dim=(-3, -2, -1), keepdim=True).clamp_min(1e-30)
        vals = vals * w_c[..., None, None, None]
        inside = (
            ((ak >= 0) & (ak < nz))[..., :, None, None]
            & ((aj >= 0) & (aj < ny))[..., None, :, None]
            & ((ai >= 0) & (ai < nx))[..., None, None, :]
        )
        flat = (
            (ak.clamp(0, nz - 1).long()[..., :, None, None] * ny
             + aj.clamp(0, ny - 1).long()[..., None, :, None]) * nx
            + ai.clamp(0, nx - 1).long()[..., None, None, :]
        )  # fmt: skip
        idx = flat + image * (nz * ny * nx)
        return torch.where(inside, vals, torch.zeros_like(vals)), idx.expand(vals.shape)

    grad = torch.is_grad_enabled() and any(t.requires_grad for t in tensors)
    for start in range(0, count, chunk):
        part = slice(start, min(start + chunk, count))
        pos_c = pos.expand(batch, frames, count, 3)[:, :, part]
        w_c = w[:, :, part]
        rot_c = None if rot is None else rot.expand(batch, frames, count, 4)[:, :, part]
        shape_c = [v[:, :, part] if v.shape[2] > 1 else v for v in shape.values()]
        if grad:
            vals, idx = checkpoint(patches, pos_c, w_c, rot_c, *shape_c, use_reentrant=False)
        else:
            vals, idx = patches(pos_c, w_c, rot_c, *shape_c)
        if deterministic:  # one object at a time: indices within a patch are unique
            for n in range(vals.shape[2]):
                out = out.index_add(0, idx[:, :, n].reshape(-1), vals[:, :, n].reshape(-1))
        else:
            out = out.index_add(0, idx.reshape(-1), vals.reshape(-1))
    return out.reshape(batch, frames, nz, ny, nx)


def label_density(
    objects: Solid,
    grid: VolumeGrid,
    *,
    sigma: float,
    reach: float,
    deterministic: bool = False,
) -> Tensor:
    """Return the photons per voxel a labelled population emits on a grid.

    Parameters
    ----------
    objects : Solid
        A population with a :class:`~gradix.Labeling`.
    grid : VolumeGrid
        The grid.
    sigma : float
        Width of the raster prefilter, µm.
    reach : float
        The largest bounding radius of any object, µm.
    deterministic : bool, default False
        Add objects in a fixed order.

    Returns
    -------
    Tensor
        ``[B, T, Z, Y, X]`` photons per voxel per exposure: a density times the voxel volume,
        or each object's photons spread over its footprint (summing to its photons where the
        grid holds all of it).

    Raises
    ------
    StructureError
        If the population has no labeling, or a surface label sits on a kind without a surface.
    """
    labeling = objects.labeling
    if not isinstance(labeling, Labeling):
        name = type(objects).__name__
        raise StructureError(f"{name} has no labeling", fix="pass labeling=gx.Labeling(...)")
    if labeling.on_surface and not objects.has_surface:
        raise StructureError(f"{type(objects).__name__} has no surface to label")
    spec = type(labeling).schema()
    voxel = grid.xy.spacing**2 * grid.dz
    if labeling.photons is not None:
        weight, normalize = canonical(labeling.photons, spec["photons"]), True
    elif labeling.surface_density is not None:
        weight = voxel * canonical(labeling.surface_density, spec["surface_density"])
        normalize = False
    elif labeling.density is not None:
        weight, normalize = voxel * canonical(labeling.density, spec["density"]), False
    else:  # pragma: no cover - Labeling requires one of the three
        raise StructureError("the labeling has no density, surface density or photons")
    return rasterise(
        objects,
        grid,
        sigma=sigma,
        reach=reach,
        surface=labeling.on_surface,
        weight=weight,
        normalize=normalize,
        deterministic=deterministic,
    )
