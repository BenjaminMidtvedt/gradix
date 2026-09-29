"""Lowering of emitter populations to the :class:`~gradix.EmitterSet` view (exact)."""

from __future__ import annotations

from collections.abc import Callable, Mapping

import torch
from torch import Tensor

from gradix import register
from gradix._core.axes import AcqIndex
from gradix._core.carriers import EmitterSet
from gradix._core.errors import StructureError
from gradix._core.precision import result_dtype
from gradix.objects.objectset import Emitters
from gradix.schema.base import iter_leaves
from gradix.schema.layout import canonical, canonical_shape, common_device

__all__ = ["emission_rows", "emitter_set", "population_counts", "populations"]


def populations(emitters: Emitters | Mapping[str, Emitters]) -> list[tuple[str, Emitters]]:
    """Return emitter populations in canonical (sorted-name) order.

    Parameters
    ----------
    emitters : Emitters or Mapping[str, Emitters]
        One population (named ``"emitters"``), or populations by name.

    Returns
    -------
    list of (str, Emitters)
        The populations, sorted by name.

    Raises
    ------
    StructureError
        If a value is not an :class:`~gradix.Emitters`, or there is none.
    """
    if isinstance(emitters, Emitters):
        return [("emitters", emitters)]
    items = sorted(emitters.items())
    for name, pop in items:
        if not isinstance(pop, Emitters):
            msg = f"population {name!r} is a {type(pop).__name__}, not gx.Emitters"
            raise StructureError(msg)
    if not items:
        raise StructureError("emitter_set needs at least one population")
    return items


@register.lowering("point->emitter_set")
def population_counts(emitters: Emitters | Mapping[str, Emitters]) -> dict[str, int]:
    """Return each population's slot count N as :func:`emitter_set` lays the populations out.

    Parameters
    ----------
    emitters : Emitters or Mapping[str, Emitters]
        One population, or populations by name.

    Returns
    -------
    dict of str to int
        N by population, in sorted-name (concatenation) order: the largest N of each
        population's per-object fields, read from shapes only.
    """
    out: dict[str, int] = {}
    for name, pop in populations(emitters):
        spec = pop.schema()
        sizes = [
            canonical_shape(spec[f], tuple(torch.as_tensor(value).shape))[2]
            for f in ("position", "photons", "presence", "id")
            if (value := getattr(pop, f)) is not None
        ]
        out[name] = _slots(sizes)
    return out


def _slots(sizes: list[int]) -> int:
    """Return a population's slot count: its fields' common N (1 broadcasts; 0 is empty)."""
    return 0 if 0 in sizes else max(sizes)


def emitter_set(
    emitters: Emitters | Mapping[str, Emitters],
    *,
    dtype: torch.dtype | None = None,
    device: torch.device | str | None = None,
    acq: AcqIndex | None = None,
) -> EmitterSet:
    """Lower one or more emitter populations to an :class:`~gradix.EmitterSet`.

    Populations are concatenated along N in sorted name order, so the result does not depend on
    mapping order; each population is one species row of the emission table. Frames (T) become
    the acquisition axis A, with a ``"time"`` axis when T > 1.

    Parameters
    ----------
    emitters : Emitters or Mapping[str, Emitters]
        One population, or populations by name.
    dtype : torch.dtype, optional
        Real dtype of the result; float64 when any input is float64, else float32.
    device : torch.device or str, optional
        Device of the result; the inputs' device by default.
    acq : AcqIndex, optional
        The acquisition's index (from ``gx.acq``); by default a ``"time"`` axis of the
        populations' T. Per-frame fields must then have T equal to its frame count, or 1.

    Returns
    -------
    EmitterSet
        Positions ``[B|1, A|1, N, 3]``, photons, presence, species ``[B|1, 1, N]`` and the
        emission table ``[B|1, S, L]``.

    Raises
    ------
    StructureError
        If populations disagree on B or T, or T does not match the acquisition.
    """
    pops = populations(emitters)
    leaves = [v for _, p in pops for _, _, v in iter_leaves(p)]
    dtype = dtype or result_dtype(*leaves)
    device = device or common_device(*leaves)
    columns: list[dict[str, Tensor | None]] = []
    for name, pop in pops:
        spec = pop.schema()

        def get(field: str, pop: Emitters = pop, spec: dict = spec, name: str = name) -> Tensor:
            value = getattr(pop, field)
            is_int = spec[field].dtype == "integer"
            return canonical(
                value,
                spec[field],
                dtype=None if is_int else dtype,
                device=device,
                where=f"{name}.{field}",
            )

        columns.append(
            {
                "position": get("position"),
                "photons": get("photons"),
                "presence": None if pop.presence is None else get("presence"),
                "id": None if pop.id is None else get("id"),
            }
        )
    lead = [t.shape for col in columns for t in col.values() if t is not None and t.ndim >= 3]
    batch = max(s[0] for s in lead)
    frames = max(s[1] for s in lead)
    for s in lead:
        if s[0] not in (1, batch) or s[1] not in (1, frames):
            msg = f"emitter populations disagree on B or T: {list(s[:2])} vs [{batch}, {frames}]"
            raise StructureError(msg)
    # a population's slot count is the largest N of its per-object fields (the others broadcast)
    counts = [_slots([t.shape[2] for t in col.values() if t is not None]) for col in columns]
    shared_b = 1 if all(s[0] == 1 for s in lead) else batch
    shared_t = 1 if all(s[1] == 1 for s in lead) else frames

    def column(
        name: str, event: tuple[int, ...], fill: Callable[[int], Tensor] | None
    ) -> Tensor | None:
        parts: list[Tensor] = []
        for col, n in zip(columns, counts, strict=True):
            t = col[name]
            if t is None:
                if fill is None:
                    return None
                t = fill(n)
            frames_t = 1 if name == "id" else shared_t
            parts.append(t.expand(shared_b, frames_t, n, *event))
        return parts[0] if len(parts) == 1 else torch.cat(parts, dim=2)

    single = len(columns) == 1
    ones = None if single else (lambda n: torch.ones(1, 1, n, dtype=dtype, device=device))
    minus_one = (
        None if single else (lambda n: torch.full((1, 1, n), -1, device=device, dtype=torch.int64))
    )
    position = column("position", (3,), None)
    photons = column("photons", (), None)
    if position is None or photons is None:  # pragma: no cover - required fields
        raise StructureError("emitters need position and photons")
    species = None
    if not single:
        species = torch.cat(
            [
                torch.full((1, 1, n), s, dtype=torch.int64, device=device)
                for s, n in enumerate(counts)
            ],
            2,
        ).expand(shared_b, 1, sum(counts))
    wavelengths, weights = _table([p.emission.arrays(dtype=dtype, device=device) for _, p in pops])
    return EmitterSet(
        position=position,
        photons=photons,
        presence=column("presence", (), ones),
        species=species,
        wavelengths=wavelengths,
        weights=weights,
        id=column("id", (), minus_one),
        acq=_acquisition(acq, frames),
    )


def _table(tables: list[tuple[Tensor, Tensor]]) -> tuple[Tensor, Tensor]:
    """Stack per-species ``[B|1, L_s]`` spectra into ``[B|1, S, L]``, padding L with weight 0."""
    batch = max(wl.shape[0] for wl, _ in tables)
    bins = max(wl.shape[1] for wl, _ in tables)
    rows_wl, rows_w = [], []
    for wl, w in tables:
        if wl.shape[1] < bins:
            pad = bins - wl.shape[1]
            wl = torch.cat([wl, wl[:, :1].expand(wl.shape[0], pad)], 1)
            w = torch.cat([w, torch.zeros(w.shape[0], pad, dtype=w.dtype, device=w.device)], 1)
        rows_wl.append(wl.expand(batch, bins))
        rows_w.append(w.expand(batch, bins))
    return torch.stack(rows_wl, 1), torch.stack(rows_w, 1)


def emission_rows(emitters: EmitterSet) -> tuple[Tensor, Tensor]:
    """Gather each emitter's emission row: its wavelengths and weights.

    Parameters
    ----------
    emitters : EmitterSet
        The lowered emitters.

    Returns
    -------
    tuple of Tensor
        Wavelengths in µm and weights, each ``[B|1, 1, N|1, L]`` (N is 1 when every emitter
        shares one species).
    """
    wl, w = emitters.wavelengths, emitters.weights  # [B|1, S, L]
    if emitters.species is None:
        return wl[:, :1, None, :].expand(-1, 1, 1, -1), w[:, :1, None, :].expand(-1, 1, 1, -1)
    species = emitters.species  # [B|1, 1, N]
    batch = max(wl.shape[0], species.shape[0])
    rows = species[:, 0, :].expand(batch, -1)  # [B, N]
    index = rows[..., None].expand(-1, -1, wl.shape[-1])  # [B, N, L]
    wl_e = torch.gather(wl.expand(batch, -1, -1), 1, index)
    w_e = torch.gather(w.expand(batch, -1, -1), 1, index)
    return wl_e[:, None], w_e[:, None]


def _acquisition(acq: AcqIndex | None, frames: int) -> AcqIndex:
    """Return the EmitterSet's acquisition index: the declared one (checked), or one from T."""
    if acq is None:
        return AcqIndex((("time", frames),)) if frames > 1 else AcqIndex()
    declared = acq.frames
    if frames != 1 and frames != declared:
        msg = f"the populations have T = {frames} frames but the acquisition declares {declared}"
        raise StructureError(msg, fix="give per-frame fields T = the acquisition's frame count")
    return acq
