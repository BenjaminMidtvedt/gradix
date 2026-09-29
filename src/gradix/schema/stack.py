"""Batching: :func:`stack` for per-image pytrees and :func:`pad` for ragged lists (§6.1, §6.9).

``stack`` turns per-image (or per-sub-batch) trees of identical structure into one batch. Every
population is padded to a slot bucket (8, 16, 32, …): padded slots get presence 0 and repeat the
values of the last present slot, or a value that satisfies the field's constraint for an empty
image, so they stay numerically harmless. A population is a node with a ``presence`` field;
per-object fields of its child nodes (labeling, materials) are padded with it. Structures must
agree; the error names the first difference. Value constraints are checked here on CPU inputs,
where the check is free (§6.1).
"""

from __future__ import annotations

import functools
from collections.abc import Sequence
from typing import TypeVar

import torch
from torch import Tensor

from gradix._core.errors import StructureError
from gradix._core.rules import slot_bucket
from gradix.schema.base import Leaf, iter_leaves, map_leaves, tree_axis_sizes
from gradix.schema.fields import Constraint, FieldSpec
from gradix.schema.layout import ROLE_RULES, canonical, canonical_shape, leading_axes
from gradix.schema.signature import signature

__all__ = ["check_constraint", "pad", "stack"]

T = TypeVar("T")


def check_constraint(value: Tensor, constraint: Constraint) -> Tensor:
    """Return a mask of the entries that satisfy a constraint.

    Parameters
    ----------
    value : Tensor
        Values (complex values are checked on their real part).
    constraint : str or tuple of float
        ``"positive"``, ``"nonnegative"``, ``"unit_interval"``, ``"finite"`` or ``(lo, hi)``.

    Returns
    -------
    Tensor
        Boolean mask, same shape as ``value``.
    """
    v = value.real if value.is_complex() else value
    if constraint == "positive":
        return v > 0
    if constraint == "nonnegative":
        return v >= 0
    if constraint == "unit_interval":
        return (v >= 0) & (v <= 1)
    if constraint == "finite":
        return torch.isfinite(v)
    if isinstance(constraint, tuple):
        lo, hi = constraint
        return (v >= lo) & (v <= hi)
    raise ValueError(f"unknown constraint {constraint!r}")


def _parent(path: str) -> str:
    return path.rpartition(".")[0]


def _join(prefix: str, name: str) -> str:
    return f"{prefix}.{name}" if prefix else name


def _populations(item: object) -> list[str]:
    """Paths of the nodes that own a presence field (populations), longest first."""
    roots = {
        _parent(path)
        for path, spec, _value in iter_leaves(item)
        if spec is not None and spec.role == "object" and path.rpartition(".")[2] == "presence"
    }
    return sorted(roots, key=len, reverse=True)


def _owner(path: str, populations: Sequence[str]) -> str:
    """Return the population of a per-object field: its nearest ancestor with presence."""
    for root in populations:
        if root == "" or path.startswith(root + "."):
            return root
    return _parent(path)


def _object_counts(items: Sequence[object], populations: Sequence[str]) -> dict[str, list[int]]:
    """Object count of every item for every population (or node with per-object fields).

    An item's count is the one size other than 1 that its per-object fields have along N (0 for
    an empty image), or 1 if every field broadcasts over objects.
    """
    found: dict[str, list[set[int]]] = {}
    for i, item in enumerate(items):
        for path, spec, value in iter_leaves(item):
            if spec is None or spec.role != "object" or not isinstance(value, Tensor):
                continue
            n = canonical_shape(spec, tuple(value.shape))[2]
            rows = found.setdefault(_owner(path, populations), [set() for _ in items])
            rows[i].add(n)
    counts: dict[str, list[int]] = {}
    for owner, rows in found.items():
        row: list[int] = []
        for i, sizes in enumerate(rows):
            real = sizes - {1}
            if len(real) > 1:
                msg = f"gx.stack: item {i}: {owner or 'the item'} has object counts {sorted(real)}"
                raise StructureError(msg, fix="give every per-object field the same N")
            row.append(real.pop() if real else 1)
        counts[owner] = row
    return counts


def _safe_value(constraint: Constraint | None) -> float:
    """Return a value that satisfies a constraint, for the slots of an empty image."""
    if constraint == "positive":
        return 1.0
    if isinstance(constraint, tuple):
        lo, hi = constraint
        return 0.0 if lo <= 0.0 <= hi else 0.5 * (lo + hi)
    return 0.0


def _pad_objects(
    t: Tensor, n_item: int, n_pad: int, presence: bool, constraint: Constraint | None
) -> Tensor:
    """Pad the N axis (dim 2 of a canonical object value) from ``n_item`` to ``n_pad`` slots."""
    if t.shape[2] == 1 and not presence:
        return t  # broadcast over objects: nothing to pad
    if t.shape[2] == 1 and n_item != 1:
        t = t.expand(*t.shape[:2], n_item, *t.shape[3:])
    n = t.shape[2]
    if n == n_pad:
        return t
    shape = list(t.shape)
    shape[2] = n_pad - n
    if presence:
        filler = torch.zeros(shape, dtype=t.dtype, device=t.device)
    elif n == 0:
        filler = torch.full(shape, _safe_value(constraint), dtype=t.dtype, device=t.device)
    else:
        filler = t[:, :, n - 1 : n].expand(shape)
    return torch.cat([t, filler], dim=2)


def _combine(
    path: str,
    spec: FieldSpec | None,
    values: list[Leaf],
    counts: list[int] | None,
    batches: list[int],
) -> Leaf:
    if all(v is None for v in values):
        return None
    if any(v is None for v in values):
        i = next(k for k, v in enumerate(values) if (v is None) != (values[0] is None))
        msg = f"gx.stack: {path} is absent in some items but present in item {i}"
        raise StructureError(msg, fix="give every item the same optional fields")
    if spec is None:
        raise StructureError(f"gx.stack: {path} is a bare tensor without a schema")
    numbers = [v for v in values if not isinstance(v, Tensor)]
    if len(numbers) == len(values) and all(v == numbers[0] for v in numbers):
        return values[0]
    if spec.role in ("shared", "carrier", "any"):
        first = values[0]
        same = all(v is first for v in values)
        tensors = [v for v in values if isinstance(v, Tensor)]
        if not same and len(tensors) == len(values):
            same = all(
                t.shape == tensors[0].shape and bool(torch.equal(t, tensors[0])) for t in tensors
            )
        if not same:
            msg = f"gx.stack: {path} is a {spec.role} field but differs between items"
            raise StructureError(msg, fix="make it equal in every item, or declare it per image")
        return first
    rule = ROLE_RULES[spec.role]
    ts = [canonical(v, spec, where=path) for v in values if v is not None]
    if spec.role == "object" and counts is not None:
        n_pad = slot_bucket(max(counts))
        is_presence = path.rpartition(".")[2] == "presence"
        ts = [
            _pad_objects(t, n, n_pad, is_presence, spec.constraint)
            for t, n in zip(ts, counts, strict=True)
        ]
    if "B" in rule.axes[:1]:
        # an item may be a sub-batch: bring every field to that item's batch before stacking
        for i, (t, b) in enumerate(zip(ts, batches, strict=True)):
            if t.shape[0] != b:
                if t.shape[0] != 1:
                    msg = f"gx.stack: item {i}: {path} has B = {t.shape[0]} in an item of {b}"
                    raise StructureError(msg)
                ts[i] = t.expand(b, *t.shape[1:])
    common = list(ts[0].shape)
    for t in ts[1:]:
        for d in range(1, t.ndim):
            if t.shape[d] == common[d] or t.shape[d] == 1:
                continue
            if common[d] != 1:
                axis = rule.axes[d] if d < len(rule.axes) else f"event dim {d - len(rule.axes)}"
                sizes = f"{t.shape[d]} in one item, {common[d]} in another"
                msg = f"gx.stack: {path} has {axis} = {sizes}"
                raise StructureError(msg)
            common[d] = t.shape[d]
    out = torch.cat([t.expand(t.shape[0], *common[1:]) for t in ts], dim=0)
    return _input_form(out, spec, values, path)


def _input_form(out: Tensor, spec: FieldSpec, values: list[Leaf], path: str) -> Tensor:
    """Drop the role axes no item used (and that stayed 1), so ``[1, N]`` items give ``[B, N]``."""
    rule = ROLE_RULES[spec.role]
    used: set[str] = {"B"}
    for v in values:
        shape = tuple(v.shape) if isinstance(v, Tensor) else ()
        used.update(leading_axes(spec, shape, where=path))
    forms = [axes for axes in rule.forms.values() if used <= set(axes)]
    if not forms:
        return out
    form = min(forms, key=len)
    keep = [i for i, axis in enumerate(rule.axes) if axis in form]
    drop = [i for i, axis in enumerate(rule.axes) if axis not in form]
    if any(out.shape[i] != 1 for i in drop):
        return out
    event = list(range(len(rule.axes), out.ndim))
    return out.reshape([out.shape[i] for i in keep + event])


def _with_presence(
    items: Sequence[object], per_path: dict[str, list[Leaf]], counts: dict[str, list[int]]
) -> None:
    """Give padded populations a presence mask: padded slots must never render."""
    for node_path, row in counts.items():
        n_pad = slot_bucket(max(row))
        if all(n == n_pad for n in row):
            continue
        ppath = _join(node_path, "presence")
        if ppath not in per_path:
            owner = node_path or "the item"
            msg = f"gx.stack: {owner} needs padding to {n_pad} slots but has no presence field"
            raise StructureError(msg, fix="give every item the same object count")
        values = per_path[ppath]
        device = next(
            (
                v.device
                for p, vs in per_path.items()
                if _parent(p) == node_path
                for v in vs
                if isinstance(v, Tensor)
            ),
            None,
        )
        per_path[ppath] = [
            torch.ones(1, n, device=device) if v is None else v
            for v, n in zip(values, row, strict=True)
        ]


def _check_values(items: Sequence[object]) -> None:
    for i, item in enumerate(items):
        for path, spec, value in iter_leaves(item):
            if spec is None or spec.constraint is None or value is None:
                continue
            t = value if isinstance(value, Tensor) else torch.as_tensor(value)
            if t.device.type != "cpu" or t.numel() == 0:
                continue
            ok = check_constraint(t.detach(), spec.constraint)
            if not bool(ok.all()):
                bad = t.detach()[~ok].reshape(-1)[0].item()
                rule = spec.constraint
                msg = f"gx.stack: item {i}: {path} = {bad!r} violates the constraint {rule!r}"
                raise StructureError(msg)


def stack(items: Sequence[T]) -> T:
    """Stack per-image trees of identical structure into one batch.

    Parameters
    ----------
    items : sequence of T
        Data objects, elements, Chains or containers, one per image (or per sub-batch). Their
        structures must agree; per-object fields may differ in their object count.

    Returns
    -------
    T
        One tree whose per-image fields have a batch axis. Populations are padded to a slot
        bucket with presence 0; values equal in every item stay shared.

    Raises
    ------
    StructureError
        If the structures differ (the message names the first difference), a shared field
        differs between items, a population needs padding but has no presence field, or a CPU
        value violates its field's constraint.
    """
    if not items:
        raise StructureError("gx.stack needs at least one item")
    first = items[0]
    base = signature(first)
    for i, item in enumerate(items[1:], start=1):
        if type(item) is not type(first):
            msg = f"gx.stack: item {i} is a {type(item).__name__}, item 0 a {type(first).__name__}"
            raise StructureError(msg)
        diff = base.first_difference(signature(item), ignore_capacity=True)
        if diff is not None:
            raise StructureError(f"gx.stack: item {i} differs from item 0 {diff}")
    _check_values(items)
    populations = _populations(first)
    counts = _object_counts(items, populations)
    batches = [tree_axis_sizes(item).get("B", 1) for item in items]
    per_path: dict[str, list[Leaf]] = {}
    for item in items:
        for path, _spec, value in iter_leaves(item):
            per_path.setdefault(path, []).append(value)
    _with_presence(items, per_path, counts)

    def combine(path: str, spec: FieldSpec | None, _value: Leaf) -> Leaf:
        owner = _owner(path, populations)
        return _combine(path, spec, per_path[path], counts.get(owner), batches)

    return map_leaves(combine, first)


def pad(*ragged: Sequence[Tensor], n: int | None = None) -> tuple[Tensor, ...]:
    """Pad ragged per-image lists into batched tensors plus a presence mask.

    Parameters
    ----------
    *ragged : sequence of Tensor
        One or more lists with one tensor ``[N_i, ...]`` per image; every list has the same
        ``N_i`` for image i.
    n : int, optional
        Slot count; defaults to the smallest bucket (8, 16, 32, …) that holds the largest N_i.

    Returns
    -------
    tuple of Tensor
        One padded tensor ``[B, n, ...]`` per list (present slots first, zeros after), then the
        presence mask ``[B, n]`` (1.0 present, 0.0 padding).

    Raises
    ------
    StructureError
        If the lists disagree in length or counts, or ``n`` is too small.
    """
    if not ragged:
        raise StructureError("gx.pad needs at least one list")
    b = len(ragged[0])
    if b == 0 or any(len(r) != b for r in ragged):
        raise StructureError("gx.pad: every list needs one entry per image")
    counts = [int(t.shape[0]) for t in ragged[0]]
    for r in ragged[1:]:
        if [int(t.shape[0]) for t in r] != counts:
            raise StructureError("gx.pad: the lists disagree in their per-image counts")
    size = slot_bucket(max(counts)) if n is None else int(n)
    if max(counts) > size:
        raise StructureError(f"gx.pad: an image has {max(counts)} objects but n = {size}")
    out: list[Tensor] = []
    for k, r in enumerate(ragged):
        ref = r[0]
        for i, t in enumerate(r):
            if tuple(t.shape[1:]) != tuple(ref.shape[1:]):
                msg = (
                    f"gx.pad: list {k}, image {i} has entries of shape {list(t.shape[1:])}, "
                    f"image 0 {list(ref.shape[1:])}"
                )
                raise StructureError(msg)
        dtype = functools.reduce(torch.promote_types, [t.dtype for t in r])
        padded = torch.zeros((b, size, *ref.shape[1:]), dtype=dtype, device=ref.device)
        for i, t in enumerate(r):
            padded[i, : t.shape[0]] = t
        out.append(padded)
    ref = ragged[0][0]
    dtype = ref.dtype if ref.is_floating_point() else torch.float32
    presence = torch.zeros((b, size), dtype=dtype, device=ref.device)
    for i, c in enumerate(counts):
        presence[i, :c] = 1.0
    out.append(presence)
    return tuple(out)
