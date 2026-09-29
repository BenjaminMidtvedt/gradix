"""Envelopes: static intervals for every shape-affecting field (§6.4).

An :class:`Envelope` maps field paths to intervals. Paths follow one grammar (§6.4):
``<population>.<field>[.<component>]`` (``beads.radius``, ``beads.position.z``),
``<part>.<field>`` (``objective.focus``, ``camera.pixel_size``, ``light.wavelength``) and
nested fields (``beads.emission.wavelengths``).

Envelopes come from three sources, later ones overriding earlier ones entry by entry:
*derived* from example inputs with headroom (:func:`envelope_of`), *probed* by a front end, and
*explicit* entries. Sources combine with ``|``.

Derived entries widen the observed span by 25 % and round outward to buckets: 0.25 µm steps for
linear lengths such as z and focus, 10 % log steps for sizes and other positive quantities. The
rounding is strict, so a value is never on the edge of its own envelope.
"""

from __future__ import annotations

import dataclasses
import difflib
import math
from collections.abc import Collection, Iterator, Mapping
from typing import NamedTuple, Protocol, runtime_checkable

import torch
from torch import Tensor

from gradix._core.errors import EnvelopeError
from gradix.schema.base import Node, iter_leaves
from gradix.schema.fields import FieldSpec
from gradix.schema.layout import canonical

__all__ = [
    "Entry",
    "Envelope",
    "Values",
    "check_paths",
    "envelope_of",
    "enveloped_paths",
    "in_envelope",
    "shape_affecting_values",
    "widen",
]

HEADROOM = 1.25
"""Factor by which derived spans are widened."""
LINEAR_BUCKET: dict[str, float] = {"length": 0.25, "opd": 0.25}
"""Bucket (µm) of linear-scale entries by quantity; other linear quantities are not bucketed."""
LOG_STEP = 1.1
"""Ratio between log-scale bucket edges (a 10 % step)."""


@dataclasses.dataclass(frozen=True)
class Entry:
    """One envelope entry: an inclusive interval and where it came from.

    Parameters
    ----------
    lo : float
        Lower bound.
    hi : float
        Upper bound.
    source : {"derived", "probed", "explicit"}, default "explicit"
        Where the entry came from; reports list it.
    observed : tuple of float, optional
        The span the inputs actually had, before headroom (derived entries). Validity reports
        errors for these values and warnings for the interval.
    """

    lo: float
    hi: float
    source: str = "explicit"
    observed: tuple[float, float] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "lo", float(self.lo))
        object.__setattr__(self, "hi", float(self.hi))
        if self.observed is not None:
            lo, hi = self.observed
            object.__setattr__(self, "observed", (float(lo), float(hi)))
        if not (math.isfinite(self.lo) and math.isfinite(self.hi)) or self.lo > self.hi:
            msg = f"an envelope entry needs finite lo <= hi, got ({self.lo}, {self.hi})"
            raise EnvelopeError(msg)


@runtime_checkable
class _HasParts(Protocol):
    """Objects whose shape-affecting fields are reached through named parts (Chains, Samples)."""

    def parts(self) -> Mapping[str, Node]:
        """Return the named parts whose fields have envelope paths.

        Returns
        -------
        Mapping[str, Node]
            Parts by path prefix, such as ``{"beads": emitters, "objective": objective}``.
        """
        ...


class Envelope(Mapping[str, Entry]):
    """Static intervals for shape-affecting fields, by field path.

    Parameters
    ----------
    entries : Mapping[str, tuple of float or Entry], optional
        Intervals by path, such as ``{"beads.position.z": (-6.0, 6.0)}``.
    source : str, default "explicit"
        Source recorded for entries given as tuples.
    """

    def __init__(
        self,
        entries: Mapping[str, tuple[float, float] | Entry] | None = None,
        *,
        source: str = "explicit",
    ) -> None:
        self._entries: dict[str, Entry] = {}
        for path, value in (entries or {}).items():
            if isinstance(value, Entry):
                entry = value
            else:
                try:
                    lo, hi = value
                except (TypeError, ValueError):
                    msg = f"envelope entry {path!r} must be a (lo, hi) pair, got {value!r}"
                    raise EnvelopeError(msg) from None
                entry = Entry(float(lo), float(hi), source)
            self._entries[str(path)] = entry

    def __getitem__(self, path: str) -> Entry:
        return self._entries[path]

    def __iter__(self) -> Iterator[str]:
        return iter(sorted(self._entries))

    def __len__(self) -> int:
        return len(self._entries)

    def __or__(self, other: Mapping[str, tuple[float, float] | Entry]) -> Envelope:
        merged: dict[str, tuple[float, float] | Entry] = dict(self._entries)
        other_env = other if isinstance(other, Envelope) else Envelope(other)
        for path, entry in other_env._entries.items():
            mine = self._entries.get(path)
            if entry.observed is None and mine is not None and mine.observed is not None:
                # an override replaces the interval; the inputs' values are still what they were
                entry = dataclasses.replace(entry, observed=mine.observed)
            merged[path] = entry
        return Envelope(merged)

    def __ror__(self, other: Mapping[str, tuple[float, float] | Entry]) -> Envelope:
        return Envelope(other) | self

    def __repr__(self) -> str:
        inner = ", ".join(f"{p!r}: ({e.lo:g}, {e.hi:g})" for p, e in sorted(self._entries.items()))
        return f"Envelope({{{inner}}})"

    def range(self, path: str) -> tuple[float, float] | None:
        """Return the interval of a path, or None if the envelope has no entry for it.

        Parameters
        ----------
        path : str
            Field path.

        Returns
        -------
        tuple of float or None
            ``(lo, hi)``.
        """
        entry = self._entries.get(path)
        return None if entry is None else (entry.lo, entry.hi)

    def observed(self, path: str) -> tuple[float, float] | None:
        """Return the span the inputs actually had at a path (before headroom), if recorded.

        Parameters
        ----------
        path : str
            Field path.

        Returns
        -------
        tuple of float or None
            ``(lo, hi)`` of the derived values; None for entries without a record.
        """
        entry = self._entries.get(path)
        return None if entry is None else entry.observed

    def union(self, other: Mapping[str, tuple[float, float] | Entry]) -> Envelope:
        """Return the envelope that covers both: each entry spans both intervals.

        Parameters
        ----------
        other : Mapping
            Another envelope or mapping of entries.

        Returns
        -------
        Envelope
            Entries present in either; shared entries widened to cover both, with the source of
            ``other`` when it widened them.
        """
        other_env = other if isinstance(other, Envelope) else Envelope(other)
        merged = dict(self._entries)
        for path, entry in other_env._entries.items():
            mine = merged.get(path)
            if mine is None:
                merged[path] = entry
                continue
            seen = [s for s in (mine.observed, entry.observed) if s is not None]
            observed = (min(s[0] for s in seen), max(s[1] for s in seen)) if seen else None
            widened = entry.lo < mine.lo or entry.hi > mine.hi
            lo, hi = min(mine.lo, entry.lo), max(mine.hi, entry.hi)
            source = entry.source if widened else mine.source
            merged[path] = Entry(lo, hi, source, observed)
        return Envelope(merged)

    def key(self) -> tuple[tuple[str, float, float], ...]:
        """Return a hashable key of the intervals (sources excluded), for caches.

        Returns
        -------
        tuple
            ``(path, lo, hi)`` triples, sorted by path.
        """
        return tuple((p, e.lo, e.hi) for p, e in sorted(self._entries.items()))

    def to_json(self) -> dict[str, dict[str, object]]:
        """Return a JSON-serialisable description.

        Returns
        -------
        dict
            ``{path: {"lo": …, "hi": …, "source": …}}``, sorted by path. Observed spans are
            left out: they describe one set of inputs, not the plan (hashes stay stable).
        """
        return {
            p: {"lo": e.lo, "hi": e.hi, "source": e.source}
            for p, e in sorted(self._entries.items())
        }

    @classmethod
    def from_json(cls, data: Mapping[str, Mapping[str, object]]) -> Envelope:
        """Rebuild an envelope from :meth:`to_json` output.

        Parameters
        ----------
        data : Mapping
            ``{path: {"lo": …, "hi": …, "source": …}}``.

        Returns
        -------
        Envelope
            The envelope.
        """
        entries: dict[str, tuple[float, float] | Entry] = {}
        for p, d in data.items():
            seen = d.get("observed")
            observed = None
            if isinstance(seen, (list, tuple)) and len(seen) == 2:
                observed = (float(str(seen[0])), float(str(seen[1])))
            source = str(d.get("source", "explicit"))
            entries[p] = Entry(float(str(d["lo"])), float(str(d["hi"])), source, observed)
        return cls(entries)


def widen(
    lo: float, hi: float, spec: FieldSpec | None, *, headroom: float = HEADROOM
) -> tuple[float, float]:
    """Widen an observed span by the headroom and round it outward to buckets.

    Parameters
    ----------
    lo : float
        Smallest observed value.
    hi : float
        Largest observed value.
    spec : FieldSpec or None
        The field's declaration; it selects the log or linear scale and the bucket. Without one,
        positive spans use the log scale and others the linear scale without buckets.
    headroom : float, default 1.25
        Factor by which the span is widened.

    Returns
    -------
    tuple of float
        The widened interval. It always reaches at least a factor ``headroom`` beyond the
        observed values (``lo / headroom`` and ``hi * headroom`` on the log scale, ``(headroom -
        1)·|value|`` on the linear scale), so a single value still has room to move; it
        strictly contains ``[lo, hi]`` except for a zero value of an un-bucketed linear quantity.
    """
    scale = spec.scale if spec is not None else ("log" if lo > 0 else "linear")
    if scale == "log" and lo > 0:
        centre = math.sqrt(lo * hi)
        half = 0.5 * math.log(hi / lo) * headroom
        lo_w = min(centre * math.exp(-half), lo / headroom)
        hi_w = max(centre * math.exp(half), hi * headroom)
        step = math.log(LOG_STEP)
        return (
            LOG_STEP ** (math.ceil(math.log(lo_w) / step) - 1),
            LOG_STEP ** (math.floor(math.log(hi_w) / step) + 1),
        )
    centre = 0.5 * (lo + hi)
    half = 0.5 * (hi - lo) * headroom
    lo_w = min(centre - half, lo - (headroom - 1.0) * abs(lo))
    hi_w = max(centre + half, hi + (headroom - 1.0) * abs(hi))
    bucket = LINEAR_BUCKET.get(spec.quantity or "", 0.0) if spec is not None else 0.0
    if bucket <= 0:
        return lo_w, hi_w
    return (math.ceil(lo_w / bucket) - 1) * bucket, (math.floor(hi_w / bucket) + 1) * bucket


def _parts(tree: object) -> Mapping[str, object]:
    if isinstance(tree, _HasParts):
        return tree.parts()
    if isinstance(tree, Mapping):
        return {str(k): v for k, v in tree.items()}
    return {"": tree}


def _bound(tree: object) -> object:
    """Return a Chain with its acquisition's settings bound (other trees unchanged)."""
    bound = getattr(tree, "acquired", None)
    return bound(keep=True) if callable(bound) else tree  # fields keep their paths


def _covered(tree: object, own: Mapping[str, Node]) -> tuple[str, ...]:
    """Tree paths of the parts other than elements' own fields (children to skip in own parts)."""
    paths = getattr(tree, "paths", None)
    if not callable(paths):
        return ()
    table = paths()
    if not isinstance(table, dict):
        return ()
    return tuple(t for name, t in table.items() if name not in own)


def _own_parts(tree: object) -> dict[str, Node]:
    """Elements whose own (non-child) fields have envelope paths, such as ``imaging.width``."""
    own = getattr(tree, "own_parts", None)
    return dict(own()) if callable(own) else {}


def _presence_of(node: Node) -> Tensor | None:
    value = getattr(node, "presence", None)
    return value if isinstance(value, Tensor) else None


def _specs_by_path(like: object) -> dict[str, FieldSpec]:
    found: dict[str, FieldSpec] = {}
    for prefix, part in _parts(like).items():
        if isinstance(part, Node):
            for path, spec, _value in iter_leaves(part, prefix):
                if spec is not None:
                    found[path] = spec
    return found


def shape_affecting_values(tree: object, *, like: object = None) -> Iterator[Values]:
    """Yield the values of every shape-affecting field of a tree, by envelope path.

    Parameters
    ----------
    tree : object
        A node, an object with :meth:`_HasParts.parts` (a Chain, a Sample, a Microscope), or a
        mapping from paths to nodes or to whole tensors (per-image tables, §6.9).
    like : object, optional
        A template with the same paths (a Chain, Sample or Microscope). Table values are then
        read with the declarations of the template's fields, so they give the same entries as
        nodes would; without it, tables are read by name (``position`` has x, y, z).

    Yields
    ------
    Values
        One entry per enveloped field or event component.

    Raises
    ------
    EnvelopeError
        If a table does not match its presence table, or names no field of ``like``.
    """
    tree = _bound(tree)
    parts = dict(_parts(tree))
    own = _own_parts(tree)
    covered = _covered(tree, own)
    parts.update(own)
    tables = {p: v for p, v in parts.items() if isinstance(v, Tensor)}
    declared = _specs_by_path(like) if like is not None and tables else None
    for prefix, part in parts.items():
        if isinstance(part, Tensor):
            if declared is None:
                yield from _table_values(prefix, part, tables)
            else:
                yield from _declared_table_values(prefix, part, tables, declared)
            continue
        if not isinstance(part, Node):
            continue
        presence = _presence_of(part)
        present = None if presence is None else canonical(presence, _MASK_SPEC) > 0
        own_only = prefix in own
        for path, spec, value in iter_leaves(part, prefix):
            if spec is None or not spec.shape_affecting or value is None:
                continue
            if own_only and any(path.startswith(f"{c}.") for c in covered):
                continue  # children that are parts of their own (objective, camera, population)
            constant = not isinstance(value, Tensor)
            exact = torch.complex128 if isinstance(value, complex) else torch.float64
            values = canonical(value, spec, dtype=exact if constant else None, where=path)
            per_object = spec.role == "object" or (
                spec.role == "carrier" and spec.dims is not None and "N" in spec.dims
            )
            mask = present if per_object else None
            yield from _components(path, spec, values, mask, constant)


_MASK_SPEC = FieldSpec(kind="tensor", quantity="dimensionless", role="object", dtype="bool")


class Values(NamedTuple):
    """The values of one enveloped field, as :func:`shape_affecting_values` yields them.

    Parameters
    ----------
    path : str
        Envelope path, with the component name for event components.
    spec : FieldSpec or None
        The field's declaration; None for a bare tensor table.
    values : Tensor
        The values in canonical layout (B leading where the role has a batch axis).
    mask : Tensor or None
        Broadcastable presence mask of per-object values.
    constant : bool
        Whether the stored value is a Python number: a constant that cannot drift.
    """

    path: str
    spec: FieldSpec | None
    values: Tensor
    mask: Tensor | None
    constant: bool


def _components(
    path: str, spec: FieldSpec | None, values: Tensor, mask: Tensor | None, constant: bool
) -> Iterator[Values]:
    names = spec.components if spec is not None else None
    if names:
        for i, name in enumerate(names):
            yield Values(f"{path}.{name}", spec, values[..., i], mask, constant)
        return
    if mask is not None and spec is not None and spec.event:
        mask = mask.reshape(tuple(mask.shape) + (1,) * len(spec.event))
    yield Values(path, spec, values, mask, constant)


def _declared_table_values(
    path: str, values: Tensor, tables: Mapping[str, Tensor], declared: Mapping[str, FieldSpec]
) -> Iterator[Values]:
    spec = declared.get(path)
    if spec is None:
        near = difflib.get_close_matches(path, list(declared), n=1)
        hint = f"; did you mean {near[0]!r}?" if near else ""
        raise EnvelopeError(f"table {path!r} names no field of the template{hint}")
    head, _, name = path.rpartition(".")
    if name == "presence" or not spec.shape_affecting:
        return
    canon = canonical(values, spec, dtype=torch.float64, where=path)
    mask = None
    presence = tables.get(f"{head}.presence") if head else None
    per_object = spec.role == "object"
    if presence is not None and per_object:
        mask = canonical(presence, _MASK_SPEC, where=f"{head}.presence") > 0
    yield from _components(path, spec, canon, mask, False)


def _table_values(path: str, values: Tensor, tables: Mapping[str, Tensor]) -> Iterator[Values]:
    head, _, name = path.rpartition(".")
    if name == "presence":
        return
    presence = tables.get(f"{head}.presence") if head else None
    mask = None
    if presence is not None:
        mask = presence > 0
        extra = values.ndim - mask.ndim
        if extra < 0 or tuple(values.shape[: mask.ndim]) != tuple(mask.shape):
            msg = (
                f"{path} has shape {list(values.shape)} "
                f"but {head}.presence has {list(presence.shape)}"
            )
            raise EnvelopeError(msg)
    if name == "position" and values.shape[-1:] == (3,):
        for i, comp in enumerate(("x", "y", "z")):
            yield Values(f"{path}.{comp}", None, values[..., i], mask, False)
        return
    if mask is not None:
        mask = mask.reshape(tuple(mask.shape) + (1,) * (values.ndim - mask.ndim))
    yield Values(path, None, values, mask, False)


def _declared(path: str, inputs: Collection[str]) -> bool:
    return any(path == p or path.startswith(p + ".") for p in inputs)


def envelope_of(
    tree: object,
    *,
    headroom: float = HEADROOM,
    like: object = None,
    inputs: Collection[str] = (),
) -> Envelope:
    """Derive an envelope from example inputs, respecting presence.

    Reads tensor values, so on GPU inputs it synchronises with the host once; on CPU inputs
    (a front end's workers) it is free.

    Parameters
    ----------
    tree : object
        A node, a Chain, Sample or Microscope, or a mapping from paths to nodes or to whole
        per-image tables (``{"beads.position": pos, "beads.presence": presence}``).
    headroom : float, default 1.25
        Factor by which observed spans are widened (see :func:`widen`).
    like : object, optional
        Template whose field declarations are used to read tables (see
        :func:`shape_affecting_values`); give it so that tables and nodes derive the same
        entries.
    inputs : collection of str, optional
        Declared input paths (or path prefixes). Their Python-number values are widened like
        tensors instead of being treated as exact constants, since they will vary.

    Returns
    -------
    Envelope
        Entries with source ``"derived"``. Fields whose present values are all absent get no
        entry. Fields stored as Python numbers are constants and get exact entries, unless
        they are declared inputs.

    Raises
    ------
    EnvelopeError
        If a present value is NaN or infinite (the message names the field).
    """
    ranges: dict[str, tuple[float, float, FieldSpec | None, bool]] = {}
    for path, spec, values, mask, constant in shape_affecting_values(tree, like=like):
        v = values.detach()
        if v.is_complex():
            v = v.real
        v = v.to(torch.float64)
        if mask is not None:
            m, v = torch.broadcast_tensors(mask.to(v.device), v)
            if not bool(m.any()):
                continue
            v = v[m]
        if v.numel() == 0:
            continue
        lo_v, hi_v = float(v.min()), float(v.max())
        if not (math.isfinite(lo_v) and math.isfinite(hi_v)):
            bad = "NaN" if math.isnan(lo_v) or math.isnan(hi_v) else "infinite"
            msg = f"{path} has {bad} values, so no envelope can be derived for it"
            raise EnvelopeError(msg, fix="check the inputs; mark absent objects with presence")
        constant = constant and not _declared(path, inputs)
        if path in ranges:
            lo0, hi0, _, constant0 = ranges[path]
            lo_v, hi_v, constant = min(lo0, lo_v), max(hi0, hi_v), constant and constant0
        ranges[path] = (lo_v, hi_v, spec, constant)
    entries: dict[str, Entry] = {}
    for p, (lo, hi, spec, constant) in ranges.items():
        bounds = (lo, hi) if constant else widen(lo, hi, spec, headroom=headroom)
        entries[p] = Entry(*bounds, "derived", (lo, hi))
    return Envelope(entries)


def enveloped_paths(tree: object) -> set[str]:
    """Return every path an envelope entry may name for a tree.

    Parameters
    ----------
    tree : object
        A node, a Chain, Sample or Microscope, as for :func:`envelope_of`.

    Returns
    -------
    set of str
        Paths of the shape-affecting fields (and their components), including fields that are
        absent or hold no present value.
    """
    paths: set[str] = set()
    own = _own_parts(tree)
    covered = _covered(tree, own)
    for prefix, part in {**_parts(tree), **own}.items():
        if not isinstance(part, Node):
            continue
        for path, spec, _value in iter_leaves(part, prefix):
            if spec is None or not spec.shape_affecting:
                continue
            if prefix in own and any(path.startswith(f"{c}.") for c in covered):
                continue
            if spec.components:
                paths.update(f"{path}.{c}" for c in spec.components)
            else:
                paths.add(path)
    return paths


def check_paths(envelope: Mapping[str, object], tree: object) -> None:
    """Raise if an explicit envelope entry names a path that no shape-affecting field has.

    Parameters
    ----------
    envelope : Mapping
        Explicit entries by path.
    tree : object
        The template, as for :func:`envelope_of`.

    Raises
    ------
    EnvelopeError
        For the first unknown path, with the nearest valid ones.
    """
    known = enveloped_paths(tree)
    for path in envelope:
        if path in known:
            continue
        parts = sorted(p for p in known if p.startswith(path + "."))
        if parts:
            fix = f"give one entry per component: {', '.join(parts)}"
        else:
            near = difflib.get_close_matches(path, sorted(known), n=3)
            fix = f"did you mean {' or '.join(map(repr, near))}?" if near else None
        msg = f"envelope entry {path!r} names no shape-affecting field"
        raise EnvelopeError(msg, fix=fix)


def in_envelope(envelope: Envelope, tree: object, batch: int) -> Tensor:
    """Return per-image in-envelope flags, computed on the device without host synchronisation.

    Parameters
    ----------
    envelope : Envelope
        The envelope.
    tree : object
        The inputs, as for :func:`envelope_of`.
    batch : int
        The batch size B.

    Returns
    -------
    Tensor
        Boolean flags, shape [B]: whether every present value of every enveloped field of
        image i lies inside its interval. Fields without an entry are ignored.
    """
    flags: Tensor | None = None
    constants_inside = True
    device: torch.device | None = None
    for path, spec, values, mask, constant in shape_affecting_values(tree):
        interval = envelope.range(path)
        if interval is None:
            continue
        v = values.detach()
        if v.is_complex():
            v = v.real
        if constant:
            # A Python-number field is the same for every image; check it on the host.
            value = float(v)
            constants_inside = constants_inside and interval[0] <= value <= interval[1]
            continue
        ok = (v >= interval[0]) & (v <= interval[1])
        if mask is not None:
            m, ok = torch.broadcast_tensors(mask.to(ok.device), ok)
            ok = ok | ~m
        per_batch = spec is None or spec.role not in ("shared", "any", "carrier")
        if per_batch and ok.ndim > 0:
            per_image = ok.reshape(ok.shape[0], -1).all(dim=1)
        else:
            per_image = ok.all().reshape(1)
        if per_image.shape[0] == 1:
            per_image = per_image.expand(batch)
        if flags is not None and flags.device != per_image.device:
            target = per_image.device if per_image.device.type != "cpu" else flags.device
            flags, per_image = flags.to(target), per_image.to(target)
        flags = per_image if flags is None else flags & per_image
        device = flags.device
    if flags is None:
        flags = torch.ones(batch, dtype=torch.bool, device=device)
    return flags if constants_inside else torch.zeros_like(flags)
