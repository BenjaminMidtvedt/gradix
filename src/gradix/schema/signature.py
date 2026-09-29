"""Structure signatures: the static part of a tree, used in cache keys (§6.1).

A signature covers node types, static field values (knobs, camera shape, acquisition sizes),
which optional fields are present, and each tensor field's dtype and role pattern: whether each
capacity axis (B and N; B for pixel maps) varies or is 1, and the exact structural sizes (T, L,
event shapes). It excludes tensor values and the sizes of capacity axes. Python numbers record
their kind (real, complex, integer, bool). Modules held in static fields are keyed by their
architecture (repr and parameter shapes), so equal architectures share a signature across
processes; other callables are keyed by identity, so a signature that holds one is stable only
within a process. Unordered mappings (populations) are sorted by key.
"""

from __future__ import annotations

import dataclasses
import enum
import hashlib
from collections.abc import Iterator, Mapping

import torch
from torch import Tensor

from gradix.schema.base import Node
from gradix.schema.fields import FieldSpec, specs
from gradix.schema.layout import ROLE_RULES, leading_axes

__all__ = [
    "Signature",
    "describe",
    "leaf_issue",
    "leaf_pattern",
    "signature",
    "type_name",
    "widened_axes",
]

_CAPACITY_AXES = frozenset({"B", "N"})
"""Capacity axes of carrier dimension patterns (role fields use their role's rule)."""


def _capacity(spec: FieldSpec | None) -> frozenset[str]:
    rule = ROLE_RULES.get(spec.role) if spec is not None else None
    return rule.capacity if rule is not None else _CAPACITY_AXES


def type_name(node_type: type) -> str:
    """Return a stable name for a node class: its registry name, or module and qualname.

    Parameters
    ----------
    node_type : type
        The class.

    Returns
    -------
    str
        The name. Only a registry name set on the class itself counts: an unregistered
        subclass of a registered element is a different element.
    """
    registered = node_type.__dict__.get("registry_name")
    if isinstance(registered, str) and registered:
        return registered
    return f"{node_type.__module__}.{node_type.__qualname__}"


def _static_repr(value: object) -> object:
    if value is None or isinstance(value, str):
        return value
    if isinstance(value, bool):
        return ("bool", value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return ("float", value)
    if isinstance(value, complex):
        return ("complex", value.real, value.imag)
    if isinstance(value, enum.Enum):
        return (type(value).__qualname__, value.value)
    if isinstance(value, (tuple, list)):
        return tuple(_static_repr(v) for v in value)
    if isinstance(value, Mapping):
        return tuple(sorted((str(k), _static_repr(v)) for k, v in value.items()))
    if isinstance(value, type):
        return f"{value.__module__}.{value.__qualname__}"
    if isinstance(value, Node):
        return (type_name(type(value)), signature(value).digest)
    if dataclasses.is_dataclass(value):
        return (type(value).__qualname__, repr(value))
    if isinstance(value, Tensor):
        return ("tensor", str(value.dtype), tuple(value.shape), id(value))
    if isinstance(value, torch.nn.Module):
        # architecture, not weights: weights are values (learned), the architecture is structure
        shapes = tuple((n, tuple(p.shape), str(p.dtype)) for n, p in value.named_parameters())
        return (f"{type(value).__module__}.{type(value).__qualname__}", repr(value), shapes)
    # Functions and other objects: structure we cannot read, so identity.
    name = getattr(value, "__qualname__", type(value).__qualname__)
    return (f"{getattr(value, '__module__', '?')}.{name}", repr(value), id(value))


def _number_kind(value: object, spec: FieldSpec | None = None) -> str:
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        # a Python int in a real field is a real number (it canonicalises to one)
        real = spec is not None and spec.dtype in ("real", "number")
        return "real" if real else "integer"
    if isinstance(value, complex):
        return "complex"
    return "real"


def leaf_pattern(spec: FieldSpec | None, value: object) -> tuple[object, ...]:
    """Return the structural pattern of one tensor-field value.

    Parameters
    ----------
    spec : FieldSpec or None
        The field's declaration.
    value : object
        The stored value.

    Returns
    -------
    tuple
        ``("absent",)``, ``("number", kind)`` or ``("tensor", dtype, axes, event)`` where
        ``axes`` records, for each role axis of the value's form, ``1`` or ``"n"`` (capacity
        axes: B and N, or B for pixel maps) or the exact size (structural axes such as T and L).
    """
    if value is None:
        return ("absent",)
    if not isinstance(value, Tensor):
        return ("number", _number_kind(value, spec))
    dtype = str(value.dtype).removeprefix("torch.")
    shape = tuple(int(s) for s in value.shape)
    if spec is None or spec.role in ("any",):
        return ("tensor", dtype, shape)
    if spec.role == "carrier":
        dims = spec.dims or ()
        pattern = []
        for size, token in zip(shape, dims, strict=False):
            label = str(token).split("|")[0]
            pattern.append((label, (1 if size == 1 else "n") if label in _CAPACITY_AXES else size))
        return ("tensor", dtype, tuple(pattern))
    capacity = _capacity(spec)
    axes = leading_axes(spec, shape)
    lead = tuple(
        (axis, (1 if size == 1 else "n") if axis in capacity else size)
        for axis, size in zip(axes, shape, strict=False)
    )
    return ("tensor", dtype, lead, shape[len(axes) :])


def _axes(entry: tuple[object, ...]) -> dict[str, object]:
    """Role axes of a leaf entry's form, with their recorded sizes (1, "n" or exact)."""
    if len(entry) < 4 or entry[1] != "tensor" or not isinstance(entry[3], tuple):
        return {}
    return {
        str(item[0]): item[1] for item in entry[3] if isinstance(item, tuple) and len(item) == 2
    }


def _varying(entry: tuple[object, ...]) -> dict[str, object]:
    """Axes along which a leaf entry varies (size other than 1), with their sizes."""
    return {axis: size for axis, size in _axes(entry).items() if size != 1}


def _is_complex(entry: tuple[object, ...]) -> bool:
    """Whether a number or tensor leaf entry (with its path) holds complex values."""
    kind = entry[2] if len(entry) > 2 else ""
    return kind == "complex" if entry[1] == "number" else str(kind).startswith("complex")


def widened_axes(template: tuple[object, ...], value: tuple[object, ...]) -> set[str]:
    """Return the axes along which a value varies but the template's value does not.

    Varying along N is allowed where the template holds a tensor with an N axis (its object
    count is a capacity); a shared or per-image template value that becomes per-object widens.

    Parameters
    ----------
    template : tuple
        The template's signature entry for the leaf: its path, then its :func:`leaf_pattern`.
    value : tuple
        The value's entry, in the same form.

    Returns
    -------
    set of str
        The widened axes; empty when the value is no wider.
    """
    had, has = _varying(template), _varying(value)
    wider = set(has) - set(had)
    if "N" in _axes(template):
        wider.discard("N")
    return wider


def describe(entry: tuple[object, ...]) -> str:
    """Return a short human-readable description of a signature entry (without its path).

    Parameters
    ----------
    entry : tuple
        A signature entry without its path: a :func:`leaf_pattern`, or a node, static, mapping
        or sequence entry.

    Returns
    -------
    str
        Such as ``"a float32 tensor [B, N, 3]"`` or ``"a Python real number"``.
    """
    kind = entry[0] if entry else "?"
    if kind == "absent":
        return "absent"
    if kind == "number":
        return f"a Python {entry[1]} number" if len(entry) > 1 else "a Python number"
    if kind == "tensor":
        dtype = entry[1]
        lead = entry[2] if len(entry) > 2 and isinstance(entry[2], tuple) else ()
        event = entry[3] if len(entry) > 3 and isinstance(entry[3], tuple) else ()
        dims = [
            (str(item[0]) if item[1] == "n" else str(item[1]))
            if isinstance(item, tuple)
            else str(item)
            for item in lead
        ]
        dims += [str(d) for d in event]
        return f"a {dtype} tensor [{', '.join(dims)}]"
    if kind == "node":
        return f"a {entry[1]} (schema v{entry[2]})"
    if kind == "static":
        return repr(entry[1])
    if kind == "mapping":
        keys = entry[1] if isinstance(entry[1], tuple) else ()
        return f"a mapping with keys {list(keys)}"
    if kind == "sequence":
        return f"a sequence of {entry[1]}"
    return repr(entry)


def leaf_issue(
    template: tuple[object, ...], value: tuple[object, ...], *, declared: bool
) -> str | None:
    """Explain why a leaf entry cannot stand in for a template's, or return None if it can.

    A value is compatible when it keeps the template's dtype (a Python number fits any
    floating dtype and vice versa), its event shape and structural sizes (T, L, …), and varies
    along no axis the template does not (object counts excepted), unless the path is declared.

    Parameters
    ----------
    template : tuple
        The template's signature entry for the leaf.
    value : tuple
        The new value's signature entry.
    declared : bool
        Whether the leaf is a declared input (any role width allowed).

    Returns
    -------
    str or None
        The reason, or None when compatible.
    """
    kind_t, kind_v = template[1], value[1]
    if kind_t == "absent" or kind_v == "absent":
        return None if kind_t == kind_v else "an optional field is present in only one of them"
    if kind_t not in ("number", "tensor") or kind_v not in ("number", "tensor"):
        if repr(template[1:]) == repr(value[1:]):
            return None
        return f"{describe(value[1:])} instead of {describe(template[1:])}"
    if kind_t == "tensor" and kind_v == "tensor":
        if template[2] != value[2]:
            return f"dtype {value[2]} instead of {template[2]}"
        if template[4:] != value[4:]:
            return f"event shape {value[4:]} instead of {template[4:]}"
    if _is_complex(template) != _is_complex(value):
        return f"{describe(value[1:])} instead of {describe(template[1:])} (real vs complex)"
    for axis, size in _axes(value).items():
        before = _axes(template).get(axis, 1)
        if isinstance(size, int) and size != 1 and isinstance(before, int) and before != size:
            if before != 1:
                return (
                    f"axis {axis} has size {size} instead of {before}; {axis} is structure, "
                    "so build a Pipeline for it"
                )
            if not declared:
                msg = f"varies along {axis} (size {size}), which the template does not"
                return f"{msg}; {axis} is structure, so build a Pipeline for it"
    wider = {a for a in widened_axes(template, value) if not isinstance(_axes(value)[a], int)}
    if wider and not declared:
        return f"varies along {sorted(wider)}, which the template does not (declare it in inputs=)"
    return None


def _entries(tree: object, prefix: str, *, ordered: bool = False) -> Iterator[tuple[object, ...]]:
    if isinstance(tree, Node):
        cls = type(tree)
        yield (prefix, "node", type_name(cls), cls.schema_version)
        for name, spec in specs(cls).items():
            value = getattr(tree, name)
            path = f"{prefix}.{name}" if prefix else name
            if spec.kind == "tensor":
                yield (path, *leaf_pattern(spec, value))
            elif spec.kind == "static":
                yield (path, "static", _static_repr(value))
            elif value is None:
                yield (path, "absent")
            else:
                yield from _entries(value, path, ordered=spec.ordered)
    elif isinstance(tree, Mapping):
        items = list(tree.items())
        if not ordered:
            items.sort(key=lambda kv: str(kv[0]))
        yield (prefix, "mapping", tuple(str(k) for k, _ in items))
        for key, value in items:
            yield from _entries(value, f"{prefix}.{key}" if prefix else str(key))
    elif isinstance(tree, (tuple, list)):
        yield (prefix, "sequence", len(tree))
        for i, value in enumerate(tree):
            yield from _entries(value, f"{prefix}.{i}" if prefix else str(i))
    elif isinstance(tree, Tensor):
        yield (prefix, *leaf_pattern(None, tree))
    else:
        yield (prefix, "value", _static_repr(tree))


@dataclasses.dataclass(frozen=True)
class Signature:
    """A structure signature.

    Parameters
    ----------
    entries : tuple of tuple
        The structural entries, in traversal order.
    """

    entries: tuple[tuple[object, ...], ...]

    @property
    def digest(self) -> str:
        """A 16-hex-digit SHA-256 digest of the entries, stable across runs.

        Returns
        -------
        str
            The digest.
        """
        return hashlib.sha256(repr(self.entries).encode()).hexdigest()[:16]

    def incompatibility(self, other: Signature, *, declared: tuple[str, ...] = ()) -> str | None:
        """Explain why a tree with signature ``other`` cannot replace this one, or return None.

        Structure (types, static values, containers) must be equal; leaves follow
        :func:`leaf_issue`. Capacities (B, N) may shrink or grow.

        Parameters
        ----------
        other : Signature
            The candidate's signature.
        declared : tuple of str, default ()
            Tree-path prefixes of declared inputs.

        Returns
        -------
        str or None
            The first reason, naming the path, or None when compatible.
        """
        if len(self.entries) != len(other.entries):
            return self.first_difference(other, ignore_capacity=True) or "the trees differ in size"
        for a, b in zip(self.entries, other.entries, strict=True):
            if a[0] != b[0]:
                return f"at {a[0] or '<root>'}: the trees differ in layout"
            path = str(a[0])
            is_leaf = a[1] in ("number", "tensor", "absent") or b[1] in (
                "number",
                "tensor",
                "absent",
            )
            if not is_leaf:
                if repr(a) != repr(b):
                    return f"at {path or '<root>'}: {describe(b[1:])} instead of {describe(a[1:])}"
                continue
            is_declared = any(path == d or path.startswith(f"{d}.") for d in declared)
            issue = leaf_issue(a, b, declared=is_declared)
            if issue is not None:
                return f"at {path}: {issue}"
        return None

    @property
    def structure_digest(self) -> str:
        """A digest that ignores capacities (whether B and N are 1 or larger).

        Returns
        -------
        str
            16 hex digits.
        """
        entries = tuple(_strip(e) for e in self.entries)
        return hashlib.sha256(repr(entries).encode()).hexdigest()[:16]

    def first_difference(self, other: Signature, *, ignore_capacity: bool = False) -> str | None:
        """Describe the first entry where two signatures differ, or None if they are equal.

        Parameters
        ----------
        other : Signature
            The other signature.
        ignore_capacity : bool, default False
            Ignore whether B and N are 1 or larger (capacities, not structure).

        Returns
        -------
        str or None
            A human-readable description naming the path.
        """
        for a, b in zip(self.entries, other.entries, strict=False):
            if repr(a) != repr(b) and (not ignore_capacity or repr(_strip(a)) != repr(_strip(b))):
                where = a[0] if a[0] == b[0] else f"{a[0]} / {b[0]}"
                return f"at {where or '<root>'}: {describe(a[1:])} vs {describe(b[1:])}"
        if len(self.entries) != len(other.entries):
            longer = self.entries if len(self.entries) > len(other.entries) else other.entries
            extra = longer[min(len(self.entries), len(other.entries))]
            return f"at {extra[0] or '<root>'}: present in only one of them"
        return None


def _strip(entry: tuple[object, ...]) -> tuple[object, ...]:
    """Drop the capacity-axis patterns (recorded as 1 or "n") of a tensor entry."""
    if len(entry) < 4 or entry[1] != "tensor" or not isinstance(entry[3], tuple):
        return entry
    axes = tuple(
        a for a in entry[3] if not (isinstance(a, tuple) and len(a) == 2 and a[1] in (1, "n"))
    )
    return (*entry[:3], axes, *entry[4:])


def signature(tree: object) -> Signature:
    """Return the structure signature of a tree.

    Parameters
    ----------
    tree : object
        A node, or a mapping or sequence of nodes.

    Returns
    -------
    Signature
        The signature; equal signatures mean interchangeable structure.
    """
    return Signature(tuple(_entries(tree, "")))
