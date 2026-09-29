"""Pytree utilities over data objects, elements, Chains and containers (``gx.tree``).

Paths are dotted: a node field by name, a mapping entry by key, a tuple entry by index, such as
``"emitters.beads.position"`` or ``"imaging.objective.pupil.0.coeffs"`` (§6.1).

Examples
--------
>>> import torch, gradix as gx
>>> beads = gx.Emitters(
...     position=torch.zeros(1, 2, 3), photons=100.0, emission=gx.Spectrum.line(0.6)
... )
>>> sorted(gx.tree.tensors(beads))
['position']
>>> gx.tree.replace(beads, {"photons": 200.0}).photons
200.0
"""

from __future__ import annotations

import builtins
from collections.abc import Callable, Mapping
from typing import Any, TypeVar, cast

from torch import Tensor

from gradix._core.errors import StructureError
from gradix.schema.base import Leaf, Node, iter_leaves, map_leaves, rebuild
from gradix.schema.fields import FieldSpec, specs

__all__ = ["get", "leaves", "map", "parameters", "paths", "replace", "tensors", "to"]

T = TypeVar("T")


def leaves(tree: object) -> dict[str, Leaf]:
    """Return every tensor field of a tree by path, including Python numbers and absent (None) ones.

    Parameters
    ----------
    tree : object
        A node, or a mapping or sequence of nodes.

    Returns
    -------
    dict of str to Leaf
        Values by dotted path, in traversal order.
    """
    return {path: value for path, _spec, value in iter_leaves(tree)}


def tensors(tree: object) -> dict[str, Tensor]:
    """Return every tensor stored in a tree, by path.

    Parameters
    ----------
    tree : object
        A node, or a mapping or sequence of nodes.

    Returns
    -------
    dict of str to Tensor
        Tensors by dotted path.
    """
    return {path: value for path, _spec, value in iter_leaves(tree) if isinstance(value, Tensor)}


def parameters(tree: object) -> dict[str, Tensor]:
    """Return the learnable tensors of a tree, by path, ready for an optimiser.

    Parameters
    ----------
    tree : object
        A node, or a mapping or sequence of nodes.

    Returns
    -------
    dict of str to Tensor
        Leaf tensors with ``requires_grad`` by dotted path. Derived tensors (views and results
        of operations) are left out, and a tensor shared by several fields appears once, under
        its first path.
    """
    out: dict[str, Tensor] = {}
    seen: set[int] = set()
    for path, t in tensors(tree).items():
        if t.requires_grad and t.is_leaf and id(t) not in seen:
            seen.add(id(t))
            out[path] = t
    return out


def paths(tree: object) -> list[str]:
    """Return the path of every tensor field of a tree.

    Parameters
    ----------
    tree : object
        A node, or a mapping or sequence of nodes.

    Returns
    -------
    list of str
        Dotted paths in traversal order.
    """
    return [path for path, _spec, _value in iter_leaves(tree)]


def map(fn: Callable[[Tensor], Tensor], tree: T, *, numbers: bool = False) -> T:
    """Return a copy of a tree with ``fn`` applied to every tensor.

    Parameters
    ----------
    fn : callable
        ``fn(tensor) -> tensor``.
    tree : T
        A node, or a mapping or sequence of nodes.
    numbers : bool, default False
        Also apply ``fn`` to Python-number leaves (as 0-d tensors).

    Returns
    -------
    T
        The mapped copy, validated.
    """
    import torch

    def apply(_path: str, _spec: FieldSpec | None, value: Leaf) -> Leaf:
        if isinstance(value, Tensor):
            return fn(value)
        if numbers and value is not None:
            return fn(torch.as_tensor(value))
        return value

    return map_leaves(apply, tree)


def to(tree: T, *args: Any, **kwargs: Any) -> T:  # noqa: ANN401 - forwards Tensor.to
    """Return a copy of a tree with every tensor moved or cast by ``Tensor.to``.

    Parameters
    ----------
    tree : T
        A node, or a mapping or sequence of nodes.
    *args : Any
        Positional arguments of :meth:`torch.Tensor.to`.
    **kwargs : Any
        Keyword arguments of :meth:`torch.Tensor.to`.

    Returns
    -------
    T
        The moved copy.
    """
    return map(lambda t: t.to(*args, **kwargs), tree)


def _split(path: str) -> list[str]:
    if not path or any(not part for part in path.split(".")):
        raise StructureError(f"invalid tree path {path!r}")
    return path.split(".")


def get(tree: object, path: str) -> object:
    """Return the value at a path: a tensor field, a static field, a child node or a mapping entry.

    Parameters
    ----------
    tree : object
        A node, or a mapping or sequence of nodes.
    path : str
        Dotted path.

    Returns
    -------
    object
        The value.

    Raises
    ------
    StructureError
        If the path does not exist; the message lists the valid names at the failing step.
    """
    current = tree
    walked: list[str] = []
    for part in _split(path):
        current = _step(current, part, ".".join(walked))
        walked.append(part)
    return current


def _step(current: object, part: str, where: str) -> object:
    if isinstance(current, Node):
        names = specs(type(current))
        if part not in names:
            msg = f"{where or '<root>'} ({type(current).__name__}) has no field {part!r}"
            raise StructureError(msg, fix=f"fields: {builtins.list(names)}")
        return getattr(current, part)
    if isinstance(current, Mapping):
        if part not in current:
            msg = f"{where or '<root>'} has no entry {part!r}"
            raise StructureError(msg, fix=f"entries: {sorted(current)}")
        return current[part]
    if isinstance(current, (tuple, builtins.list)):
        try:
            return current[int(part)]
        except (ValueError, IndexError):
            msg = f"{where or '<root>'} has no index {part!r} (length {len(current)})"
            raise StructureError(msg) from None
    msg = f"cannot descend into {type(current).__name__} at {where!r}"
    raise StructureError(msg)


def replace(tree: T, changes: Mapping[str, object]) -> T:
    """Return a copy of a tree with the values at some paths replaced.

    Any field may be replaced (tensor, static or child); the copies are validated, so a change
    that breaks the structure raises.

    Parameters
    ----------
    tree : T
        A node, or a mapping or sequence of nodes.
    changes : Mapping[str, object]
        New values by dotted path, such as ``{"imaging.objective.focus": 0.5}``.

    Returns
    -------
    T
        The edited copy; the input is unchanged.
    """
    nested: dict[str, object] = {}
    for path, value in changes.items():
        _insert(nested, _split(path), value)
    return cast("T", _apply(tree, nested, ""))


_LEAF = object()


def _insert(nested: dict[str, object], parts: list[str], value: object) -> None:
    head, rest = parts[0], parts[1:]
    if not rest:
        if isinstance(nested.get(head), dict):
            raise StructureError(f"{head!r} is replaced both whole and in part")
        nested[head] = (_LEAF, value)
        return
    sub = nested.setdefault(head, {})
    if not isinstance(sub, dict):
        raise StructureError(f"{head!r} is replaced both whole and in part")
    _insert(sub, rest, value)


def _apply(current: object, nested: Mapping[str, object], where: str) -> object:
    """Apply every change below one node at once, so only the final state is validated."""
    updates: dict[str, object] = {}
    for head, change in nested.items():
        child = _step(current, head, where)
        if isinstance(change, tuple) and len(change) == 2 and change[0] is _LEAF:
            updates[head] = change[1]
        elif isinstance(change, dict):
            updates[head] = _apply(child, change, f"{where}.{head}" if where else head)
    if isinstance(current, Node):
        return rebuild(current, updates)
    if isinstance(current, Mapping):
        return {**current, **updates}
    if isinstance(current, (tuple, builtins.list)):
        seq = builtins.list(current)
        for head, value in updates.items():
            seq[int(head)] = value
        return tuple(seq) if isinstance(current, tuple) else seq
    raise StructureError(f"cannot replace inside {type(current).__name__}")
