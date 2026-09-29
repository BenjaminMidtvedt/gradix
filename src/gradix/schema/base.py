"""The pytree base of data objects, elements and carriers.

A :class:`Node` is a frozen dataclass whose fields are declared with :func:`gradix.field`,
:func:`gradix.knob` or :func:`gradix.child`. It stores exactly the values it is given (§6.1):
construction checks structure (ranks, event shapes, dtype kinds, axis agreement between fields)
and never reads tensor values, so it costs no host synchronisation. Canonical views are made
inside each call (:mod:`gradix.schema.layout`), never at construction, so an element built once
around an ``nn.Parameter`` sees every optimiser update.

Nodes are edited by copy (:meth:`Node.replace`, :func:`gradix.tree.replace`) and form pytrees:
tensor fields are leaves, static fields are structure, and child fields nest.
"""

from __future__ import annotations

import copy
import dataclasses
import numbers
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from typing import Any, ClassVar, TypeVar, cast

import torch
from torch import Tensor

from gradix._core.errors import StructureError
from gradix._core.names import check_name
from gradix.schema.fields import FieldSpec, specs
from gradix.schema.layout import axis_sizes, check_dims, leading_axes, merge_sizes

__all__ = [
    "DataObject",
    "Leaf",
    "Node",
    "iter_leaves",
    "map_leaves",
    "node_axis_sizes",
    "rebuild",
    "subtree_axis_sizes",
    "tree_axis_sizes",
]

N = TypeVar("N", bound="Node")
T = TypeVar("T")

Leaf = Tensor | float | int | complex | bool | None
"""The value of a tensor field: a tensor, a Python number, or None when absent."""


def _dtype_kind(value: Tensor | numbers.Number) -> str:
    if isinstance(value, Tensor):
        if value.dtype == torch.bool:
            return "bool"
        if value.is_complex():
            return "complex"
        if value.is_floating_point():
            return "real"
        return "integer"
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, numbers.Integral):
        return "integer"
    if isinstance(value, numbers.Real):
        return "real"
    return "complex"


_ACCEPTS: dict[str, frozenset[str]] = {
    "real": frozenset({"real", "integer"}),
    "complex": frozenset({"real", "integer", "complex"}),
    "number": frozenset({"real", "integer", "complex"}),
    "integer": frozenset({"integer"}),
    "bool": frozenset({"bool"}),
}


def _check_tensor_field(owner: str, name: str, spec: FieldSpec, value: object) -> None:
    where = f"{owner}.{name}"
    if value is None:
        if not spec.optional:
            raise StructureError(f"{where} is required", fix=f"pass {name}=...")
        return
    if not isinstance(value, (Tensor, numbers.Number)):
        msg = f"{where}: expected a tensor or a Python number, got {type(value).__name__}"
        fix = "wrap sequences with torch.tensor(...)" if isinstance(value, (list, tuple)) else None
        raise StructureError(msg, fix=fix)
    kind = _dtype_kind(value)
    if kind not in _ACCEPTS[spec.dtype]:
        msg = f"{where}: dtype kind {kind!r} is not accepted; the field takes {spec.dtype!r} values"
        raise StructureError(msg)
    if spec.role in ("carrier", "any"):
        return
    shape = tuple(value.shape) if isinstance(value, Tensor) else ()
    leading_axes(spec, shape, where=where)


def _check_child_field(owner: str, name: str, spec: FieldSpec, value: object) -> object:
    """Validate a child field and return the value to store (containers are copied)."""
    where = f"{owner}.{name}"
    if value is None:
        if not spec.optional:
            raise StructureError(f"{where} is required", fix=f"pass {name}=...")
        return None
    if spec.container == "node":
        if not isinstance(value, Node):
            msg = f"{where}: expected a gradix data object or element, got {type(value).__name__}"
            raise StructureError(msg)
        return value
    if spec.container == "mapping":
        if not isinstance(value, Mapping):
            msg = f"{where}: expected a mapping from names to nodes, got {type(value).__name__}"
            raise StructureError(msg)
        out: dict[str, Node] = {}
        for key, item in value.items():
            check_name(key, what=f"{where}: key")  # keys become path segments (ADR-42)
            if not isinstance(item, Node):
                kind = type(item).__name__
                msg = f"{where}[{key!r}]: expected a data object or element, got {kind}"
                raise StructureError(msg)
            out[key] = item
        return out
    if not isinstance(value, (tuple, list)):
        msg = f"{where}: expected a tuple of nodes, got {type(value).__name__}"
        raise StructureError(msg)
    for i, item in enumerate(value):
        if not isinstance(item, Node):
            msg = f"{where}[{i}]: expected a data object or element, got {type(item).__name__}"
            raise StructureError(msg)
    return tuple(value)


def _check_static_field(owner: str, name: str, spec: FieldSpec, value: object) -> object:
    where = f"{owner}.{name}"
    if value is None and spec.optional:
        return None
    if isinstance(value, Tensor):
        msg = f"{where} is static structure and cannot hold a tensor"
        raise StructureError(msg, fix="pass a Python value")
    if isinstance(value, list):
        value = tuple(value)
    if spec.choices is not None and value not in spec.choices:
        msg = f"{where}: {value!r} is not one of {list(spec.choices)}"
        raise StructureError(msg)
    return value


def validate(node: Node) -> None:
    """Check a node's structure; called by every node's ``__post_init__``.

    Parameters
    ----------
    node : Node
        The node to check. Container child fields are replaced by copies (a list becomes a
        tuple, a mapping a dict), and static lists become tuples.

    Raises
    ------
    StructureError
        If a field has the wrong kind, rank, event shape or dtype kind, or two fields disagree
        on an axis size.
    TypeError
        If a dataclass field of the node lacks a schema declaration.
    """
    cls = type(node)
    owner = cls.__qualname__
    sizes: dict[str, int] = {}
    origins: dict[str, str] = {}
    carrier_dims: dict[str, tuple[Tensor, Sequence[str | int]]] = {}
    for name, spec in specs(cls).items():
        value = getattr(node, name)
        if spec.kind == "tensor":
            _check_tensor_field(owner, name, spec, value)
            if isinstance(value, Tensor):
                if spec.role == "carrier" and spec.dims is not None:
                    carrier_dims[name] = (value, spec.dims)
                elif spec.role not in ("carrier", "any"):
                    merge_sizes(
                        axis_sizes(spec, tuple(value.shape), where=f"{owner}.{name}"),
                        sizes,
                        where=owner,
                        source=name,
                        origins=origins,
                    )
        elif spec.kind == "child":
            stored = _check_child_field(owner, name, spec, value)
            if stored is not value:
                object.__setattr__(node, name, stored)
            for child_name, child in _children(name, stored):
                child_sizes = subtree_axis_sizes(child)
                shared = {"B", "T", "N"} if spec.container == "node" else {"B", "T"}
                merge_sizes(
                    {a: s for a, s in child_sizes.items() if a in shared},
                    sizes,
                    where=owner,
                    source=child_name,
                    origins=origins,
                )
        else:
            stored = _check_static_field(owner, name, spec, value)
            if stored is not value:
                object.__setattr__(node, name, stored)
    if carrier_dims:
        check_dims(carrier_dims, where=owner)


def _children(name: str, value: object) -> Iterator[tuple[str, Node]]:
    if isinstance(value, Node):
        yield name, value
    elif isinstance(value, Mapping):
        for key, item in value.items():
            if isinstance(item, Node):
                yield f"{name}.{key}", item
    elif isinstance(value, tuple):
        for i, item in enumerate(value):
            if isinstance(item, Node):
                yield f"{name}.{i}", item


def subtree_axis_sizes(node: Node) -> dict[str, int]:
    """Return the B, T and N sizes of a node and its single-node children (not collections).

    Parameters
    ----------
    node : Node
        The node.

    Returns
    -------
    dict of str to int
        Sizes of the axes found (1 where only broadcast sizes were seen). N comes from
        per-object fields only.
    """
    sizes: dict[str, int] = {}
    owner = type(node).__qualname__
    for name, spec in specs(type(node)).items():
        value = getattr(node, name)
        if spec.kind == "tensor" and isinstance(value, Tensor):
            if spec.role in ("carrier", "any"):
                continue
            found = axis_sizes(spec, tuple(value.shape), where=f"{owner}.{name}")
            keep = {"B", "T", "N"} if spec.role == "object" else {"B", "T"}
            merge_sizes(
                {a: s for a, s in found.items() if a in keep}, sizes, where=owner, source=name
            )
        elif spec.kind == "child" and spec.container == "node" and isinstance(value, Node):
            merge_sizes(subtree_axis_sizes(value), sizes, where=owner, source=name)
    return sizes


def _replace(node: N, changes: Mapping[str, object]) -> N:
    """Apply ``dataclasses.replace``, resetting init-only arguments that are not given.

    ``dataclasses.replace`` passes every init-only argument the current attribute value;
    classes whose init-only arguments are read back through a property (``Chain.camera``) list
    them in ``init_only`` so a copy starts from None instead.
    """
    reset = {name: None for name in type(node).init_only if name not in changes}
    return dataclasses.replace(node, **reset, **changes)


def _convert(value: Tensor, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Tensor:
    """Apply ``Tensor.to``, casting only compatible dtypes.

    Floats take a float target, complex tensors its complex counterpart; integer and bool
    tensors only move.
    """
    dtype = kwargs.get("dtype")
    rest = []
    for a in args:
        if isinstance(a, torch.dtype):
            dtype = a
        else:
            rest.append(a)
    kw = {k: v for k, v in kwargs.items() if k != "dtype"}
    if dtype is None:
        return value.to(*rest, **kw)
    if value.is_floating_point() and dtype.is_floating_point:
        return value.to(*rest, dtype=dtype, **kw)
    if value.is_complex() and dtype.is_floating_point:
        target = torch.complex128 if dtype == torch.float64 else torch.complex64
        return value.to(*rest, dtype=target, **kw)
    if value.is_complex() and dtype.is_complex:
        return value.to(*rest, dtype=dtype, **kw)
    return value.to(*rest, **kw)


def _register_pytree(cls: type) -> None:
    """Register a node class with torch's pytree, so torch.func transforms see its tensors."""
    try:
        from torch.utils import _pytree as pytree
    except ImportError:  # pragma: no cover - torch always ships it
        return

    def flatten(
        node: Node,
    ) -> tuple[list[object], tuple[type, tuple[tuple[str, bool, object], ...]]]:
        # Tensors and child nodes are children; Python numbers, None and static fields are
        # context. A node rebuilt by torch (possibly around placeholder leaves) keeps the layout
        # it was rebuilt with, so flatten after unflatten preserves the structure.
        recorded = node.__dict__.get("_gx_dynamic")
        children: list[object] = []
        layout: list[tuple[str, bool, object]] = []
        for name, spec in specs(type(node)).items():
            value = getattr(node, name)
            if recorded is not None:
                dynamic = name in recorded
            else:
                dynamic = (spec.kind == "tensor" and isinstance(value, Tensor)) or (
                    spec.kind == "child" and value is not None
                )
            layout.append((name, dynamic, None if dynamic else value))
            if dynamic:
                children.append(value)
        return children, (type(node), tuple(layout))

    def unflatten(
        children: Iterable[object], context: tuple[type, tuple[tuple[str, bool, object], ...]]
    ) -> Node:
        node_type, layout = context
        values = iter(children)
        new = cast("Node", object.__new__(node_type))
        for name, dynamic, static in layout:
            object.__setattr__(new, name, next(values) if dynamic else static)
        object.__setattr__(new, "_gx_dynamic", frozenset(n for n, d, _ in layout if d))
        return new

    try:
        pytree.register_pytree_node(cls, flatten, unflatten)
    except ValueError:
        pass  # already registered (a class re-created in the same process)


@dataclasses.dataclass(frozen=True, kw_only=True, eq=False)
class Node:
    """Base of every gradix data object, element and carrier.

    Subclasses are frozen dataclasses (``@dataclass(frozen=True, kw_only=True, eq=False)``) whose
    fields are all declared with :func:`gradix.field`, :func:`gradix.knob` or
    :func:`gradix.child`. Nodes compare by identity, as ``nn.Module`` does.
    """

    schema_version: ClassVar[int] = 1
    """Version of the class's schema; bumped on every schema change (§11.10)."""
    registry_name: ClassVar[str | None] = None
    """Registry name, set by the ``gx.register`` decorators."""
    init_only: ClassVar[tuple[str, ...]] = ()
    """Init-only arguments (``dataclasses.InitVar``) that copies reset to None."""

    def __init_subclass__(cls, **kwargs: Any) -> None:  # noqa: ANN401 - forwards class keywords
        super().__init_subclass__(**kwargs)
        _register_pytree(cls)

    def __post_init__(self) -> None:
        validate(self)

    @classmethod
    def schema(cls) -> dict[str, FieldSpec]:
        """Return the field declarations of this class.

        Returns
        -------
        dict of str to FieldSpec
            The declarations by field name, in declaration order.
        """
        return specs(cls)

    def replace(self: N, **changes: object) -> N:
        """Return a copy with some fields replaced; the copy is validated again.

        Parameters
        ----------
        **changes : object
            New field values by name.

        Returns
        -------
        Node
            A new node of the same class.
        """
        return _replace(self, changes)

    def to(self: N, *args: Any, **kwargs: Any) -> N:  # noqa: ANN401 - forwards Tensor.to
        """Return a copy with every tensor moved or cast by ``Tensor.to(*args, **kwargs)``.

        A dtype applies only where it fits: floating tensors take a floating dtype, complex
        tensors the matching complex dtype, and integer or bool tensors (keys, identities) only
        move. Shared sub-nodes stay shared.

        Parameters
        ----------
        *args : Any
            Positional arguments of :meth:`torch.Tensor.to`.
        **kwargs : Any
            Keyword arguments of :meth:`torch.Tensor.to`.

        Returns
        -------
        Node
            A new node; Python numbers are unchanged.
        """
        return map_leaves(
            lambda _path, _spec, v: _convert(v, args, kwargs) if isinstance(v, Tensor) else v,
            self,
            share=True,
        )

    def select(self: N, index: int | slice | Tensor) -> N:
        """Return the node for a subset of images: index every per-image field along B.

        Fields that do not vary over images (no B axis, or B = 1) are kept as they are.

        Parameters
        ----------
        index : int, slice or Tensor
            Images to keep. An int keeps the batch axis (``select(3)`` has B = 1).

        Returns
        -------
        Node
            A new node.
        """
        if isinstance(index, Tensor) and index.ndim == 0:
            index = int(index)
        if isinstance(index, int):
            index = slice(index, index + 1)

        def pick(path: str, spec: FieldSpec | None, value: Leaf) -> Leaf:
            if not isinstance(value, Tensor) or spec is None or value.ndim == 0:
                return value
            if spec.role == "carrier":
                dims = spec.dims or ()
                has_b = bool(dims) and str(dims[0]).split("|")[0] == "B"
            elif spec.role == "any":
                has_b = False
            else:
                has_b = "B" in leading_axes(spec, tuple(value.shape), where=path)
            if not has_b or value.shape[0] == 1:
                return value
            return value[index]

        return map_leaves(pick, self, share=True)


@dataclasses.dataclass(frozen=True, kw_only=True, eq=False)
class DataObject(Node):
    """Base of data objects: object sets, spectra, media, devices and other described values."""


def _field_names(node: Node, kind: str) -> Iterator[tuple[str, FieldSpec]]:
    for name, spec in specs(type(node)).items():
        if spec.kind == kind:
            yield name, spec


def _join(prefix: str, name: str) -> str:
    return f"{prefix}.{name}" if prefix else name


def iter_leaves(tree: object, prefix: str = "") -> Iterator[tuple[str, FieldSpec | None, Leaf]]:
    """Yield ``(path, spec, value)`` for every tensor field of a tree, depth first.

    Parameters
    ----------
    tree : object
        A node, a mapping or sequence of trees, or a bare tensor.
    prefix : str, optional
        Path prefix of ``tree``.

    Yields
    ------
    tuple of (str, FieldSpec or None, Leaf)
        Dotted path, the field's declaration (None for a bare tensor), and the stored value
        (possibly None or a Python number).
    """
    if isinstance(tree, Node):
        for name, spec in specs(type(tree)).items():
            value = getattr(tree, name)
            path = _join(prefix, name)
            if spec.kind == "tensor":
                yield path, spec, value
            elif spec.kind == "child" and value is not None:
                yield from iter_leaves(value, path)
    elif isinstance(tree, Mapping):
        for key, value in tree.items():
            yield from iter_leaves(value, _join(prefix, str(key)))
    elif isinstance(tree, (tuple, list)):
        for i, value in enumerate(tree):
            yield from iter_leaves(value, _join(prefix, str(i)))
    elif isinstance(tree, Tensor):
        yield prefix, None, tree


def rebuild(node: N, changes: Mapping[str, object], *, validate: bool = True) -> N:
    """Return a copy of a node with some fields replaced.

    Parameters
    ----------
    node : Node
        The node to copy.
    changes : Mapping[str, object]
        New field values by name.
    validate : bool, default True
        Run the structure checks on the copy. Internal transformations that cannot change
        structure (moving devices, torch function transforms) may skip them.

    Returns
    -------
    Node
        The copy.
    """
    if not changes:
        return node
    if validate:
        return _replace(node, changes)
    new = copy.copy(node)
    for name, value in changes.items():
        object.__setattr__(new, name, value)
    return new


def map_leaves(
    fn: Callable[[str, FieldSpec | None, Leaf], Leaf],
    tree: T,
    prefix: str = "",
    *,
    validate: bool = True,
    share: bool = False,
) -> T:
    """Return a copy of a tree with ``fn`` applied to every tensor field.

    Parameters
    ----------
    fn : callable
        ``fn(path, spec, value) -> new_value``, called for every tensor field, including absent
        (None) ones and Python numbers.
    tree : object
        A node, or a mapping, tuple or list of trees, or a bare tensor.
    prefix : str, optional
        Path prefix of ``tree``.
    validate : bool, default True
        Validate rebuilt nodes.
    share : bool, default False
        Map a node that appears several times once, so sharing survives. Only for functions
        that do not depend on the path.

    Returns
    -------
    object
        A tree of the same structure.
    """
    return _map_leaves(fn, tree, prefix, validate, {} if share else None)


def _map_leaves(
    fn: Callable[[str, FieldSpec | None, Leaf], Leaf],
    tree: T,
    prefix: str,
    validate: bool,
    memo: dict[int, object] | None,
) -> T:
    if isinstance(tree, Node) and memo is not None and id(tree) in memo:
        return cast("T", memo[id(tree)])
    if isinstance(tree, Node):
        changes: dict[str, object] = {}
        for name, spec in specs(type(tree)).items():
            value = getattr(tree, name)
            path = _join(prefix, name)
            if spec.kind == "tensor":
                new = fn(path, spec, value)
            elif spec.kind == "child" and value is not None:
                new = _map_leaves(fn, value, path, validate, memo)
            else:
                continue
            if new is not value:
                changes[name] = new
        out = rebuild(tree, changes, validate=validate)
        if memo is not None:
            memo[id(tree)] = out
        return out
    if isinstance(tree, Mapping):
        items = {
            k: _map_leaves(fn, v, _join(prefix, str(k)), validate, memo) for k, v in tree.items()
        }
        return cast("T", type(tree)(items) if isinstance(tree, dict) else items)
    if isinstance(tree, (tuple, list)):
        seq = [
            _map_leaves(fn, v, _join(prefix, str(i)), validate, memo) for i, v in enumerate(tree)
        ]
        return cast("T", type(tree)(seq))
    if isinstance(tree, Tensor):
        return cast("T", fn(prefix, None, tree))
    return tree


def node_axis_sizes(node: Node, *, where: str | None = None) -> dict[str, int]:
    """Return the sizes of the role axes (B, T, N, L, …) of a node's own tensor fields.

    Parameters
    ----------
    node : Node
        The node.
    where : str, optional
        Name for error messages; defaults to the class name.

    Returns
    -------
    dict of str to int
        Size of each axis, 1 where no field varies along it.
    """
    owner = where or type(node).__qualname__
    sizes: dict[str, int] = {}
    for name, spec in _field_names(node, "tensor"):
        value = getattr(node, name)
        if isinstance(value, Tensor) and spec.role not in ("carrier", "any"):
            merge_sizes(axis_sizes(spec, tuple(value.shape)), sizes, where=owner, source=name)
    return sizes


def tree_axis_sizes(tree: object) -> dict[str, int]:
    """Return the batch and frame sizes (B, T) shared by every tensor field of a tree.

    Parameters
    ----------
    tree : object
        A node, or a mapping or sequence of nodes.

    Returns
    -------
    dict of str to int
        ``{"B": …, "T": …}`` where some field has the axis; 1 where no field varies along it.
        Object counts (N) and wavelength bins (L) may differ between parts and are excluded.

    Raises
    ------
    StructureError
        If two fields disagree on an axis (neither being 1), naming both.
    """
    sizes: dict[str, int] = {}
    origins: dict[str, str] = {}
    for path, spec, value in iter_leaves(tree):
        if not isinstance(value, Tensor) or spec is None or spec.role in ("carrier", "any"):
            continue
        found = axis_sizes(spec, tuple(value.shape), where=path)
        shared = {axis: size for axis, size in found.items() if axis in ("B", "T")}
        merge_sizes(shared, sizes, where="inputs", source=path, origins=origins)
    return sizes
