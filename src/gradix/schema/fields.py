"""Field declarations: :func:`field`, :func:`knob` and :func:`child` (``gx.field`` and friends).

Every field of a data object, element or carrier is declared with one of three constructors:

- :func:`field` declares a *tensor field*: a leaf that holds a tensor or a Python number, with a
  physical quantity, a role (how it varies over images, frames, objects and wavelengths), an event
  shape, and whether it is shape-affecting (whether it enters the envelope, §6.4).
- :func:`knob` declares a *static field*: structure such as an element's options, a camera
  shape or an enum choice. Static fields are part of the structure signature and never hold
  tensors.
- :func:`child` declares a *child field*: a nested data object or element, a mapping from names
  to them, or a tuple of them.

The declarations are stored in the dataclass field metadata under the key ``"gradix"`` as a
:class:`FieldSpec`; :func:`specs` reads them back for a class.
"""

from __future__ import annotations

import dataclasses
import functools
from collections.abc import Callable
from typing import Any, Literal, TypeAlias

from gradix.units import QUANTITIES

__all__ = [
    "CONSTRAINTS",
    "DTYPES",
    "ROLES",
    "Constraint",
    "FieldSpec",
    "child",
    "field",
    "knob",
    "spec_of",
    "specs",
]

METADATA_KEY = "gradix"

ROLES: list[str] = [
    "shared",
    "image",
    "frame",
    "setting",
    "object",
    "wavelength",
    "layer",
    "pixels",
    "carrier",
    "any",
]
"""Recognised roles; :mod:`gradix.schema.layout` defines the input ranks of each, and
:func:`gradix.schema.layout.register_role` adds more."""

DTYPES: tuple[str, ...] = ("real", "complex", "number", "integer", "bool")
"""Recognised dtype kinds. ``"number"`` accepts real or complex values."""

CONSTRAINTS: tuple[str, ...] = ("positive", "nonnegative", "unit_interval", "finite")
"""Named value constraints; a ``(lo, hi)`` tuple is an inclusive interval."""

Constraint: TypeAlias = "str | tuple[float, float]"

Kind: TypeAlias = Literal["tensor", "static", "child"]
Container: TypeAlias = Literal["node", "mapping", "tuple"]


@dataclasses.dataclass(frozen=True)
class FieldSpec:
    """The schema entry of one field.

    Parameters
    ----------
    kind : {"tensor", "static", "child"}
        Tensor leaf, static structure, or nested node(s).
    quantity : str or None
        Physical quantity of a tensor field (a key of :data:`gradix.units.QUANTITIES`).
    role : str
        How a tensor field varies: one of :data:`ROLES`.
    event : tuple of int
        Trailing event shape of a tensor field, such as ``(3,)`` for positions; ``-1`` matches
        any size.
    dims : tuple of str or int, or None
        Full dimension pattern of a carrier field, such as ``("B|1", "A|1", "N", 3)``.
    shape_affecting : bool
        Whether the field's values enter the envelope (§6.4).
    constraint : str or tuple of float, or None
        Value constraint, checked on CPU inputs only (§6.1).
    scale : {"linear", "log"}
        How the envelope widens and buckets the field's range.
    components : tuple of str or None
        Names of the event components, such as ``("x", "y", "z")``; used by envelope paths.
    dtype : str
        Accepted dtype kind: one of :data:`DTYPES`.
    optional : bool
        Whether ``None`` is accepted (it means "absent").
    choices : tuple or None
        Accepted values of a static field.
    container : {"node", "mapping", "tuple"}
        Shape of a child field.
    ordered : bool
        Whether a mapping child's order is meaningful (stages applied in sequence). Unordered
        mappings (populations) are compared and traversed in sorted key order.
    doc : str
        One-line description.
    """

    kind: Kind
    quantity: str | None = None
    role: str = "shared"
    event: tuple[int, ...] = ()
    dims: tuple[str | int, ...] | None = None
    shape_affecting: bool = False
    constraint: Constraint | None = None
    scale: Literal["linear", "log"] = "linear"
    components: tuple[str, ...] | None = None
    dtype: str = "real"
    optional: bool = False
    choices: tuple[object, ...] | None = None
    container: Container = "node"
    ordered: bool = False
    doc: str = ""

    def __post_init__(self) -> None:
        if self.kind == "tensor":
            if self.quantity not in QUANTITIES:
                msg = f"unknown quantity {self.quantity!r}; known: {sorted(QUANTITIES)}"
                raise ValueError(msg)
            if self.role not in ROLES:
                msg = f"unknown role {self.role!r}; known: {ROLES}"
                raise ValueError(msg)
            if self.dtype not in DTYPES:
                msg = f"unknown dtype kind {self.dtype!r}; known: {DTYPES}"
                raise ValueError(msg)
            if (self.role == "carrier") != (self.dims is not None):
                msg = "carrier fields need dims=..., and only carrier fields take dims"
                raise ValueError(msg)
            if self.components is not None and len(self.components) != (
                self.event[-1] if self.event else 0
            ):
                msg = f"components {self.components} do not match the event shape {self.event}"
                raise ValueError(msg)
            constraint = self.constraint
            if isinstance(constraint, str) and constraint not in CONSTRAINTS:
                msg = f"unknown constraint {constraint!r}; known: {CONSTRAINTS} or (lo, hi)"
                raise ValueError(msg)


def _dataclass_field(
    spec: FieldSpec,
    default: object,
    default_factory: Callable[[], object] | None,
) -> Any:  # noqa: ANN401 - field specifiers must type as Any to fit every annotation
    metadata = {METADATA_KEY: spec}
    if default_factory is not None:
        return dataclasses.field(default_factory=default_factory, metadata=metadata)
    if default is not dataclasses.MISSING:
        return dataclasses.field(default=default, metadata=metadata)
    return dataclasses.field(metadata=metadata)


def field(
    *,
    quantity: str,
    role: str = "shared",
    event: tuple[int, ...] = (),
    dims: tuple[str | int, ...] | None = None,
    shape_affecting: bool = False,
    constraint: Constraint | None = None,
    scale: Literal["linear", "log"] | None = None,
    components: tuple[str, ...] | None = None,
    dtype: str = "real",
    default: object = dataclasses.MISSING,
    default_factory: Callable[[], object] | None = None,
    doc: str = "",
) -> Any:  # noqa: ANN401 - field specifiers must type as Any to fit every annotation
    """Declare a tensor field of a data object, element or carrier.

    Parameters
    ----------
    quantity : str
        Physical quantity (a key of :data:`gradix.units.QUANTITIES`), such as ``"length"``.
    role : str, default "shared"
        How the field varies (§6.1): ``"shared"``, ``"image"`` ``[B]``, ``"frame"`` ``[B, T]``,
        ``"object"`` ``[B, T|1, N]``, ``"wavelength"`` ``[B|1, L]``, ``"layer"``, ``"pixels"``
        (a constant, ``[B]``, a map ``[H, W]`` or ``[B, H, W]``), ``"carrier"`` (fixed
        ``dims``) or ``"any"``.
    event : tuple of int, default ()
        Trailing event shape, such as ``(3,)`` for positions; ``-1`` matches any size.
    dims : tuple of str or int, optional
        Full dimension pattern of a carrier field, such as ``("B|1", "A|1", "N", 3)``.
    shape_affecting : bool, default False
        Whether the field's values enter the envelope (§6.4).
    constraint : str or tuple of float, optional
        ``"positive"``, ``"nonnegative"``, ``"unit_interval"``, ``"finite"`` or ``(lo, hi)``;
        checked on CPU inputs only.
    scale : {"linear", "log"}, optional
        How the envelope widens the field's range; ``"log"`` by default for positive fields.
    components : tuple of str, optional
        Names of the event components, such as ``("x", "y", "z")``.
    dtype : str, default "real"
        Accepted dtype kind: ``"real"``, ``"complex"``, ``"number"``, ``"integer"`` or ``"bool"``.
    default : object, optional
        Default value; ``None`` makes the field optional.
    default_factory : callable, optional
        Zero-argument factory of the default value.
    doc : str, default ""
        One-line description, used by schema exports.

    Returns
    -------
    dataclasses.Field
        The dataclass field carrying the :class:`FieldSpec` in its metadata.
    """
    if scale is None:
        scale = "log" if constraint == "positive" else "linear"
    spec = FieldSpec(
        kind="tensor",
        quantity=quantity,
        role=role,
        event=tuple(event),
        dims=None if dims is None else tuple(dims),
        shape_affecting=shape_affecting,
        constraint=constraint,
        scale=scale,
        components=components,
        dtype=dtype,
        optional=default is None,
        doc=doc,
    )
    return _dataclass_field(spec, default, default_factory)


def knob(
    default: object = dataclasses.MISSING,
    *,
    choices: tuple[object, ...] | None = None,
    doc: str = "",
) -> Any:  # noqa: ANN401 - field specifiers must type as Any to fit every annotation
    """Declare a static field: an option or other structure that never holds a tensor.

    Static fields are part of the structure signature (§6.1). Lists are stored as tuples.

    Parameters
    ----------
    default : object, optional
        Default value; ``None`` makes the field optional.
    choices : tuple, optional
        Accepted values.
    doc : str, default ""
        One-line description.

    Returns
    -------
    dataclasses.Field
        The dataclass field carrying the :class:`FieldSpec` in its metadata.
    """
    spec = FieldSpec(
        kind="static",
        optional=default is None,
        choices=None if choices is None else tuple(choices),
        doc=doc,
    )
    return _dataclass_field(spec, default, None)


def child(
    *,
    container: Container = "node",
    default: object = dataclasses.MISSING,
    default_factory: Callable[[], object] | None = None,
    ordered: bool = False,
    doc: str = "",
) -> Any:  # noqa: ANN401 - field specifiers must type as Any to fit every annotation
    """Declare a child field: a nested data object or element, or a collection of them.

    Parameters
    ----------
    container : {"node", "mapping", "tuple"}, default "node"
        A single node, a mapping from names to nodes, or a tuple of nodes.
    default : object, optional
        Default value; ``None`` makes the field optional.
    default_factory : callable, optional
        Zero-argument factory of the default value.
    ordered : bool, default False
        Whether a mapping's order is meaningful (a sequence of stages); unordered mappings
        (populations) are compared in sorted key order.
    doc : str, default ""
        One-line description.

    Returns
    -------
    dataclasses.Field
        The dataclass field carrying the :class:`FieldSpec` in its metadata.
    """
    spec = FieldSpec(
        kind="child", container=container, optional=default is None, ordered=ordered, doc=doc
    )
    return _dataclass_field(spec, default, default_factory)


def spec_of(f: dataclasses.Field[Any]) -> FieldSpec | None:
    """Return the :class:`FieldSpec` of a dataclass field, or None if it has none.

    Parameters
    ----------
    f : dataclasses.Field
        A field of a dataclass.

    Returns
    -------
    FieldSpec or None
        The declaration made with :func:`field`, :func:`knob` or :func:`child`.
    """
    spec = f.metadata.get(METADATA_KEY)
    return spec if isinstance(spec, FieldSpec) else None


@functools.cache
def specs(cls: type) -> dict[str, FieldSpec]:
    """Return the field declarations of a dataclass, in declaration order.

    Parameters
    ----------
    cls : type
        A dataclass whose fields are all declared with :func:`field`, :func:`knob` or
        :func:`child`.

    Returns
    -------
    dict of str to FieldSpec
        The declarations by field name.

    Raises
    ------
    TypeError
        If ``cls`` is not a dataclass, or a field lacks a declaration.
    """
    if not dataclasses.is_dataclass(cls):
        msg = f"{cls.__qualname__} is not a dataclass; decorate it with @dataclass(frozen=True)"
        raise TypeError(msg)
    out: dict[str, FieldSpec] = {}
    for f in dataclasses.fields(cls):
        spec = spec_of(f)
        if spec is None:
            msg = (
                f"field {cls.__qualname__}.{f.name} has no schema; declare it with gx.field(...) "
                "for tensors, gx.knob(...) for options or gx.child(...) for nested nodes"
            )
            raise TypeError(msg)
        out[f.name] = spec
    return out
