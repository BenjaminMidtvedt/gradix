"""Roles, input ranks and the canonical per-call layout (§6.1).

A tensor field's *role* says how it varies. Each role accepts a few input forms, told apart by
rank alone, so the layout never depends on sizes (the B == N trap of revision 1 cannot occur):

========== ================================== ===========================================
role       input forms (event dims omitted)   canonical form inside a call
========== ================================== ===========================================
shared     ``[]``                             ``[]``
image      ``[]``, ``[B]``                    ``[B|1]``
frame      ``[]``, ``[B, T]``                 ``[B|1, T|1]``
setting    ``[]``, ``[B]``, ``[B, T]``        ``[B|1, T|1]``
object     ``[]``, ``[B, N]``, ``[B, T, N]``  ``[B|1, T|1, N|1]``
wavelength ``[]``, ``[L]``, ``[B, L]``        ``[B|1, L]``
layer      ``[]``, ``[K]``, ``[B, K]``        ``[B|1, K]``
pixels     ``[]``, ``[B]``, ``[H, W]``,       ``[B|1, H|1, W|1]``
           ``[B, H, W]``
========== ================================== ===========================================

Per-object and per-frame fields always carry an explicit batch axis: ``[1, N]`` for one image,
``[B, 1]`` for one value per image. A bare ``[B]``, ``[N]`` or ``[T]`` vector is rejected,
because right-aligned broadcasting would silently mix images, frames and objects when their
counts coincide. Per-image camera values are ``[B]`` or ``[B, 1, 1]``; a 2-d value is a map.

Canonical forms are per role; to combine fields of different roles, :func:`align` places each on
a common axis frame such as ``("B", "T", "N")``. Carrier fields declare a full dimension pattern
such as ``("B|1", "A|1", "N", 3)``, checked by :func:`check_dims`. Plugins add roles with
:func:`register_role`.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence

import torch
from torch import Tensor

from gradix._core.errors import StructureError
from gradix.schema.fields import FieldSpec

__all__ = [
    "ROLE_RULES",
    "RoleRule",
    "align",
    "axis_sizes",
    "canonical",
    "canonical_shape",
    "check_dims",
    "common_device",
    "leading_axes",
    "merge_sizes",
    "register_role",
]


@dataclasses.dataclass(frozen=True)
class RoleRule:
    """Input forms and canonical axes of a role.

    Parameters
    ----------
    axes : tuple of str
        Canonical leading axes, in order.
    forms : Mapping[int, tuple of str]
        For each accepted number of leading (non-event) dims, the axes they are.
    hint : str
        Human-readable accepted forms, for error messages.
    capacity : frozenset of str, default {"B", "N"}
        Axes whose sizes are capacities (they may shrink between calls) rather than structure;
        structure signatures record only whether they are 1.
    """

    axes: tuple[str, ...]
    forms: Mapping[int, tuple[str, ...]]
    hint: str
    capacity: frozenset[str] = frozenset({"B", "N"})


ROLE_RULES: dict[str, RoleRule] = {
    "shared": RoleRule((), {0: ()}, "a scalar (or an event-shaped tensor)"),
    "image": RoleRule(("B",), {0: (), 1: ("B",)}, "a scalar or [B]"),
    "frame": RoleRule(
        ("B", "T"),
        {0: (), 2: ("B", "T")},
        "a scalar or [B, T] (use [B, 1] for per-image values, [1, T] for per-frame values)",
    ),
    "setting": RoleRule(
        ("B", "T"),
        {0: (), 1: ("B",), 2: ("B", "T")},
        "a scalar, per image [B], or per image and frame [B, T] (use [1, T] for per-frame "
        "values shared by every image)",
    ),
    "object": RoleRule(
        ("B", "T", "N"),
        {0: (), 2: ("B", "N"), 3: ("B", "T", "N")},
        "a scalar, [B, N] or [B, T, N] (use [1, N] for one image, [B, 1] for per-image values)",
    ),
    "wavelength": RoleRule(
        ("B", "L"), {0: (), 1: ("L",), 2: ("B", "L")}, "a scalar, [L] or [B, L]"
    ),
    "layer": RoleRule(("B", "K"), {0: (), 1: ("K",), 2: ("B", "K")}, "a scalar, [K] or [B, K]"),
    "pixels": RoleRule(
        ("B", "H", "W"),
        {0: (), 1: ("B",), 2: ("H", "W"), 3: ("B", "H", "W")},
        "a scalar, [B] (or [B, 1, 1]) per image, a map [H, W] or per-image maps [B, H, W]",
    ),
}
"""The role table. ``carrier`` and ``any`` fields have no role rule."""


def register_role(name: str, rule: RoleRule, *, override: bool = False) -> RoleRule:
    """Register a role so that schema fields may declare it.

    Parameters
    ----------
    name : str
        Role name, such as ``"channel"``.
    rule : RoleRule
        Its input forms and canonical axes.
    override : bool, default False
        Replace an existing role instead of raising.

    Returns
    -------
    RoleRule
        The registered rule.

    Raises
    ------
    ValueError
        If the name is taken (or reserved) and ``override`` is False.
    """
    from gradix.schema.fields import ROLES

    if name in ("carrier", "any") or (name in ROLE_RULES and not override):
        msg = f"role {name!r} is already defined; pass override=True to replace it"
        raise ValueError(msg)
    ROLE_RULES[name] = rule
    if name not in ROLES:
        ROLES.append(name)
    return rule


def _shape_str(shape: Sequence[int]) -> str:
    return "[" + ", ".join(str(int(s)) for s in shape) + "]"


def leading_axes(spec: FieldSpec, shape: Sequence[int], *, where: str = "") -> tuple[str, ...]:
    """Name the leading (non-event) axes of a value of a role field.

    Parameters
    ----------
    spec : FieldSpec
        The field's declaration; its role must be in :data:`ROLE_RULES`.
    shape : sequence of int
        Shape of the value (``()`` for a Python number).
    where : str, optional
        Field name for error messages, such as ``"Emitters.photons"``.

    Returns
    -------
    tuple of str
        The axes of the leading dims, such as ``("B", "N")``.

    Raises
    ------
    StructureError
        If the rank is not an accepted form, or the event dims do not match.
    """
    rule = ROLE_RULES.get(spec.role)
    if rule is None:
        msg = f"{where}: role {spec.role!r} has no rank rule"
        raise StructureError(msg)
    n_event = len(spec.event)
    n_lead = len(shape) - n_event
    if n_lead not in rule.forms:
        event = f" followed by the event shape {list(spec.event)}" if spec.event else ""
        msg = (
            f"{where}: got shape {_shape_str(shape)}; a {spec.role} field takes {rule.hint}{event}"
        )
        fix = None
        if spec.role == "object" and n_lead == 1:
            fix = "add the batch axis: x[None] for one image, or x[:, None] for per-image values"
        raise StructureError(msg, fix=fix)
    for size, want in zip(shape[n_lead:], spec.event, strict=True):
        if want != -1 and int(size) != want:
            event = list(spec.event)
            msg = f"{where}: got shape {_shape_str(shape)}; the event shape must be {event}"
            raise StructureError(msg)
    return rule.forms[n_lead]


def axis_sizes(spec: FieldSpec, shape: Sequence[int], *, where: str = "") -> dict[str, int]:
    """Return the sizes of the named leading axes of a role field's value.

    Parameters
    ----------
    spec : FieldSpec
        The field's declaration.
    shape : sequence of int
        Shape of the value.
    where : str, optional
        Field name for error messages.

    Returns
    -------
    dict of str to int
        Size of each leading axis present, such as ``{"B": 4, "N": 8}``.
    """
    axes = leading_axes(spec, shape, where=where)
    return {axis: int(size) for axis, size in zip(axes, shape, strict=False)}


def canonical_shape(spec: FieldSpec, shape: Sequence[int], *, where: str = "") -> tuple[int, ...]:
    """Return the canonical shape of a role field's value: every role axis present.

    Parameters
    ----------
    spec : FieldSpec
        The field's declaration.
    shape : sequence of int
        Shape of the value.
    where : str, optional
        Field name for error messages.

    Returns
    -------
    tuple of int
        The canonical shape, with size 1 on the axes the value does not vary along, followed by
        the event shape.
    """
    rule = ROLE_RULES[spec.role]
    sizes = axis_sizes(spec, shape, where=where)
    event = tuple(int(s) for s in shape[len(shape) - len(spec.event) :]) if spec.event else ()
    return tuple(sizes.get(axis, 1) for axis in rule.axes) + event


def canonical(
    value: Tensor | float | complex | bool,
    spec: FieldSpec,
    *,
    dtype: torch.dtype | None = None,
    device: torch.device | str | None = None,
    where: str = "",
) -> Tensor:
    """Return a value in the canonical layout of its role, as a view where possible.

    Parameters
    ----------
    value : Tensor or Python number
        The stored value of a role field.
    spec : FieldSpec
        The field's declaration.
    dtype : torch.dtype, optional
        Dtype of the result. Python numbers become tensors of this dtype; tensors are cast.
    device : torch.device or str, optional
        Device of the result; tensors elsewhere are moved (a differentiable copy).
    where : str, optional
        Field name for error messages.

    Returns
    -------
    Tensor
        The value reshaped to its canonical shape (see :data:`ROLE_RULES`).
    """
    if not isinstance(value, Tensor) and dtype is None and spec.dtype in ("real", "number"):
        # a Python int in a real field is a real number, not an integer tensor
        if isinstance(value, int) and not isinstance(value, bool):
            dtype = torch.get_default_dtype()
    if isinstance(value, Tensor):
        t = value
    else:
        # torch.full fills on the device without a host-to-device copy (no sync, capturable)
        kind = dtype if dtype is not None else torch.as_tensor(value).dtype
        t = torch.full((), value, dtype=kind, device=device)
    if device is not None and t.device != torch.device(device):
        t = t.to(device)
    if dtype is not None and t.dtype != dtype:
        t = t.to(dtype)
    if spec.role in ("carrier", "any"):
        return t
    return t.reshape(canonical_shape(spec, tuple(t.shape), where=where))


def align(
    value: Tensor | float | complex | bool,
    spec: FieldSpec,
    frame: Sequence[str],
    *,
    dtype: torch.dtype | None = None,
    device: torch.device | str | None = None,
    where: str = "",
) -> Tensor:
    """Place a role field's value on a common axis frame, to combine fields of different roles.

    Every axis of the role must appear in ``frame``; the frame's other axes become size 1. For
    example an image field ``[B]`` on the frame ``("B", "T", "N")`` becomes ``[B|1, 1, 1]``, and
    an object field ``[B, N]`` becomes ``[B|1, 1, N]``, so the two broadcast image by image.

    Parameters
    ----------
    value : Tensor or Python number
        The stored value.
    spec : FieldSpec
        The field's declaration.
    frame : sequence of str
        The target axes, in order.
    dtype : torch.dtype, optional
        Dtype of the result.
    device : torch.device or str, optional
        Device of the result.
    where : str, optional
        Field name for error messages.

    Returns
    -------
    Tensor
        ``[*frame sizes, *event]``.

    Raises
    ------
    StructureError
        If the role has an axis that ``frame`` lacks.
    """
    t = canonical(value, spec, dtype=dtype, device=device, where=where)
    rule = ROLE_RULES.get(spec.role)
    if rule is None:
        raise StructureError(f"{where}: {spec.role} fields have no axes to align")
    missing = [a for a in rule.axes if a not in frame]
    if missing:
        msg = f"{where}: axis {missing} of a {spec.role} field is not in the frame {tuple(frame)}"
        raise StructureError(msg)
    n_axes = len(rule.axes)
    order = sorted(range(n_axes), key=lambda i: list(frame).index(rule.axes[i]))
    if order != list(range(n_axes)):
        t = t.permute(*order, *range(n_axes, t.ndim))
    sizes = {rule.axes[i]: t.shape[k] for k, i in enumerate(order)}
    event = tuple(t.shape[n_axes:])
    return t.reshape(*(sizes.get(a, 1) for a in frame), *event)


def merge_sizes(
    sizes: Mapping[str, int],
    into: dict[str, int],
    *,
    where: str = "",
    source: str = "",
    origins: dict[str, str] | None = None,
) -> None:
    """Merge axis sizes into a running table, checking that sizes other than 1 agree.

    Parameters
    ----------
    sizes : Mapping[str, int]
        Sizes of one value's axes.
    into : dict of str to int
        The running table; updated in place.
    where : str, optional
        Owner name for error messages.
    source : str, optional
        Name of the value the sizes come from.
    origins : dict of str to str, optional
        Which value set each size in ``into``; updated in place and used in error messages.

    Raises
    ------
    StructureError
        If an axis has two different sizes, neither of them 1.
    """
    for axis, size in sizes.items():
        if size == 1:
            into.setdefault(axis, 1)
            continue
        have = into.get(axis, 1)
        if have not in (1, size):
            other = origins.get(axis, "another field") if origins is not None else "another field"
            msg = f"{where}: axis {axis} is {size} in {source} but {have} in {other}"
            raise StructureError(msg, fix=f"give every field the same {axis} (or 1 to broadcast)")
        into[axis] = size
        if origins is not None and have == 1:
            origins[axis] = source


def check_dims(
    values: Mapping[str, tuple[Tensor, Sequence[str | int]]],
    *,
    where: str = "",
) -> dict[str, int]:
    """Check tensors against dimension patterns and return the resolved named sizes.

    A pattern token is an int (an exact size), a name such as ``"N"`` (the same size in every
    field that names it), a name with ``"|1"`` such as ``"B|1"`` (that size, or 1 to broadcast),
    or ``"*"`` (any size).

    Parameters
    ----------
    values : Mapping[str, tuple of (Tensor, sequence of str or int)]
        For each field name, its tensor and its dimension pattern.
    where : str, optional
        Owner name for error messages.

    Returns
    -------
    dict of str to int
        The size of every named dimension (1 where only broadcast sizes were seen).

    Raises
    ------
    StructureError
        If a rank, an exact size or a named size disagrees.
    """
    exact: dict[str, dict[int, str]] = {}
    flexible: dict[str, dict[int, str]] = {}
    for name, (t, dims) in values.items():
        if t.ndim != len(dims):
            msg = f"{where}.{name}: got shape {_shape_str(t.shape)}; expected {list(dims)}"
            raise StructureError(msg)
        for size, token in zip(t.shape, dims, strict=True):
            if isinstance(token, int):
                if int(size) != token:
                    msg = f"{where}.{name}: got shape {_shape_str(t.shape)}; expected {list(dims)}"
                    raise StructureError(msg)
                continue
            if token == "*":
                continue
            label, _, one = token.partition("|")
            table = flexible if one else exact
            if one and int(size) == 1:
                table.setdefault(label, {})
                continue
            table.setdefault(label, {})[int(size)] = name
    resolved: dict[str, int] = {}
    for label in {*exact, *flexible}:
        seen = {**flexible.get(label, {}), **exact.get(label, {})}
        if len(seen) > 1:
            detail = ", ".join(f"{n}: {s}" for s, n in seen.items())
            msg = f"{where}: dimension {label} disagrees between fields ({detail})"
            raise StructureError(msg)
        resolved[label] = next(iter(seen), 1)
    return resolved


def common_device(*values: object) -> torch.device:
    """Return the device a call computes on: the first non-CPU device among tensor inputs.

    Tensors on other devices are moved there by :func:`canonical` (a differentiable copy);
    keep inputs on the render device to avoid the copies.

    Parameters
    ----------
    *values : object
        Tensors or other values; non-tensors are ignored.

    Returns
    -------
    torch.device
        The device (CPU when every tensor is on the CPU or there is none).

    Raises
    ------
    StructureError
        If tensors live on two different non-CPU devices.
    """
    found: torch.device | None = None
    for v in values:
        if isinstance(v, Tensor) and v.device.type != "cpu":
            if found is not None and v.device != found:
                msg = f"inputs live on {found} and {v.device}"
                raise StructureError(msg, fix="move every input to one device with .to(device)")
            found = v.device
    return found if found is not None else torch.device("cpu")
