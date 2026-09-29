"""Schemas as data: :func:`gx.schema_of <schema_of>` (§6.1, §10.2).

Every registered data object and element publishes its schema, which any front end can use to
construct, wrap or document it without adapters maintained in gradix.
"""

from __future__ import annotations

import dataclasses
import inspect
from typing import Any

from gradix.schema.base import Node
from gradix.schema.fields import specs
from gradix.schema.signature import type_name

__all__ = ["schema_of"]


def _json_default(value: object) -> object:
    if value is dataclasses.MISSING:
        return None
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, tuple):
        return [_json_default(v) for v in value]
    return repr(value)


def schema_of(obj: type[Node] | Node) -> dict[str, Any]:
    """Return the schema of a data object or element class as JSON-serialisable data.

    Parameters
    ----------
    obj : type or Node
        A node class or instance.

    Returns
    -------
    dict
        ``{"name", "class", "version", "doc", "fields"}``; each field entry holds its kind,
        quantity and unit, role, event shape, dimension pattern, shape-affecting flag,
        constraint, dtype kind, whether it is optional, its default and its description.
    """
    from gradix.units import QUANTITIES

    cls = obj if isinstance(obj, type) else type(obj)
    fields = []
    defaults = {f.name: f for f in dataclasses.fields(cls)}
    for name, spec in specs(cls).items():
        f = defaults[name]
        if f.default_factory is not dataclasses.MISSING:
            default: object = "<factory>"
        else:
            default = _json_default(f.default)
        entry: dict[str, object] = {
            "name": name,
            "kind": spec.kind,
            "optional": spec.optional,
            "required": f.default is dataclasses.MISSING
            and f.default_factory is dataclasses.MISSING,
            "default": default,
            "doc": spec.doc,
        }
        if spec.kind == "tensor":
            quantity = QUANTITIES[spec.quantity or "dimensionless"]
            entry.update(
                quantity=quantity.name,
                unit=quantity.unit,
                role=spec.role,
                event=list(spec.event),
                dims=None if spec.dims is None else [str(d) for d in spec.dims],
                shape_affecting=spec.shape_affecting,
                constraint=list(spec.constraint)
                if isinstance(spec.constraint, tuple)
                else spec.constraint,
                dtype=spec.dtype,
                components=None if spec.components is None else list(spec.components),
            )
        elif spec.kind == "static":
            entry["choices"] = (
                None if spec.choices is None else [_json_default(c) for c in spec.choices]
            )
        else:
            entry["container"] = spec.container
        fields.append(entry)
    doc = inspect.getdoc(cls) or ""
    return {
        "name": type_name(cls),
        "class": f"{cls.__module__}.{cls.__qualname__}",
        "version": cls.schema_version,
        "doc": doc.split("\n\n")[0],
        "fields": fields,
    }
