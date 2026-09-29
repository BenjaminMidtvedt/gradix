"""Outputs of a render: specs (``gx.out``) and the dict-like :class:`Output` (§6.5).

``outputs=`` names what a render computes: a tuple of reserved names (``"image"``,
``"expected"``), or a mapping from names to specs (``{"mu": "expected", "pos":
gx.labels.Positions("beads")}``). Only requested outputs are computed. Every returned tensor is
an ordinary tensor the caller owns.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterator, Mapping
from typing import Any, Union

from torch import Tensor

from gradix._core.errors import StructureError
from gradix.schema.base import Node

__all__ = ["RESERVED", "Expected", "Image", "Output", "OutputSpec", "normalize_outputs"]


@dataclasses.dataclass(frozen=True)
class Image:
    """The camera image in the camera's unit; needs a key when the camera has noise."""

    def to_json(self) -> dict[str, Any]:
        """Return a JSON-serialisable description.

        Returns
        -------
        dict
            ``{"type": "image"}``.
        """
        return {"type": "image"}


@dataclasses.dataclass(frozen=True)
class Expected:
    """The noise-free expected image in the camera's unit."""

    def to_json(self) -> dict[str, Any]:
        """Return a JSON-serialisable description.

        Returns
        -------
        dict
            ``{"type": "expected"}``.
        """
        return {"type": "expected"}


OutputSpec = Union[Image, Expected, Node]  # noqa: UP007 - a runtime alias used in isinstance
"""An output spec: :class:`Image`, :class:`Expected`, or a label renderer (``gx.labels.*``)."""

RESERVED: dict[str, OutputSpec] = {"image": Image(), "expected": Expected()}
"""Reserved output names and their specs."""


def normalize_outputs(outputs: object) -> dict[str, OutputSpec]:
    """Normalise an ``outputs=`` argument into a mapping from names to specs.

    Parameters
    ----------
    outputs : str, tuple of str, or Mapping[str, object]
        Reserved names, or names mapped to specs (a spec may be a reserved name string).

    Returns
    -------
    dict of str to OutputSpec
        The specs by output name.

    Raises
    ------
    StructureError
        If a name or spec is unknown.
    """
    if isinstance(outputs, str):
        outputs = (outputs,)
    if isinstance(outputs, Mapping):
        items = list(outputs.items())
    elif isinstance(outputs, (tuple, list)):
        items = [(name, name) for name in outputs]
    else:
        raise StructureError(f"outputs must be names or a mapping, got {type(outputs).__name__}")
    out: dict[str, OutputSpec] = {}
    for name, spec in items:
        if not isinstance(name, str) or not name:
            raise StructureError(f"output names must be non-empty strings, got {name!r}")
        if isinstance(spec, str):
            if spec not in RESERVED:
                msg = f"unknown output {spec!r}; reserved names are {sorted(RESERVED)}"
                raise StructureError(msg, fix="pass a spec such as gx.labels.Positions('beads')")
            spec = RESERVED[spec]
        from gradix.labels.positions import Label

        if not isinstance(spec, (Image, Expected, Label)):
            kind = type(spec).__name__
            msg = f"output {name!r}: a {kind} is not an output spec"
            fix = "use 'image', 'expected', gx.out.Image(...), gx.out.Expected(...) or a label"
            raise StructureError(msg, fix=fix)
        out[name] = spec
    if not out:
        raise StructureError("request at least one output")
    return out


class Output(Mapping[str, Tensor]):
    """The result of a render: output tensors by name, plus ``meta``.

    Parameters
    ----------
    values : Mapping[str, Tensor]
        Output tensors by name.
    meta : Mapping[str, object]
        Pipeline hash, chunk sizes, key kind, per-image in-envelope flags and diagnostics.
    """

    def __init__(self, values: Mapping[str, Tensor], meta: Mapping[str, object]) -> None:
        self._values = dict(values)
        self.meta: dict[str, Any] = dict(meta)

    def __getitem__(self, name: str) -> Tensor:
        try:
            return self._values[name]
        except KeyError:
            msg = f"no output {name!r}; this render produced {sorted(self._values)}"
            raise KeyError(msg) from None

    def __iter__(self) -> Iterator[str]:
        return iter(self._values)

    def __len__(self) -> int:
        return len(self._values)

    def __repr__(self) -> str:
        shapes = ", ".join(f"{k}: {list(v.shape)}" for k, v in self._values.items())
        return f"Output({shapes})"
