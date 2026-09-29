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
from gradix._core.names import check_name
from gradix.schema.base import Node

__all__ = [
    "LAYOUTS",
    "NORMALIZE",
    "RESERVED",
    "Expected",
    "Field",
    "Image",
    "Output",
    "OutputSpec",
    "normalize_outputs",
]


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


NORMALIZE = ("background", "incident", "none")
"""How :class:`Field` normalises the complex field."""

LAYOUTS = ("complex", "re_im", "phase", "amplitude")
"""How :class:`Field` lays the complex field out."""

SAMPLINGS = ("centre", "mean")
"""How :class:`Field` takes one value per camera pixel."""


@dataclasses.dataclass(frozen=True)
class Field:
    """The complex image field on the camera grid: what a reconstruction should recover.

    Coherent Chains only, for one incoherent mode and one wavelength bin. The field is formed
    on the detection samples; each camera pixel takes the sample at its centre (the
    band-limited field that a reconstruction recovers) or the complex mean over the pixel.

    Parameters
    ----------
    normalize : {"background", "incident", "none"}, default "background"
        ``"background"`` divides by the unscattered field at each pixel, so an empty field is 1
        and a scatterer contributes E_s/E_b; ``"incident"`` divides by the amplitude √I₀ of the
        illumination (for epi geometries, where no background reaches the camera);
        ``"none"`` keeps √(photons/µm²).
    layout : {"complex", "re_im", "phase", "amplitude"}, default "complex"
        Complex ``[B, A, H, W]``; real ``[B, A, H, W, 2]``; ``arg`` of the field in rad; or
        ``|E|``.
    sampling : {"centre", "mean"}, default "centre"
        The value at each pixel's centre, or the complex mean over the pixel.

    Raises
    ------
    StructureError
        If ``normalize`` or ``layout`` is unknown.
    """

    normalize: str = "background"
    layout: str = "complex"
    sampling: str = "centre"

    def __post_init__(self) -> None:
        if self.normalize not in NORMALIZE:
            raise StructureError(f"unknown normalize={self.normalize!r}", fix=f"one of {NORMALIZE}")
        if self.layout not in LAYOUTS:
            raise StructureError(f"unknown layout={self.layout!r}", fix=f"one of {LAYOUTS}")
        if self.sampling not in SAMPLINGS:
            raise StructureError(f"unknown sampling={self.sampling!r}", fix=f"one of {SAMPLINGS}")

    def to_json(self) -> dict[str, Any]:
        """Return a JSON-serialisable description.

        Returns
        -------
        dict
            ``{"type": "field", "normalize": …, "layout": …, "sampling": …}``.
        """
        return {
            "type": "field",
            "normalize": self.normalize,
            "layout": self.layout,
            "sampling": self.sampling,
        }


OutputSpec = Union[Image, Expected, Field, Node]  # noqa: UP007 - a runtime alias in isinstance
"""An output spec: :class:`Image`, :class:`Expected`, :class:`Field`, or a label renderer."""

RESERVED: dict[str, OutputSpec] = {"image": Image(), "expected": Expected(), "field": Field()}
"""Reserved output names and their specs."""

ATTRIBUTES: frozenset[str] = frozenset({"get", "items", "keys", "meta", "values"})
""":class:`Output`'s own attributes, which output names may not shadow (ADR-42)."""


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
        check_name(name, what="output", reserved=ATTRIBUTES)
        if isinstance(spec, str):
            if spec not in RESERVED:
                msg = f"unknown output {spec!r}; reserved names are {sorted(RESERVED)}"
                raise StructureError(msg, fix="pass a spec such as gx.labels.Positions('beads')")
            spec = RESERVED[spec]
        from gradix.labels.positions import Label

        if not isinstance(spec, (Image, Expected, Field, Label)):
            kind = type(spec).__name__
            msg = f"output {name!r}: a {kind} is not an output spec"
            fix = "use 'image', 'expected', 'field', gx.out.Field(...) or a label"
            raise StructureError(msg, fix=fix)
        out[name] = spec
    if not out:
        raise StructureError("request at least one output")
    return out


class Output(Mapping[str, Tensor]):
    """The result of a render: output tensors by name, plus ``meta``.

    Every output is an attribute and a key: ``out.image`` is ``out["image"]``, and an output
    requested as ``outputs={"pos": gx.labels.Positions("beads")}`` is ``out.pos``. Output names
    are identifiers that do not shadow this class's own attributes (``keys``, ``values``,
    ``items``, ``get``, ``meta``; checked when outputs are requested), so ``out.<name>`` is
    always the output. A label's extra tensors keep dotted keys (``out["pos.in_fov"]``).

    Parameters
    ----------
    values : Mapping[str, Tensor]
        Output tensors by name.
    meta : Mapping[str, object]
        Pipeline hash, chunk sizes, key kind, per-image in-envelope flags and diagnostics.

    Attributes
    ----------
    image : Tensor
        Noisy frames in the camera unit, ``[B, (T,) C, H, W]``, when ``"image"`` is requested.
    expected : Tensor
        Noise-free frames in the camera unit, same layout, when ``"expected"`` is requested.
    meta : dict
        Pipeline hash, chunk sizes, key kind, in-envelope flags and diagnostics.

    Examples
    --------
    >>> import torch
    >>> out = Output({"image": torch.zeros(1, 1, 4, 4)}, {"key": "none"})
    >>> out.image is out["image"]
    True
    """

    image: Tensor
    expected: Tensor
    meta: dict[str, Any]

    def __init__(self, values: Mapping[str, Tensor], meta: Mapping[str, object]) -> None:
        self._values = dict(values)
        self.meta = dict(meta)

    def __getattr__(self, name: str) -> Tensor:
        """Return the output called ``name`` (attribute access to the outputs).

        Parameters
        ----------
        name : str
            Output name.

        Returns
        -------
        Tensor
            The output.

        Raises
        ------
        AttributeError
            If the render produced no output of that name.
        """
        values = self.__dict__.get("_values")
        if values is None or name.startswith("__"):  # copies and pickling look up dunders first
            raise AttributeError(name)
        try:
            return values[name]
        except KeyError:
            msg = f"no output {name!r}; this render produced {sorted(values)}"
            raise AttributeError(msg) from None

    def __dir__(self) -> list[str]:
        names = (k for k in self._values if k.isidentifier())
        return sorted({*super().__dir__(), *names})

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
