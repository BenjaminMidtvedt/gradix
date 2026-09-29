"""The element contract (§5.1): slots, capabilities, violations, costs and the Element base.

An element is a public class that can be called directly (L2), wired into a Chain (L3) or chosen
by the planner (L4). Its static methods see structure and the envelope, never tensor values:

- :meth:`Element.validity` returns :class:`Violation` s for an envelope;
- :meth:`Element.configure` resolves the knobs left ``"auto"`` and sizes grids, returning a
  :class:`Static`;
- :meth:`Element.cost` (optional) ranks numerically equivalent strategies.

At run time :meth:`Element.forward` is pure torch on carriers, data-object and element tensors
and explicit keys. An eager call (:meth:`Element.__call__`) configures on the envelope of its own
inputs (or on ``grid=``) and then runs ``forward``.
"""

from __future__ import annotations

import dataclasses
import enum
import warnings
from collections.abc import Iterable, Mapping
from typing import Any, ClassVar, Generic, Literal, TypeAlias, TypeVar

from gradix._core.envelope import Envelope, envelope_of
from gradix._core.errors import GradixWarning, ValidityError
from gradix._core.rules import Decision
from gradix.schema.base import Node

__all__ = [
    "GRAD_QUALITIES",
    "Capabilities",
    "Cost",
    "Description",
    "Edge",
    "Element",
    "GradQuality",
    "Slot",
    "Static",
    "Violation",
    "report",
    "worst_quality",
]


class Slot(str, enum.Enum):
    """The slots of the render skeleton (§5.1).

    Parameters
    ----------
    *values : str
        The slot name, as in ``Slot("emit")``.
    """

    SOURCE = "source"
    STAGE = "stage"
    INTERACT = "interact"
    OBJECTIVE = "objective"
    TRANSDUCE = "transduce"
    EMIT = "emit"
    DETECT = "detect"


GradQuality: TypeAlias = Literal["exact", "exact-a.e.", "biased", "zero"]
GRAD_QUALITIES: tuple[str, ...] = ("exact", "exact-a.e.", "biased", "zero")
"""Gradient qualities from best to worst (§6.2)."""


def worst_quality(*qualities: str) -> str:
    """Return the worst of several gradient qualities (``"exact"`` for none).

    Parameters
    ----------
    *qualities : str
        Qualities from :data:`GRAD_QUALITIES`.

    Returns
    -------
    str
        The worst one.
    """
    return max(qualities, key=GRAD_QUALITIES.index, default="exact")


@dataclasses.dataclass(frozen=True)
class Edge:
    """An approximation edge: this element approximates ``target`` within a regime (§5.9).

    Parameters
    ----------
    target : str
        Registry name of the higher-fidelity element.
    regime : str, default ""
        Where the approximation holds, in words.
    tolerance : str, optional
        Tolerance-table identifier, such as ``"ladders/gaussian_pupil@v1.0"``.
    """

    target: str
    regime: str = ""
    tolerance: str | None = None


def _lookup(table: Mapping[str, str], path: str, default: str) -> str:
    """Return the entry of the longest dotted prefix of ``path`` in ``table``."""
    parts = path.split(".")
    for end in range(len(parts), 0, -1):
        hit = table.get(".".join(parts[:end]))
        if hit is not None:
            return hit
    return default


@dataclasses.dataclass(frozen=True)
class Capabilities:
    """What an element consumes, produces and guarantees (§5.1).

    ``accepts`` and ``produces`` are checked along a Chain's wiring before it renders, and
    ``grad_quality`` and ``reads`` feed the gradient table (and the conformance suite's
    ``declared`` check). ``ports``, ``polarization``, ``linear_in_field``,
    ``linear_in_sample``, ``space_variant`` and ``grad`` are declarations for the coherent
    paths and rewrites of M3 and later: nothing enforces them yet.

    Parameters
    ----------
    accepts : frozenset of type
        Carrier or view types consumed.
    produces : type or tuple of type
        Carrier type produced.
    ports : frozenset of str, default {"+z"}
        Travel directions filled.
    polarization : frozenset of int, default {1}
        Supported P values.
    linear_in_field : bool, default True
        Linear in the incident field (enables mode chunking).
    linear_in_sample : bool, default False
        Linear in the sample (enables WOTF/Born-type rewrites, v1.x).
    space_variant : bool, default False
        An objective that depends on the field position.
    grad : frozenset of str, default {"autograd"}
        How gradients are computed: ``autograd``, ``checkpoint``, ``adjoint``, ``implicit``.
    grad_quality : Mapping[str, str], optional
        Gradient quality of the element's output with respect to its own fields, by field path
        (dotted prefixes such as ``"objective.pupil"`` cover every field below them; the
        longest match wins). Unlisted fields are ``"exact"``, or take the ``"*"`` entry.
    reads : Mapping[str, str], optional
        What the element consumes from its inputs, with the gradient quality of its output with
        respect to each: population fields by name (``"position"``, ``"photons"``), carrier
        fields, and ``"environment"``. Inputs it does not list reach its output with zero
        gradient. None (the default) means every input field is consumed exactly.
    approximates : tuple of Edge, default ()
        Approximation edges to higher-fidelity elements.
    """

    accepts: frozenset[type] = frozenset()
    produces: type | tuple[type, ...] = object
    ports: frozenset[str] = frozenset({"+z"})
    polarization: frozenset[int] = frozenset({1})
    linear_in_field: bool = True
    linear_in_sample: bool = False
    space_variant: bool = False
    grad: frozenset[str] = frozenset({"autograd"})
    grad_quality: Mapping[str, str] = dataclasses.field(default_factory=dict)
    reads: Mapping[str, str] | None = None
    approximates: tuple[Edge, ...] = ()

    def quality(self, field: str) -> str:
        """Return the declared gradient quality with respect to one of the element's fields.

        Parameters
        ----------
        field : str
            Field path within the element, such as ``"NA"`` or ``"objective.pupil.coeffs"``.

        Returns
        -------
        str
            The quality of the longest declared prefix; ``"exact"`` when none is declared.
        """
        return _lookup(self.grad_quality, field, self.grad_quality.get("*", "exact"))

    def read_quality(self, name: str) -> str:
        """Return the gradient quality with respect to a consumed input field.

        Parameters
        ----------
        name : str
            Field path within the input, such as ``"position"`` or ``"emission.weights"``, or
            ``"environment"``.

        Returns
        -------
        str
            The quality of the longest declared prefix; ``"zero"`` for inputs the element does
            not read, ``"exact"`` for every input when :attr:`reads` is None.
        """
        if self.reads is None:
            return "exact"
        return _lookup(self.reads, name, "zero")

    def to_json(self) -> dict[str, Any]:
        """Return a JSON-serialisable description.

        Returns
        -------
        dict
            The fields, with types as names.
        """
        produces = self.produces if isinstance(self.produces, tuple) else (self.produces,)
        return {
            "accepts": sorted(t.__name__ for t in self.accepts),
            "produces": [t.__name__ for t in produces],
            "ports": sorted(self.ports),
            "polarization": sorted(self.polarization),
            "linear_in_field": self.linear_in_field,
            "grad": sorted(self.grad),
            "grad_quality": dict(sorted(self.grad_quality.items())),
            "reads": None if self.reads is None else dict(sorted(self.reads.items())),
            "approximates": [dataclasses.asdict(e) for e in self.approximates],
        }


@dataclasses.dataclass(frozen=True)
class Violation:
    """A validity finding (§5.8).

    Parameters
    ----------
    severity : {"info", "warn", "error"}
        How serious it is; ``error`` raises under ``on_invalid="raise"`` and by default.
    message : str
        What is wrong.
    entry : str, optional
        The envelope entry or field path involved.
    value : float, optional
        The offending value.
    limit : float, optional
        The threshold.
    element : str, optional
        Path of the element that reported it.
    fix : str, optional
        How to fix it.
    calibration : {"literature", "calibrated"}, optional
        Where the threshold comes from.
    """

    severity: Literal["info", "warn", "error"]
    message: str
    entry: str | None = None
    value: float | None = None
    limit: float | None = None
    element: str | None = None
    fix: str | None = None
    calibration: Literal["literature", "calibrated"] | None = None

    def __str__(self) -> str:
        where = f"{self.element}: " if self.element else ""
        detail = ""
        if self.entry is not None:
            detail = f" [{self.entry}"
            if self.value is not None:
                detail += f" = {self.value:.4g}"
            if self.limit is not None:
                detail += f", limit {self.limit:.4g}"
            detail += "]"
        fix = f" (fix: {self.fix})" if self.fix else ""
        return f"{self.severity:5s} {where}{self.message}{detail}{fix}"

    def to_json(self) -> dict[str, Any]:
        """Return a JSON-serialisable description.

        Returns
        -------
        dict
            The fields.
        """
        return dataclasses.asdict(self)


def report(violations: Iterable[Violation], on_invalid: str = "warn") -> list[Violation]:
    """Act on validity findings according to a policy.

    Errors raise unless the policy is ``"ignore"``; warnings raise under ``"raise"``, are issued
    as :class:`~gradix._core.errors.GradixWarning` under ``"warn"`` and are dropped under
    ``"ignore"``. Info findings are only returned.

    Parameters
    ----------
    violations : iterable of Violation
        The findings.
    on_invalid : {"warn", "raise", "ignore"}, default "warn"
        The policy.

    Returns
    -------
    list of Violation
        The findings, for reports.

    Raises
    ------
    ValidityError
        On errors (unless ignored), or on warnings under ``"raise"``.
    """
    found = list(violations)
    if on_invalid not in ("warn", "raise", "ignore"):
        msg = f"on_invalid must be 'warn', 'raise' or 'ignore', got {on_invalid!r}"
        raise ValueError(msg)
    fatal = [v for v in found if v.severity == "error" and on_invalid != "ignore"]
    if on_invalid == "raise":
        fatal += [v for v in found if v.severity == "warn"]
    if fatal:
        raise ValidityError("; ".join(str(v) for v in fatal), fix=fatal[0].fix)
    if on_invalid == "warn":
        for v in found:
            if v.severity == "warn":
                warnings.warn(str(v), GradixWarning, stacklevel=3)
    return found


@dataclasses.dataclass(frozen=True)
class Cost:
    """An estimated cost, used only to rank numerically equivalent strategies.

    Parameters
    ----------
    flops : float, default 0.0
        Floating-point operations.
    bytes : float, default 0.0
        Bytes moved.
    """

    flops: float = 0.0
    bytes: float = 0.0

    def __add__(self, other: Cost) -> Cost:
        return Cost(self.flops + other.flops, self.bytes + other.bytes)

    @staticmethod
    def fft2(shape: tuple[int, int], count: int = 1, batch: int = 1) -> Cost:
        """Return the cost of ``count × batch`` complex 2-D FFTs of a shape.

        Parameters
        ----------
        shape : tuple of int
            ``(Y, X)``.
        count : int, default 1
            FFTs per image.
        batch : int, default 1
            Images.

        Returns
        -------
        Cost
            5·n·log2(n) flops and 16·n bytes per FFT, n = Y·X.
        """
        import math

        n = shape[0] * shape[1]
        k = count * batch
        return Cost(5.0 * n * math.log2(max(n, 2)) * k, 16.0 * n * k)


@dataclasses.dataclass(frozen=True)
class Description:
    """The static description an element's ``validity`` and ``configure`` see.

    Parameters
    ----------
    path : str, default ""
        The element's path in the Chain (``"imaging"``, ``"camera"``); empty for eager calls.
    batch : int, default 1
        Batch size or capacity.
    slots : Mapping[str, int], optional
        Slot capacity of each population.
    frames : int, default 1
        Size of the acquisition axis A.
    populations : tuple of str, default ()
        Names of the populations the element renders; their envelope paths start with them.
    parts : Mapping[str, str], optional
        Envelope path prefix of each named part the element reads, such as
        ``{"objective": "objective", "camera": "camera"}``.
    nodes : Mapping[str, Node], optional
        The template's parts by the same names (the environment, the camera, populations),
        for structure only: types, static fields and shapes, never tensor values.
    bins : int, default 1
        Largest number of spectral bins L of the populations the element renders.
    deterministic : bool, default False
        Whether calls must be bit-exact (no atomic accumulation).
    upstream : Mapping[str, Static], optional
        Configurations of the elements before this one in the Chain, by path.
    """

    path: str = ""
    batch: int = 1
    slots: Mapping[str, int] = dataclasses.field(default_factory=dict)
    frames: int = 1
    populations: tuple[str, ...] = ()
    parts: Mapping[str, str] = dataclasses.field(default_factory=dict)
    nodes: Mapping[str, Node] = dataclasses.field(default_factory=dict)
    bins: int = 1
    deterministic: bool = False
    upstream: Mapping[str, Static] = dataclasses.field(default_factory=dict)

    def part(self, name: str) -> str:
        """Return the envelope path prefix of a part (the part's name by default).

        Parameters
        ----------
        name : str
            Part name, such as ``"objective"``.

        Returns
        -------
        str
            The prefix.
        """
        return self.parts.get(name, name)


@dataclasses.dataclass(frozen=True)
class Static:
    """Base of the static configuration an element's ``configure`` returns.

    Parameters
    ----------
    decisions : tuple of Decision, default ()
        Sizing decisions with provenance, for ``explain()`` and the SamplingPlan.
    """

    decisions: tuple[Decision, ...] = ()

    def to_json(self) -> dict[str, Any]:
        """Return a JSON-serialisable description.

        Returns
        -------
        dict
            Every field; grids and decisions through their own ``to_json``.
        """
        out: dict[str, object] = {"type": type(self).__name__}
        for f in dataclasses.fields(self):
            value = getattr(self, f.name)
            if f.name == "decisions":
                out[f.name] = [d.to_json() for d in value]
            elif hasattr(value, "to_json"):
                out[f.name] = value.to_json()
            else:
                out[f.name] = list(value) if isinstance(value, tuple) else value
        return out


Out = TypeVar("Out")
"""The carrier type an element produces."""


@dataclasses.dataclass(frozen=True, kw_only=True, eq=False)
class Element(Node, Generic[Out]):
    """Base of every element (§5.1).

    Subclasses set the class attributes :attr:`slot` and :attr:`caps`, declare their fields and
    knobs, and implement :meth:`forward`. They override :meth:`validity` and :meth:`configure`
    when they have predicates or knobs, and :meth:`eager_parts` when an eager call needs
    envelope entries from its inputs.
    """

    slot: ClassVar[Slot]
    """The slot the element fills."""
    caps: ClassVar[Capabilities] = Capabilities()
    """The element's capabilities."""

    def field_quality(self, path: str, output: str = "expected") -> str:
        """Return the gradient quality of an output with respect to one of the element's fields.

        Parameters
        ----------
        path : str
            Field path within the element.
        output : {"expected", "image"}, default "expected"
            The output kind the element's result feeds.

        Returns
        -------
        str
            By default the declared :meth:`Capabilities.quality`.
        """
        return self.caps.quality(path)

    def input_quality(self, name: str, output: str = "expected") -> str:
        """Return the gradient quality of an output with respect to a consumed input field.

        Parameters
        ----------
        name : str
            Field path within the input, or ``"environment"``.
        output : {"expected", "image"}, default "expected"
            The output kind the element's result feeds.

        Returns
        -------
        str
            By default the declared :meth:`Capabilities.read_quality`.
        """
        return self.caps.read_quality(name)

    def validity(self, desc: Description, envelope: Envelope) -> list[Violation]:
        """Return the element's validity findings on an envelope.

        Parameters
        ----------
        desc : Description
            Static description of the element's context.
        envelope : Envelope
            The envelope.

        Returns
        -------
        list of Violation
            Findings; empty when valid.
        """
        return []

    def configure(self, desc: Description, envelope: Envelope) -> Static:
        """Resolve ``"auto"`` knobs and size grids from an envelope.

        Parameters
        ----------
        desc : Description
            Static description of the element's context.
        envelope : Envelope
            The envelope.

        Returns
        -------
        Static
            The static configuration :meth:`forward` runs with.
        """
        return Static()

    def cost(self, desc: Description, static: Static) -> Cost | None:
        """Estimate the cost of a call; optional.

        Parameters
        ----------
        desc : Description
            Static description of the element's context.
        static : Static
            The configuration.

        Returns
        -------
        Cost or None
            The estimate, or None when not declared.
        """
        return None

    def memory(self, desc: Description, static: Static) -> int | None:
        """Estimate the peak bytes one image needs in a forward and backward pass; optional.

        The Pipeline's memory pass (P6) sizes batch chunks from the largest estimate of its
        elements, so elements with large intermediates (per-emitter pupils) declare them.

        Parameters
        ----------
        desc : Description
            Static description of the element's context (slot capacities, frames, bins).
        static : Static
            The configuration.

        Returns
        -------
        int or None
            Bytes per image, or None when the element's intermediates are small.
        """
        return None

    def forward(self, *inputs: object, static: Static) -> Out:
        """Run the element: pure torch on its inputs and its own tensors.

        Parameters
        ----------
        *inputs : object
            The slot's carriers, then the environment.
        static : Static
            The configuration from :meth:`configure`.

        Returns
        -------
        object
            The produced carrier.
        """
        raise NotImplementedError(f"{type(self).__name__}.forward")

    def eager_parts(self, *inputs: object) -> dict[str, Node]:
        """Return the named parts an eager call derives its envelope from.

        Parameters
        ----------
        *inputs : object
            The inputs of the eager call.

        Returns
        -------
        dict of str to Node
            Parts by envelope path prefix; by default every child field of the element.
        """
        out: dict[str, Node] = {}
        for name, spec in self.schema().items():
            value = getattr(self, name)
            if spec.kind == "child" and isinstance(value, Node):
                out[name] = value
        return out

    def eager_description(self, *inputs: object) -> Description:
        """Return the static description of an eager call.

        Parameters
        ----------
        *inputs : object
            The inputs of the eager call.

        Returns
        -------
        Description
            By default an empty description (no populations).
        """
        return Description()

    def eager_envelope(self, *inputs: object) -> Envelope:
        """Return the envelope an eager call sizes its grids from: its inputs', without headroom.

        Parameters
        ----------
        *inputs : object
            The inputs of the eager call.

        Returns
        -------
        Envelope
            Derived from :meth:`eager_parts`.
        """
        return envelope_of(_Parts(self.eager_parts(*inputs)), headroom=1.0)

    def eager_static(self, *inputs: object, envelope: Envelope | None = None) -> Static:
        """Configure for an eager call: on ``envelope``, or on the envelope of the inputs.

        Validity is checked on the same envelope; warnings are issued, errors raise.

        Parameters
        ----------
        *inputs : object
            The inputs of the eager call.
        envelope : Envelope, optional
            An explicit envelope; derived from :meth:`eager_parts` without headroom otherwise.

        Returns
        -------
        Static
            The configuration.
        """
        if envelope is None:
            envelope = self.eager_envelope(*inputs)
        desc = self.eager_description(*inputs)
        report(self.validity(desc, envelope), "warn")
        return self.configure(desc, envelope)

    def static_for_grid(self, grid: object) -> Static:
        """Return a configuration that renders on a given grid; for ``__call__(grid=...)``.

        Parameters
        ----------
        grid : object
            A grid from :mod:`gradix._core.grid`.

        Returns
        -------
        Static
            The configuration.

        Raises
        ------
        NotImplementedError
            If the element has no grid to set.
        """
        msg = f"{type(self).__name__} takes no grid=; pass static= from a Pipeline instead"
        raise NotImplementedError(msg)

    def __call__(self, *inputs: object, grid: object = None, static: Static | None = None) -> Out:
        """Run the element eagerly.

        Parameters
        ----------
        *inputs : object
            The slot's carriers, then the environment.
        grid : object, optional
            A grid to render on (see :meth:`static_for_grid`).
        static : Static, optional
            A configuration to run with, such as a Pipeline's (for bit-identical results);
            derived from the inputs by :meth:`eager_static` otherwise.

        Returns
        -------
        object
            The produced carrier.
        """
        if static is None:
            static = self.static_for_grid(grid) if grid is not None else self.eager_static(*inputs)
        return self.forward(*inputs, static=static)


class _Parts:
    """Adapter that exposes a dict of parts through the ``HasParts`` protocol."""

    def __init__(self, parts: Mapping[str, Node]) -> None:
        self._parts = dict(parts)

    def parts(self) -> Mapping[str, Node]:
        return self._parts
