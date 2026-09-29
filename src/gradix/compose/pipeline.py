"""The Pipeline: a Chain fixed to static grids from an envelope, then fed data (§5.7, §6.9).

Building a Pipeline runs the analysis passes on any Chain, hand-built or planned:

- **P0 normalise**: structure checks, the structure signature, capacities, declared inputs;
- **P1 envelope**: derived from the template with headroom, overridden by explicit entries;
- **P3 validity**: element predicates on the envelope (``on_invalid`` decides what raises);
- **P4 gradients**: the gradient-quality table;
- **P5 sampling**: every element's static configuration (the :class:`SamplingPlan`);
- **P6 memory**: dominant-tensor estimates and recorded chunk sizes;
- **P7 rewrites**: none on the M0 emission path;
- **P8 emit**: the frozen state as JSON, and the Pipeline hash.

A built Pipeline is immutable. Calls take any structure-compatible Chain, or a mapping from
field paths to new values bound onto the template (:meth:`Pipeline.bind`). The template's batch
size and slot counts are capacities: calls may use fewer images or objects, never more.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import time
from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import Tensor

from gradix._core.carriers import EmitterSet
from gradix._core.contract import Description, Element, Static, Violation, report
from gradix._core.envelope import Entry, Envelope, check_paths, envelope_of, in_envelope
from gradix._core.errors import (
    BindingError,
    CapacityError,
    EnvelopeError,
    PlanError,
    StructureError,
)
from gradix._core.keys import check_key
from gradix._core.precision import DEFAULT, result_dtype
from gradix._core.rules import batch_bucket, slot_bucket
from gradix._version import __version__
from gradix.compose import execute
from gradix.compose.chain import Chain
from gradix.compose.gradients import GradientTable, gradient_table, warn_partial
from gradix.compose.memory import MemoryPlan, plan_memory
from gradix.compose.outputs import Expected, Image, Output, OutputSpec, normalize_outputs
from gradix.compose.rewrites import rewrites
from gradix.compose.sampling import SamplingPlan, sampling_plan
from gradix.compose.validity import check as check_validity
from gradix.detect.camera import Camera
from gradix.objects.acquisition import Acquisition
from gradix.objects.volumes import Voxels
from gradix.schema.base import Node, iter_leaves, tree_axis_sizes
from gradix.schema.layout import canonical_shape, common_device
from gradix.schema.signature import leaf_pattern, signature, widened_axes
from gradix.tree import get as tree_get
from gradix.tree import replace as tree_replace

__all__ = ["Pipeline"]

JSON_VERSION = 1
"""Version of the Pipeline JSON format."""


def _slot_counts(chain: Chain) -> dict[str, int]:
    """Object count of every population (the largest N over its per-object fields)."""
    counts: dict[str, int] = {}
    for name in sorted({*chain.emitters, *chain.scatterers}):
        pop = chain.emitters[name] if name in chain.emitters else chain.population(name)
        if isinstance(pop, Voxels):
            continue  # a volume has no slots
        n = 1
        for _path, spec, value in iter_leaves(pop):
            if spec is not None and spec.role == "object" and isinstance(value, Tensor):
                n = max(n, canonical_shape(spec, tuple(value.shape))[2])
        counts[name] = n
    return counts


def _spec_json(spec: OutputSpec) -> dict[str, object]:
    if isinstance(spec, (Image, Expected)):
        return spec.to_json()
    fields: dict[str, object] = {}
    tensors: dict[str, object] = {}
    for name, s in type(spec).schema().items():
        value = getattr(spec, name)
        if s.kind == "static":
            fields[name] = value
        elif s.kind == "tensor" and value is not None:
            if isinstance(value, complex):  # a Python number stays a number
                tensors[name] = {"number": [value.real, value.imag], "complex": True}
                continue
            if not isinstance(value, Tensor):
                tensors[name] = {"number": value}
                continue
            data = value.detach().cpu()
            entry: dict[str, object] = {
                "dtype": str(data.dtype).split(".")[-1],
                "shape": list(data.shape),  # empty tensors keep their shape
            }
            if data.is_complex():  # JSON has no complex numbers: store (re, im) pairs
                entry["values"] = torch.view_as_real(data.resolve_conj()).tolist()
            else:
                entry["values"] = data.tolist()
            tensors[name] = entry
    return {
        "type": spec.registry_name or type(spec).__qualname__,
        "label": True,
        "fields": fields,
        "tensors": tensors,
        "signature": signature(spec).digest,
    }


def _spec_from_json(data: Mapping[str, Any]) -> object:
    """Rebuild an output spec: a reserved name, or a registered label from its static fields."""
    if not data.get("label"):
        return data["type"]
    from gradix._core.registry import labels

    cls = labels.get(str(data["type"]))
    values: dict[str, object] = dict(data.get("fields", {}))
    for name, entry in dict(data.get("tensors", {})).items():
        if "number" in entry:
            number = entry["number"]
            values[name] = complex(*number) if entry.get("complex") else number
            continue
        dtype = getattr(torch, entry["dtype"])
        shape = tuple(entry.get("shape", ()))
        if dtype.is_complex:
            real = torch.tensor(entry["values"], dtype=dtype.to_real())
            tensor = torch.view_as_complex(real.reshape(*shape, 2).contiguous())
        else:
            tensor = torch.tensor(entry["values"], dtype=dtype)
        values[name] = tensor.reshape(shape) if "shape" in entry else tensor
    return cls(**values)


def _device_class(device: torch.device) -> str:
    if device.type == "cuda":
        return f"cuda:{torch.cuda.get_device_name(device)}"
    return device.type


def _elements(chain: Chain) -> dict[str, Element]:
    return chain.elements()


def _bins(chain: Chain) -> int:
    spectra = [getattr(p, "emission", None) for p in chain.emitters.values()]
    return max((s.bins for s in spectra if s is not None), default=1)


def _descriptions(
    chain: Chain, batch: int, slots: Mapping[str, int], frames: int, deterministic: bool
) -> dict[str, Description]:
    parts = {"objective": "objective", "camera": "camera", "environment": "environment"}
    nodes = chain.parts()
    populations = tuple(sorted(chain.emitters))
    return {
        path: Description(
            path=path,
            batch=batch,
            slots=dict(slots),
            frames=frames,
            populations=(
                populations
                if path == "imaging"
                else (path.split(".", 1)[1],)
                if path.startswith("scatterers.")
                else ()
            ),
            parts=parts,
            nodes=nodes,
            bins=_bins(chain),
            deterministic=deterministic,
        )
        for path in chain.elements()
    }


def _module_parameters(tree: object) -> list[Tensor]:
    """Parameters of the ``nn.Module`` objects held in static fields of a tree's nodes."""
    out: list[Tensor] = []
    stack = [tree]
    while stack:
        item = stack.pop()
        if isinstance(item, Node):
            for name, spec in type(item).schema().items():
                value = getattr(item, name)
                if spec.kind == "static" and isinstance(value, torch.nn.Module):
                    out.extend(value.parameters())
                elif spec.kind == "child" and value is not None:
                    stack.append(value)
        elif isinstance(item, Mapping):
            stack.extend(item.values())
        elif isinstance(item, (tuple, list)):
            stack.extend(item)
    return out


def _is_declared(path: str, inputs: Sequence[str]) -> bool:
    return any(path == d or path.startswith(f"{d}.") or d.startswith(f"{path}.") for d in inputs)


def _check_input_path(chain: Chain, path: str, tree_path: str) -> None:
    """Refuse a declared input that names a static field (structure cannot vary per call)."""
    try:
        tree_get(chain, tree_path)
    except (BindingError, StructureError, KeyError, AttributeError) as err:
        msg = f"inputs= names {path!r}, which is not a field of the Chain"
        detail = getattr(err, "message", None) or str(err)
        hint = getattr(err, "fix", None)
        raise BindingError(f"{msg}: {detail}", fix=hint) from None
    parent_path, _, name = tree_path.rpartition(".")
    parent = tree_get(chain, parent_path) if parent_path else chain
    spec = type(parent).schema().get(name) if isinstance(parent, Node) else None
    if spec is not None and spec.kind == "static":
        msg = f"inputs= names {path!r}, a static field: structure cannot vary between calls"
        raise BindingError(msg, fix="build one Pipeline per value of it")


class Pipeline:
    """A static render of a Chain: grids from an envelope, analysis, and a pure call.

    Parameters
    ----------
    chain : Chain
        The template Chain: its structure, its batch size and slot counts (capacities), and
        the tensors that calls do not bind.
    envelope : Envelope or Mapping, optional
        Explicit envelope entries; they override the entries derived from the template.
    outputs : str, tuple of str, or Mapping, default ("image",)
        What calls compute (see :func:`gradix.compose.outputs.normalize_outputs`).
    inputs : sequence of str, optional
        Field paths that calls may bind at any role width their schema allows (§6.9).
    batch : int, optional
        Batch capacity; the template's batch size rounded up to a power of two by default.
    slots : Mapping[str, int], optional
        Slot capacity per population; the template's count rounded up to 8, 16, 32, … by
        default.
    on_invalid : {"warn", "raise", "ignore"}, default "warn"
        What validity findings do when the Pipeline is built.
    memory_budget : float, default 0.5
        Fraction of the device's total memory a call may use (total, not free, so chunk sizes
        are reproducible).
    deterministic : bool, default False
        Record that calls must be bit-exact (sort-based splatting where atomics would be used).
    recorded_chunks : Mapping[str, int], optional
        Chunk sizes to reuse (set by :meth:`from_json`).

    Attributes
    ----------
    template : Chain
        The template Chain.
    outputs : Mapping[str, OutputSpec]
        The normalised outputs.
    inputs : tuple of str
        The declared input paths.
    envelope : Envelope
        The envelope entries, derived and explicit, that fixed the grids.
    sampling : SamplingPlan
        The grids and knob values the elements chose, with provenance (:meth:`static` reads
        one element's).
    gradient_table : GradientTable
        The gradient quality of every field for every output (:meth:`gradients` prints it).
    batch : int
        The batch capacity.
    slots : dict of str to int
        The slot capacity of every population.
    hash : str
        A short hash of the build state.

    Examples
    --------
    >>> import torch, gradix as gx
    >>> from gradix.units import nm, um
    >>> camera = gx.Camera(pixel_size=6.5 * um, shape=(32, 32), noise=gx.noise.Ideal())
    >>> chain = gx.Chain(
    ...     emitters={"beads": gx.Emitters(
    ...         position=torch.tensor([[[2.6, 2.6, 0.0]]]),
    ...         photons=torch.tensor([[1000.0]]),
    ...         emission=gx.Spectrum.line(600 * nm),
    ...     )},
    ...     imaging=gx.imaging.Sprites(gx.Objective(NA=0.7, magnification=40), camera),
    ...     environment=gx.env.Homogeneous(1.33),
    ... )
    >>> pipe = gx.Pipeline(chain, outputs=("image", "expected"))
    >>> out = pipe(chain, key=torch.tensor([7]))
    >>> tuple(out["image"].shape)
    (1, 1, 32, 32)
    """

    def __init__(
        self,
        chain: Chain,
        envelope: Envelope | Mapping[str, tuple[float, float] | Entry] | None = None,
        outputs: object = ("image",),
        inputs: Sequence[str] | None = None,
        batch: int | None = None,
        slots: Mapping[str, int] | None = None,
        on_invalid: str = "warn",
        memory_budget: float = 0.5,
        deterministic: bool = False,
        recorded_chunks: Mapping[str, int] | None = None,
    ) -> None:
        started = time.perf_counter()
        if not isinstance(chain, Chain):
            raise StructureError(f"gx.Pipeline takes a gx.Chain, got {type(chain).__name__}")
        # P0 normalise
        execute.check_supported(chain)
        self.template = chain
        self.outputs = normalize_outputs(outputs)
        self.inputs: tuple[str, ...] = tuple(inputs or ())
        self._declared_tree_paths = tuple(p for path in self.inputs for p in chain.resolve(path))
        for path, tree_path in zip(self.inputs, self._declared_tree_paths, strict=False):
            _check_input_path(chain, path, tree_path)
        self.signature = signature(chain)
        sizes = tree_axis_sizes(chain)
        b = sizes.get("B", 1)
        self.batch = int(batch) if batch is not None else batch_bucket(b)
        if self.batch < b:
            raise CapacityError(f"batch={self.batch} is smaller than the template's {b} images")
        counts = _slot_counts(chain)
        requested = dict(slots or {})
        unknown = set(requested) - set(counts)
        if unknown:
            raise StructureError(f"slots= names unknown populations {sorted(unknown)}")
        self.slots = {p: int(requested.get(p, slot_bucket(n))) for p, n in counts.items()}
        for p, n in counts.items():
            if self.slots[p] < n:
                raise CapacityError(f"slots[{p!r}] = {self.slots[p]} is below the template's {n}")
        self.frames = chain.acq_index().size
        self.on_invalid = on_invalid
        self.memory_budget = float(memory_budget)
        self.deterministic = bool(deterministic)
        leaves = [v for _, _, v in iter_leaves(chain)]
        self.device = common_device(*leaves)
        self.dtype = result_dtype(*leaves)
        # P1 envelope
        explicit = envelope if isinstance(envelope, Envelope) else Envelope(envelope or {})
        check_paths(explicit, chain)
        self.envelope = envelope_of(chain, inputs=self.inputs) | explicit
        # P3 validity
        elements = _elements(chain)
        descs = _descriptions(chain, self.batch, self.slots, self.frames, self.deterministic)
        self.violations: tuple[Violation, ...] = tuple(
            report(check_validity(chain, elements, self.envelope, descs), on_invalid)
        )
        # P4 gradient table
        self.gradient_table: GradientTable = gradient_table(chain, self.outputs)
        # P5 sampling
        self.sampling: SamplingPlan = sampling_plan(elements, self.envelope, descs)
        # P6 memory
        grid = self.sampling.grid()
        shape = grid.shape if grid is not None else (1, 1)
        bins = _bins(chain)
        self.memory: MemoryPlan = plan_memory(
            batch=self.batch,
            frames=self.frames,
            slots=self.slots,
            bins=bins,
            shape=shape,
            itemsize=torch.finfo(self.dtype).bits // 8,
            device=self.device,
            budget=self.memory_budget,
            recorded=recorded_chunks,
            per_image_hint=max(
                (
                    hint
                    for path, element in elements.items()
                    if (hint := element.memory(descs[path], self.sampling.statics[path]))
                    is not None
                ),
                default=None,
            ),
        )
        # P7 rewrites
        self.rewrites = rewrites(elements)
        # P8 emit
        self.build_ms = 1e3 * (time.perf_counter() - started)
        self.hash = hashlib.sha256(json.dumps(self._state(), sort_keys=True).encode()).hexdigest()[
            :12
        ]

    # ---- frozen state ------------------------------------------------------------------------

    def _state(self) -> dict[str, object]:
        """Return the frozen state that defines the Pipeline (hashed; the JSON adds reports)."""
        return {
            "format": JSON_VERSION,
            "gradix": __version__,
            "signature": self.signature.digest,
            "capacities": {"batch": self.batch, "slots": dict(sorted(self.slots.items()))},
            "frames": self.frames,
            "inputs": list(self.inputs),
            "outputs": {name: _spec_json(spec) for name, spec in self.outputs.items()},
            "envelope": self.envelope.to_json(),
            "sampling": self.sampling.to_json(),
            "chunks": dict(sorted(self.memory.chunks.items())),
            "deterministic": self.deterministic,
            "device": _device_class(self.device),
            "precision": DEFAULT.describe(),
            "stages": {
                path: getattr(type(e), "registry_name", None) or type(e).__qualname__
                for path, e in _elements(self.template).items()
            },
        }

    def to_json(self) -> str:
        """Freeze the Pipeline as JSON, so a dataset can be regenerated exactly (P8).

        Returns
        -------
        str
            The state (structure signature, capacities, envelope, SamplingPlan, recorded chunk
            sizes, outputs, flags), the gradient table, the validity report and the hash.
        """
        state = self._state()
        state["hash"] = self.hash
        state["gradients"] = self.gradient_table.to_json()
        state["validity"] = [v.to_json() for v in self.violations]
        state["memory"] = self.memory.to_json()
        return json.dumps(state, indent=2, sort_keys=True)

    @classmethod
    def from_json(cls, text: str, template: Chain) -> Pipeline:
        """Rebuild a frozen Pipeline around a template with the same structure.

        Parameters
        ----------
        text : str
            Output of :meth:`to_json`.
        template : Chain
            A Chain with the recorded structure signature.

        Returns
        -------
        Pipeline
            A Pipeline with the recorded envelope, capacities, outputs and chunk sizes.

        Raises
        ------
        StructureError
            If the template's structure differs from the recorded one.
        """
        state = json.loads(text)
        digest = signature(template).digest
        if state["signature"] != digest:
            msg = (
                f"the template's structure {digest} differs from the recorded {state['signature']}"
            )
            raise StructureError(msg, fix="rebuild the template the Pipeline was frozen from")
        outputs = {name: _spec_from_json(spec) for name, spec in state["outputs"].items()}
        pipe = cls(
            template,
            envelope=Envelope.from_json(state["envelope"]),
            outputs=outputs,
            inputs=state["inputs"],
            batch=state["capacities"]["batch"],
            slots=state["capacities"]["slots"],
            deterministic=state["deterministic"],
            recorded_chunks=state["chunks"],
            on_invalid="ignore",
        )
        if pipe.hash != state["hash"]:
            msg = f"the rebuilt Pipeline hash {pipe.hash} differs from the recorded {state['hash']}"
            raise StructureError(msg, fix="use the same gradix version and device class")
        return pipe

    # ---- inspection ------------------------------------------------------------------------

    def static(self, path: str = "imaging") -> Static:
        """Return an element's static configuration, for L2 calls with the Pipeline's grids.

        Parameters
        ----------
        path : str, default "imaging"
            Element path in the Chain.

        Returns
        -------
        Static
            The configuration; pass it as ``element(..., static=...)``.

        Raises
        ------
        PlanError
            If the Chain has no configured element at ``path``.
        """
        try:
            return self.sampling.statics[path]
        except KeyError:
            known = sorted(self.sampling.statics)
            raise PlanError(f"no configured element at {path!r}", fix=f"known: {known}") from None

    def gradients(self, chain: Chain | None = None) -> list[str]:
        """Return the gradient table as lines (only fields requiring gradients with ``chain``).

        Parameters
        ----------
        chain : Chain, optional
            Restrict to the fields of this Chain that require gradients.

        Returns
        -------
        list of str
            One line per field.
        """
        return self.gradient_table.lines(chain)

    def explain(self) -> str:
        """Return a report: elements, grids with provenance, gradients, validity and memory.

        Returns
        -------
        str
            The report.
        """
        lines = [
            f"Pipeline {self.hash} · B≤{self.batch} · {self.device} · {DEFAULT.describe()}"
            f" · built in {self.build_ms:.0f} ms"
        ]
        if self.inputs:
            lines.append(f"inputs     {', '.join(self.inputs)}")
        slots = ", ".join(f"{p} ≤ {n}" for p, n in self.slots.items())
        if slots:
            lines.append(f"slots      {slots}")
        for i, (path, entry) in enumerate(self.envelope.items()):
            head = "envelope   " if i == 0 else "           "
            lines.append(f"{head}{path} ∈ [{entry.lo:.4g}, {entry.hi:.4g}] ({entry.source})")
        for i, line in enumerate(self.sampling.lines()):
            lines.append(("sampling   " if i == 0 else "           ") + line)
        lines.append(" # stage     implementation")
        imaging = self.template.imaging
        dense = isinstance(imaging, Element) and EmitterSet not in imaging.caps.accepts
        carrier = "EmitterDensity" if dense else "EmitterSet"
        lines.append(
            f" 1 lower     {', '.join(sorted(self.template.emitters))} → {carrier} (exact)"
        )
        for i, (path, element) in enumerate(_elements(self.template).items(), start=2):
            name = getattr(type(element), "registry_name", None) or type(element).__name__
            lines.append(f" {i} {path:9s} {name}")
            for edge in element.caps.approximates:
                tol = f", tolerance {edge.tolerance}" if edge.tolerance else ""
                lines.append(f"   {'':9s} ≈ {edge.target} when {edge.regime}{tol}")
        grads = self.gradient_table.lines(self.template)
        for i, line in enumerate(grads or ["(no field requires gradients)"]):
            lines.append(("gradients  " if i == 0 else "           ") + line)
        for i, v in enumerate(self.violations or ()):
            lines.append(("validity   " if i == 0 else "           ") + str(v))
        chunks = " ".join(f"{k}:{v}" for k, v in self.memory.chunks.items())
        mib = self.memory.estimate_bytes / 2**20
        lines.append(
            f"memory     est. {mib:.1f} MiB per call at capacity · chunks {chunks} (recorded)"
        )
        return "\n".join(lines)

    # ---- feeding data ------------------------------------------------------------------------

    def bind(self, values: Mapping[str, object]) -> Chain:
        """Bind values by field path onto the template (§6.9).

        Parameters
        ----------
        values : Mapping[str, object]
            New values by logical path (``"beads.position"``, ``"objective.focus"``, or a whole
            data object at ``"beads"``). Unnamed fields keep the template's tensors.

        Returns
        -------
        Chain
            The bound Chain; ``pipe(values)`` equals ``pipe(pipe.bind(values))``.

        Raises
        ------
        BindingError
            If a path names nothing, a static field, or (when not declared in ``inputs=``) a
            value wider than the template's role.
        """
        changes: dict[str, object] = {}
        for path, value in values.items():
            declared = _is_declared(path, self.inputs)
            for tree_path in self.template.resolve(path):
                try:
                    current = tree_get(self.template, tree_path)
                except BindingError:
                    raise
                except StructureError as err:  # "beads.nonexistent": the node has no such field
                    msg = f"{path!r} names nothing in the Chain: {err.message}"
                    raise BindingError(msg, fix=err.fix) from err
                self._check_binding(path, tree_path, current, value, declared)
                changes[tree_path] = value
        try:
            return tree_replace(self.template, changes)
        except BindingError:
            raise
        except StructureError as err:
            msg = f"the bound values do not fit the template's other fields: {err.message}"
            fix = (
                "bind the population's other per-image or per-object fields too, "
                "or give the template one value for all images"
            )
            raise BindingError(msg, fix=fix) from err

    def _check_binding(
        self, path: str, tree_path: str, current: object, value: object, declared: bool
    ) -> None:
        parent_path, _, name = tree_path.rpartition(".")
        parent = tree_get(self.template, parent_path) if parent_path else self.template
        spec = type(parent).schema().get(name) if isinstance(parent, Node) else None
        if spec is not None and spec.kind == "static":
            msg = f"{path!r} is structure (a static field) and cannot be bound"
            raise BindingError(msg, fix="build a new Pipeline for a different structure")
        if isinstance(current, Node):
            if not isinstance(value, type(current)):
                msg = f"{path!r} holds a {type(current).__name__}; got {type(value).__name__}"
                raise BindingError(msg)
            if not declared:
                self._check_node_width(path, current, value)
            return
        if spec is None or spec.kind != "tensor":
            return
        if current is None and value is not None:
            msg = f"{path!r} is absent in the template, so binding it would change the structure"
            raise BindingError(msg, fix=f"give the template a value for {path!r}")
        if declared or value is None:
            return
        wider = widened_axes(("", *leaf_pattern(spec, current)), ("", *leaf_pattern(spec, value)))
        if wider:
            msg = f"{path!r} varies along {sorted(wider)} but the template's value does not"
            raise BindingError(msg, fix=f"declare it: gx.Pipeline(..., inputs=({path!r},))")

    def _check_node_width(self, path: str, current: Node, value: Node) -> None:
        old = {p: ("", *leaf_pattern(s, v)) for p, s, v in iter_leaves(current)}
        for p, s, v in iter_leaves(value):
            wider = widened_axes(old.get(p, ("", "absent")), ("", *leaf_pattern(s, v)))
            if wider:
                msg = f"{path}.{p} varies along {sorted(wider)} but the template's value does not"
                raise BindingError(msg, fix=f"declare it: gx.Pipeline(..., inputs=({path!r},))")

    def _conform(self, chain: Chain) -> tuple[int, dict[str, int]]:
        """Check a call's Chain against the template; return its batch size and slot counts."""
        if not isinstance(chain, Chain):
            raise StructureError(
                f"a Pipeline call takes a Chain or a mapping, got {type(chain).__name__}"
            )
        device = common_device(*(v for _, _, v in iter_leaves(chain)))
        if device is not None and self.device is not None and device != self.device:
            msg = f"the Chain is on {device} but the Pipeline was built for {self.device}"
            raise StructureError(msg, fix="move the Chain with chain.to(device), or build anew")
        if chain is not self.template:
            mine = signature(chain)
            if mine.digest != self.signature.digest:
                diff = self.signature.incompatibility(mine, declared=self._declared_tree_paths)
                if diff is not None:
                    msg = f"the Chain does not fit the template {diff}"
                    raise StructureError(
                        msg, fix="build a Pipeline for this structure (gx.render caches them)"
                    )
        sizes = tree_axis_sizes(chain)
        b = sizes.get("B", 1)
        frames = sizes.get("T", 1)
        declared = isinstance(chain.acquisition, Acquisition)
        if frames != self.frames and not (declared and frames == 1):
            msg = f"the call has {frames} frames but the Pipeline has {self.frames}"
            raise StructureError(msg, fix="the frame count is structure: build a Pipeline for it")
        if b > self.batch:
            msg = f"the call has {b} images but the batch capacity is {self.batch}"
            raise CapacityError(msg, fix="split the batch, or build with batch=...")
        counts = _slot_counts(chain)
        for pop, n in counts.items():
            if n > self.slots.get(pop, 0):
                have = self.slots.get(pop)
                msg = f"population {pop!r} has {n} objects but its slot capacity is {have}"
                raise CapacityError(msg, fix=f"build with slots={{{pop!r}: {slot_bucket(n)}}}")
        return b, counts

    def __call__(
        self,
        inputs: Chain | Mapping[str, object] | None = None,
        *,
        key: int | Tensor | None = None,
        check: str | None = None,
    ) -> Output:
        """Render a batch.

        Parameters
        ----------
        inputs : Chain or Mapping[str, object], optional
            A structure-compatible Chain, or values bound by field path; the template if None.
        key : int or Tensor, optional
            Detector-noise key: an ``int`` (batch-keyed) or an int64 ``Tensor[B]`` (image-keyed).
            Required for ``"image"`` when the camera has noise.
        check : {"raise"}, optional
            ``"raise"`` synchronises and raises when an image leaves the envelope; by default
            the flags are only returned in ``out.meta["in_envelope"]``.

        Returns
        -------
        Output
            The requested outputs and ``meta`` (``hash``, ``chunks``, ``key``,
            ``in_envelope``, ``deterministic``).
        """
        if inputs is None:
            chain = self.template
        elif isinstance(inputs, Mapping):
            chain = self.bind(inputs)
        else:
            chain = inputs
        batch, _counts = self._conform(chain)
        if isinstance(key, Tensor):
            check_key(key, batch)
        elif key is not None:
            check_key(key)
        wants_image = any(isinstance(s, Image) for s in self.outputs.values())
        camera = chain.camera
        if wants_image and key is None and isinstance(camera, Camera) and camera.noise is not None:
            raise StructureError(
                "the image output needs a key (the camera has noise)",
                fix="pass key=..., or request outputs=('expected',)",
            )
        self.gradient_table.check(chain)
        warn_partial(self.gradient_table, chain)
        flags = in_envelope(self.envelope, chain, batch)
        if check == "raise" and not bool(flags.all()):
            bad = torch.nonzero(~flags).flatten().tolist()
            raise EnvelopeError(
                f"images {bad} leave the envelope", fix="grow the envelope, or use gx.render"
            )
        needs_grad = any(
            isinstance(v, Tensor) and v.requires_grad for _, _, v in iter_leaves(chain)
        ) or any(p.requires_grad for p in _module_parameters(chain))
        mode = contextlib.nullcontext() if needs_grad else torch.no_grad()
        with mode:
            values = self._run_chunked(chain, batch, key)
        meta = {
            "hash": self.hash,
            "chunks": dict(self.memory.chunks),
            "key": "none" if key is None else ("image" if isinstance(key, Tensor) else "batch"),
            "in_envelope": flags,
            "deterministic": self.deterministic,
        }
        return Output(values, meta)

    def _run_chunked(self, chain: Chain, batch: int, key: int | Tensor | None) -> dict[str, Tensor]:
        """Render in batch chunks; the noise is drawn once, so chunking never changes results."""
        statics = self.sampling.statics
        chunk = int(self.memory.chunks.get("batch", batch))
        if chunk >= batch:
            return execute.run(chain, statics, key=key, outputs=self.outputs, batch=batch)
        wants_frames = any(isinstance(s, (Image, Expected)) for s in self.outputs.values())
        frames: list[Tensor] = []
        label_parts: list[dict[str, Tensor]] = []
        for start in range(0, batch, chunk):
            size = min(chunk, batch - start)
            sub = chain.select(slice(start, start + size))
            if wants_frames:
                mu = execute.expected(sub, statics)
                frames.append(mu.expand(size, *mu.shape[1:]) if mu.shape[0] != size else mu)
            label_parts.append(execute.labels(sub, self.outputs, size))
        mu_all = torch.cat(frames, 0) if frames else None
        values = execute.finish(chain, mu_all, key, self.outputs)
        values.update(execute.concat(label_parts))
        return execute.fresh(values, batch, inputs=chain)
