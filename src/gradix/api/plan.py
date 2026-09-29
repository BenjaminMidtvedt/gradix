"""L4 planning API: :func:`plan`, :class:`Plan` and :func:`render` (§5.7, §6.4).

``gx.plan`` chooses an element for every population (pass P2), builds the Chain a user could
have built by hand, and wraps it in a :class:`~gradix.Pipeline`. It builds nothing that is not
available at L3: ``plan.chain(sample, microscope)`` returns the Chain and ``plan.pipeline`` the
Pipeline. ``gx.render`` adds a plan cache and the pre-render envelope check.
"""

from __future__ import annotations

import collections
import logging
from collections.abc import Mapping, Sequence

from torch import Tensor

from gradix._core.contract import Element
from gradix._core.envelope import Entry, Envelope, envelope_of, in_envelope
from gradix._core.errors import PlanError, StructureError
from gradix._core.rules import batch_bucket, slot_bucket
from gradix.compose.chain import Chain
from gradix.compose.outputs import Output, normalize_outputs
from gradix.compose.pipeline import Pipeline
from gradix.containers.containers import Microscope, Sample
from gradix.objects.objectset import ObjectSet
from gradix.planner.fidelity import Fidelity
from gradix.planner.rules import Route, build, route
from gradix.schema.base import iter_leaves, tree_axis_sizes
from gradix.schema.layout import canonical_shape, common_device
from gradix.schema.signature import signature

__all__ = ["Plan", "clear_cache", "plan", "render"]

log = logging.getLogger("gradix")

CACHE_SIZE = 16
"""Plans kept by :func:`render`."""


class Plan:
    """A planned render of a Sample through a Microscope at a Fidelity.

    Parameters
    ----------
    sample : Sample
        Example inputs: their structure, capacities and envelope.
    microscope : Microscope
        The optics.
    fidelity : Fidelity or str, default "standard"
        The policy.
    outputs : str, tuple of str, or Mapping, default ("image",)
        What calls compute.
    envelope : Envelope or Mapping, optional
        Explicit envelope entries.
    methods : Mapping[str, str], optional
        Elements pinned per population (registry names), bypassing the rules.
    inputs : sequence of str, optional
        Field paths that calls may bind (§6.9).
    batch : int, optional
        Batch capacity.
    slots : Mapping[str, int], optional
        Slot capacity per population.
    deterministic : bool, default False
        Record that calls must be bit-exact.
    """

    def __init__(
        self,
        sample: Sample,
        microscope: Microscope,
        fidelity: Fidelity | str = "standard",
        *,
        outputs: object = ("image",),
        envelope: Envelope | Mapping[str, tuple[float, float] | Entry] | None = None,
        methods: Mapping[str, str] | None = None,
        inputs: Sequence[str] | None = None,
        batch: int | None = None,
        slots: Mapping[str, int] | None = None,
        deterministic: bool = False,
    ) -> None:
        if not isinstance(sample, Sample) or not isinstance(microscope, Microscope):
            raise StructureError("gx.plan takes a gx.Sample and a gx.Microscope")
        self.fidelity = Fidelity.coerce(fidelity)
        self.methods = dict(methods or {})
        unknown = set(self.fidelity.per_population) - set(sample.populations)
        if unknown:
            known = sorted(sample.populations)
            msg = f"per_population names unknown populations {sorted(unknown)}"
            raise PlanError(msg, fix=f"populations: {known}")
        # P0/P1 before P2: rules may depend on the envelope and the capacities
        explicit = envelope if isinstance(envelope, Envelope) else Envelope(envelope or {})
        derived = envelope_of({**sample.parts(), **microscope.parts()}, inputs=tuple(inputs or ()))
        self.routes: dict[str, Route] = route(
            sample.populations,
            self.fidelity,
            self.methods,
            envelope=derived | explicit,
            counts=_counts(sample),
        )
        self.pipeline = Pipeline(
            self.chain(sample, microscope),
            envelope=envelope,
            outputs=outputs,
            inputs=inputs,
            batch=batch,
            slots=slots,
            on_invalid=self.fidelity.on_invalid,
            memory_budget=self.fidelity.memory_budget,
            deterministic=deterministic,
        )

    def chain(self, sample: Sample, microscope: Microscope) -> Chain:
        """Return the Chain this plan builds for a Sample and a Microscope.

        Parameters
        ----------
        sample : Sample
            The sample.
        microscope : Microscope
            The optics.

        Returns
        -------
        Chain
            The Chain a user could build by hand: ``plan(s, m) == plan.pipeline(plan.chain(s, m))``.
        """
        emit = sorted({r.element for r in self.routes.values() if r.element.startswith("emit.")})
        interact = {
            name: r.element for name, r in self.routes.items() if r.element.startswith("interact.")
        }
        other = sorted(
            {
                r.element
                for r in self.routes.values()
                if not r.element.startswith(("emit.", "interact."))
            }
        )
        if other:
            raise PlanError(
                f"elements {other} are not available in this version",
                fix="Born, projection and multislice arrive with M3 and M4",
            )
        if interact:
            if emit:
                msg = "the Sample mixes emitters and scatterers, which one Chain cannot image"
                raise PlanError(msg, fix="plan them as two Samples")
            return self._coherent_chain(sample, microscope, interact)
        if len(emit) > 1:
            msg = f"emitter populations route to different elements {emit}"
            raise PlanError(
                msg, fix="pin the same element for every emitter population with methods="
            )
        imaging: Element | None = None
        if emit:
            parts = {"objective": microscope.objective, "camera": microscope.camera}
            emitting = sorted(r.population for r in self.routes.values() if r.element == emit[0])
            _check_shared_knobs(self.fidelity, emitting)
            population = emitting[0] if emitting else None
            imaging = build(emit[0], parts, self.fidelity, population)
        excite = None
        if microscope.light is not None and emit:
            from gradix.excite.linear import Linear

            excite = Linear()  # emitters are excited by the microscope's light
        return Chain(
            light=microscope.light,
            emitters=dict(sample.populations),
            excite=excite,
            imaging=imaging,
            references=dict(microscope.references),  # refused by the executor: not coherent
            detection_optics=dict(microscope.detection_optics),
            camera=microscope.camera,
            background=microscope.background,
            acquisition=microscope.acquisition,
            environment=sample.environment,
        )

    def _coherent_chain(
        self, sample: Sample, microscope: Microscope, interact: Mapping[str, str]
    ) -> Chain:
        """Build the coherent Chain: the light, one interaction element per population, imaging."""
        if microscope.light is None:
            raise PlanError(
                "scatterers need coherent light",
                fix="use gx.presets.InlineHolography(light=gx.light.PlaneWave(λ), ...)",
            )
        scatterers = {
            name: build(choice, {"objects": sample.populations[name]}, self.fidelity, name)
            for name, choice in sorted(interact.items())
        }
        parts = {"objective": microscope.objective, "camera": microscope.camera}
        return Chain(
            light=microscope.light,
            scatterers=scatterers,
            references=dict(microscope.references),
            detection_optics=dict(microscope.detection_optics),
            imaging=build("coherent.pupil", parts, self.fidelity),
            background=microscope.background,
            acquisition=microscope.acquisition,
            environment=sample.environment,
        )

    def __call__(
        self,
        sample: Sample | Mapping[str, object] | None = None,
        microscope: Microscope | None = None,
        *,
        key: int | Tensor | None = None,
        check: str | None = None,
    ) -> Output:
        """Render: a new Sample and Microscope, values bound by field path, or the template.

        Parameters
        ----------
        sample : Sample or Mapping[str, object], optional
            A structure-compatible Sample (with ``microscope``), or values by field path.
        microscope : Microscope, optional
            The optics for ``sample``.
        key : int or Tensor, optional
            Detector-noise key.
        check : {"raise"}, optional
            Raise when an image leaves the envelope.

        Returns
        -------
        Output
            The requested outputs.
        """
        if sample is None:
            return self.pipeline(key=key, check=check)
        if isinstance(sample, Sample):
            if microscope is None:
                raise StructureError("pass the microscope with the sample")
            return self.pipeline(self.chain(sample, microscope), key=key, check=check)
        return self.pipeline(sample, key=key, check=check)

    def explain(self) -> str:
        """Return the Pipeline report headed by the fidelity and the element choices.

        Returns
        -------
        str
            The report.
        """
        lines = [f"Plan · fidelity={self.fidelity.preset}"]
        for name, r in sorted(self.routes.items()):
            lines.append(f"choice     {name} → {r.element}  ({r.reason})")
        return "\n".join(lines) + "\n" + self.pipeline.explain()

    def why(self, population: str) -> str:
        """Return the decision trace of one population.

        Parameters
        ----------
        population : str
            Population name.

        Returns
        -------
        str
            Which element renders it and the rule or pin that chose it.
        """
        r = self.routes.get(population)
        if r is None:
            raise StructureError(
                f"no population {population!r}", fix=f"populations: {sorted(self.routes)}"
            )
        return f"{population} → {r.element}: {r.reason}"


def plan(
    sample: Sample,
    microscope: Microscope,
    fidelity: Fidelity | str = "standard",
    *,
    outputs: object = ("image",),
    envelope: Envelope | Mapping[str, tuple[float, float] | Entry] | None = None,
    methods: Mapping[str, str] | None = None,
    inputs: Sequence[str] | None = None,
    batch: int | None = None,
    slots: Mapping[str, int] | None = None,
    deterministic: bool = False,
) -> Plan:
    """Plan a render: choose elements by fidelity, build a Chain, wrap it in a Pipeline.

    Parameters
    ----------
    sample : Sample
        Example inputs.
    microscope : Microscope
        The optics.
    fidelity : Fidelity or str, default "standard"
        The policy.
    outputs : str, tuple of str, or Mapping, default ("image",)
        What calls compute.
    envelope : Envelope or Mapping, optional
        Explicit envelope entries.
    methods : Mapping[str, str], optional
        Elements pinned per population.
    inputs : sequence of str, optional
        Field paths that calls may bind.
    batch : int, optional
        Batch capacity.
    slots : Mapping[str, int], optional
        Slot capacity per population.
    deterministic : bool, default False
        Record that calls must be bit-exact.

    Returns
    -------
    Plan
        The plan.
    """
    return Plan(
        sample,
        microscope,
        fidelity,
        outputs=outputs,
        envelope=envelope,
        methods=methods,
        inputs=inputs,
        batch=batch,
        slots=slots,
        deterministic=deterministic,
    )


_CACHE: collections.OrderedDict[tuple[object, ...], Plan] = collections.OrderedDict()


def clear_cache() -> None:
    """Forget every plan cached by :func:`render`."""
    _CACHE.clear()


def _check_shared_knobs(fidelity: Fidelity, populations: list[str]) -> None:
    """Refuse per-population knobs that differ across populations sharing one element."""
    for knob in ("psf", "raster_sigma", "emitter_path"):
        values = {repr(fidelity.knob(knob, population=p)) for p in populations}
        if len(values) > 1:
            msg = f"populations {populations} share one imaging element but set {knob} differently"
            raise PlanError(msg, fix="use one value, or render the populations separately")


def _counts(sample: Sample) -> dict[str, int]:
    out: dict[str, int] = {}
    for name, pop in sample.populations.items():
        if not isinstance(pop, ObjectSet):
            continue  # a volume has no slots
        n = 1
        for _p, spec, value in iter_leaves(pop):
            if spec is not None and spec.role == "object" and isinstance(value, Tensor):
                n = max(n, canonical_shape(spec, tuple(value.shape))[2])
        out[name] = n
    return out


def render(
    sample: Sample,
    microscope: Microscope,
    fidelity: Fidelity | str = "standard",
    *,
    key: int | Tensor | None = None,
    outputs: object = ("image",),
    envelope: Envelope | Mapping[str, tuple[float, float] | Entry] | None = None,
    methods: Mapping[str, str] | None = None,
    deterministic: bool = False,
) -> Output:
    """Render with a cached plan, re-planning when the inputs no longer fit it.

    The pre-render check reads the inputs' values (a host synchronisation on GPU inputs):
    values outside the cached plan's envelope, or more images or objects than its capacities,
    trigger a re-plan with a grown envelope or capacity, logged on the ``gradix`` logger.

    Parameters
    ----------
    sample : Sample
        The sample.
    microscope : Microscope
        The optics.
    fidelity : Fidelity or str, default "standard"
        The policy.
    key : int or Tensor, optional
        Detector-noise key.
    outputs : str, tuple of str, or Mapping, default ("image",)
        What to compute.
    envelope : Envelope or Mapping, optional
        Explicit envelope entries.
    methods : Mapping[str, str], optional
        Elements pinned per population.
    deterministic : bool, default False
        Record that calls must be bit-exact.

    Returns
    -------
    Output
        The rendered outputs.
    """
    fid = Fidelity.coerce(fidelity)
    specs = normalize_outputs(outputs)
    explicit = envelope if isinstance(envelope, Envelope) else Envelope(envelope or {})
    device = common_device(*(v for _, _, v in iter_leaves((sample, microscope))))
    cache_key = (
        signature((sample, microscope)).structure_digest,
        repr(fid.to_json()),
        repr({k: repr(v) for k, v in specs.items()}),
        tuple(sorted((methods or {}).items())),
        explicit.key(),
        deterministic,
        str(device),
    )
    batch = tree_axis_sizes((sample, microscope)).get("B", 1)
    counts = _counts(sample)
    cached = _CACHE.get(cache_key)
    if cached is not None:
        pipe = cached.pipeline
        fits = batch <= pipe.batch and all(n <= pipe.slots.get(p, 0) for p, n in counts.items())
        chain = cached.chain(sample, microscope)
        inside = bool(in_envelope(pipe.envelope, chain, batch).all()) if fits else False
        if fits and inside:
            _CACHE.move_to_end(cache_key)
            return pipe(chain, key=key)
        # The union also grows explicit entries: re-applying them unchanged would re-plan on
        # every call once values leave them.
        grown = pipe.envelope.union(envelope_of(chain, inputs=pipe.inputs))
        log.info(
            "gx.render: re-planning (%s)",
            "values left the envelope" if fits else "capacity exceeded",
        )
        batch_cap = max(pipe.batch, batch_bucket(batch))
        slots = {p: max(pipe.slots.get(p, 0), slot_bucket(n)) for p, n in counts.items()}
    else:
        grown, batch_cap, slots = explicit, None, None
    planned = plan(
        sample,
        microscope,
        fid,
        outputs=specs,
        envelope=grown,
        methods=methods,
        batch=batch_cap,
        slots=slots,
        deterministic=deterministic,
    )
    _CACHE[cache_key] = planned
    _CACHE.move_to_end(cache_key)
    while len(_CACHE) > CACHE_SIZE:
        _CACHE.popitem(last=False)
    return planned.pipeline(planned.chain(sample, microscope), key=key)
