"""The planner's rule table (pass P2): deterministic routing of populations to elements (§5.7).

Every automatic choice between implementations with different approximation error is a rule on
static quantities (the fidelity knobs, the envelope, the slot counts), never on device cost, so
numerics do not differ between machines. Routing runs after the envelope and the capacities are
known (P0, P1), so a rule may depend on them: a :data:`RULES` entry is a registry name or a
callable of a :class:`RouteContext`.

Knob values double as registry names. A value with a dot (``"emit.pupil_mft"``) names an element
directly; a bare value that no rule knows (``"tmatrix"``) is looked up as ``"<family>.<value>"``
(``"interact.tmatrix"``), so a registered plugin is selected exactly like a built-in.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Callable, Mapping

from gradix._core.carriers import EmitterDensity, EmitterSet, PlaneWaves
from gradix._core.contract import Element
from gradix._core.envelope import Envelope
from gradix._core.errors import PlanError, RegistryError
from gradix._core.registry import elements
from gradix.objects.objectset import ObjectSet, Solid
from gradix.objects.volumes import Voxels
from gradix.planner.fidelity import Fidelity

__all__ = [
    "BUILDERS",
    "ELEMENT_KNOBS",
    "FAMILIES",
    "KNOBS",
    "RULES",
    "Route",
    "RouteContext",
    "build",
    "route",
    "routing",
]


@dataclasses.dataclass(frozen=True)
class RouteContext:
    """What a routing rule may look at: static quantities only.

    Parameters
    ----------
    name : str
        Population name.
    population : ObjectSet
        The population (its type and static fields; tensor values are not to be read).
    fidelity : Fidelity
        The policy.
    envelope : Envelope
        The envelope of the inputs (P1).
    count : int
        The population's slot count.
    """

    name: str
    population: ObjectSet | Voxels
    fidelity: Fidelity
    envelope: Envelope
    count: int


Rule = str | Callable[[RouteContext], str]

DIPOLE_X_MAX = 0.75
"""Largest size parameter k·r for which ``spheres="auto"`` takes the Mie-dressed dipole (within
5 % of Mie over all angles for dielectrics, §5.2)."""


def _sphere_auto(ctx: RouteContext) -> str:
    """Choose Mie-dressed dipoles for small spheres and Mie otherwise, from the envelope."""
    radius = ctx.envelope.range(f"{ctx.name}.radius")
    wavelength = ctx.envelope.range("light.wavelength")
    index = ctx.envelope.range("environment.n")
    if radius is None or wavelength is None or index is None:
        return "interact.mie"
    x_max = 2.0 * math.pi * index[1] * radius[1] / wavelength[0]
    return "interact.dipole" if x_max <= DIPOLE_X_MAX else "interact.mie"


"""A rule's result: a registry name, or a callable choosing one from the context."""

RULES: dict[tuple[str, str, str, str], Rule] = {
    ("point", "emission", "emitters", "gaussian"): "emit.gaussian",
    ("point", "emission", "emitters", "pupil"): "emit.pupil_mft",
    ("sphere", "coherent", "spheres", "auto"): _sphere_auto,
    ("sphere", "coherent", "spheres", "dipole"): "interact.dipole",
    ("sphere", "coherent", "spheres", "mie"): "interact.mie",
    ("sphere", "coherent", "spheres", "born"): "interact.born_ff",
    ("compact", "coherent", "compact", "born"): "interact.born_ff",
    ("compact", "coherent", "compact", "projection"): "interact.projection",
    ("volume", "coherent", "volumes", "projection"): "interact.projection",
    ("volume", "coherent", "volumes", "multislice"): "interact.multislice",
    ("labeled", "emission", "emitter_path", "auto"): "emit.strata_otf",
    ("labeled", "emission", "emitter_path", "strata"): "emit.strata_otf",
    ("voxels", "emission", "emitter_path", "auto"): "emit.strata_otf",
    ("voxels", "emission", "emitter_path", "strata"): "emit.strata_otf",
}
"""``(population kind, contrast, knob, value) → rule`` (§5.7)."""

KNOBS: dict[tuple[str, str], str] = {
    ("point", "emission"): "emitters",
    ("sphere", "coherent"): "spheres",
    ("compact", "coherent"): "compact",
    ("volume", "coherent"): "volumes",
    ("voxels", "emission"): "emitter_path",
    ("labeled", "emission"): "emitter_path",
}
"""The fidelity knob that routes each ``(kind, contrast)``."""

ELEMENT_KNOBS: tuple[str, ...] = ("source", "dz", "pad", "roi")
"""Fidelity knobs without preset values that are written into elements declaring them."""

FAMILIES: dict[str, str] = {"emission": "emit", "coherent": "interact"}
"""Registry family of the elements that render each contrast; bare knob values resolve in it."""

_ACCEPTS: dict[tuple[str, str] | str, type] = {
    "coherent": PlaneWaves,
    "emission": EmitterSet,
    ("voxels", "emission"): EmitterDensity,
    ("labeled", "emission"): EmitterDensity,
}
"""The carrier an element must accept to render a population: by ``(kind, contrast)``, else by
contrast (point emitters by default)."""


def _build_emit(
    cls: type[Element], parts: Mapping[str, object], knobs: Mapping[str, object]
) -> Element:
    """Build an emission element by keyword: its objective, camera and fidelity knobs."""
    schema = cls.schema()
    kwargs = {name: parts[name] for name in ("objective", "camera") if name in schema}
    kwargs.update({k: v for k, v in knobs.items() if k in schema})
    try:
        return cls(**kwargs)
    except TypeError as err:
        msg = f"cannot build {cls.__name__} from {sorted(kwargs)}: {err}"
        raise PlanError(
            msg, fix="give the element keyword fields named objective and camera"
        ) from None


def _build_interact(
    cls: type[Element], parts: Mapping[str, object], knobs: Mapping[str, object]
) -> Element:
    """Build an interaction element around its population (``parts["objects"]``)."""
    schema = cls.schema()
    kwargs = {k: v for k, v in knobs.items() if k in schema}
    kwargs[getattr(cls, "population_field", "objects")] = parts["objects"]
    try:
        return cls(**kwargs)
    except TypeError as err:
        msg = f"cannot build {cls.__name__} around its population: {err}"
        raise PlanError(msg, fix="name the population field with population_field") from None


BUILDERS: dict[
    str, Callable[[type[Element], Mapping[str, object], Mapping[str, object]], Element]
] = {
    "emit": _build_emit,
    "coherent": _build_emit,
    "interact": _build_interact,
}
"""How the planner constructs an element of each registry family from the microscope's parts."""


@dataclasses.dataclass(frozen=True)
class Route:
    """The element chosen for a population, and why.

    Parameters
    ----------
    population : str
        Population name.
    element : str
        Registry name of the chosen element.
    reason : str
        The rule or pin that chose it.
    """

    population: str
    element: str
    reason: str


def _resolve(value: str, family: str | None) -> str | None:
    """Return the registry name a knob value or pin stands for, if one is registered."""
    if "." in value:
        return value
    if family is not None and f"{family}.{value}" in elements:
        return f"{family}.{value}"
    return None


def routing(pop: ObjectSet | Voxels) -> tuple[str, str]:
    """Return the ``(kind, contrast)`` a population routes on.

    Parameters
    ----------
    pop : ObjectSet or Voxels
        The population.

    Returns
    -------
    tuple of str
        ``("labeled", "emission")`` for a solid with a labeling (it renders as a fluorescent
        density), else the population's own kind and contrast.
    """
    if isinstance(pop, Solid) and pop.labeling is not None:
        return "labeled", "emission"
    return pop.kind, pop.contrast


def _check_element(name: str, choice: str, pop: ObjectSet | Voxels) -> None:
    try:
        cls = elements.get(choice)
    except RegistryError:
        msg = f"population {name!r} routes to {choice!r}, which is not available in this version"
        if routing(pop)[1] == "emission":
            pin = {name: "emit.gaussian"}
            fix = f"use fidelity='draft' (Gaussian sprites) or pin methods={pin!r}"
        else:
            fix = "Born, projection and multislice arrive with M3 and M4; Mie is available"
        raise PlanError(msg, fix=fix) from None
    kind, contrast = routing(pop)
    carrier = _ACCEPTS.get((kind, contrast), _ACCEPTS.get(contrast))
    if not isinstance(cls, type) or not issubclass(cls, Element):
        raise PlanError(f"{choice!r} is not an element class (population {name!r})")
    if carrier is not None and carrier not in cls.caps.accepts:
        msg = f"{choice!r} does not accept {carrier.__name__}, so it cannot render {name!r}"
        raise PlanError(msg, fix="pin an element that renders this population")


def route(
    populations: Mapping[str, ObjectSet | Voxels],
    fidelity: Fidelity,
    methods: Mapping[str, str] | None = None,
    *,
    envelope: Envelope | None = None,
    counts: Mapping[str, int] | None = None,
) -> dict[str, Route]:
    """Choose an element for every population (pass P2).

    Parameters
    ----------
    populations : Mapping[str, ObjectSet or Voxels]
        Populations by name.
    fidelity : Fidelity
        The policy.
    methods : Mapping[str, str], optional
        Pinned elements per population (registry names, or bare names in the population's
        family); they bypass the rules.
    envelope : Envelope, optional
        The envelope of the inputs, for rules that depend on it.
    counts : Mapping[str, int], optional
        Slot count per population, for rules that depend on it.

    Returns
    -------
    dict of str to Route
        The choice per population.

    Raises
    ------
    PlanError
        If a pin names an unknown population, no rule applies, or the chosen element is not
        registered or does not accept the population's carrier.
    """
    methods = dict(methods or {})
    unknown = set(methods) - set(populations)
    if unknown:
        raise PlanError(f"methods= names unknown populations {sorted(unknown)}")
    env = envelope if envelope is not None else Envelope()
    out: dict[str, Route] = {}
    for name, pop in sorted(populations.items()):
        kind, contrast = routing(pop)
        family = FAMILIES.get(contrast)
        if name in methods:
            pinned = methods[name]
            choice = _resolve(pinned, family) or pinned
            reason = "pinned by methods="
        else:
            knob = KNOBS.get((kind, contrast))
            if knob is None:
                msg = f"no rule renders population {name!r} ({kind}, {contrast})"
                raise PlanError(msg, fix="pin an element with methods={...}")
            value = str(fidelity.knob(knob, population=name))
            rule = RULES.get((kind, contrast, knob, value))
            if rule is not None:
                ctx = RouteContext(name, pop, fidelity, env, int((counts or {}).get(name, 1)))
                choice = str(rule(ctx)) if callable(rule) else str(rule)
                reason = f"{knob}={value!r} at fidelity {fidelity.preset!r}"
            else:
                resolved = _resolve(value, family)
                if resolved is None:
                    msg = (
                        f"no rule or registered element for {knob}={value!r} (population {name!r})"
                    )
                    raise PlanError(
                        msg, fix="choose another knob value, register it, or pin methods="
                    )
                choice, reason = resolved, f"{knob}={value!r} (registry name)"
        _check_element(name, choice, pop)
        out[name] = Route(name, choice, reason)
    return out


def build(
    choice: str, parts: Mapping[str, object], fidelity: Fidelity, population: str | None = None
) -> Element:
    """Construct a routed element, writing the fidelity's explicit knob values into it (§5.6).

    Parameters
    ----------
    choice : str
        Registry name of the element.
    parts : Mapping[str, object]
        The microscope's parts (``objective``, ``camera``).
    fidelity : Fidelity
        The policy; every knob the element declares as a field and the policy does not leave
        ``"auto"`` is passed to it.
    population : str, optional
        Population whose per-population overrides apply.

    Returns
    -------
    Element
        The element.

    Raises
    ------
    PlanError
        If the family has no builder or the element cannot be built.
    """
    cls = elements.get(choice)
    family = choice.split(".")[0]
    builder = BUILDERS.get(family)
    if builder is None:
        raise PlanError(f"the planner cannot build {family!r} elements yet ({choice!r})")
    schema = cls.schema()
    knobs = {}
    for knob, target in cls.fidelity_knobs.items():  # only the knobs the element declares
        value = fidelity.knob(knob, population=population)
        name, values = (target, None) if isinstance(target, str) else target
        if name not in schema:
            msg = (
                f"{cls.__name__} declares the fidelity knob {knob!r} for a field {name!r} it lacks"
            )
            raise PlanError(msg, fix="fix the element's fidelity_knobs")
        if values is not None:
            if isinstance(value, str) and value in values:
                knobs[name] = values[value]
        elif value is not None and value != "auto":
            knobs[name] = value
    return builder(cls, parts, knobs)
