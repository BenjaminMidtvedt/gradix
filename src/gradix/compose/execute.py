"""Execution of a Chain: the one code path shared by eager Chain calls and Pipelines.

Both an eager ``chain()`` and a ``Pipeline`` call run the functions here; they differ only in
where the static configurations come from (the Chain's own inputs, or the Pipeline's
envelope). Given the same statics, the two levels are therefore bit-identical by construction
(§11.10).

The executor runs the emission path: emitter populations are lowered to what the imaging
element accepts (an :class:`~gradix.EmitterSet` or an :class:`~gradix.EmitterDensity`),
excited when the Chain has light and an excitation element, rendered into an
:class:`~gradix.Irradiance` and detected by the camera. The coherent path (M3a) runs the light
source, lowers every scatterer population to its spectra (:class:`~gradix.ObjectSpectra`) and
images them with the plane waves through a coherent imaging element; optical stages and
references join it in later M3a phases.

A render has three steps, so a chunked Pipeline can run the deterministic ones per chunk and
draw the noise once for the whole batch (identical results with batch or image keys):
:func:`expected` (the camera's expected frames), :func:`labels`, and :func:`finish` (noise
and output assembly). :func:`run` chains them.
"""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import Tensor

from gradix._core.carriers import (
    EmitterDensity,
    EmitterSet,
    Irradiance,
    ObjectSpectra,
    PlaneWaves,
)
from gradix._core.contract import Description, Element, Static, report
from gradix._core.envelope import envelope_of
from gradix._core.errors import PlanError, StructureError
from gradix.compose.chain import Chain
from gradix.compose.outputs import Expected, Image, OutputSpec
from gradix.compose.outputs import Field as FieldOutput
from gradix.compose.wiring import check_wiring
from gradix.detect.camera import Camera, CameraStatic
from gradix.labels.positions import Label
from gradix.lower.density import emitter_density
from gradix.lower.emitters import emitter_set, population_counts
from gradix.objects.acquisition import Acquisition, FocusStack
from gradix.objects.labeling import Labeling
from gradix.objects.objectset import Emitters, Solid
from gradix.objects.volumes import Voxels
from gradix.optics.objective import Objective
from gradix.schema.base import iter_leaves
from gradix.tree import replace as tree_replace

__all__ = [
    "acquired",
    "check_supported",
    "coherent",
    "concat",
    "eager_statics",
    "expected",
    "fields",
    "finish",
    "fresh",
    "irradiance",
    "labels",
    "run",
]


def check_supported(chain: Chain) -> None:
    """Raise :class:`~gradix._core.errors.PlanError` if a Chain uses slots M0 cannot render.

    Parameters
    ----------
    chain : Chain
        The Chain.

    Raises
    ------
    PlanError
        On coherent slots, a missing imaging element or camera, or an imaging element that does
        not consume emitters.
    """
    if chain.illumination_optics or chain.detection_optics:
        msg = "optical stages are not available yet (later M3a phases)"
        raise PlanError(msg, fix="render the Chain without them for now")
    if chain.references and not coherent(chain):
        raise PlanError("references need a coherent Chain", fix="use gx.imaging.Coherent")
    if coherent(chain):
        _check_coherent(chain)
        return
    if chain.scatterers:
        raise PlanError(
            "the Chain has scatterers but its imaging element is not coherent",
            fix="use gx.imaging.Coherent(objective, camera)",
        )
    if chain.excite is not None:
        if not isinstance(chain.excite, Element) or PlaneWaves not in chain.excite.caps.accepts:
            raise PlanError("excite= must be a transduction element such as gx.excite.Linear()")
        if not isinstance(chain.light, Element):
            raise PlanError("excitation needs a light source", fix="add light=gx.light.Uniform()")
    if not chain.emitters:
        raise PlanError("the Chain has nothing to render", fix="add emitters={...}")
    if not isinstance(chain.imaging, Element):
        raise PlanError("the Chain has emitters but no imaging element", fix="add imaging=...")
    accepts = chain.imaging.caps.accepts
    imaging = type(chain.imaging).__name__
    if EmitterSet not in accepts and EmitterDensity not in accepts:
        raise PlanError(f"{imaging} does not render emitters", fix="use gx.imaging.Sprites")
    points = EmitterSet in accepts
    grids = set()
    for name, pop in chain.emitters.items():
        kind = type(pop).__name__
        if isinstance(pop, Solid) and not isinstance(pop.labeling, Labeling):
            msg = f"emitters[{name!r}] is a {kind} without a labeling: it emits nothing"
            raise PlanError(msg, fix="pass labeling=gx.Labeling(...), or render it as a scatterer")
        if not isinstance(pop, Emitters if points else (Voxels, Solid)):
            wanted = "gx.Emitters" if points else "gx.Voxels densities and labelled solids"
            msg = f"emitters[{name!r}] is a {kind}; {imaging} renders {wanted}"
            fix = "points render with Sprites/PointPSF; densities and solids with gx.imaging.Strata"
            raise PlanError(msg, fix=fix)
        if isinstance(pop, Voxels):
            if pop.quantity != "density":
                msg = f"emitters[{name!r}] holds quantity={pop.quantity!r}, not an emitter density"
                raise PlanError(msg, fix="use quantity='density' (photons per voxel)")
            grids.add(pop.grid())
    if len(grids) > 1:
        raise PlanError("density populations must share one grid (shape, spacing, origin)")
    if not isinstance(chain.camera, Camera):
        raise PlanError(
            "the Chain has no camera", fix="add camera=... or an imaging element with one"
        )
    if chain.environment is None:
        raise PlanError("the Chain has no environment", fix="add environment=gx.env.Homogeneous(n)")
    check_wiring(chain)
    acquisition = chain.acquisition
    if isinstance(acquisition, Acquisition):
        _check_drives(acquisition, chain)


def coherent(chain: Chain) -> bool:
    """Return whether a Chain renders scattered light: its imaging element takes spectra.

    Parameters
    ----------
    chain : Chain
        The Chain.

    Returns
    -------
    bool
        True when the imaging element accepts :class:`~gradix.ObjectSpectra`.
    """
    imaging = chain.imaging
    return isinstance(imaging, Element) and ObjectSpectra in imaging.caps.accepts


def _makes(element: Element, carrier: type) -> bool:
    produced = element.caps.produces
    return any(
        issubclass(p, carrier) for p in (produced if isinstance(produced, tuple) else (produced,))
    )


def _check_coherent(chain: Chain) -> None:
    """Raise :class:`~gradix._core.errors.PlanError` for coherent Chains M3a cannot render."""
    if chain.emitters:
        msg = "a coherent Chain cannot also hold emitters (fluorescence and scattering together)"
        raise PlanError(msg, fix="render emitters and scatterers in separate Chains")
    if chain.excite is not None:
        raise PlanError("a coherent Chain has no excitation element", fix="drop excite=")
    light = chain.light
    if not isinstance(light, Element) or not _makes(light, PlaneWaves):
        fix = "add light=gx.light.PlaneWave(λ)"
        raise PlanError("a coherent Chain needs plane-wave light", fix=fix)
    for name, element in chain.scatterers.items():
        if not isinstance(element, Element) or not _makes(element, ObjectSpectra):
            msg = f"scatterers[{name!r}] must be an interaction element such as gx.interact.Mie"
            raise PlanError(msg, fix=f"scatterers={{{name!r}: gx.interact.Mie(spheres)}}")
    for name, element in chain.references.items():
        if not isinstance(element, Element) or not _makes(element, PlaneWaves):
            msg = f"references[{name!r}] must be a reference element such as gx.light.ReferenceBeam"
            raise PlanError(msg, fix=f"references={{{name!r}: gx.light.ReferenceBeam(...)}}")
    if not isinstance(chain.camera, Camera):
        raise PlanError("the Chain has no camera", fix="give the imaging element a camera")
    if chain.environment is None:
        raise PlanError("the Chain has no environment", fix="add environment=gx.env.Homogeneous(n)")
    check_wiring(chain)
    acquisition = chain.acquisition
    if isinstance(acquisition, Acquisition):
        _check_drives(acquisition, chain)


def _check_drives(acquisition: Acquisition, chain: Chain) -> None:
    """Refuse an acquisition whose declared ``drives`` and ``bindings()`` disagree.

    The gradient table routes an acquisition's fields only through ``drives``: a declared
    target it never binds is stale, and a bound setting that none of its floating tensor
    fields is declared to drive would report ``zero`` for a field that trains.
    """
    name = type(acquisition).__name__

    def logical(path: str) -> str:  # "imaging.objective.focus" and "objective.focus" agree
        return chain.logical(chain.resolve(path)[0])

    declared = {logical(p) for p in acquisition.drives.values()}
    # which bindings depend on the acquisition's own floating tensors (probed with autograd)
    probes = {
        path: value.detach().clone().requires_grad_()
        for path, _spec, value in iter_leaves(acquisition)
        if isinstance(value, Tensor) and value.is_floating_point()
    }
    probe = tree_replace(acquisition, probes) if probes else acquisition
    with torch.enable_grad():
        values = probe.bindings(chain)
    bound = {logical(p) for p in values}
    stale = sorted(declared - bound)
    if stale:
        msg = f"{name} declares drives for {stale} but binds only {sorted(bound)}"
        raise PlanError(msg, fix="make drives name the paths bindings() returns")
    dependent = {logical(p) for p, v in values.items() if isinstance(v, Tensor) and v.requires_grad}
    undeclared = sorted(dependent - declared)
    if undeclared:
        msg = f"{name} binds {undeclared} from its own fields but declares no drives for them"
        raise PlanError(msg, fix="declare drives = {field: path} for the fields that set them")


def acquired(chain: Chain) -> Chain:
    """Return the Chain with its acquisition's per-frame settings bound (§6.7).

    Parameters
    ----------
    chain : Chain
        The Chain.

    Returns
    -------
    Chain
        The Chain whose ``setting`` fields carry the acquisition's per-frame values (a focus
        stack's focal planes), or the Chain itself when the acquisition binds nothing.
    """
    return chain.acquired()


def _emission(
    chain: Chain, statics: Mapping[str, Static] | None = None
) -> EmitterSet | EmitterDensity:
    """Lower the emitting populations to what the imaging element accepts (on its grid)."""
    imaging = chain.imaging
    if isinstance(imaging, Element) and EmitterSet not in imaging.caps.accepts:
        pops = {k: v for k, v in chain.emitters.items() if isinstance(v, (Voxels, Solid))}
        static = (statics or {}).get("imaging")
        request = imaging.density_request(static) if static is not None else None
        return emitter_density(pops, request=request, acq=chain.acq_index())
    return _emitters(chain)


def _emitters(chain: Chain) -> EmitterSet:
    pops = {k: v for k, v in chain.emitters.items() if isinstance(v, Emitters)}
    acquisition = chain.acquisition
    if isinstance(getattr(acquisition, "source", acquisition), FocusStack):
        lowered = emitter_set(pops)
        if lowered.position.shape[1] != 1:
            msg = "a focus stack images a static scene: per-frame fields (T > 1) are not allowed"
            raise StructureError(msg, fix="give positions, photons and presence T = 1")
        return lowered.replace(acq=chain.acq_index())
    # without an acquisition, the time axis is the T of every per-frame field and setting
    return emitter_set(pops, acq=chain.acq_index())


def eager_statics(chain: Chain) -> dict[str, Static]:
    """Configure every element of a Chain on the envelope of its own inputs (eager, L2 semantics).

    Parameters
    ----------
    chain : Chain
        The Chain.

    Returns
    -------
    dict of str to Static
        Static configurations by element path.
    """
    check_supported(chain)
    chain = acquired(chain)  # size the grids for the settings the acquisition drives
    imaging = chain.imaging
    if not isinstance(imaging, Element):  # pragma: no cover - checked above
        raise PlanError("no imaging element")
    if coherent(chain):
        return _coherent_statics(chain, imaging)
    if any(isinstance(p, Solid) for p in chain.emitters.values()):
        # solids are rasterised onto the grid the element chooses (ADR-41): configure first,
        # on the populations' own envelope (no headroom), as a Pipeline does on its envelope
        envelope = envelope_of(chain, headroom=1.0)
        desc = Description(
            path="imaging",
            populations=tuple(sorted(chain.emitters)),
            parts={"objective": "objective", "camera": "camera", "environment": "environment"},
            nodes=chain.parts(),
        )
        report(imaging.validity(desc, envelope), "warn")
        return {"imaging": imaging.configure(desc, envelope)}
    return {"imaging": imaging.eager_static(_emission(chain), chain.environment)}


def _coherent_statics(chain: Chain, imaging: Element) -> dict[str, Static]:
    """Configure a coherent Chain's scatterers and imaging element on their own inputs."""
    waves = _waves(chain, {})
    statics: dict[str, Static] = {}
    for name, element in sorted(chain.scatterers.items()):
        statics[f"scatterers.{name}"] = element.eager_static(waves, chain.environment)
    contributions = _contributions(chain, waves, statics)
    references = _references(chain, waves, statics)
    statics["imaging"] = imaging.eager_static(contributions, waves, chain.environment, references)
    return statics


def _waves(chain: Chain, statics: Mapping[str, Static]) -> PlaneWaves:
    light = chain.light
    if not isinstance(light, Element):  # pragma: no cover - checked by check_supported
        raise PlanError("no light source")
    waves = light.forward(static=statics.get("light", Static()))
    if not isinstance(waves, PlaneWaves):
        raise StructureError(f"{type(light).__name__} returned {type(waves).__name__}")
    return waves


def _references(
    chain: Chain, waves: PlaneWaves, statics: Mapping[str, Static]
) -> tuple[PlaneWaves, ...]:
    """Evaluate the Chain's references at the light's wavelengths (in their declared order)."""
    out: list[PlaneWaves] = []
    for name, element in chain.references.items():
        reference = element.forward(waves, static=statics.get(f"references.{name}", Static()))
        if not isinstance(reference, PlaneWaves):
            kind = type(reference).__name__
            raise StructureError(f"references[{name!r}] returned {kind}, not PlaneWaves")
        out.append(reference)
    return tuple(out)


def _contributions(
    chain: Chain, waves: PlaneWaves, statics: Mapping[str, Static]
) -> tuple[ObjectSpectra, ...]:
    """Lower every scatterer population to its spectra (in sorted name order)."""
    out: list[ObjectSpectra] = []
    for name, element in sorted(chain.scatterers.items()):
        static = statics.get(f"scatterers.{name}", Static())
        for contribution in element.forward(waves, chain.environment, static=static):
            if not isinstance(contribution, ObjectSpectra):
                kind = type(contribution).__name__
                raise StructureError(f"scatterers[{name!r}] returned {kind}, not ObjectSpectra")
            out.append(contribution)
    return tuple(out)


def _excited(chain: Chain, statics: Mapping[str, Static]) -> EmitterSet | EmitterDensity:
    """Return the emission the imaging element sees: excited by the light when there is some."""
    emitters = _emission(chain, statics)
    if chain.excite is None or not isinstance(chain.light, Element):
        return emitters
    light = chain.light
    waves = light.forward(static=statics.get("light", Static()))
    excite = chain.excite
    if not isinstance(excite, Element):  # pragma: no cover - checked by check_supported
        raise PlanError("no excitation element")
    out = excite.forward(emitters, waves, chain.environment, static=statics.get("excite", Static()))
    if not isinstance(out, (EmitterSet, EmitterDensity)):
        raise StructureError(f"{type(excite).__name__} returned {type(out).__name__}")
    return out


def irradiance(chain: Chain, statics: Mapping[str, Static]) -> Irradiance:
    """Render the Chain's light up to the camera.

    Parameters
    ----------
    chain : Chain
        The Chain.
    statics : Mapping[str, Static]
        Static configurations by element path.

    Returns
    -------
    Irradiance
        Photons per pixel on the camera grid.
    """
    imaging = chain.imaging
    if not isinstance(imaging, Element):  # pragma: no cover - checked by check_supported
        raise PlanError("no imaging element")
    if coherent(chain):
        waves = _waves(chain, statics)
        contributions = _contributions(chain, waves, statics)
        references = _references(chain, waves, statics)
        out = imaging.forward(
            contributions, waves, chain.environment, references, static=statics["imaging"]
        )
        if isinstance(out, Irradiance):  # the frames the acquisition declares (a focus stack)
            out = out.replace(acq=chain.acq_index())
    else:
        out = imaging.forward(
            _excited(chain, statics), chain.environment, static=statics["imaging"]
        )
    if not isinstance(out, Irradiance):
        raise StructureError(
            f"{type(imaging).__name__} returned {type(out).__name__}, not Irradiance"
        )
    return out


def _camera(chain: Chain) -> Camera:
    camera = chain.camera
    if not isinstance(camera, Camera):  # pragma: no cover - checked by check_supported
        raise PlanError("no camera")
    return camera


def expected(chain: Chain, statics: Mapping[str, Static]) -> Tensor:
    """Return the camera's expected frames of a Chain (deterministic: no noise).

    Parameters
    ----------
    chain : Chain
        The Chain.
    statics : Mapping[str, Static]
        Static configurations by element path.

    Returns
    -------
    Tensor
        Expected frames in the camera's unit, ``[B|1, C, H, W]`` or ``[B|1, T, C, H, W]``.
    """
    chain = acquired(chain)
    camera = _camera(chain)
    return camera.forward(
        irradiance(chain, statics),
        chain.background,
        static=statics.get("camera", CameraStatic()),
    )


def labels(
    chain: Chain, outputs: Mapping[str, OutputSpec], batch: int | None = None
) -> dict[str, Tensor]:
    """Render the label outputs of a Chain.

    Parameters
    ----------
    chain : Chain
        The Chain.
    outputs : Mapping[str, OutputSpec]
        Output specs by name; non-label specs are skipped.
    batch : int, optional
        Expand labels that do not vary over images to this many rows.

    Returns
    -------
    dict of str to Tensor
        Label tensors; a label ``name`` adds ``name + "." + suffix`` entries for its extras.
    """
    camera = _camera(chain)
    values: dict[str, Tensor] = {}
    views: dict[str, EmitterSet] | None = None
    for name, spec in outputs.items():
        if not isinstance(spec, Label):
            continue
        population = getattr(spec, "population", None)
        objective = chain.objective
        if not isinstance(population, str) or not isinstance(objective, Objective):
            raise StructureError(f"label {name!r} needs a population and an objective")
        if views is None:
            views = _rendered_views(chain)
        rendered = spec.render(
            chain.population(population), camera, objective, emitters=views.get(population)
        )
        for suffix, tensor in rendered.items():
            values[f"{name}.{suffix}" if suffix else name] = tensor
    return _expand(values, batch)


def fields(
    chain: Chain,
    statics: Mapping[str, Static],
    outputs: Mapping[str, OutputSpec],
    batch: int | None = None,
) -> dict[str, Tensor]:
    """Render the field outputs (:class:`~gradix.out.Field`) of a coherent Chain.

    Parameters
    ----------
    chain : Chain
        The Chain.
    statics : Mapping[str, Static]
        Static configurations by element path.
    outputs : Mapping[str, OutputSpec]
        Output specs by name; specs other than fields are skipped.
    batch : int, optional
        Expand fields that do not vary over images to this many rows.

    Returns
    -------
    dict of str to Tensor
        Complex ``[B, A, H, W]`` (or its ``layout``) by output name.

    Raises
    ------
    StructureError
        If the Chain is not coherent, has several modes or bins, or normalises by a background
        that does not reach the camera.
    """
    specs = {name: spec for name, spec in outputs.items() if isinstance(spec, FieldOutput)}
    if not specs:
        return {}
    if not coherent(chain):
        msg = "field outputs need a coherent Chain (scatterers and gx.imaging.Coherent)"
        raise StructureError(msg)
    chain = acquired(chain)
    imaging = chain.imaging
    image_field = getattr(imaging, "image_field", None)
    if not callable(image_field):
        raise StructureError(f"{type(imaging).__name__} does not produce an image field")
    static = statics["imaging"]
    waves = _waves(chain, statics)
    contributions = _contributions(chain, waves, statics)
    total = image_field(contributions, waves, chain.environment, static=static)
    if total.shape[2] != 1 or total.shape[3] != 1:
        msg = "a field output needs one incoherent mode and one wavelength bin"
        raise StructureError(msg, fix="a partially coherent image has no single field")
    s = int(getattr(static, "oversample", 1))

    def pool(field: Tensor, sampling: str) -> Tensor:  # one value per camera pixel
        f = field[:, :, 0, 0]
        if sampling == "centre":  # the first sample of each pixel lies at its centre
            return f[..., ::s, ::s]
        b, a, hs, ws = f.shape
        return f.reshape(b, a, hs // s, s, ws // s, s).mean((3, 5))

    backgrounds: dict[str, Tensor] = {}
    out: dict[str, Tensor] = {}
    for name, spec in specs.items():
        value = pool(total, spec.sampling)
        if spec.normalize == "background":
            background = backgrounds.get(spec.sampling)
            if background is None:
                empty = image_field((), waves, chain.environment, static=static)
                background = backgrounds[spec.sampling] = pool(empty, spec.sampling)
            if bool((background.abs() == 0).any()):
                msg = "no background reaches the camera (epi or darkfield illumination)"
                raise StructureError(msg, fix="use gx.out.Field(normalize='incident')")
            value = value / background
        elif spec.normalize == "incident":
            amplitude = waves.amplitude[:, :, 0, 0, 0, 0].to(value.dtype)  # [B|1, A|1]
            value = value / amplitude[:, :, None, None]
        if spec.layout == "re_im":
            value = torch.view_as_real(value)
        elif spec.layout == "phase":
            value = torch.angle(value)
        elif spec.layout == "amplitude":
            value = value.abs()
        out[name] = value
    return _expand(out, batch)


def _rendered_views(chain: Chain) -> dict[str, EmitterSet]:
    """Return each emitter population's slice of the rendered (excited) EmitterSet."""
    if not chain.emitters or not isinstance(chain.imaging, Element):
        return {}
    excited = _excited(acquired(chain), {})
    if not isinstance(excited, EmitterSet):
        return {}
    views: dict[str, EmitterSet] = {}
    offset = 0
    pops = {k: v for k, v in chain.emitters.items() if isinstance(v, Emitters)}
    for name, count in population_counts(pops).items():  # emitter_set's concatenation order
        part = slice(offset, offset + count)
        views[name] = excited.replace(
            position=excited.position[:, :, part],
            photons=excited.photons[..., part],
            presence=None if excited.presence is None else excited.presence[..., part],
            species=None if excited.species is None else excited.species[..., part],
            dipole=None if excited.dipole is None else excited.dipole[:, :, part],
            id=None if excited.id is None else excited.id[..., part],
        )
        offset += count
    return views


def finish(
    chain: Chain,
    mu: Tensor | None,
    key: int | Tensor | None,
    outputs: Mapping[str, OutputSpec],
) -> dict[str, Tensor]:
    """Assemble the frame outputs from expected frames: noise is drawn here, once.

    Parameters
    ----------
    chain : Chain
        The Chain (its camera draws the noise).
    mu : Tensor or None
        Expected frames of the whole batch; None when no frame output is requested.
    key : int or Tensor, optional
        The detector-noise key.
    outputs : Mapping[str, OutputSpec]
        Output specs by name; label specs are skipped.

    Returns
    -------
    dict of str to Tensor
        The ``expected`` and ``image`` outputs by name.
    """
    values: dict[str, Tensor] = {}
    if mu is None:
        return values
    image: Tensor | None = None
    for name, spec in outputs.items():
        if isinstance(spec, Expected):
            values[name] = mu
        elif isinstance(spec, Image):
            if image is None:
                image = _camera(chain).sample(mu, key, stream="camera")
            values[name] = image
    return values


def _expand(values: Mapping[str, Tensor], batch: int | None) -> dict[str, Tensor]:
    out: dict[str, Tensor] = {}
    for name, t in values.items():
        if batch is not None and t.ndim > 0 and t.shape[0] == 1 and batch != 1:
            t = t.expand(batch, *t.shape[1:]).contiguous()
        out[name] = t
    return out


def _storage(t: Tensor) -> int:
    """Identify a tensor's memory (by identity for tensors without storage, e.g. under vmap)."""
    if not t.numel():
        return id(t)
    try:
        return t.untyped_storage().data_ptr()
    except (NotImplementedError, RuntimeError):  # functorch wrappers have no storage
        return id(t)


def fresh(
    values: Mapping[str, Tensor], batch: int | None = None, *, inputs: object = None
) -> dict[str, Tensor]:
    """Make outputs safe to hand out: one row per image, no aliasing.

    Parameters
    ----------
    values : Mapping[str, Tensor]
        Output tensors by name.
    batch : int, optional
        The call's batch size; outputs with one row are expanded to it.
    inputs : object, optional
        The inputs (a Chain); outputs that share storage with one of its tensors are cloned.

    Returns
    -------
    dict of str to Tensor
        Tensors that alias neither the inputs nor each other (repeated outputs are cloned).
    """
    out = _expand(values, batch)
    seen: set[int] = set()
    if inputs is not None:
        seen.update(_storage(v) for _p, _s, v in iter_leaves(inputs) if isinstance(v, Tensor))
    for name, t in out.items():
        storage = _storage(t)
        if storage in seen:
            out[name] = t.clone()
        else:
            seen.add(storage)
    return out


def run(
    chain: Chain,
    statics: Mapping[str, Static],
    *,
    key: int | Tensor | None,
    outputs: Mapping[str, OutputSpec],
    batch: int | None = None,
) -> dict[str, Tensor]:
    """Render the requested outputs of a Chain.

    Parameters
    ----------
    chain : Chain
        The Chain.
    statics : Mapping[str, Static]
        Static configurations by element path.
    key : int or Tensor, optional
        The detector-noise key.
    outputs : Mapping[str, OutputSpec]
        Output specs by name.
    batch : int, optional
        The call's batch size; every output gets one row per image.

    Returns
    -------
    dict of str to Tensor
        Output tensors; each is a fresh tensor that aliases neither the inputs nor another
        output. A label ``name`` adds ``name + "." + suffix`` entries for its extra tensors.
    """
    wants_frames = any(isinstance(s, (Image, Expected)) for s in outputs.values())
    mu = expected(chain, statics) if wants_frames else None
    values = finish(chain, mu, key, outputs)
    values.update(labels(chain, outputs))
    values.update(fields(chain, statics, outputs))
    return fresh(values, batch, inputs=chain)


def concat(parts: list[dict[str, Tensor]]) -> dict[str, Tensor]:
    """Concatenate per-chunk outputs along the batch axis.

    Parameters
    ----------
    parts : list of dict
        Outputs of consecutive chunks, each with one row per image.

    Returns
    -------
    dict of str to Tensor
        The batch's outputs.
    """
    if not parts:
        return {}
    return {name: torch.cat([p[name] for p in parts], 0) for name in parts[0]}
