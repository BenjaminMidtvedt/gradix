"""Regressions for the M1 audits: focus stacks in layered media, pupil edges, AD, sensors."""

import math

import pytest
import torch
from scenes import emitters, optics

import gradix as gx


def _bead(z, dtype=torch.float64):
    return gx.Emitters(
        position=torch.tensor([[[1.04, 1.04, z]]], dtype=dtype),
        photons=1000.0,
        emission=gx.Spectrum.line(0.68),
    )


def test_a_focus_stack_refocuses_instead_of_moving_emitters_in_a_layered_medium():
    camera = gx.Camera(pixel_size=6.5, shape=(32, 32))
    objective = gx.Objective(NA=1.4, magnification=100)
    medium = gx.env.WaterOnCoverslip(sample=1.33)
    steps = torch.tensor([-0.4, 0.0, 0.4], dtype=torch.float64)
    bead = _bead(-0.2)

    def chain(obj, acquisition=None):
        return gx.Chain(
            emitters={"b": bead},
            imaging=gx.imaging.PointPSF(obj, camera, psf="scalar", roi=32),
            environment=medium,
            acquisition=acquisition,
        )

    stack = chain(objective, gx.acq.FocusStack(focus=steps))(outputs=("expected",))["expected"]
    for a, step in enumerate(steps.tolist()):
        single = chain(objective.replace(focus=step))(outputs=("expected",))["expected"]
        assert torch.allclose(stack[0, a], single[0, 0], rtol=1e-6, atol=1e-9), a


def test_focus_stacks_leave_the_excitation_at_the_true_depth():
    objective, camera = optics(shape=(24, 24))
    bead = gx.Emitters(
        position=torch.tensor([[[0.78, 0.78, -0.1]]]),
        photons=1000.0,
        emission=gx.Spectrum.line(0.6),
    )
    light = gx.light.Evanescent(angle=math.radians(70.0))
    stack = gx.Chain(
        light=light,
        emitters={"b": bead},
        excite=gx.excite.Linear(),
        imaging=gx.imaging.Sprites(objective, camera),
        environment=gx.env.Homogeneous(1.33),
        acquisition=gx.acq.FocusStack(focus=torch.tensor([-0.1, 0.1])),
    )(outputs=("expected",))["expected"]
    totals = stack.sum((-2, -1))[0]
    assert float(totals[0]) == pytest.approx(float(totals[1]), rel=1e-4)  # same excitation


def test_homogeneous_na_close_to_n_keeps_the_collection_efficiency():
    camera = gx.Camera(pixel_size=6.5, shape=(128, 128))  # wide enough for the PSF wings
    bead = _bead(0.0, dtype=torch.float32)
    bead = bead.replace(position=torch.tensor([[[4.16, 4.16, 0.0]]]))
    for na, tolerance in ((1.32, 0.02), (1.329, 0.05)):
        psf = gx.imaging.PointPSF(gx.Objective(NA=na, magnification=100), camera, psf="scalar")
        total = float(psf(gx.lower.emitter_set(bead), gx.env.Homogeneous(1.33)).data.sum())
        eta = float(gx.conventions.collection_efficiency(na, 1.33))
        # was 240× too bright; within a pupil cell of n the soft edge is under-resolved
        assert total == pytest.approx(1000.0 * eta, rel=tolerance), na


def test_gibson_lanni_mismatch_stays_finite_in_float32():
    camera = gx.Camera(pixel_size=6.5, shape=(24, 24))
    medium = gx.env.LayeredMedium(sample=1.33, design_coverslip=1.515)
    bead = _bead(-0.3, dtype=torch.float32)
    psf = gx.imaging.PointPSF(gx.Objective(NA=1.4, magnification=100), camera, psf="scalar")
    image = psf(gx.lower.emitter_set(bead), medium).data
    assert torch.isfinite(image).all() and image.sum() > 0


def test_learnable_na_and_index_are_usable_in_a_pipeline():
    camera = gx.Camera(pixel_size=6.5, shape=(24, 24))
    objective = gx.Objective(NA=torch.tensor(1.2), magnification=100)
    chain = gx.Chain(
        emitters={"b": _bead(0.0, dtype=torch.float32)},
        imaging=gx.imaging.PointPSF(objective, camera, psf="scalar"),
        environment=gx.env.Homogeneous(torch.tensor(1.33)),
    )
    with pytest.warns(gx.GradixWarning, match="NA at or above"):
        pipe = gx.Pipeline(chain, outputs=("expected",))
    assert pipe(chain)["expected"].sum() > 0


def test_forward_mode_through_layered_media_and_pixel_pupils():
    camera = gx.Camera(pixel_size=6.5, shape=(12, 12))
    bead = _bead(-0.1)
    bead = bead.replace(position=torch.tensor([[[0.39, 0.39, -0.1]]], dtype=torch.float64))

    def render(n_i, opd):
        medium = gx.env.LayeredMedium(sample=1.33, immersion=n_i, coverslip=1.518)
        pupil = (gx.pupil.PixelPupil(opd=opd),)
        objective = gx.Objective(NA=1.3, magnification=100, pupil=pupil)
        psf = gx.imaging.PointPSF(objective, camera, psf="scalar", roi=12, pupil_samples=32)
        return psf(gx.lower.emitter_set(bead), medium).data

    n_i = torch.tensor(1.518, dtype=torch.float64)
    opd = torch.zeros(8, 8, dtype=torch.float64)
    jac_n, jac_opd = torch.func.jacfwd(render, argnums=(0, 1))(n_i, opd)
    assert torch.isfinite(jac_n).all() and torch.isfinite(jac_opd).all()
    assert (jac_opd != 0).any()


def test_batch_keyed_poisson_handles_non_finite_and_huge_rates():
    rates = torch.tensor([[1.0, float("nan"), float("inf"), 1e10, -float("inf")]])
    counts = gx.ops.detect.poisson(rates, 3, "p")
    assert torch.isnan(counts[0, 1]) and torch.isnan(counts[0, 2]) and counts[0, 4] == 0
    assert float(counts[0, 3]) == pytest.approx(1e10, rel=1e-3)


def test_gradient_table_sees_emccd_gain_and_focus_stack_steps():
    objective, camera = optics(shape=(24, 24), noise=gx.noise.EMCCD(gain=torch.tensor(50.0)))
    chain = gx.Chain(
        emitters={"b": emitters(1, 2, fov=3.0)},
        imaging=gx.imaging.Sprites(objective, camera),
        environment=gx.env.Homogeneous(1.33),
        acquisition=gx.acq.FocusStack(focus=torch.tensor([-0.1, 0.1])),
    )
    table = gx.Pipeline(chain, outputs=("expected",)).gradient_table
    assert table.quality("imaging.camera.noise.gain") == "exact"
    assert table.quality("acquisition.focus") == "exact"


def test_focus_stacks_need_static_scenes_and_scmos_crops_stay_inside():
    base = emitters(1, 2, fov=3.0)
    moving = base.replace(position=base.position[:, None].repeat(1, 2, 1, 1))
    objective, camera = optics(shape=(24, 24))
    chain = gx.Chain(
        emitters={"b": moving},
        imaging=gx.imaging.Sprites(objective, camera),
        environment=gx.env.Homogeneous(1.33),
        acquisition=gx.acq.FocusStack(focus=torch.tensor([-0.1, 0.1])),
    )
    with pytest.raises(gx.StructureError, match="static scene"):
        chain(outputs=("expected",))
    maps = {"gain": torch.ones(8, 8), "offset": torch.zeros(8, 8), "read": torch.ones(8, 8)}
    with pytest.raises(gx.StructureError, match="leaves"):
        gx.Camera.scmos(maps=maps, pixel_size=6.5, roi=(4, 4, 8, 8))
    with pytest.raises(gx.StructureError, match="leaves"):
        gx.Camera.scmos(maps=maps, pixel_size=6.5, roi=torch.tensor([[-1, 0]]), shape=(4, 4))


# ---- skeleton review regressions --------------------------------------------------------------


def test_fidelity_knobs_reach_the_planned_element():
    objective, camera = optics(shape=(24, 24))
    sample = gx.Sample({"b": emitters(1, 2, fov=3.0)}, environment=gx.env.Homogeneous(1.33))
    scope = gx.presets.Widefield(objective=objective, camera=camera)
    plan = gx.plan(sample, scope, gx.Fidelity("standard", roi=20), outputs=("expected",))
    imaging = plan.chain(sample, scope).imaging
    assert isinstance(imaging, gx.imaging.PointPSF) and imaging.roi == 20
    assert imaging.method == "roi"
    wide = gx.plan(sample, scope, gx.Fidelity("standard", emitter_path="global"))
    assert getattr(wide.chain(sample, scope).imaging, "method", None) == "global"
    two = gx.Sample(
        {"a": emitters(1, 2), "b": emitters(1, 2, seed=1)}, environment=sample.environment
    )
    mixed = gx.Fidelity("standard", per_population={"a": {"psf": "scalar"}})
    with pytest.raises(gx.PlanError, match="share one imaging element"):
        gx.plan(two, scope, mixed)


def test_pupil_modifiers_may_return_plain_maps():
    import dataclasses

    @dataclasses.dataclass(frozen=True, kw_only=True, eq=False)
    class Stop(gx.PupilModifier):
        """A central stop returned as a [P, P] map."""

        def __call__(self, fx, fy, wl, ctx):
            u2 = (fx * wl) ** 2 + (fy * wl) ** 2
            return (u2[0, 0] > 0.1).to(torch.complex64)

    objective, camera = optics(shape=(24, 24))
    psf = gx.imaging.PointPSF(objective.with_pupil(Stop()), camera, psf="scalar")
    beads = emitters(1, 1, fov=3.0)
    assert psf(gx.lower.emitter_set(beads), gx.env.Homogeneous(1.33)).data.sum() > 0


def test_conformance_reports_zero_gradient_paths():
    from gradix.testing import conformance

    objective, camera = optics(shape=(16, 16))
    beads = emitters(2, 2, fov=2.0)
    element = gx.imaging.Sprites(objective, camera)
    probe = {"unused": torch.tensor(1.0, requires_grad=True)}
    report = conformance(
        element,
        (gx.lower.emitter_set(beads), gx.env.Homogeneous(1.33)),
        probes=probe,
        make_inputs=lambda p: (element, (gx.lower.emitter_set(beads), gx.env.Homogeneous(1.33))),
    )
    assert "no gradient path" in report["gradients"]


def test_module_signatures_are_structural_and_core_names_stable():
    from gradix.schema.signature import _static_repr, type_name

    assert _static_repr(torch.nn.Linear(2, 3)) == _static_repr(torch.nn.Linear(2, 3))
    assert _static_repr(torch.nn.Linear(2, 3)) != _static_repr(torch.nn.Linear(3, 3))
    assert type_name(gx.Objective) == "optics.objective"
    assert gx.registry.data_objects.get("chain") is gx.Chain


def test_elements_own_shape_affecting_fields_are_enveloped():
    import dataclasses

    @dataclasses.dataclass(frozen=True, eq=False)
    class WideSprites(gx.imaging.Sprites):
        """Sprites with a shape-affecting blur width of their own."""

        _: dataclasses.KW_ONLY
        width: torch.Tensor | float = gx.field(
            quantity="length", role="image", shape_affecting=True, default=1.0
        )

    objective, camera = optics(shape=(24, 24))
    chain = gx.Chain(
        emitters={"b": emitters(2, 2, fov=3.0)},
        imaging=WideSprites(objective, camera, width=torch.tensor([1.0, 2.0])),
        environment=gx.env.Homogeneous(1.33),
    )
    pipe = gx.Pipeline(chain, outputs=("expected",), envelope={"imaging.width": (0.5, 3.0)})
    assert pipe.envelope.range("imaging.width") == (0.5, 3.0)
    assert pipe.envelope.range("imaging.objective.NA") is None  # children are parts of their own
    wide = pipe({"imaging.width": torch.tensor([1.0, 6.0])})
    assert wide.meta["in_envelope"].tolist() == [True, False]


def test_one_path_table_behind_resolve_logical_parts_and_gradient_names():
    from gradix.compose.gradients import gradient_table
    from gradix.compose.outputs import Expected

    objective, camera = optics(shape=(16, 16))
    spheres = gx.Spheres(position=torch.zeros(1, 1, 3), radius=0.02, material=1.59)
    chain = gx.Chain(
        light=gx.light.Uniform(irradiance=1.0, wavelength=0.532),
        scatterers={"cells": gx.interact.Dipole(spheres)},
        emitters={"beads": emitters(1, 2)},
        imaging=gx.imaging.Sprites(objective, camera),
        environment=gx.env.Homogeneous(1.33),
    )
    for tree_path in gx.tree.paths(chain):
        assert chain.resolve(chain.logical(tree_path)) == [tree_path], tree_path
    assert chain.logical("scatterers.cells.objects.radius") == "cells.radius"
    assert chain.logical("imaging.camera.gain") == "camera.gain"
    assert chain.logical("imaging.objective.NA") == "objective.NA"
    assert set(chain.parts()) == {"beads", "cells", "light", "objective", "camera", "environment"}
    assert set(chain.own_parts()) == {"imaging", "scatterers.cells"}
    table = gradient_table(chain, {"expected": Expected()})
    assert table.name("scatterers.cells.objects.radius") == "cells.radius"
    assert table.name("emitters.beads.photons") == "beads.photons"
    detector = gx.Chain(camera=camera)
    assert detector.logical("detector.gain") == "camera.gain"
    assert detector.resolve("camera.gain") == ["detector.gain"]
    with pytest.raises(gx.BindingError, match="no objective"):
        detector.resolve("objective.NA")


def test_acquisition_bindings_size_eager_and_planned_grids():
    import dataclasses

    from gradix.compose import execute
    from gradix.imaging.point_psf import PointPSFStatic

    objective, camera = optics(shape=(48, 48), na=1.2, magnification=100)
    beads = emitters(1, 1, fov=3.0, z=0.0)
    medium = gx.env.Homogeneous(1.33)
    stack = gx.Chain(
        emitters={"b": beads},
        environment=medium,
        acquisition=gx.acq.FocusStack(focus=torch.tensor([-2.0, 0.0, 2.0])),
        imaging=gx.imaging.PointPSF(objective, camera, psf="scalar"),
    )
    moved = objective.replace(focus=2.0)
    single = gx.Chain(
        emitters={"b": beads},
        environment=medium,
        imaging=gx.imaging.PointPSF(moved, camera, psf="scalar"),
    )
    planned, reference = (execute.eager_statics(c)["imaging"] for c in (stack, single))
    assert isinstance(planned, PointPSFStatic) and isinstance(reference, PointPSFStatic)
    assert planned.roi >= reference.roi  # the ±2 µm planes fit

    @dataclasses.dataclass(frozen=True, kw_only=True, eq=False)
    class Biplane(gx.acq.Acquisition):
        def index(self):
            return gx.AcqIndex((("plane", 2),))

        def bindings(self, chain):
            return {"objective.focus": torch.tensor([[-2.0, 2.0]])}

    pipe = gx.Pipeline(stack.replace(acquisition=Biplane()), outputs=("expected",))
    focus = pipe.envelope.range("objective.focus")
    assert focus is not None and focus[0] <= -2.0 and focus[1] >= 2.0


def test_per_frame_settings_without_an_acquisition_render_every_frame():
    objective, camera = optics(shape=(24, 24))
    drift = objective.replace(focus=torch.tensor([[0.0, 0.15, 0.3]]))  # [B, T]
    chain = gx.Chain(
        emitters={"b": emitters(1, 2, fov=3.0)},
        imaging=gx.imaging.Sprites(drift, camera),
        environment=gx.env.Homogeneous(1.33),
    )
    eager = chain(outputs=("expected",))["expected"]
    assert eager.shape == (1, 3, 1, 24, 24)  # [B, T, C, H, W]
    assert not torch.equal(eager[:, 0], eager[:, 2])  # each frame at its own focus
    piped = gx.Pipeline(chain, outputs=("expected",))(chain)["expected"]
    assert piped.shape == eager.shape
    stack = chain.replace(acquisition=gx.acq.FocusStack(focus=torch.tensor([-1.0, 0.0, 1.0, 2.0])))
    with pytest.raises(gx.StructureError, match="focus stack has 4 steps"):
        stack(outputs=("expected",))


def test_element_children_that_are_not_parts_are_enveloped():
    import dataclasses

    @dataclasses.dataclass(frozen=True, kw_only=True, eq=False)
    class Blur(gx.DataObject):
        width: torch.Tensor | float = gx.field(
            quantity="length", role="image", shape_affecting=True, default=0.1
        )

    @dataclasses.dataclass(frozen=True, eq=False)
    class Blurred(gx.imaging.Sprites):
        blur: Blur = gx.child(default_factory=Blur)

    objective, camera = optics(shape=(16, 16))
    element = Blurred(objective, camera, blur=Blur(width=torch.tensor([0.1, 0.2])))
    chain = gx.Chain(
        emitters={"b": emitters(2, 2, fov=2.0)},
        imaging=element,
        environment=gx.env.Homogeneous(1.33),
    )
    envelope = gx.envelope_of(chain)
    assert envelope.range("imaging.blur.width") is not None
    assert envelope.range("imaging.objective.NA") is None  # the objective is a part of its own
    gx.Pipeline(chain, outputs=("expected",), envelope={"imaging.blur.width": (0.05, 0.5)})


def test_acquisition_fields_route_only_through_the_settings_they_drive():
    objective, camera = optics(shape=(16, 16))
    base = gx.Chain(
        emitters={"b": emitters(1, 2, fov=2.0)},
        imaging=gx.imaging.Sprites(objective, camera),
        environment=gx.env.Homogeneous(1.33),
    )
    interval = torch.tensor(0.02, requires_grad=True)
    timed = base.replace(acquisition=gx.acq.Frames(2, interval=interval))
    pipe = gx.Pipeline(timed, outputs=("expected",))
    assert pipe.gradient_table.quality("acquisition.interval") == "zero"
    with pytest.raises(gx.GradientPathError, match=r"acquisition\.interval"):
        pipe(timed)
    steps = torch.tensor([-0.1, 0.1], requires_grad=True)
    stack = base.replace(acquisition=gx.acq.FocusStack(focus=steps))
    table = gx.Pipeline(stack, outputs=("expected",)).gradient_table
    assert table.quality("acquisition.focus") != "zero"


def test_validity_errors_use_the_inputs_values_not_the_widened_envelope():
    camera = gx.Camera(pixel_size=6.5, shape=(24, 24))

    def chain(na, medium, z=0.0):
        objective = gx.Objective(NA=torch.tensor(na), magnification=100)
        return gx.Chain(
            emitters={
                "b": emitters(1, 1, fov=1.5, z=0.0).replace(
                    position=torch.tensor([[[0.8, 0.8, z]]])
                )
            },
            imaging=gx.imaging.PointPSF(objective, camera),
            environment=medium,
        )

    with pytest.warns(gx.GradixWarning):  # the widened NA reaches n: a warning, not an error
        pipe = gx.Pipeline(chain(1.25, gx.env.Homogeneous(1.33)), outputs=("expected",))
    # psf="auto" falling back to scalar is an info finding
    assert any(v.severity == "info" and "vectorial" in v.message for v in pipe.violations)
    with pytest.raises(gx.ValidityError, match="supercritical"):
        gx.Pipeline(chain(1.45, gx.env.Homogeneous(1.33)), outputs=("expected",))
    layered = gx.env.LayeredMedium(sample=1.33)
    with pytest.raises(gx.ValidityError, match="above the coverslip"):
        gx.Pipeline(
            chain(1.2, layered, z=0.15),
            outputs=("expected",),
            envelope={"b.position.z": (-0.5, 0.5)},
        )


def test_conformance_compares_gradients_with_declared_qualities():
    import dataclasses
    from typing import ClassVar

    from gradix.testing import conformance

    objective, camera = optics(shape=(16, 16))
    inputs = (gx.lower.emitter_set(emitters(2, 2, fov=2.0)), gx.env.Homogeneous(1.33))
    assert conformance(gx.imaging.Sprites(objective, camera), inputs)["declared"] == "ok"

    @dataclasses.dataclass(frozen=True, eq=False)
    class MisDeclared(gx.imaging.Sprites):
        """Claims the NA reaches nothing and the magnification is exact."""

        caps: ClassVar = dataclasses.replace(
            gx.imaging.Sprites.caps, grad_quality={"objective.NA": "zero", "camera": "zero"}
        )

    report = conformance(MisDeclared(objective, camera), inputs)["declared"]
    assert "objective.NA: declared zero, but its gradient is non-zero" in report
    assert "camera.pixel_size: declared zero" in report  # the pitch sets the sprite grid


def test_linked_elements_must_agree_on_their_carriers():
    import dataclasses
    from typing import ClassVar

    @dataclasses.dataclass(frozen=True, kw_only=True, eq=False)
    class PlaneWavesOnly(gx.excite.Linear):
        """An excitation that only understands plane waves."""

        caps: ClassVar = dataclasses.replace(
            gx.excite.Linear.caps, accepts=frozenset({gx.EmitterSet, gx.PlaneWaves})
        )

    objective, camera = optics(shape=(16, 16))
    chain = gx.Chain(
        light=gx.light.Sheet(waist=0.5, wavelength=0.488),
        emitters={"b": emitters(1, 2, fov=2.0)},
        excite=PlaneWavesOnly(),
        imaging=gx.imaging.Sprites(objective, camera),
        environment=gx.env.Homogeneous(1.33),
    )
    with pytest.raises(gx.PlanError, match="light produces GaussianSheet, but excite accepts"):
        chain(outputs=("expected",))


def test_binding_is_idempotent_and_drives_must_match_bindings():
    import dataclasses
    from typing import ClassVar

    objective, camera = optics(shape=(16, 16))
    chain = gx.Chain(
        emitters={"b": emitters(1, 2, fov=2.0)},
        imaging=gx.imaging.Sprites(objective, camera),
        environment=gx.env.Homogeneous(1.33),
        acquisition=gx.acq.FocusStack(focus=torch.tensor([-0.2, 0.2])),
    )
    once = chain(outputs=("expected",))["expected"]
    twice = chain.acquired()(outputs=("expected",))["expected"]
    assert torch.equal(once, twice)  # the stack is not applied twice

    @dataclasses.dataclass(frozen=True, kw_only=True, eq=False)
    class Stale(gx.acq.Frames):
        drives: ClassVar = {"interval": "objective.focus"}

    stale = chain.replace(acquisition=Stale(2))
    with pytest.raises(gx.PlanError, match="declares drives"):
        stale(outputs=("expected",))


def test_envelope_unions_keep_every_observed_span():
    a = gx.Envelope({"x": gx.Entry(0.0, 2.0, "derived", (1.0, 1.0))})
    b = gx.Envelope({"x": gx.Entry(0.5, 1.5, "derived", (1.05, 1.05))})
    assert a.union(b).observed("x") == (1.0, 1.05)


@pytest.mark.parametrize("photons", ["tensor", "scalar"])
def test_empty_populations_render_empty_frames(photons):
    objective, camera = optics(shape=(12, 12), na=0.8)
    beads = gx.Emitters(
        position=torch.zeros(1, 0, 3),
        photons=torch.zeros(1, 0) if photons == "tensor" else 1000.0,
        emission=gx.Spectrum.line(0.6),
    )
    lowered = gx.lower.emitter_set(beads)
    assert lowered.position.shape[2] == 0
    medium = gx.env.Homogeneous(1.33)
    for element in (
        gx.imaging.PointPSF(objective, camera, psf="scalar"),
        gx.imaging.PointPSF(objective, camera, psf="scalar", method="global"),
    ):
        out = element(lowered, medium).data
        assert out.shape[-2:] == (12, 12) and float(out.abs().sum()) == 0.0
    spheres = gx.Spheres(
        position=torch.zeros(1, 0, 3), radius=torch.zeros(1, 0) + 0.02, material=1.5
    )
    waves = gx.light.PlaneWave(0.532)()
    coherent = gx.imaging.Coherent(objective, camera, pupil_samples=16)
    alone = coherent(gx.interact.Dipole(spheres)(waves, medium), waves, medium).data
    assert torch.allclose(alone, coherent((), waves, medium).data)  # the background alone


def test_bound_envelopes_keep_the_acquisitions_paths():
    objective, camera = optics(shape=(16, 16))
    chain = gx.Chain(
        emitters={"b": emitters(1, 1, fov=1.0)},
        imaging=gx.imaging.Sprites(objective, camera),
        environment=gx.env.Homogeneous(1.33),
        acquisition=gx.acq.FocusStack(focus=torch.tensor([-0.2, 0.2])),
    )
    paths = set(gx.envelope_of(chain))
    assert "acquisition.focus" in paths and "acquisition.source.focus" not in paths


def test_per_image_pitch_renders_every_image_on_the_global_path():
    camera = gx.Camera(pixel_size=torch.tensor([6.5, 13.0], dtype=torch.float64), shape=(24, 24))
    objective = gx.Objective(NA=1.2, magnification=65)
    medium = gx.env.Homogeneous(1.33)
    centres = torch.tensor([[[1.2, 1.2, 0.0]], [[2.4, 2.4, 0.0]]], dtype=torch.float64)
    beads = gx.Emitters(position=centres, photons=1000.0, emission=gx.Spectrum.line(0.6))
    psf = gx.imaging.PointPSF(objective, camera, psf="scalar", method="global")
    batch = psf(gx.lower.emitter_set(beads), medium).data
    for b, size in enumerate((6.5, 13.0)):
        alone = gx.imaging.PointPSF(
            objective, gx.Camera(pixel_size=size, shape=(24, 24)), psf="scalar", method="global"
        )
        one = beads.replace(position=centres[b : b + 1])
        single = alone(gx.lower.emitter_set(one), medium).data[0]
        # equal up to the pupil grid each render sizes for itself (the soft edge is O(du))
        assert float((batch[b] - single).norm() / single.norm()) < 0.03


def test_likelihood_benchmark_roots_are_robust():
    from gradix.testing import likelihoods

    exact = likelihoods.evaluate("convolution", 100.0, 0.8)
    assert all(abs(b) < 1e-6 for b in exact.bias_se.values())
    shifted = likelihoods.evaluate("shifted_poisson", 0.1, 0.8)
    assert shifted.bias["offset"] == math.inf  # the estimate diverges: no root


def test_coherent_refuses_na_at_the_index_and_is_differentiable_at_normal_incidence():
    camera = gx.Camera(pixel_size=6.5, shape=(9, 9))
    medium = gx.env.Homogeneous(1.33)
    direction = torch.zeros(2, dtype=torch.float64, requires_grad=True)
    waves = gx.light.PlaneWave(0.532, direction=direction)()
    wide = gx.imaging.Coherent(gx.Objective(NA=1.4, magnification=100), camera)
    with pytest.raises(gx.ValidityError, match="index"):
        wide((), waves, medium)
    element = gx.imaging.Coherent(gx.Objective(NA=1.0, magnification=100), camera)
    element((), waves, medium).data.sum().backward()
    assert direction.grad is not None and bool(torch.isfinite(direction.grad).all())


def test_strata_mark_images_rendered_at_another_pitch():
    medium = gx.env.Homogeneous(1.33)
    camera = gx.Camera(pixel_size=6.5, shape=(16, 16))
    objective = gx.Objective(NA=1.0, magnification=65)
    cells = gx.Voxels(
        values=torch.rand(2, 12, 12),
        spacing=(0.3, 0.1, 0.1),
        origin=(0.25, 0.25, -0.3),
        quantity="density",
        emission=gx.Spectrum.line(0.6),
    )
    chain = gx.Chain(
        emitters={"c": cells}, imaging=gx.imaging.Strata(objective, camera), environment=medium
    )
    pipe = gx.Pipeline(chain, outputs=("expected",), inputs=("camera.pixel_size",))
    assert bool(torch.isfinite(pipe(chain)["expected"]).all())
    moved = pipe({"camera.pixel_size": torch.tensor([7.0])})["expected"]
    assert bool(torch.isnan(moved).all())


def test_batch_keyed_poisson_keeps_integers_at_huge_rates_and_drives_cover_bindings():
    import dataclasses
    from typing import ClassVar

    from gradix.ops.detect import poisson

    counts = poisson(torch.full((4096,), 2e7, dtype=torch.float64), 3, "huge")
    assert bool((counts % 2 == 1).any())  # not only even counts (float32 ran out of integers)

    @dataclasses.dataclass(frozen=True, kw_only=True, eq=False)
    class Offsets(gx.acq.Acquisition):
        shift: torch.Tensor = gx.field(quantity="length", role="shared", event=(-1,))
        drives: ClassVar = {}

        def index(self):
            return gx.AcqIndex((("plane", 2),))

        def bindings(self, chain):
            return {"objective.focus": self.shift[None]}

    objective, camera = optics(shape=(16, 16))
    chain = gx.Chain(
        emitters={"b": emitters(1, 1, fov=1.0)},
        imaging=gx.imaging.Sprites(objective, camera),
        environment=gx.env.Homogeneous(1.33),
        acquisition=Offsets(shift=torch.tensor([-0.2, 0.2])),
    )
    with pytest.raises(gx.PlanError, match="declares no drives"):
        chain(outputs=("expected",))


def test_the_convolution_handles_bright_outliers_with_large_read_noise():
    from gradix.ops.detect import poisson_gaussian_log_prob

    k, lam, var = 1e4, 1e-3, 400.0
    n = torch.arange(0, 20000, dtype=torch.float64)
    log_pois = n * math.log(lam) - lam - torch.lgamma(n + 1)
    gauss = -0.5 * ((k - n) ** 2 / var + math.log(2 * math.pi * var))
    brute = float(torch.logsumexp(log_pois + gauss, 0))
    got = poisson_gaussian_log_prob(torch.tensor(k, dtype=torch.float64), torch.tensor(lam), var)
    assert float(got) == pytest.approx(brute, rel=1e-8)
    inf = poisson_gaussian_log_prob(torch.tensor([1.0]), torch.tensor([math.inf]), 1.0)
    assert float(inf) == -math.inf


def test_calls_beyond_the_envelope_do_not_ghost_into_the_frame():
    camera = gx.Camera(pixel_size=6.5, shape=(24, 24))
    objective = gx.Objective(NA=1.2, magnification=65)
    inside = torch.tensor([[[1.2, 1.2, 0.0], [1.0, 1.0, 0.0]]], dtype=torch.float64)
    beads = gx.Emitters(
        position=inside,
        photons=torch.tensor([[1000.0, 0.0]], dtype=torch.float64),
        emission=gx.Spectrum.line(0.6),
    )
    chain = gx.Chain(
        emitters={"b": beads},
        imaging=gx.imaging.PointPSF(objective, camera, psf="scalar", method="global"),
        environment=gx.env.Homogeneous(1.33),
    )
    pipe = gx.Pipeline(chain, outputs=("expected",), inputs=("b.position", "b.photons"))
    photons = torch.tensor([[1000.0, 1e5]], dtype=torch.float64)
    alone = pipe({"b.photons": photons * torch.tensor([1.0, 0.0])})["expected"]
    far = inside.clone()
    far[0, 1, 0] = 40.0  # far beyond the envelope (and its period)
    with_far = pipe({"b.position": far, "b.photons": photons})
    assert not bool(with_far.meta["in_envelope"].all())
    assert float((with_far["expected"] - alone).abs().sum()) < 1.0  # no ghost of 1e5 photons


def test_last_verification_round_regressions():
    import dataclasses
    from typing import ClassVar

    from gradix.ops.detect import poisson, poisson_gaussian_log_prob

    # a NaN read noise gives a NaN likelihood, not a crash
    out = poisson_gaussian_log_prob(
        torch.tensor([1.0, 2.0]), torch.tensor([1.0, 1.0]), torch.tensor([1.0, math.nan])
    )
    assert bool(torch.isfinite(out[0])) and bool(torch.isnan(out[1]))
    # float32 PTRS keeps odd counts right below its limit
    counts = poisson(torch.full((4096,), 2.0**23 - 4096.0, dtype=torch.float64), 5, "top")
    assert bool((counts % 2 == 1).any())
    # Coherent validity is per image: every image here has NA < n
    camera = gx.Camera(pixel_size=6.5, shape=(9, 9))
    na = torch.tensor([0.9, 1.2], dtype=torch.float64)
    medium = gx.env.Homogeneous(torch.tensor([1.0, 1.33], dtype=torch.float64))
    element = gx.imaging.Coherent(gx.Objective(NA=na, magnification=100), camera)
    element((), gx.light.PlaneWave(0.532)(), medium)
    # constant bindings are fine next to unrelated tensor fields; drives compare logically

    @dataclasses.dataclass(frozen=True, kw_only=True, eq=False)
    class Timed(gx.acq.Acquisition):
        interval: torch.Tensor = gx.field(quantity="time", role="shared", default=0.1)
        drives: ClassVar = {}

        def index(self):
            return gx.AcqIndex((("plane", 2),))

        def bindings(self, chain):
            return {"imaging.objective.focus": torch.tensor([[-0.2, 0.2]])}

    objective, camera = optics(shape=(16, 16))
    chain = gx.Chain(
        emitters={"b": emitters(1, 1, fov=1.0)},
        imaging=gx.imaging.Sprites(objective, camera),
        environment=gx.env.Homogeneous(1.33),
        acquisition=Timed(interval=torch.tensor(0.1)),
    )
    assert chain(outputs=("expected",))["expected"].shape[1] == 2
    # Strata accept one object-space pitch from different pixel sizes and magnifications
    cells = gx.Voxels(
        values=torch.rand(2, 2, 10, 10),
        spacing=(0.3, 0.1, 0.1),
        origin=(0.25, 0.25, -0.3),
        quantity="density",
        emission=gx.Spectrum.line(0.6),
    )
    strata = gx.imaging.Strata(
        gx.Objective(NA=1.0, magnification=torch.tensor([65.0, 130.0])),
        gx.Camera(pixel_size=torch.tensor([6.5, 13.0]), shape=(16, 16)),
    )
    image = strata(gx.lower.emitter_density(cells), gx.env.Homogeneous(1.33)).data
    assert bool(torch.isfinite(image).all())
