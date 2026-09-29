"""User names never share a namespace with the framework's (ADR-42): one rule, checked early."""

import dataclasses
from typing import ClassVar

import pytest
import torch
from scenes import emitters, optics

import gradix as gx
from gradix._core.registry import elements


def _scene():
    objective, camera = optics(shape=(16, 16))
    return objective, camera, gx.env.Homogeneous(1.33)


@pytest.mark.parametrize(
    "name", ["camera", "objective", "scatterers", "my.beads", "2beads", "class"]
)
def test_population_names_are_identifiers_that_do_not_start_a_reserved_path(name):
    objective, camera, medium = _scene()
    beads = emitters(1, 2, fov=1.0)
    with pytest.raises(gx.StructureError, match=r"reserved|not an identifier"):
        gx.Chain(
            emitters={name: beads},
            imaging=gx.imaging.Sprites(objective, camera),
            environment=medium,
        )
    with pytest.raises(gx.StructureError, match=r"reserved|not an identifier"):
        gx.Sample({name: beads}, environment=medium)


@pytest.mark.parametrize("name", ["keys", "values", "meta", "pos.in_fov", ""])
def test_output_names_are_identifiers_that_do_not_shadow_the_output(name):
    objective, camera, medium = _scene()
    chain = gx.Chain(
        emitters={"beads": emitters(1, 2, fov=1.0)},
        imaging=gx.imaging.Sprites(objective, camera),
        environment=medium,
    )
    with pytest.raises(gx.StructureError, match=r"reserved|not an identifier"):
        gx.Pipeline(chain, outputs={name: "expected"})


def test_fidelity_knobs_reach_only_the_fields_an_element_declares():
    @dataclasses.dataclass(frozen=True, kw_only=True, eq=False)
    class Plugin(gx.imaging.Sprites):
        """A plugin whose field happens to share a fidelity knob's name."""

        roi: int = gx.knob(default=7, doc="something else entirely")

    gx.register.element("emit.name_probe_test")(Plugin)
    try:
        objective, camera, medium = _scene()
        sample = gx.Sample({"beads": emitters(1, 2, fov=1.0)}, environment=medium)
        scope = gx.presets.Widefield(objective=objective, camera=camera)
        fidelity = gx.Fidelity("draft", roi=48)
        planned = gx.plan(sample, scope, fidelity, methods={"beads": "name_probe_test"})
        built = planned.chain(sample, scope).imaging
        assert isinstance(built, Plugin) and built.roi == 7  # not declared: untouched

        @dataclasses.dataclass(frozen=True, kw_only=True, eq=False)
        class Declared(Plugin):
            fidelity_knobs: ClassVar = {"roi": "roi"}

        gx.register.element("emit.name_probe_declared_test")(Declared)
        try:
            planned = gx.plan(
                sample, scope, fidelity, methods={"beads": "name_probe_declared_test"}
            )
            imaging = planned.chain(sample, scope).imaging
            assert isinstance(imaging, Declared) and imaging.roi == 48
        finally:
            elements.unregister("emit.name_probe_declared_test")
    finally:
        elements.unregister("emit.name_probe_test")


def test_a_declared_knob_must_name_a_field():
    @dataclasses.dataclass(frozen=True, kw_only=True, eq=False)
    class Broken(gx.imaging.Sprites):
        fidelity_knobs: ClassVar = {"roi": "window"}

    gx.register.element("emit.name_probe_broken_test")(Broken)
    try:
        objective, camera, medium = _scene()
        sample = gx.Sample({"beads": emitters(1, 2, fov=1.0)}, environment=medium)
        scope = gx.presets.Widefield(objective=objective, camera=camera)
        with pytest.raises(gx.PlanError, match="field 'window'"):
            gx.plan(sample, scope, "draft", methods={"beads": "name_probe_broken_test"})
    finally:
        elements.unregister("emit.name_probe_broken_test")


def test_any_shape_field_name_works():
    # shape fields travel as one mapping: a Gaussian blob's "sigma" never meets the blur argument
    blob = gx.Gaussians(position=torch.zeros(1, 1, 3), sigma=torch.tensor([[[0.2, 0.2, 0.2]]]))
    peak = blob.occupancy(torch.zeros(3), 0.05, {"sigma": blob.sigma})
    assert float(peak.squeeze()) == pytest.approx((0.2 / (0.2**2 + 0.05**2) ** 0.5) ** 3)
