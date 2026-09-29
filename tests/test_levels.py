"""Level equivalence (§11.10, M0 exit): L2, L3 and L4 renders are bit-identical."""

import pytest
import torch
from scenes import DEVICES, emitters, optics

import gradix as gx
from gradix.testing import levels


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("noise", [None, "poisson_gaussian"])
def test_levels_are_bit_identical(device, noise):
    model = gx.noise.PoissonGaussian(read=2.0) if noise else None
    objective, camera = optics(
        shape=(128, 128), noise=model, gain=2.0, offset=100.0, magnification=100
    )
    b = 64
    beads = emitters(b, 8, device=device, fov=8.0)
    sample = gx.Sample({"beads": beads}, environment=gx.env.Homogeneous(1.33))
    scope = gx.presets.Widefield(objective=objective, camera=camera, background=5.0)
    key = torch.arange(b, device=device) if noise else None
    out = levels.level_outputs(sample, scope, key=key)
    same = levels.identical(out)
    assert all(same.values()), levels.max_differences(out)
    assert out["L3"]["expected"].shape == (64, 1, 128, 128)


def test_level_harness_catches_a_planner_that_builds_the_wrong_element(monkeypatch):
    import sys

    plan_module = sys.modules["gradix.api.plan"]

    objective, camera = optics(shape=(32, 32))
    sample = gx.Sample({"beads": emitters(4, 3, fov=4.0)}, environment=gx.env.Homogeneous(1.33))
    scope = gx.presets.Widefield(objective=objective, camera=camera)
    real = plan_module.build

    def wrong(choice, parts, fidelity, population=None):
        return real(choice, {**parts, "objective": objective.replace(NA=0.63)}, fidelity)

    monkeypatch.setattr(plan_module, "build", wrong)
    with pytest.raises(ValueError, match="hand-built"):
        levels.level_outputs(sample, scope, key=None)


def test_max_differences_does_not_hide_broadcasting():
    a = {"expected": torch.zeros(4, 1, 8, 8)}
    b = {"expected": torch.zeros(1, 1, 8, 8)}
    diffs = levels.max_differences({"L2": a, "L3": b, "L4": b})
    assert diffs["L2-L3/expected"] == float("inf")


def test_eager_chain_statics_equal_pipeline_statics_for_constant_optics():
    objective, camera = optics(shape=(32, 32))
    c = gx.Chain(
        emitters={"b": emitters(2, 3, fov=4.0)},
        imaging=gx.imaging.Sprites(objective, camera),
        environment=gx.env.Homogeneous(1.33),
    )
    eager = gx.compose.execute.eager_statics(c)["imaging"]
    planned = gx.Pipeline(c).static("imaging")
    assert isinstance(eager, gx.imaging.SpritesStatic)
    assert isinstance(planned, gx.imaging.SpritesStatic)
    assert eager.grid == planned.grid


def test_l2_call_with_a_grid():
    objective, camera = optics(shape=(32, 32))
    sprites = gx.imaging.Sprites(objective, camera)
    es = gx.lower.emitter_set(emitters(2, 3, fov=4.0))
    env = gx.env.Homogeneous(1.33)
    by_grid = sprites(es, env, grid=gx.Grid2D((32, 32), 0.13))
    assert torch.equal(by_grid.data, sprites(es, env).data)


def test_level_harness_compares_the_elements_knobs():
    # the plan pins PointPSF by registry name but builds method="roi": a hand-built global
    # element is not what the plan built, and the harness says so instead of reporting a gap
    objective, camera = optics(shape=(32, 32))
    sample = gx.Sample({"beads": emitters(2, 3, fov=3.0)}, environment=gx.env.Homogeneous(1.33))
    scope = gx.presets.Widefield(objective=objective, camera=camera)
    hand = gx.imaging.PointPSF(objective, camera, psf="scalar", method="global")
    with pytest.raises(ValueError, match="hand-built"):
        levels.level_outputs(sample, scope, key=None, imaging=hand, fidelity="standard")


@pytest.mark.parametrize("device", DEVICES)
def test_point_psf_levels_are_bit_identical(device):
    objective, camera = optics(shape=(32, 32))
    sample = gx.Sample(
        {"beads": emitters(3, 4, device=device, fov=3.0)}, environment=gx.env.Homogeneous(1.33)
    )
    scope = gx.presets.Widefield(objective=objective, camera=camera)
    hand = gx.imaging.PointPSF(objective, camera, psf="scalar")
    out = levels.level_outputs(
        sample, scope, key=None, imaging=hand, fidelity=gx.Fidelity("standard", psf="scalar")
    )
    same = levels.identical(out)
    assert all(same.values()), levels.max_differences(out)
