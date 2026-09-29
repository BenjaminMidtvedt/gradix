"""L4 planning (§5.6, §5.7): fidelity, routing, plans, presets and gx.render's cache."""

import logging

import pytest
import torch
from scenes import emitters, optics

import gradix as gx


def _inputs(b=4, n=3, seed=0, **kw):
    objective, camera = optics(shape=(32, 32), **kw)
    sample = gx.Sample(
        {"beads": emitters(b, n, seed=seed, fov=4.0)}, environment=gx.env.Homogeneous(1.33)
    )
    return sample, gx.presets.Widefield(objective=objective, camera=camera)


def test_fidelity_presets_and_overrides():
    assert gx.Fidelity("draft").knob("emitters") == "gaussian"
    assert gx.Fidelity("standard").knob("emitters") == "pupil"
    assert gx.Fidelity("standard", emitters="gaussian").knob("emitters") == "gaussian"
    fid = gx.Fidelity("standard", per_population={"beads": {"emitters": "gaussian"}})
    assert (
        fid.knob("emitters", population="beads") == "gaussian" and fid.knob("emitters") == "pupil"
    )
    with pytest.raises(gx.PlanError, match="global knobs"):
        gx.Fidelity(per_population={"beads": {"source": 3}})
    with pytest.raises(NotImplementedError):
        gx.Fidelity(backward="standard")
    with pytest.raises(gx.PlanError):
        gx.Fidelity.coerce("fast")


def test_routing():
    sample, scope = _inputs()
    assert gx.plan(sample, scope, "standard").routes["beads"].element == "emit.pupil_mft"
    plan = gx.plan(sample, scope, "draft", outputs=("expected",))
    assert plan.routes["beads"].element == "emit.gaussian"
    assert "beads → emit.gaussian" in plan.why("beads")
    pinned = gx.plan(
        sample, scope, "standard", methods={"beads": "emit.gaussian"}, outputs=("expected",)
    )
    assert pinned.routes["beads"].reason == "pinned by methods="
    by_name = gx.plan(
        sample, scope, gx.Fidelity("standard", emitters="emit.gaussian"), outputs=("expected",)
    )
    assert by_name.routes["beads"].element == "emit.gaussian"
    with pytest.raises(gx.PlanError, match="unknown populations"):
        gx.plan(sample, scope, "draft", methods={"cells": "emit.gaussian"})


def test_plan_builds_the_chain_a_user_would_build():
    sample, scope = _inputs()
    plan = gx.plan(sample, scope, "draft", outputs=("expected",))
    chain = plan.chain(sample, scope)
    assert isinstance(chain.imaging, gx.imaging.Sprites) and chain.camera is scope.camera
    assert torch.equal(plan(sample, scope)["expected"], plan.pipeline(chain)["expected"])
    assert "fidelity=draft" in plan.explain()


def test_plan_binds_values_by_path():
    sample, scope = _inputs(b=8, n=4)
    beads = sample.populations["beads"].replace(photons=1000.0)  # shared, so batches may shrink
    sample = gx.Sample({"beads": beads}, environment=sample.environment)
    plan = gx.plan(
        sample, scope, "draft", outputs=("expected",), inputs=("beads.position", "objective.focus")
    )
    out = plan({"beads.position": torch.rand(5, 4, 3) * 4, "objective.focus": torch.zeros(5)})
    assert out["expected"].shape == (5, 1, 32, 32)


def test_render_caches_and_replans(caplog):
    gx.api.clear_cache()
    sample, scope = _inputs(b=4, n=3)
    first = gx.render(sample, scope, "draft", outputs=("expected",))
    assert len(gx.api.plan.__globals__["_CACHE"]) == 1
    gx.render(_inputs(b=4, n=3, seed=0)[0], scope, "draft", outputs=("expected",))
    assert len(gx.api.plan.__globals__["_CACHE"]) == 1
    far = sample.populations["beads"].replace(
        position=sample.populations["beads"].position + torch.tensor([0.0, 0.0, 5.0])
    )
    with (
        caplog.at_level(logging.INFO, logger="gradix"),
        pytest.warns(gx.GradixWarning, match="DOF"),
    ):
        moved = gx.render(
            gx.Sample({"beads": far}, environment=sample.environment),
            scope,
            "draft",
            outputs=("expected",),
        )
    assert "re-planning" in caplog.text
    assert moved["expected"].shape == first["expected"].shape
    gx.api.clear_cache()
    gx.render(sample, scope, "draft", outputs=("expected",))
    bigger = _inputs(b=9, n=3)[0]
    with caplog.at_level(logging.INFO, logger="gradix"):
        out = gx.render(bigger, scope, "draft", outputs=("expected",))
    assert out["expected"].shape[0] == 9 and "capacity" in caplog.text


def test_presets_are_registered():
    assert gx.registry.presets.get("widefield") is gx.presets.Widefield
    with pytest.raises(gx.RegistryError, match="did you mean"):
        gx.registry.presets.get("widefeld")


def test_plugins_are_selected_by_bare_name_and_built_by_keyword():
    import dataclasses

    from gradix._core.registry import elements

    @dataclasses.dataclass(frozen=True, kw_only=True, eq=False)
    class KwSprites(gx.imaging.Sprites):
        """A keyword-only plugin emission element."""

    gx.register.element("emit.kw_sprites_test")(KwSprites)
    try:
        _check_plugin_routing()
    finally:
        elements.unregister("emit.kw_sprites_test")


def _check_plugin_routing():
    sample, scope = _inputs()
    planned = gx.plan(sample, scope, gx.Fidelity("draft", emitters="kw_sprites_test"))
    assert planned.routes["beads"].element == "emit.kw_sprites_test"
    pinned = gx.plan(sample, scope, "draft", methods={"beads": "kw_sprites_test"})
    assert type(pinned.chain(sample, scope).imaging).__name__ == "KwSprites"
    with pytest.raises(gx.PlanError, match="does not accept"):
        gx.plan(sample, scope, "draft", methods={"beads": "detect.camera"})
    with pytest.raises(gx.PlanError, match="per_population"):
        gx.plan(sample, scope, gx.Fidelity("draft", per_population={"bead": {"emitters": "x"}}))
    assert hash(gx.Fidelity("draft")) == hash(gx.Fidelity("draft"))
