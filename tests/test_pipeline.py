"""The Pipeline (§5.7, §6.3, §6.5, §6.9): passes, feeding data, capacities, purity, ownership."""

import dataclasses
import json
from typing import ClassVar

import pytest
import torch
from scenes import DEVICES, chain, emitters

import gradix as gx


def _noisy_chain(b=8, n=6, device="cpu", **kw):
    return chain(
        b, n, device=device, noise=gx.noise.PoissonGaussian(read=1.5), gain=2.0, offset=100.0, **kw
    )


@pytest.mark.parametrize("device", DEVICES)
def test_eager_chain_equals_pipeline(device):
    c = _noisy_chain(device=device)
    pipe = gx.Pipeline(c, outputs=("image", "expected"))
    key = torch.arange(8, device=device)
    out = pipe(c, key=key)
    assert out["image"].shape == (8, 1, 64, 64)
    eager = gx.compose.execute.run(
        c, gx.compose.execute.eager_statics(c), key=key, outputs=pipe.outputs
    )
    assert torch.equal(eager["expected"], out["expected"]) and torch.equal(
        eager["image"], out["image"]
    )
    assert set(out.meta) >= {"hash", "chunks", "key", "in_envelope", "deterministic"}
    assert out.meta["key"] == "image"


@pytest.mark.parametrize("device", DEVICES)
def test_purity_same_inputs_same_key_same_bits(device):
    c = _noisy_chain(device=device)
    pipe = gx.Pipeline(c, outputs=("image",), deterministic=True)
    a = pipe(c, key=torch.arange(8, device=device))["image"]
    b = pipe(c, key=torch.arange(8, device=device))["image"]
    assert torch.equal(a, b)
    c_ = pipe(c, key=torch.arange(8, device=device) + 1)["image"]
    assert not torch.equal(a, c_)
    batch_keyed = pipe(c, key=5)["image"]
    assert torch.equal(batch_keyed, pipe(c, key=5)["image"])


@pytest.mark.parametrize("device", DEVICES)
def test_image_keyed_noise_is_independent_of_batch_composition(device):
    big = _noisy_chain(64, 6, device=device)
    keys = torch.arange(64, device=device) * 11 + 5
    pipe = gx.Pipeline(big, outputs=("image",))
    full = pipe(big, key=keys)["image"]
    for sl in (slice(17, 18), slice(0, 32), slice(16, 48)):
        part = pipe(big.select(sl), key=keys[sl])["image"]
        assert torch.equal(part, full[sl])


def test_gradient_set_never_changes_forward_values():
    c = _noisy_chain()
    pipe = gx.Pipeline(c, outputs=("image", "expected"))
    plain = pipe(c, key=torch.arange(8))
    beads = c.emitters["beads"]
    grad_beads = beads.replace(position=beads.position.clone().requires_grad_())
    with_grad = pipe(c.replace(emitters={"beads": grad_beads}), key=torch.arange(8))
    assert torch.equal(plain["image"], with_grad["image"].detach())
    assert with_grad["image"].requires_grad and not plain["image"].requires_grad


def test_outputs_are_fresh_tensors():
    c = chain(2, 3)  # a noise-free camera: the image equals the expected frames
    out = (
        gx.Pipeline(
            c,
            outputs=("image", "expected", "pos"),
        )(c)
        if False
        else None
    )
    pipe = gx.Pipeline(
        c,
        outputs={
            "image": "image",
            "mu": "expected",
            "pos": gx.labels.Positions("beads", unit="um"),
        },
    )
    out = pipe(c)
    assert torch.equal(out["image"], out["mu"])
    assert out["image"].data_ptr() != out["mu"].data_ptr()
    assert out["pos"].data_ptr() != c.emitters["beads"].position.data_ptr()


def test_capacities():
    c = _noisy_chain(40, 5)
    pipe = gx.Pipeline(c, outputs=("expected",))
    assert pipe.batch == 64 and pipe.slots == {"beads": 8}
    with pytest.raises(gx.CapacityError, match="batch capacity"):
        pipe(_noisy_chain(65, 5), key=torch.arange(65))
    with pytest.raises(gx.CapacityError, match="slot capacity"):
        pipe(_noisy_chain(4, 9))
    small = gx.Pipeline(c, outputs=("expected",), batch=40, slots={"beads": 5})
    assert small.batch == 40
    with pytest.raises(gx.CapacityError):
        gx.Pipeline(c, batch=8)


def test_ragged_last_batch_and_slot_padding_match_whole_chains():
    full = _noisy_chain(40, 5)
    pipe64 = gx.Pipeline(full, outputs=("expected", "image"), batch=64, slots={"beads": 16})
    pipe40 = gx.Pipeline(full, outputs=("expected", "image"))
    key = torch.arange(40)
    a, b = pipe64(full, key=key), pipe40(full, key=key)
    assert torch.equal(a["expected"], b["expected"]) and torch.equal(a["image"], b["image"])
    beads = full.emitters["beads"]
    padded = beads.replace(
        position=torch.cat([beads.position, beads.position[:, :3]], 1),
        photons=torch.cat([beads.photons, beads.photons[:, :3]], 1),
        presence=torch.cat([torch.ones(40, 5), torch.zeros(40, 3)], 1),
    )
    with_presence = full.replace(emitters={"beads": beads.replace(presence=torch.ones(40, 5))})
    pipe_p = gx.Pipeline(with_presence, outputs=("expected",), slots={"beads": 16})
    assert torch.equal(
        pipe_p(full.replace(emitters={"beads": padded}))["expected"],
        pipe_p(with_presence)["expected"],
    )


def _binding_template(b=64, n=8):
    beads = emitters(b, n, presence=torch.ones(b, n))
    beads = beads.replace(photons=1000.0)
    objective, camera = (
        gx.Objective(NA=0.7, magnification=50),
        gx.Camera(pixel_size=6.5, shape=(48, 48)),
    )
    return gx.Chain(
        emitters={"beads": beads},
        imaging=gx.imaging.Sprites(objective, camera),
        environment=gx.env.Homogeneous(1.33),
    )


def test_values_bound_by_field_path():
    template = _binding_template()
    pipe = gx.Pipeline(
        template,
        outputs={"mu": "expected"},
        inputs=("beads.position", "beads.presence", "objective.focus"),
    )
    pos = torch.rand(40, 5, 3) * 5
    values = {
        "beads.position": pos,
        "beads.presence": torch.ones(40, 5),
        "objective.focus": torch.linspace(-0.1, 0.1, 40),
    }
    via_values = pipe(values)["mu"]
    via_bind = pipe(pipe.bind(values))["mu"]
    whole = gx.tree.replace(
        template,
        {
            "emitters.beads.position": pos,
            "emitters.beads.presence": torch.ones(40, 5),
            "imaging.objective.focus": torch.linspace(-0.1, 0.1, 40),
        },
    )
    assert torch.equal(via_values, via_bind) and torch.equal(via_values, pipe(whole)["mu"])
    assert via_values.shape == (40, 1, 48, 48)
    # a whole data object at its path
    new_beads = gx.Emitters(
        position=pos, photons=1000.0, presence=torch.ones(40, 5), emission=gx.Spectrum.line(0.6)
    )
    assert torch.allclose(
        pipe({"beads": new_beads, "objective.focus": torch.linspace(-0.1, 0.1, 40)})["mu"],
        via_values,
    )


def test_binding_rules():
    template = _binding_template()
    pipe = gx.Pipeline(template, outputs=("expected",), inputs=("beads.position", "beads.presence"))
    with pytest.raises(gx.BindingError, match="declare it"):
        pipe({"objective.focus": torch.zeros(64)})  # undeclared and wider than the shared template
    pipe({"objective.focus": 0.25})  # undeclared but the same (shared) role
    with pytest.raises(gx.BindingError, match="static"):
        pipe({"camera.unit": "e"})
    with pytest.raises(gx.BindingError, match="names nothing"):
        pipe({"cells.position": torch.zeros(1, 1, 3)})
    with pytest.raises(gx.BindingError, match="absent in the template"):
        pipe({"beads.rotation": torch.zeros(64, 8, 4)})
    with pytest.raises(gx.BindingError, match="other per-image"):
        gx.Pipeline(chain(8, 6), inputs=("beads.position",))(
            {"beads.position": torch.zeros(4, 6, 3)}
        )


def test_structure_mismatch_is_refused():
    pipe = gx.Pipeline(chain(4, 3), outputs=("expected",))
    other = gx.tree.replace(chain(4, 3), {"imaging.camera.unit": "e"})
    with pytest.raises(gx.StructureError, match="does not fit the template"):
        pipe(other)


def test_zero_gradient_paths_are_refused_at_call_time():
    c = chain(2, 3)
    pipe = gx.Pipeline(c, outputs=("expected",))
    beads = c.emitters["beads"]
    rotating = beads.replace(rotation=torch.zeros(2, 3, 4, requires_grad=True))
    pipe_r = gx.Pipeline(
        c.replace(emitters={"beads": beads.replace(rotation=torch.zeros(2, 3, 4))}),
        outputs=("expected",),
    )
    with pytest.raises(gx.GradientPathError, match=r"beads\.rotation"):
        pipe_r(c.replace(emitters={"beads": rotating}))
    table = pipe.gradient_table
    assert table.quality("emitters.beads.position") == "exact"
    assert table.quality("beads.position") == "exact"  # logical paths, as explain() prints them
    assert "imaging.objective.pupil" not in table.rows  # an empty pupil has no leaves
    with pytest.raises(KeyError, match="names no field"):
        table.quality("imaging.objective.pupil")
    with pytest.raises(KeyError, match="did you mean"):
        table.quality("beads.positon")


def test_binding_an_unknown_field_is_a_binding_error():
    c = chain(2, 3)
    pipe = gx.Pipeline(c, outputs=("expected",))
    with pytest.raises(gx.BindingError, match="names nothing"):
        pipe({"beads.nonexistent": torch.zeros(2, 3)})
    with pytest.raises(gx.BindingError, match="names nothing"):
        pipe.bind({"nonexistent": torch.zeros(2, 3)})


def test_validity_policies():
    c = chain(2, 3, na=0.9)
    with pytest.warns(gx.GradixWarning, match="high NA"):
        gx.Pipeline(c)
    with pytest.raises(gx.ValidityError, match="high NA"):
        gx.Pipeline(c, on_invalid="raise")
    pipe = gx.Pipeline(c, on_invalid="ignore")
    assert any(v.severity == "warn" for v in pipe.violations)


def test_explain_and_json_roundtrip():
    c = _noisy_chain()
    with pytest.warns(gx.GradixWarning, match="DOF"):
        pipe = gx.Pipeline(c, outputs=("image",), envelope={"beads.position.z": (-1.0, 1.0)})
    text = pipe.explain()
    for needle in (
        "Pipeline",
        "envelope",
        "beads.position.z",
        "(explicit)",
        "sampling",
        "emit.gaussian",
        "memory",
    ):
        assert needle in text
    frozen = pipe.to_json()
    state = json.loads(frozen)
    assert state["chunks"] == dict(pipe.memory.chunks) and state["hash"] == pipe.hash
    again = gx.Pipeline.from_json(frozen, c)
    assert again.hash == pipe.hash
    assert torch.equal(
        again(c, key=torch.arange(8))["image"], pipe(c, key=torch.arange(8))["image"]
    )
    with pytest.raises(gx.StructureError):
        gx.Pipeline.from_json(frozen, chain(8, 6, noise=gx.noise.Ideal()))


def test_hash_is_stable_across_builds():
    assert gx.Pipeline(chain(4, 3)).hash == gx.Pipeline(chain(4, 3)).hash
    # the envelope is part of the hash: different values hash equal under the same envelope
    box = {f"beads.position.{c}": (-1.0, 9.0) for c in "xyz"}
    first = gx.Pipeline(chain(4, 3), envelope=box, on_invalid="ignore")
    second = gx.Pipeline(chain(4, 3, seed=9), envelope=box, on_invalid="ignore")
    assert first.hash == second.hash
    assert gx.Pipeline(chain(4, 3)).hash != gx.Pipeline(chain(4, 3, seed=9)).hash


def test_chunked_execution_matches_a_single_pass():
    c = _noisy_chain(16, 4)
    whole = gx.Pipeline(c, outputs=("image", "expected"))
    chunked = gx.Pipeline(c, outputs=("image", "expected"), recorded_chunks={"batch": 4})
    key = torch.arange(16)
    a, b = whole(c, key=key), chunked(c, key=key)
    assert torch.equal(a["image"], b["image"]) and torch.equal(a["expected"], b["expected"])


def test_image_needs_a_key_for_noisy_cameras():
    c = _noisy_chain()
    pipe = gx.Pipeline(c)
    with pytest.raises(gx.StructureError, match="needs a key"):
        pipe(c)
    assert gx.Pipeline(c, outputs=("expected",))(c)["expected"].shape == (8, 1, 64, 64)


def test_unsupported_chains_raise_plan_errors():
    objective, camera = gx.Objective(NA=0.7), gx.Camera(pixel_size=6.5, shape=(8, 8))
    with pytest.raises(gx.PlanError, match="nothing to render"):
        gx.Pipeline(
            gx.Chain(
                imaging=gx.imaging.Sprites(objective, camera), environment=gx.env.Homogeneous(1.0)
            )
        )
    with pytest.raises(gx.PlanError, match="environment"):
        gx.Pipeline(
            gx.Chain(emitters={"b": emitters(1, 1)}, imaging=gx.imaging.Sprites(objective, camera))
        )


def test_chain_reserved_names():
    with pytest.raises(gx.StructureError, match="reserved"):
        gx.Chain(emitters={"camera": emitters(1, 1)})


def test_the_camera_has_one_home():
    template = _binding_template(4, 3)
    cam = template.camera
    assert cam is template.imaging.camera and template.detector is None
    assert template.resolve("camera.pixel_size") == ["imaging.camera.pixel_size"]
    with pytest.raises(gx.StructureError, match="imaging element's camera"):
        template.replace(camera=gx.Camera(pixel_size=6.5, shape=(48, 48)))
    template.replace(camera=cam)  # repeating the same object is fine
    moved = template.to(torch.float64)
    assert moved.camera is moved.imaging.camera
    # binding the pitch moves the image and the labels together
    pipe = gx.Pipeline(template, outputs={"mu": "expected", "pos": gx.labels.Positions("beads")})
    out = pipe(template)
    wide = pipe({"camera.pixel_size": 13.0})
    assert not torch.equal(out["mu"], wide["mu"])
    assert torch.allclose(wide["pos"], (out["pos"] + 0.5) / 2 - 0.5, atol=1e-5)


def test_a_learnable_pixel_pitch_gets_a_gradient():
    template = _binding_template(2, 3)
    pitch = torch.tensor(6.5, requires_grad=True)
    chain_ = gx.tree.replace(template, {"imaging.camera.pixel_size": pitch})
    mu = gx.Pipeline(chain_, outputs=("expected",))(chain_)["expected"]
    weights = torch.rand_like(mu, generator=torch.Generator().manual_seed(0))
    (grad,) = torch.autograd.grad((mu * weights).sum(), pitch)
    assert torch.isfinite(grad) and grad != 0


# ---- composition audit regressions -----------------------------------------------------------


def test_declared_inputs_are_checked():
    with pytest.raises(gx.BindingError, match="postion"):
        gx.Pipeline(chain(2, 3), inputs=("beads.postion",))
    with pytest.raises(gx.BindingError, match="static"):
        gx.Pipeline(chain(2, 3), inputs=("camera.shape",))


def test_output_specs_are_strict_and_repeated_outputs_do_not_alias():
    with pytest.raises(gx.StructureError, match="not an output spec"):
        gx.Pipeline(chain(2, 3), outputs={"x": gx.Objective(NA=1.0)})
    out = gx.Pipeline(chain(2, 3), outputs={"a": "expected", "b": "expected"})(chain(2, 3))
    assert torch.equal(out["a"], out["b"])
    out["a"].zero_()
    assert out["b"].abs().sum() > 0


def test_labels_get_one_row_per_image_also_when_chunked():
    beads = gx.Emitters(
        position=torch.tensor([[[2.0, 2.0, 0.0], [4.0, 4.0, 0.0], [6.0, 5.0, 0.0]]]),
        photons=torch.full((16, 3), 500.0),
        emission=gx.Spectrum.line(0.6),
    )
    template = chain(16, 3).replace(emitters={"beads": beads})
    outputs = {"mu": "expected", "pos": gx.labels.Positions("beads")}
    whole = gx.Pipeline(template, outputs=outputs)(template)
    chunked = gx.Pipeline(template, outputs=outputs, recorded_chunks={"batch": 4})(template)
    assert whole["pos"].shape == chunked["pos"].shape == (16, 3, 2)
    assert torch.equal(whole["mu"], chunked["mu"]) and torch.equal(whole["pos"], chunked["pos"])


def test_chunking_never_changes_noise_even_with_batch_keys():
    noisy = chain(16, 3, noise=gx.noise.PoissonGaussian(read=1.0))
    whole = gx.Pipeline(noisy, outputs=("image",))(noisy, key=7)["image"]
    pipe = gx.Pipeline(noisy, outputs=("image",), recorded_chunks={"batch": 4})
    assert pipe.memory.chunks["batch"] == 4
    assert torch.equal(whole, pipe(noisy, key=7)["image"])
    keys = torch.arange(16) + 100
    image_keyed = gx.Pipeline(noisy, outputs=("image",))(noisy, key=keys)["image"]
    assert torch.equal(image_keyed, pipe(noisy, key=keys)["image"])


def test_pipelines_with_labels_round_trip_through_json():
    template = chain(4, 3)
    pipe = gx.Pipeline(template, outputs={"mu": "expected", "pos": gx.labels.Positions("beads")})
    again = gx.Pipeline.from_json(pipe.to_json(), template)
    assert again.hash == pipe.hash
    assert torch.equal(again(template)["pos"], pipe(template)["pos"])


def test_eager_chain_call_equals_the_pipeline():
    c = chain(4, 3)
    eager = c(outputs=("expected",))["expected"]
    assert eager.shape[0] == 4
    with pytest.raises(gx.PlanError, match="no configured element"):
        gx.Pipeline(c).static("light")


def test_gradient_qualities_come_from_the_elements():
    @dataclasses.dataclass(frozen=True, kw_only=True, eq=False)
    class BlindSprites(gx.imaging.Sprites):
        """Sprites that declare they ignore photons and the NA."""

        caps: ClassVar = dataclasses.replace(
            gx.imaging.Sprites.caps,
            grad_quality={**gx.imaging.Sprites.caps.grad_quality, "objective.NA": "zero"},
            reads={"position": "exact", "presence": "exact", "emission": "exact"},
        )

    base = chain(2, 3)
    c = base.replace(imaging=BlindSprites(base.objective, base.camera))
    table = gx.Pipeline(c, outputs=("expected",)).gradient_table
    assert table.quality("emitters.beads.position") == "exact"
    assert table.quality("emitters.beads.photons") == "zero"
    assert table.quality("imaging.objective.NA") == "zero"
    assert table.quality("imaging.camera.gain") == "exact"  # through the camera's own route
    noisy = chain(2, 3, noise=gx.noise.PoissonGaussian(read=1.0))
    rows = gx.Pipeline(noisy, outputs=("expected", "image")).gradient_table.rows
    assert rows["imaging.camera.noise.read"] == {"expected": "zero", "image": "exact"}
    assert rows["emitters.beads.position"] == {"expected": "exact", "image": "biased"}
    assert rows["imaging.camera.gain"]["image"] == "exact"  # after the shot noise


def test_module_parameters_in_static_fields_keep_gradients():
    @dataclasses.dataclass(frozen=True, kw_only=True, eq=False)
    class ModuleSprites(gx.imaging.Sprites):
        """Sprites scaled by a learnable module."""

        scale: torch.nn.Module | None = gx.knob(default=None)

        def forward(self, *inputs, static):
            out = super().forward(*inputs, static=static)
            assert self.scale is not None
            return out.replace(data=out.data * self.scale(torch.ones(1)))

    base = chain(2, 3)
    module = torch.nn.Linear(1, 1)
    c = base.replace(imaging=ModuleSprites(base.objective, base.camera, scale=module))
    mu = gx.Pipeline(c, outputs=("expected",))(c)["expected"]
    assert mu.requires_grad
    mu.sum().backward()
    assert module.weight.grad is not None


def test_label_tensor_fields_keep_their_kind_through_json():
    @dataclasses.dataclass(frozen=True, eq=False)
    class Weighted(gx.labels.Positions):
        weight: torch.Tensor | complex = gx.field(
            quantity="dimensionless", role="shared", dtype="number", default=1.0
        )

    gx.register.label("test.weighted_positions")(Weighted)
    try:
        template = chain(2, 2)
        for weight in (torch.tensor(2.0), torch.tensor(1 + 2j), 3.0, 1 - 1j):
            label = Weighted("beads", weight=weight)
            pipe = gx.Pipeline(template, outputs={"mu": "expected", "w": label})
            again = gx.Pipeline.from_json(pipe.to_json(), template)
            assert again.hash == pipe.hash
            rebuilt = again.outputs["w"]
            assert isinstance(rebuilt, Weighted)
            assert type(rebuilt.weight) is type(weight)
    finally:
        gx.registry.labels.unregister("test.weighted_positions")
