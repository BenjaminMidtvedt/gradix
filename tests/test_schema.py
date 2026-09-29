"""Schemas, the pytree base, layout, stacking, SI units and signatures (§6.1, §11.9, §11.10)."""

import dataclasses
import json

import pytest
import torch
from scenes import chain, emitters, tensor

import gradix as gx
from gradix.schema.layout import canonical
from gradix.units import nm, um


def test_object_fields_need_a_batch_axis():
    with pytest.raises(gx.StructureError, match="batch axis"):
        gx.Emitters(position=torch.zeros(5, 3), photons=1.0, emission=gx.Spectrum.line(0.6))
    pos = torch.zeros(1, 5, 3)
    e = gx.Emitters(position=pos, photons=1.0, emission=gx.Spectrum.line(0.6))
    assert e.position is pos


def test_canonical_layout_resists_the_b_equals_n_trap():
    # B == N == 8: a per-image focus and per-object positions must not mix
    b = n = 8
    objective = gx.Objective(NA=0.7, focus=torch.arange(b, dtype=torch.float32))
    beads = gx.Emitters(
        position=torch.zeros(b, n, 3), photons=torch.ones(b, n), emission=gx.Spectrum.line(0.6)
    )
    focus = canonical(objective.focus, objective.schema()["focus"])
    photons = canonical(beads.photons, beads.schema()["photons"])
    assert focus.shape == (8, 1) and photons.shape == (8, 1, 8)  # focus is a setting: [B, T]
    assert torch.equal((photons * focus.reshape(-1, 1, 1))[3, 0], torch.full((8,), 3.0))


def test_frames_are_an_explicit_axis():
    beads = gx.Emitters(
        position=torch.zeros(2, 5, 4, 3), photons=1.0, emission=gx.Spectrum.line(0.6)
    )
    assert canonical(beads.position, beads.schema()["position"]).shape == (2, 5, 4, 3)


def test_structure_errors_name_the_field_and_fix():
    with pytest.raises(gx.StructureError, match=r"event shape must be \[3\]"):
        gx.Emitters(position=torch.zeros(1, 5, 2), photons=1.0, emission=gx.Spectrum.line(0.6))
    with pytest.raises(gx.StructureError, match="dtype kind 'complex'"):
        gx.Emitters(
            position=torch.zeros(1, 5, 3, dtype=torch.complex64),
            photons=1.0,
            emission=gx.Spectrum.line(0.6),
        )
    with pytest.raises(gx.StructureError, match=r"torch.tensor"):
        gx.Objective(NA=[0.7])  # ty: ignore[invalid-argument-type] - deliberately invalid
    with pytest.raises(gx.StructureError, match="axis N is 4 in photons but 5 in position"):
        gx.Emitters(
            position=torch.zeros(1, 5, 3), photons=torch.ones(1, 4), emission=gx.Spectrum.line(0.6)
        )
    with pytest.raises(gx.StructureError, match="static structure"):
        gx.Camera(pixel_size=6.5, shape=(4, 4), unit=torch.zeros(1))  # ty: ignore[invalid-argument-type]
    with pytest.raises(gx.StructureError, match="not one of"):
        gx.Camera(pixel_size=6.5, shape=(4, 4), unit="volts")


def test_undeclared_fields_are_refused():
    @dataclasses.dataclass(frozen=True, kw_only=True, eq=False)
    class Bad(gx.DataObject):
        radius: float = 1.0

    with pytest.raises(TypeError, match=r"gx\.field"):
        Bad()


def test_nodes_store_exactly_what_they_are_given():
    pos = torch.zeros(1, 3, 3)
    beads = gx.Emitters(position=pos, photons=2.0, emission=gx.Spectrum.line(0.6))
    assert beads.position is pos and beads.photons == 2.0
    moved = beads.to(torch.float64)
    assert isinstance(moved.position, torch.Tensor) and moved.position.dtype == torch.float64
    assert moved.photons == 2.0
    assert beads.replace(photons=3.0).photons == 3.0 and beads.photons == 2.0


def test_static_lists_become_tuples():
    obj = gx.Objective(NA=0.8, pupil=[])  # ty: ignore[invalid-argument-type] - a list on purpose
    assert obj.pupil == ()


def test_element_holding_a_parameter_sees_optimizer_updates():
    focus = torch.nn.Parameter(torch.tensor(0.4))
    objective, camera = (
        gx.Objective(NA=0.7, magnification=50, focus=focus),
        gx.Camera(pixel_size=6.5, shape=(32, 32)),
    )
    sprites = gx.imaging.Sprites(objective, camera)
    env = gx.env.Homogeneous(1.33)
    target_beads = emitters(1, 3, seed=1, z=0.0, fov=4.0)  # inside the 4.16 µm field of view
    target = camera.expected(
        sprites.replace(objective=objective.replace(focus=0.0))(
            gx.lower.emitter_set(target_beads), env
        )
    )
    opt = torch.optim.Adam([focus], lr=0.05)
    losses = []
    for _ in range(10):
        mu = camera.expected(
            sprites(gx.lower.emitter_set(target_beads), env)
        )  # the same element every step
        loss = ((mu - target) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        assert focus.grad is not None and torch.isfinite(focus.grad)
        opt.step()
        losses.append(float(loss.detach()))
    assert losses[-1] < losses[0]


def test_tree_utilities():
    c = chain(2, 3)
    paths = gx.tree.paths(c)
    assert "emitters.beads.position" in paths and "imaging.objective.focus" in paths
    assert gx.tree.get(c, "imaging.objective.NA") == 0.7
    c2 = gx.tree.replace(c, {"imaging.objective.NA": 0.5, "environment.n": 1.4})
    assert c2.imaging.objective.NA == 0.5 and c2.environment.n == 1.4 and c.environment.n == 1.33
    with pytest.raises(gx.StructureError, match="has no field"):
        gx.tree.get(c, "imaging.objective.na")
    p = torch.nn.Parameter(torch.tensor(0.1))
    c3 = gx.tree.replace(c, {"imaging.objective.focus": p})
    assert gx.tree.parameters(c3) == {"imaging.objective.focus": p}
    assert gx.tree.map(lambda t: t * 0, c).emitters["beads"].position.abs().max() == 0


def test_replace_validates_only_the_final_state():
    beads = emitters(2, 3)
    bigger = gx.tree.replace(beads, {"position": torch.zeros(5, 4, 3), "photons": torch.ones(5, 4)})
    assert bigger.position.shape == (5, 4, 3)


def test_stack_pads_with_presence_zero():
    items = [emitters(1, n, seed=n) for n in (3, 5, 9)]
    batch = gx.stack(items)
    assert batch.position.shape == (3, 16, 3)
    assert batch.presence.shape == (3, 16)
    assert batch.presence.sum(1).tolist() == [3.0, 5.0, 9.0]
    assert torch.equal(
        batch.position[0, 3], batch.position[0, 2]
    )  # padded slots repeat the last one


def test_stack_keeps_equal_shared_numbers_and_rejects_different_structure():
    a = gx.Emitters(position=torch.zeros(1, 2, 3), photons=5.0, emission=gx.Spectrum.line(0.6))
    b = gx.Emitters(position=torch.ones(1, 2, 3), photons=5.0, emission=gx.Spectrum.line(0.6))
    assert gx.stack([a, b]).photons == 5.0
    c = gx.Emitters(position=torch.ones(1, 2, 3), photons=7.0, emission=gx.Spectrum.line(0.6))
    stacked = gx.stack([a, c]).photons
    assert isinstance(stacked, torch.Tensor) and stacked.tolist() == [[5.0], [7.0]]
    d = a.replace(presence=torch.ones(1, 2))
    with pytest.raises(gx.StructureError, match="item 1 differs from item 0 at presence"):
        gx.stack([a, d])


def test_stack_checks_values_on_cpu():
    bad = gx.Emitters(
        position=torch.zeros(1, 2, 3),
        photons=torch.tensor([[1.0, -2.0]]),
        emission=gx.Spectrum.line(0.6),
    )
    with pytest.raises(gx.StructureError, match="violates the constraint 'nonnegative'"):
        gx.stack([bad])


def test_pad_roundtrips_ragged_lists():
    ragged = [torch.randn(n, 3) for n in (0, 2, 7)]
    pos, presence = gx.pad(ragged)
    assert pos.shape == (3, 8, 3) and presence.shape == (3, 8)
    for i, t in enumerate(ragged):
        n = t.shape[0]
        assert torch.equal(pos[i, :n], t) and presence[i].sum() == n
    with pytest.raises(gx.StructureError):
        gx.pad(ragged, n=4)


def test_si_roundtrip_is_identity():
    c = chain(2, 3)
    back = gx.to_si(gx.from_si(c))
    for path, value in gx.tree.leaves(c).items():
        other = gx.tree.leaves(back)[path]
        if isinstance(value, torch.Tensor):
            assert isinstance(other, torch.Tensor) and torch.allclose(value, other)
        else:
            assert value == pytest.approx(other)


def test_from_si_scales_by_quantity():
    beads = gx.Emitters(
        position=torch.full((1, 1, 3), 2e-6), photons=10.0, emission=gx.Spectrum.line(600e-9)
    )
    converted = gx.from_si(beads)
    assert isinstance(converted.position, torch.Tensor)
    assert torch.allclose(converted.position, torch.full((1, 1, 3), 2.0))
    assert converted.emission.wavelengths == pytest.approx(0.6)
    assert converted.photons == 10.0
    wave = gx.from_si(gx.light.PlaneWave(532e-9, irradiance=1e12))
    assert wave.irradiance == pytest.approx(1.0)  # photons/m² → photons/µm²


def test_signatures():
    a, b = chain(2, 3, seed=0), chain(2, 3, seed=1)
    assert gx.signature(a).digest == gx.signature(b).digest  # same structure, different values
    c = gx.tree.replace(a, {"imaging.objective.NA": torch.full((2,), 0.7)})
    assert gx.signature(c).digest != gx.signature(a).digest
    difference = gx.signature(a).first_difference(gx.signature(c))
    assert difference is not None and "imaging.objective.NA" in difference
    smaller = chain(1, 5, seed=0)
    assert gx.signature(smaller).structure_digest == gx.signature(a).structure_digest


def test_schema_export():
    for cls in [
        gx.Emitters,
        gx.Spheres,
        gx.Camera,
        gx.imaging.Sprites,
        gx.Objective,
        gx.light.PlaneWave,
    ]:
        schema = gx.schema_of(cls)
        json.dumps(schema)
        fields = {f["name"]: f for f in schema["fields"]}
        for f in fields.values():
            if f["kind"] == "tensor":
                assert f["quantity"] and f["unit"] and f["role"]
    photons = {f["name"]: f for f in gx.schema_of(gx.Emitters)["fields"]}["photons"]
    assert photons["unit"] == "photons" and photons["role"] == "object" and photons["required"]


def test_select_images():
    beads = emitters(4, 3)
    one = beads.select(2)
    assert one.position.shape == (1, 3, 3) and torch.equal(one.position[0], beads.position[2])


def test_spectrum_constructors():
    assert gx.Spectrum.line(500 * nm).bins == 1
    band = gx.Spectrum.band(680 * nm, 30 * nm, bins=5)
    wl, w = band.arrays()
    assert (
        wl.shape == (1, 5)
        and abs(float(w.sum()) - 1) < 1e-6
        and abs(float((wl * w).sum()) - 0.68) < 1e-6
    )
    per_image = gx.Spectrum.line(torch.tensor([0.5, 0.6]))
    assert per_image.arrays()[0].shape == (2, 1)
    table = gx.Spectrum.table(torch.tensor([0.5, 0.6]), torch.tensor([1.0, 3.0]))
    assert torch.allclose(table.arrays()[1], torch.tensor([[0.25, 0.75]]))


def test_units_module():
    assert 532 * nm == pytest.approx(0.532) and 6.5 * um == 6.5


# ---- audit regressions: stacking sub-batches, empty images, nested per-object children --------

_LINE = gx.Spectrum.line(0.6)


@dataclasses.dataclass(frozen=True, kw_only=True, eq=False)
class _PerObjectLabel(gx.DataObject):
    """Test labeling with a per-object field."""

    photons: torch.Tensor = gx.field(quantity="photons", role="object", constraint="nonnegative")


def test_stack_of_sub_batches_keeps_images_aligned():
    a = gx.Emitters(
        position=torch.zeros(2, 8, 3), photons=torch.full((1, 8), 100.0), emission=_LINE
    )
    b = gx.Emitters(
        position=torch.ones(1, 8, 3),
        photons=torch.tensor([[300.0] * 8, [400.0] * 8]),
        emission=_LINE,
    )
    s = gx.stack([a, b])
    assert tensor(s.position).shape == (4, 8, 3)
    assert tensor(s.photons)[:, 0].tolist() == [100.0, 100.0, 300.0, 400.0]
    assert tensor(s.position)[:, 0, 0].tolist() == [0.0, 0.0, 1.0, 1.0]


def test_an_empty_image_gets_no_phantom_object():
    empty = gx.Emitters(
        position=torch.zeros(1, 0, 3), photons=torch.tensor([[1e3]]), emission=_LINE
    )
    full = gx.Emitters(position=torch.ones(1, 3, 3), photons=torch.tensor([[1e3]]), emission=_LINE)
    s = gx.stack([empty, full])
    presence = tensor(s.presence)
    assert presence[0].sum() == 0 and presence[1].sum() == 3


def test_empty_images_are_padded_with_values_that_satisfy_constraints():
    empty = gx.Spheres(position=torch.zeros(1, 0, 3), radius=torch.zeros(1, 0))
    full = gx.Spheres(position=torch.ones(1, 2, 3), radius=torch.full((1, 2), 0.5))
    s = gx.stack([empty, full])
    assert bool((tensor(s.radius) > 0).all())  # log(radius) stays finite on padded slots


def test_python_int_values_stack_to_floats():
    a = gx.Emitters(position=torch.zeros(1, 2, 3), photons=1000, emission=_LINE)
    b = gx.Emitters(position=torch.zeros(1, 2, 3), photons=2000, emission=_LINE)
    assert tensor(gx.stack([a, b]).photons).dtype == torch.get_default_dtype()


def test_nested_per_object_children_are_padded_with_their_population():
    a = gx.Emitters(
        position=torch.zeros(1, 2, 3),
        photons=1.0,
        emission=_LINE,
        labeling=_PerObjectLabel(photons=torch.ones(1, 2)),
    )
    b = gx.Emitters(
        position=torch.zeros(1, 5, 3),
        photons=1.0,
        emission=_LINE,
        labeling=_PerObjectLabel(photons=torch.ones(1, 5)),
    )
    s = gx.stack([a, b])
    assert isinstance(s.labeling, _PerObjectLabel)
    assert s.labeling.photons.shape == tensor(s.presence).shape == (2, 8)
    assert tensor(s.presence)[0].sum() == 2


def test_a_child_must_agree_with_its_population_on_the_object_count():
    with pytest.raises(gx.StructureError):
        gx.Emitters(
            position=torch.zeros(1, 3, 3),
            photons=1.0,
            emission=_LINE,
            labeling=_PerObjectLabel(photons=torch.ones(1, 5)),
        )


def test_pad_checks_trailing_shapes_and_promotes_dtypes():
    with pytest.raises(gx.StructureError, match="shape"):
        gx.pad([torch.zeros(3, 3), torch.zeros(3, 1)])
    (out, presence) = gx.pad([torch.zeros(2, 3), torch.zeros(1, 3, dtype=torch.float64)])
    assert out.dtype == torch.float64 and presence.shape == (2, 8)


def test_select_with_a_zero_dim_tensor_keeps_the_batch_axis():
    e = gx.Emitters(position=torch.zeros(4, 2, 3), photons=1.0, emission=_LINE)
    assert tensor(e.select(torch.tensor(2)).position).shape == (1, 2, 3)


def test_nodes_are_torch_pytrees():
    e = gx.Emitters(position=torch.zeros(1, 2, 3), photons=torch.ones(1, 2), emission=_LINE)
    jac = torch.func.jacfwd(lambda em: (em.position**2).sum())(e)
    assert jac.position.shape == (1, 2, 3)
    batched = e.replace(photons=torch.ones(4, 2), position=torch.zeros(4, 2, 3))
    assert torch.func.vmap(lambda em: em.photons.sum())(batched).shape == (4,)
    grad = torch.func.grad(lambda em: (em.photons**2).sum())(e)
    assert torch.equal(grad.photons, torch.full((1, 2), 2.0))


def test_signatures_key_callables_modules_and_number_kinds():
    from gradix.schema.signature import _static_repr, signature

    assert _static_repr(lambda x: x) != _static_repr(lambda x: x)
    assert _static_repr(torch.nn.Linear(2, 3)) != _static_repr(torch.nn.Linear(5, 7))
    assert _static_repr(True) != _static_repr(1) and _static_repr(1) != _static_repr(1.0)
    real = gx.Emitters(position=torch.zeros(1, 3, 3), photons=1000.0, emission=_LINE)
    integer = gx.Emitters(position=torch.zeros(1, 3, 3), photons=1000, emission=_LINE)
    assert signature(real).digest == signature(integer).digest  # an int in a real field is real
    assert signature(integer).incompatibility(signature(real)) is None


def test_an_unregistered_subclass_is_a_different_element():
    from gradix.schema.signature import type_name

    class MySprites(gx.imaging.Sprites):
        pass

    assert type_name(MySprites) != type_name(gx.imaging.Sprites)


def test_population_order_is_not_structure_but_stage_order_is():
    from gradix.schema.signature import signature

    base = chain(1, 2)
    a, b = emitters(1, 2), emitters(1, 3, seed=1)
    ab = base.replace(emitters={"a": a, "b": b})
    ba = base.replace(emitters={"b": b, "a": a})
    assert signature(ab).structure_digest == signature(ba).structure_digest


def test_structural_sizes_are_exact_and_n_widening_needs_a_per_object_template():
    from gradix.schema.signature import signature

    t1 = emitters(2, 3)
    frames = t1.replace(position=t1.position[:, None].expand(2, 4, 3, 3).clone())
    issue = signature(t1).incompatibility(signature(frames))
    assert issue is not None and "structure" in issue  # T is structure, not a capacity
    shared = gx.Emitters(position=torch.zeros(1, 8, 3), photons=1000.0, emission=_LINE)
    per_object = shared.replace(photons=torch.ones(1, 8))
    assert "declare" in str(signature(shared).incompatibility(signature(per_object)))
    assert signature(per_object).incompatibility(signature(per_object.select(0))) is None


def test_signature_differences_read_like_sentences():
    from gradix.schema.signature import signature

    a = gx.Emitters(position=torch.zeros(1, 3, 3), photons=1000.0, emission=_LINE)
    b = a.replace(photons=torch.ones(1, 3))
    diff = signature(a).first_difference(signature(b))
    assert diff == "at photons: a Python real number vs a float32 tensor [1, N]"


def test_b_equals_n_equals_t_renders_every_image_and_frame_correctly():
    """§12.1: with B = N = T = 3, per-image focus and per-frame positions stay aligned."""
    b = n = t = 3
    g = torch.Generator().manual_seed(1)
    position = torch.cat(
        [
            1.0 + 2.5 * torch.rand(b, t, n, 2, generator=g),
            0.2 * torch.rand(b, t, n, 1, generator=g),
        ],
        -1,
    )
    photons = 500.0 + 500.0 * torch.rand(b, n, generator=g)
    presence = (torch.rand(b, t, n, generator=g) > 0.3).float()
    focus = torch.tensor([-0.2, 0.0, 0.3])
    objective = gx.Objective(NA=0.7, magnification=50.0, focus=focus)
    camera = gx.Camera(pixel_size=6.5, shape=(32, 32))
    beads = gx.Emitters(position=position, photons=photons, presence=presence, emission=_LINE)
    batched = gx.Chain(
        emitters={"beads": beads},
        imaging=gx.imaging.Sprites(objective, camera),
        environment=gx.env.Homogeneous(1.33),
    )
    pipe = gx.Pipeline(batched, outputs=("expected",))
    mu = pipe(batched)["expected"]
    assert mu.shape[:2] == (b, t)
    statics = pipe.static("imaging")
    for i in range(b):
        for k in range(t):
            one = gx.Emitters(
                position=position[i : i + 1, k],
                photons=photons[i : i + 1],
                presence=presence[i : i + 1, k],
                emission=_LINE,
            )
            element = gx.imaging.Sprites(objective.replace(focus=float(focus[i])), camera)
            irr = element(gx.lower.emitter_set({"beads": one}), batched.environment, static=statics)
            ref = camera.expected(irr)
            assert torch.allclose(mu[i, k], ref[0], rtol=1e-5, atol=1e-6), (i, k)
