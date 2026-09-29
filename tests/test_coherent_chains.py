"""Coherent Chains and Pipelines (M3a phase 1): Mie holograms through the three levels."""

import pytest
import torch

import gradix as gx


def hologram_chain(b=3, n=2, dtype=torch.float64, noise=None):
    camera = gx.Camera(pixel_size=6.5, shape=(32, 32), noise=noise or gx.noise.Ideal())
    objective = gx.Objective(NA=0.8, magnification=60)
    fov = gx.coords.field_of_view(camera, objective).to(dtype)
    g = torch.Generator().manual_seed(3)
    position = torch.cat(
        [
            fov * (0.25 + 0.5 * torch.rand(b, n, 2, generator=g, dtype=dtype)),
            2.0 * torch.rand(b, n, 1, generator=g, dtype=dtype) - 1.0,
        ],
        -1,
    )
    beads = gx.Spheres(
        position=position,
        radius=0.05 + 0.1 * torch.rand(b, n, generator=g, dtype=dtype),
        material=1.59,
    )
    return gx.Chain(
        light=gx.light.PlaneWave(0.532, irradiance=1e4),
        scatterers={"beads": gx.interact.Mie(beads)},
        imaging=gx.imaging.Coherent(objective, camera),
        environment=gx.env.Homogeneous(1.33),
    )


def test_eager_and_pipeline_holograms_agree():
    chain = hologram_chain()
    eager = chain(outputs=("expected",)).expected
    pipe = gx.Pipeline(chain, outputs=("expected",))
    # the Pipeline's envelope has headroom, so Mie keeps a few more (negligible) orders
    assert torch.allclose(pipe(chain).expected, eager, rtol=1e-12, atol=0.0)
    assert eager.shape == (3, 1, 32, 32)
    # values bound by field path equal a whole new Chain
    radius = torch.full((3, 2), 0.12, dtype=torch.float64)
    bound = pipe({"beads.radius": radius}).expected
    replaced = gx.tree.replace(chain, {"scatterers.beads.objects.radius": radius})
    assert torch.equal(bound, pipe(replaced).expected)


def test_the_envelope_sizes_the_mie_orders():
    chain = hologram_chain()
    pipe = gx.Pipeline(chain, outputs=("expected",), envelope={"beads.radius": (0.02, 0.5)})
    static = pipe.static("scatterers.beads")
    assert isinstance(static, gx.interact.MieStatic)
    x_max = 2 * 3.141592653589793 * 1.33 * 0.5 / 0.532
    assert static.terms == gx.special.mie.terms(x_max)
    assert "mie.terms" in pipe.explain() and "lower" not in pipe.explain()


def test_gradients_reach_the_spheres_and_the_light():
    chain = hologram_chain()
    table = gx.Pipeline(chain, outputs=("expected",)).gradient_table
    for path in ("beads.radius", "beads.material", "beads.position", "light.wavelength"):
        assert table.quality(path) == "exact", path
    assert table.quality("beads.id") == "zero"


def test_the_field_output_is_the_normalised_image_field():
    chain = hologram_chain(b=1, n=1)
    out = chain(outputs={"E": "field", "phase": gx.out.Field(layout="phase"), "mu": "expected"})
    field = out.E
    assert field.shape == (1, 1, 32, 32) and field.dtype == torch.complex128
    far = field[0, 0, 0, 0]  # a corner, far from the bead: the background alone
    assert abs(complex(far) - 1.0) < 0.05
    # a polystyrene bead in water delays the light: the phase at its centre is positive
    pitch = 6.5 / 60
    x, y, _z = chain.population("beads").position[0, 0].tolist()
    i, j = int(y / pitch), int(x / pitch)
    assert float(out.phase[0, 0, i, j]) > 0.0
    re_im = chain(outputs={"E": gx.out.Field(layout="re_im")}).E
    assert torch.allclose(torch.view_as_complex(re_im), field)


def test_epi_fields_normalise_by_the_incident_amplitude():
    chain = hologram_chain(b=1, n=1)
    epi = chain.replace(light=gx.light.Uniform(wavelength=0.532, irradiance=1e4))
    field = epi(outputs={"E": gx.out.Field(normalize="incident")}).E
    assert float(field.abs().max()) > 0.0 and torch.isfinite(field).all()
    with pytest.raises(gx.StructureError, match="no background reaches the camera"):
        epi(outputs={"E": "field"})


def test_labels_and_focus_stacks_work_on_scatterers():
    chain = hologram_chain(b=2, n=2)
    out = chain(outputs={"xy": gx.labels.Positions("beads", unit="px", dims="xy")})
    assert out.xy.shape == (2, 2, 2)
    stack = chain.replace(acquisition=gx.acq.FocusStack(focus=torch.tensor([-0.5, 0.0, 0.5])))
    frames = stack(outputs=("expected",)).expected
    assert frames.shape == (2, 3, 32, 32)
    single = chain.replace(
        imaging=chain.imaging.replace(objective=chain.objective.replace(focus=0.5))
    )
    assert torch.allclose(frames[:, 2], single(outputs=("expected",)).expected[:, 0])


def test_incoherent_and_coherent_parts_do_not_mix():
    chain = hologram_chain()
    emitters = gx.Emitters(
        position=torch.zeros(1, 1, 3), photons=100.0, emission=gx.Spectrum.line(0.6)
    )
    with pytest.raises(gx.PlanError, match="cannot also hold emitters"):
        chain.replace(emitters={"dye": emitters})(outputs=("expected",))
    sprites = gx.imaging.Sprites(chain.objective, chain.camera)
    with pytest.raises(gx.PlanError, match="not coherent"):
        chain.replace(imaging=sprites)(outputs=("expected",))
    with pytest.raises(gx.PlanError, match="plane-wave light"):
        chain.replace(light=None)(outputs=("expected",))


def test_crlb_of_radius_and_index_from_holograms():
    chain = hologram_chain(b=2, n=1)
    pipe = gx.Pipeline(chain, outputs=("expected",))
    bound = gx.crlb(pipe, chain, wrt=("beads.radius", "beads.material"))
    assert bound["beads.radius"].shape == (2, 1) and bound["beads.material"].shape == ()
    assert torch.isfinite(bound["beads.radius"]).all() and float(bound["beads.material"]) > 0
    central = gx.crlb(pipe, chain, wrt=("beads.radius", "beads.material"), method="central")
    assert torch.allclose(central["beads.radius"], bound["beads.radius"], rtol=1e-4)


def test_the_planner_builds_the_hand_built_hologram():
    chain = hologram_chain(b=2, n=2)
    beads = chain.population("beads")
    sample = gx.Sample({"beads": beads}, environment=chain.environment)
    scope = gx.presets.InlineHolography(
        light=chain.light, objective=chain.objective, camera=chain.camera
    )
    planned = gx.plan(sample, scope, "standard", outputs=("expected",))
    assert planned.routes["beads"].element == "interact.mie"
    hand = gx.Pipeline(chain, outputs=("expected",))
    assert torch.equal(planned(sample, scope).expected, hand(chain).expected)
    assert "MiB" in hand.explain() and hand.memory.estimate_bytes > 0


def test_draft_routes_small_spheres_to_dipoles():
    chain = hologram_chain(b=1, n=1)
    small = chain.population("beads").replace(radius=torch.tensor([[0.02]], dtype=torch.float64))
    scope = gx.presets.InlineHolography(
        light=chain.light, objective=chain.objective, camera=chain.camera
    )
    sample = gx.Sample({"beads": small}, environment=chain.environment)
    assert gx.plan(sample, scope, "draft").routes["beads"].element == "interact.dipole"
    big = gx.Sample({"beads": chain.population("beads")}, environment=chain.environment)
    assert gx.plan(big, scope, "draft").routes["beads"].element == "interact.mie"


def test_holography_presets_need_plane_waves():
    chain = hologram_chain(b=1, n=1)
    sheet = gx.light.Sheet(waist=1.0)
    with pytest.raises(gx.StructureError, match="plane-wave light"):
        gx.presets.InlineHolography(light=sheet, objective=chain.objective, camera=chain.camera)
