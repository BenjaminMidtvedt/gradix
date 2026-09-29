"""Labels and coordinates (§6.6, §11.10): one pinned pixel convention, in-FOV flags."""

import math

import pytest
import torch
from scenes import emitters, optics

import gradix as gx


def test_pixel_conversion_roundtrips_and_pins_centres():
    objective, camera = optics(shape=(16, 16), magnification=10)  # pitch 0.65 µm
    x = torch.tensor([[0.325, 0.975, 1.5], [0.0, 0.65, -2.0]])
    px = gx.coords.to_pixels(x, camera, objective)
    assert torch.allclose(px[0], torch.tensor([0.0, 1.0, 1.5]))  # centres are integers; z unchanged
    assert torch.allclose(gx.coords.from_pixels(px, camera, objective), x)


def test_per_image_optics_need_a_batch_axis():
    camera = gx.Camera(pixel_size=torch.tensor([6.5, 13.0]), shape=(8, 8))
    objective = gx.Objective(NA=0.5, magnification=10.0)
    x = torch.tensor([[[0.65, 0.65]], [[1.3, 1.3]]])  # [B, N, 2]
    px = gx.coords.to_pixels(x, camera, objective)
    assert torch.allclose(px, torch.full((2, 1, 2), 0.5))


def test_field_of_view_and_pitch_express_pixels_in_micrometres():
    objective, camera = optics(shape=(20, 30), magnification=65)  # pitch 0.1 µm
    fov = gx.coords.field_of_view(camera, objective)
    assert torch.allclose(fov, torch.tensor([3.0, 2.0]))  # (width, height)
    anywhere = torch.rand(4, 50, 2) * fov  # "anywhere in the image"
    px = gx.coords.to_pixels(anywhere, camera, objective)
    assert bool((px >= -0.5).all()) and bool((px < torch.tensor([29.5, 19.5])).all())
    px_unit = gx.coords.pitch(camera, objective)  # a pixel, as a length unit
    assert math.isclose(float(20 * px_unit), 2.0, rel_tol=1e-6)
    per_image = gx.Objective(NA=0.5, magnification=torch.tensor([65.0, 130.0]))
    assert gx.coords.field_of_view(camera, per_image).shape == (2, 2)
    assert gx.coords.pitch(camera, per_image).shape == (2,)
    wide = gx.coords.field_of_view(camera, objective, like=torch.zeros((), dtype=torch.float64))
    assert wide.dtype == torch.float64 and wide.tolist() == [3.0, 2.0]


def test_positions_label():
    objective, camera = optics(shape=(20, 30), magnification=50)
    beads = emitters(3, 4, fov=5.0)
    label = gx.labels.render(gx.labels.Positions("beads"), beads, camera, objective)
    assert label[""].shape == (3, 4, 2) and label["in_fov"].shape == (3, 4)
    px = label[""]
    inside = (px[..., 0] >= -0.5) & (px[..., 0] < 29.5) & (px[..., 1] >= -0.5) & (px[..., 1] < 19.5)
    assert torch.equal(label["in_fov"], inside)
    um = gx.labels.Positions("beads", unit="um", dims="xyz").render(beads, camera, objective)
    assert torch.equal(um[""], beads.position)


def test_positions_as_a_pipeline_output():
    objective, camera = optics(shape=(32, 32))
    chain = gx.Chain(
        emitters={"beads": emitters(2, 3, fov=4.0)},
        imaging=gx.imaging.Sprites(objective, camera),
        environment=gx.env.Homogeneous(1.33),
    )
    out = gx.Pipeline(chain, outputs={"mu": "expected", "pos": gx.labels.Positions("beads")})(chain)
    assert out["pos"].shape == (2, 3, 2) and out["pos.in_fov"].dtype == torch.bool
    # outputs are attributes too, by the names they were requested under
    assert out.pos is out["pos"] and out.mu is out["mu"]
    assert "pos" in dir(out) and "pos.in_fov" not in dir(out)
    with pytest.raises(AttributeError, match=r"no output 'heat'.*\['mu', 'pos', 'pos.in_fov'\]"):
        _ = out.heat
    eager = chain(outputs=("expected",))
    assert eager.expected is eager["expected"]


def test_heatmap_and_emitter_table_labels():
    objective, camera = optics(shape=(24, 32))
    p = 6.5 / objective.magnification
    beads = gx.Emitters(
        position=torch.tensor([[[5 * p + 0.5 * p, 7 * p + 0.5 * p, 0.1], [-1.0, -1.0, 0.0]]]),
        photons=torch.tensor([[800.0, 900.0]]),
        presence=torch.tensor([[1.0, 0.0]]),
        emission=gx.Spectrum.line(0.6),
    )
    heat = gx.labels.render(gx.labels.Heatmap("b", sigma_px=1.0), beads, camera, objective)[""]
    assert heat.shape == (1, 24, 32)
    assert float(heat[0, 7, 5]) == pytest.approx(1.0, abs=1e-5)  # unit peak at (row 7, col 5)
    assert float(heat.sum()) == pytest.approx(2 * math.pi, rel=1e-3)  # only the present bead
    table = gx.labels.render(gx.labels.EmitterTable("b"), beads, camera, objective)
    assert table[""].shape == (1, 2, 5)
    assert table[""][0, 0].tolist() == pytest.approx([5.0, 7.0, 0.1, 800.0, 1.0], abs=1e-4)
    assert table["in_fov"].tolist() == [[True, False]]


def test_masks_and_distance_maps_of_spheres():
    objective, camera = optics(shape=(24, 32))
    p = 6.5 / objective.magnification
    cells = gx.Spheres(
        position=torch.tensor(
            [[[8.5 * p, 7.5 * p, 0.0], [12.5 * p, 7.5 * p, 0.2], [0.0, 0.0, 0.0]]]
        ),
        radius=torch.tensor([[3.0 * p, 3.0 * p, 1.0]]),
        presence=torch.tensor([[1.0, 1.0, 0.0]]),
    )
    inst = gx.labels.render(gx.labels.InstanceMask("c"), cells, camera, objective)[""]
    assert inst.shape == (1, 24, 32) and inst.dtype == torch.int64
    assert int(inst[0, 7, 8]) == 1 and int(inst[0, 7, 12]) == 2  # centres
    assert int(inst[0, 7, 10]) == 2  # overlap: the higher (z = 0.2) sphere wins
    assert int(inst[0, 0, 0]) == 0  # the absent sphere draws nothing
    sem = gx.labels.render(gx.labels.SemanticMask("c"), cells, camera, objective)[""]
    assert torch.equal(sem > 0, inst > 0)
    dist = gx.labels.render(gx.labels.DistanceMap("c"), cells, camera, objective)[""]
    assert float(dist[0, 7, 8]) == pytest.approx(-3.0) and float(dist[0, 7, 2]) == pytest.approx(
        3.0
    )
    with pytest.raises(gx.StructureError, match="no radius"):
        gx.labels.render(gx.labels.InstanceMask("b"), emitters(1, 2), camera, objective)


def test_masks_of_an_empty_population_are_background():
    camera = gx.Camera(pixel_size=6.5, shape=(16, 16))
    objective = gx.Objective(NA=0.7, magnification=50)
    empty = gx.Spheres(position=torch.zeros(2, 0, 3), radius=torch.zeros(2, 0) + 0.5)
    instance = gx.labels.render(gx.labels.InstanceMask("s"), empty, camera, objective)[""]
    assert instance.shape == (2, 16, 16) and not bool(instance.any())
    semantic = gx.labels.render(gx.labels.SemanticMask("s"), empty, camera, objective)[""]
    assert not bool(semantic.any())
    distance = gx.labels.render(gx.labels.DistanceMap("s"), empty, camera, objective)[""]
    assert bool(torch.isinf(distance).all())
