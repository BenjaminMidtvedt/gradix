"""Excitation (§4.5, §5.1, M1): analytic sources and linear transduction of point emitters."""

import math

import pytest
import torch
from scenes import optics

import gradix as gx


def _beads(z, x=None):
    n = len(z)
    xs = torch.full((n,), 2.08) if x is None else torch.as_tensor(x, dtype=torch.float32)
    pos = torch.stack([xs, torch.full((n,), 2.08), torch.as_tensor(z, dtype=torch.float32)], -1)
    return gx.Emitters(position=pos[None], photons=1000.0, emission=gx.Spectrum.line(0.6))


def _excite(element, lowered, waves, medium):
    out = element(lowered, waves, medium)
    assert isinstance(out, gx.EmitterSet)
    return out


def _excited_photons(light, beads, medium):
    return _excite(gx.excite.Linear(), gx.lower.emitter_set(beads), light(), medium).photons


def test_uniform_excitation_scales_photons_by_the_irradiance():
    photons = _excited_photons(gx.light.Uniform(irradiance=3.0), _beads([0.0, -1.0]), _WATER)
    assert torch.allclose(photons, torch.full_like(photons, 3000.0), rtol=1e-5)
    lowered = gx.lower.emitter_set(_beads([0.0]))
    light = gx.light.Uniform(irradiance=100.0)()
    saturated = _excite(gx.excite.Linear(saturation=100.0), lowered, light, _WATER).photons
    assert float(saturated) == pytest.approx(1000.0 * 100.0 / (1.0 + 1.0), rel=1e-5)


_WATER = gx.env.Homogeneous(1.33)


def test_tirf_excitation_decays_with_the_penetration_depth():
    angle = math.radians(70.0)
    depth = gx.light.penetration_depth(angle, 1.518, 1.33, 0.488)
    light = gx.light.Evanescent(angle=angle, wavelength=0.488)
    photons = _excited_photons(light, _beads([0.0, -depth, -2 * depth]), _WATER)[0, 0]
    ratios = (photons / photons[0]).tolist()
    assert ratios[1] == pytest.approx(math.exp(-1.0), rel=1e-4)
    assert ratios[2] == pytest.approx(math.exp(-2.0), rel=1e-4)


def test_sim_beams_make_fringes_of_the_requested_period_and_modulation():
    period, m = 0.4, 0.8
    xs = torch.linspace(0.0, 0.8, 9)
    light = gx.light.SIMBeams(period=period, modulation=m, phase=0.3)
    photons = _excited_photons(light, _beads([0.0] * 9, xs), _WATER)[0, 0] / 1000.0
    expected = 1.0 + m * torch.cos(2 * math.pi * xs / period + 0.3)
    assert torch.allclose(photons, expected, atol=1e-4)


def test_tirf_preset_plans_excitation_and_renders():
    objective, camera = optics(shape=(32, 32))
    sample = gx.Sample({"beads": _beads([0.0, -0.3])}, environment=_WATER)
    scope = gx.presets.TIRF(objective=objective, camera=camera, angle=math.radians(70.0))
    plan = gx.plan(sample, scope, "draft", outputs=("expected",))
    chain = plan.chain(sample, scope)
    assert isinstance(chain.excite, gx.excite.Linear)
    deep = gx.Sample({"beads": _beads([-3.0, -3.3])}, environment=_WATER)
    near_total = float(plan(sample, scope)["expected"].sum())
    deep_total = float(plan(deep, scope)["expected"].sum())
    assert deep_total < 1e-3 * near_total  # the evanescent field does not reach 3 µm


def test_excitation_gradients_reach_the_light():
    angle = torch.tensor(math.radians(70.0), requires_grad=True)
    light = gx.light.Evanescent(angle=angle)
    photons = _excited_photons(light, _beads([-0.1]), _WATER)
    (grad,) = torch.autograd.grad(photons.sum(), angle)
    assert torch.isfinite(grad) and grad < 0  # steeper incidence → shallower field → fewer photons


def test_gradient_table_routes_light_through_the_excitation():
    objective, camera = optics(shape=(32, 32))
    light = gx.light.Evanescent(angle=torch.tensor(math.radians(70.0)))
    chain = gx.Chain(
        light=light,
        emitters={"beads": _beads([0.0, -0.3])},
        excite=gx.excite.Linear(),
        imaging=gx.imaging.Sprites(objective, camera),
        environment=_WATER,
    )
    table = gx.Pipeline(chain, outputs=("expected",)).gradient_table
    assert table.quality("light.angle") == "exact"
    assert table.quality("light.irradiance") == "exact"


def test_emitter_tables_report_the_photons_as_rendered():
    objective, camera = optics(shape=(32, 32))
    beads = _beads([0.0, -0.3])
    chain = gx.Chain(
        light=gx.light.Evanescent(angle=math.radians(70.0)),
        emitters={"beads": beads},
        excite=gx.excite.Linear(),
        imaging=gx.imaging.Sprites(objective, camera),
        environment=_WATER,
    )
    out = gx.Pipeline(chain, outputs={"table": gx.labels.EmitterTable("beads")})(chain)
    photons = out["table"][0, :, 3]
    excited = _excited_photons(chain.light, beads, _WATER)[0, 0]
    assert torch.allclose(photons, excited) and photons[1] < photons[0]


def test_emitter_tables_train_the_light_through_the_excitation():
    # the table's photons column is the excited photons: light, excitation and medium fields
    # reach it, so an EmitterTable-only output must not refuse a trainable light
    objective, camera = optics(shape=(32, 32))
    angle = torch.tensor(math.radians(70.0), requires_grad=True)
    chain = gx.Chain(
        light=gx.light.Evanescent(angle=angle),
        emitters={"beads": _beads([0.0, -0.3])},
        excite=gx.excite.Linear(),
        imaging=gx.imaging.Sprites(objective, camera),
        environment=_WATER,
    )
    pipe = gx.Pipeline(chain, outputs={"table": gx.labels.EmitterTable("beads")})
    for path in ("light.angle", "light.irradiance", "environment.n"):
        assert pipe.gradient_table.quality(path) == "exact", path
    table = pipe(chain)["table"]
    (grad,) = torch.autograd.grad(table[..., 3].sum(), angle)
    assert torch.isfinite(grad) and grad < 0
    masks = gx.Pipeline(chain, outputs={"heat": gx.labels.Heatmap("beads")}).gradient_table
    assert masks.quality("light.angle") == "zero"  # heatmaps read positions only


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_cuda_emitters_are_excited_by_lights_with_scalar_parameters():
    objective, camera = optics(shape=(16, 16))
    beads = gx.Emitters(
        position=torch.tensor([[[0.6, 0.6, 0.0]]], device="cuda"),
        photons=torch.tensor([[1000.0]], device="cuda"),
        emission=gx.Spectrum.line(0.6),
    )
    chain = gx.Chain(
        light=gx.light.Uniform(irradiance=2.0, wavelength=0.488),
        emitters={"b": beads},
        excite=gx.excite.Linear(),
        imaging=gx.imaging.Sprites(objective, camera),
        environment=gx.env.Homogeneous(1.33),
    )
    out = chain(outputs=("expected",))["expected"]
    assert out.device.type == "cuda" and float(out.sum()) > 0


def test_dispersive_samples_are_read_at_the_excitation_wavelength():
    water = gx.materials.Water()
    n_488 = float(water.index(0.488))
    angle = math.radians(70.0)
    light = gx.light.Evanescent(angle=angle, wavelength=0.488)
    layered = gx.env.LayeredMedium(sample=water)
    beads = _beads([0.0, -0.2])
    dispersive = _excited_photons(light, beads, layered)[0, 0]
    constant = _excited_photons(light, beads, gx.env.Homogeneous(n_488))[0, 0]
    assert torch.allclose(dispersive / dispersive[0], constant / constant[0], rtol=1e-5)
    assert tuple(layered.index(wavelength=torch.tensor([0.45, 0.5, 0.6])).shape) == (1, 3)


def test_a_light_sheet_is_gaussian_in_z_and_spreads_past_its_waist():
    n, wl, w0 = 1.33, 0.5, 1.0
    sheet = gx.light.Sheet(irradiance=2.0, waist=w0, center=-1.0, focus=5.0, wavelength=wl)()
    rayleigh = math.pi * w0**2 * n / wl
    points = torch.tensor(
        [[[[5.0, 3.0, -1.0], [5.0, 3.0, 0.0], [5.0 + rayleigh, 3.0, -1.0 + math.sqrt(2)]]]],
        dtype=torch.float64,
    )
    irradiance = (sheet.at(points, n).abs() ** 2).reshape(-1)
    expected = torch.tensor([2.0, 2.0 * math.exp(-2), math.sqrt(2) * math.exp(-2)])
    assert torch.allclose(irradiance.float(), expected, rtol=1e-6)


def test_light_sheet_excitation_selects_the_plane_and_moves_with_it():
    objective, camera = optics(shape=(24, 24))
    beads = gx.Emitters(
        position=torch.tensor([[[0.9, 0.6, 0.0], [0.9, 1.2, -0.7]]]),
        photons=1000.0,
        emission=gx.Spectrum.line(0.6),
    )
    center = torch.tensor(0.1, requires_grad=True)
    chain = gx.Chain(
        light=gx.light.Sheet(waist=0.3, center=center, focus=0.9, wavelength=0.488),
        emitters={"b": beads},
        excite=gx.excite.Linear(),
        imaging=gx.imaging.Sprites(objective, camera),
        environment=gx.env.Homogeneous(1.33),
    )
    table = gx.labels.EmitterTable("b")
    out = chain(outputs={"mu": "expected", "t": table})
    photons = out["t"][0, :, 3].detach()
    assert float(photons[1] / photons[0]) < 1e-5  # 0.8 µm below a 0.3 µm sheet: dark
    out["mu"].sum().backward()
    assert center.grad is not None and float(center.grad) != 0.0
