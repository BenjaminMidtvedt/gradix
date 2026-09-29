"""iSCAT in the layered geometry (M3a phase 3): coverslip reference, backscatter, interface."""

import math

import pytest
import torch

import gradix as gx

WL = 0.532
PITCH = 0.065
GLASS = gx.env.LayeredMedium(sample=1.33, coverslip=1.52, immersion=1.52)


def iscat_chain(
    depth=0.1, radius=0.03, na=1.4, focus=0.0, shape=32, medium=GLASS, irradiance=1.0, **scope
):
    camera = gx.Camera(pixel_size=6.5, shape=(shape, shape), noise=gx.noise.Ideal(), unit="e")
    c = shape * PITCH / 2 + PITCH / 2  # a pixel centre
    beads = gx.Spheres(
        position=torch.tensor([[[c, c, -depth]]], dtype=torch.float64),
        radius=radius,
        material=1.59,
    )
    microscope = gx.presets.ISCAT(
        light=gx.light.PlaneWave(WL, irradiance=irradiance, travel=-1),
        objective=gx.Objective(NA=na, magnification=100, focus=focus),
        camera=camera,
        **scope,
    )
    chain = gx.Chain(
        light=microscope.light,
        scatterers={"beads": gx.interact.Mie(beads)},
        imaging=gx.imaging.Coherent(microscope.objective, camera),
        environment=medium,
    )
    return chain, microscope


def centre_contrast(chain):
    image = chain(outputs=("expected",)).expected
    reference = chain.replace(scatterers={})(outputs=("expected",)).expected
    h = image.shape[-1] // 2
    return float(image[0, 0, h, h] / reference[0, 0, h, h] - 1.0)


def test_the_coverslip_reflection_is_the_reference():
    chain, _scope = iscat_chain()
    reference = chain.replace(scatterers={})(outputs=("expected",)).expected
    r = (1.52 - 1.33) / (1.52 + 1.33)  # glass onto water, normal incidence
    assert torch.allclose(reference, torch.full_like(reference, r * r * PITCH**2), rtol=1e-6)


def test_equal_indices_reduce_to_the_homogeneous_medium():
    # no interface: transmitted and epi light alike see a homogeneous medium
    same = gx.env.LayeredMedium(sample=1.33, coverslip=1.33, immersion=1.33)
    for travel in (1, -1):
        chain, _scope = iscat_chain(depth=0.5, radius=0.1, na=1.0, focus=-0.3, medium=same)
        chain = chain.replace(light=gx.light.PlaneWave(WL, irradiance=1.0, travel=travel))
        layered = chain(outputs=("expected",)).expected
        homogeneous = chain.replace(environment=gx.env.Homogeneous(1.33))
        flat = homogeneous(outputs=("expected",)).expected
        assert torch.allclose(layered, flat, rtol=1e-10, atol=1e-18), travel


def test_the_contrast_oscillates_at_the_round_trip_period():
    # with the focus on the bead, the reference and the backscatter differ by the round trip
    # 2·k·n·depth: period λ/(2n), lengthened by the collected ⟨cos θ⟩ at high NA
    depths = torch.arange(0.05, 1.25, 0.01, dtype=torch.float64)
    values = torch.tensor(
        [centre_contrast(iscat_chain(depth=d, radius=0.02, na=0.9, focus=-d)[0]) for d in depths]
    )
    v = values - values.mean()
    freqs = torch.linspace(3.0, 8.0, 2000, dtype=torch.float64)
    power = torch.stack([(v * torch.exp(-2j * math.pi * f * depths)).sum().abs() for f in freqs])
    period = 1.0 / float(freqs[int(power.argmax())])
    assert period == pytest.approx(WL / (2 * 1.33), rel=0.02)


def test_supercritical_backscatter_converges_with_the_pupil_grid():
    chain, _scope = iscat_chain(depth=0.1, radius=0.05, na=1.4)  # NA 1.4 > n = 1.33
    waves, spectra = chain.light(), chain.scatterers["beads"](chain.light(), GLASS)
    powers = []
    for samples in (64, 128, 256):
        element = chain.imaging.replace(pupil_samples=samples)
        static = element.eager_static(spectra, waves, GLASS)
        scattered = element.image_field(spectra, waves, GLASS, static=static)
        scattered = scattered - element.image_field((), waves, GLASS, static=static)
        powers.append(float((scattered.abs() ** 2).sum()))
    first, second = abs(powers[1] - powers[0]), abs(powers[2] - powers[1])
    assert second < 0.5 * first and second < 5e-3 * powers[2]


def test_a_pupil_filter_attenuates_the_reference_and_raises_the_contrast():
    plain, _scope = iscat_chain(radius=0.02)
    filtered, scope = iscat_chain(radius=0.02, attenuation=0.1, filter_radius=0.05)
    assert isinstance(scope.objective.pupil[0], gx.pupil.Filter)
    reference = filtered.replace(scatterers={})(outputs=("expected",)).expected
    r = (1.52 - 1.33) / (1.52 + 1.33)
    assert float(reference.mean()) == pytest.approx(0.01 * r * r * PITCH**2, rel=1e-5)
    assert abs(centre_contrast(filtered)) > 5.0 * abs(centre_contrast(plain))


def test_iscat_is_differentiable_and_bounds_depth_and_radius():
    chain, _scope = iscat_chain(radius=0.03, shape=16, irradiance=1e7)  # ~2000 e reference
    pipe = gx.Pipeline(chain, outputs=("expected",))
    table = pipe.gradient_table
    for path in ("beads.position", "beads.radius", "beads.material", "environment.sample.n"):
        assert table.quality(path) == "exact", path
    bound = gx.crlb(pipe, chain, wrt=("beads.position.z", "beads.radius"))
    for path in ("beads.position.z", "beads.radius"):
        assert torch.isfinite(bound[path]).all(), path


def test_the_planner_builds_the_iscat_chain_and_refuses_beads_in_the_glass():
    chain, scope = iscat_chain(shape=16)
    sample = gx.Sample({"beads": chain.population("beads")}, environment=GLASS)
    planned = gx.plan(sample, scope, "standard", outputs=("expected",))
    direct = gx.Pipeline(chain, outputs=("expected",))(chain).expected
    assert torch.equal(planned(sample, scope).expected, direct)
    above = chain.population("beads").replace(position=torch.tensor([[[0.5, 0.5, 0.2]]]))
    with pytest.raises(gx.ValidityError, match="reach z > 0"):
        gx.Pipeline(chain.replace(scatterers={"beads": gx.interact.Mie(above)}))


def test_iscat_lite_is_a_constant_reference_in_a_homogeneous_medium():
    chain, scope = iscat_chain(shape=16, na=1.2)  # below water's index
    water = gx.Sample({"beads": chain.population("beads")}, environment=gx.env.Homogeneous(1.33))
    with pytest.warns(gx.GradixWarning, match="nothing reflects a reference"):
        gx.plan(water, scope, "standard", outputs=("expected",))
    r = (1.52 - 1.33) / (1.52 + 1.33)
    lite = scope.replace(references={"reference": gx.light.ReferenceBeam(r * r, phase=0.3)})
    planned = gx.plan(water, lite, "standard", outputs=("expected",))
    notes = [v for v in planned.pipeline.violations if "iSCAT-lite" in v.message]
    assert [v.severity for v in notes] == ["info"]
    empty = planned.chain(water, lite).replace(scatterers={})
    reference = empty(outputs=("expected",)).expected
    assert torch.allclose(reference, torch.full_like(reference, r * r * PITCH**2), rtol=1e-10)


def test_beads_near_the_coverslip_get_an_info_note():
    near, scope = iscat_chain(depth=0.05, shape=16)
    far, _scope = iscat_chain(depth=2.0, shape=16)
    for chain, expected in ((near, ["info"]), (far, [])):
        sample = gx.Sample({"beads": chain.population("beads")}, environment=GLASS)
        planned = gx.plan(sample, scope, "standard", outputs=("expected",))
        notes = [v for v in planned.pipeline.violations if "within λ" in v.message]
        assert [v.severity for v in notes] == expected
