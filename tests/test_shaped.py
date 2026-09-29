"""Shaped illumination (M3a phase 5): gx.light.Shaped, gx.devices.PhaseSLM, Mie over J waves."""

import math

import pytest
import torch

import gradix as gx
from gradix.imaging import coherent

WL, MAG, PIXEL = 0.532, 60.0, 6.5
PITCH = PIXEL / MAG
F64 = torch.float64
WATER = gx.env.Homogeneous(1.33)


def scene(light, shape=24, radius=0.15, na=0.5):
    camera = gx.Camera(pixel_size=PIXEL, shape=(shape, shape), noise=gx.noise.Ideal(), unit="e")
    c = shape * PITCH / 2
    beads = gx.Spheres(
        position=torch.tensor([[[c + 0.01, c - 0.02, 0.2]]], dtype=F64),
        radius=torch.tensor([[[radius]]], dtype=F64),
        material=torch.tensor([[[1.59]]], dtype=F64),
    )
    return gx.Chain(
        light=light,
        scatterers={"beads": gx.interact.Mie(beads)},
        imaging=gx.imaging.Coherent(gx.Objective(NA=na, magnification=MAG), camera),
        environment=WATER,
    )


def frames(chain):
    return chain(outputs=("expected",)).expected


def test_a_single_cell_pupil_is_a_plane_wave():
    shaped = gx.light.Shaped(WL, na=0.3, samples=1, irradiance=1e3)
    plane = gx.light.PlaneWave(WL, irradiance=1e3)
    assert torch.allclose(frames(scene(shaped)), frames(scene(plane)), rtol=1e-12)


def test_a_phase_only_pupil_keeps_the_irradiance():
    phase = torch.rand(12, 12, generator=torch.Generator().manual_seed(0), dtype=F64) * 6.0
    slm = gx.devices.PhaseSLM(phase=phase)
    waves = gx.light.Shaped(WL, na=0.4, device=slm, irradiance=1e3)(WATER)
    cos = torch.sqrt(1.0 - (waves.u**2).sum(-1) / 1.33**2)  # [1, 1, 1, J]
    flux = (waves.amplitude[..., 0, 0].abs() ** 2 * cos).sum()
    assert float(flux) == pytest.approx(1e3, rel=1e-9)
    dimmed = gx.devices.PhaseSLM(phase=phase, zero_order=0.5)  # |t| ≤ 1: a passive mask
    waves = gx.light.Shaped(WL, na=0.4, device=dimmed, irradiance=1e3)(WATER)
    assert float((waves.amplitude[..., 0, 0].abs() ** 2 * cos).sum()) < 1e3


def test_a_pupil_phase_ramp_shifts_the_focal_field():
    q, na, shift = 16, 0.4, 0.7
    cells = ((torch.arange(q, dtype=F64) + 0.5) * 2 / q - 1) * na  # u_x of each column
    ramp = (2 * math.pi * cells * shift / WL).expand(q, q)  # φ(u) = 2π·u_x·Δ/λ
    flat = gx.light.Shaped(
        WL, na=na, device=gx.devices.PhaseSLM(phase=torch.zeros(q, q, dtype=F64))
    )
    tilted = flat.replace(device=gx.devices.PhaseSLM(phase=ramp.clone()))
    points = torch.tensor([[[0.3, -0.2, 0.0], [1.1, 0.4, 0.0]]], dtype=F64)
    moved = points + torch.tensor([shift, 0.0, 0.0], dtype=F64)
    at_moved = flat(WATER).at(moved[:, None], 1.33)
    assert torch.allclose(tilted(WATER).at(points[:, None], 1.33), at_moved, atol=1e-12)


def test_the_field_is_linear_in_the_pupil():
    q = 4
    one = torch.zeros(q, q, dtype=torch.complex128)
    two = one.clone()
    one[1, 2] = 1.0
    two[2, 0] = torch.tensor(0.5j)

    def field(pupil):
        chain = scene(gx.light.Shaped(WL, na=0.4, pupil_field=pupil, irradiance=1e3))
        return chain(outputs={"E": gx.out.Field(normalize="none")}).E

    assert torch.allclose(field(one + two), field(one) + field(two), atol=1e-12)
    assert torch.allclose(field(1j * two), 1j * field(two), atol=1e-12)  # complex pupils


def test_mie_sums_the_waves_in_chunks(monkeypatch):
    slm = gx.devices.PhaseSLM(phase=torch.linspace(0, 3, 64, dtype=F64).reshape(8, 8))
    chain = scene(gx.light.Shaped(WL, na=0.4, device=slm, irradiance=1e3))
    whole = frames(chain)
    monkeypatch.setattr(coherent, "_WORKING_SET", 1)  # one wave per chunk
    assert torch.allclose(frames(chain), whole, rtol=1e-12)


def test_the_bound_is_differentiable_in_the_slm_phase():
    phase = torch.zeros(6, 6, dtype=F64, requires_grad=True)
    slm = gx.devices.PhaseSLM(phase=phase)
    chain = scene(gx.light.Shaped(WL, na=0.4, device=slm, irradiance=1e4))
    pipe = gx.Pipeline(chain, outputs=("expected",))
    recon = gx.recon.Inline.from_chain(chain, normalize=True, pad=0)

    def information(p):
        moved = chain.replace(light=chain.light.replace(device=gx.devices.PhaseSLM(phase=p)))
        bound = gx.crlb(pipe, moved, wrt=("beads.radius",), reconstruction=recon, tolerance=1e-12)
        return bound.fisher[0, 0, 0]

    (grad,) = torch.autograd.grad(information(phase), phase)
    # a flat pupil focuses on the axis, far from the bead: the frame is dark but for the
    # spot, the variances span ~1e15 and the information converges to ~1e-4 (hence the step)
    step = torch.zeros(6, 6, dtype=F64)
    step[2, 3] = 1e-3
    with torch.no_grad():
        numeric = (information(step) - information(-step)) / 2e-3
    assert float(grad[2, 3]) == pytest.approx(float(numeric), rel=1e-3)  # as built 3e-5


def test_a_pupil_beyond_the_medium_index_is_refused():
    chain = scene(gx.light.Shaped(WL, na=1.4, samples=4))
    with pytest.raises(gx.ValidityError, match="beyond the medium's index"):
        gx.Pipeline(chain, outputs=("expected",))


def test_the_planner_takes_shaped_light():
    light = gx.light.Shaped(WL, na=0.3, samples=4, irradiance=1e3)
    chain = scene(light)
    scope = gx.presets.InlineHolography(light=light, objective=chain.objective, camera=chain.camera)
    sample = gx.Sample({"beads": chain.population("beads")}, environment=WATER)
    planned = gx.plan(sample, scope, "standard", outputs=("expected",))
    assert torch.allclose(planned(sample, scope).expected, frames(chain), rtol=1e-12)
