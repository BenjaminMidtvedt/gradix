"""Off-axis digital holography (M3a phase 2): image-side references and gx.recon.OffAxis."""

import dataclasses
import math

import pytest
import torch

import gradix as gx

WL, NA, MAG, PIXEL = 0.532, 0.5, 60.0, 6.5
PITCH = PIXEL / MAG


def offaxis_chain(shape=96, cycles=26, phase=0.0, irradiance=1e4, radius=0.15):
    """Beads under an off-axis reference whose carrier has ``cycles`` periods across the frame."""
    u = WL * cycles / (shape * PITCH)  # on the DFT grid: no leakage of pure tones
    angle = math.asin(u / MAG)
    camera = gx.Camera(pixel_size=PIXEL, shape=(shape, shape))
    objective = gx.Objective(NA=NA, magnification=MAG)
    c = shape * PITCH / 2
    beads = gx.Spheres(
        position=torch.tensor([[[c, c, 0.5]]], dtype=torch.float64), radius=radius, material=1.59
    )
    scope = gx.presets.OffAxisHolography(
        light=gx.light.PlaneWave(WL, irradiance=irradiance),
        objective=objective,
        camera=camera,
        angle=torch.tensor([angle, angle], dtype=torch.float64),
        reference_phase=phase,
    )
    return gx.Chain(
        light=scope.light,
        scatterers={"beads": gx.interact.Mie(beads)},
        references=dict(scope.references),
        imaging=gx.imaging.Coherent(objective, camera),
        environment=gx.env.Homogeneous(1.33),
    ), scope


def band_limited(field, recon, background):
    """The true field limited to the reconstruction's window (what any off-axis method keeps)."""
    h, w = field.shape[-2:]
    fy = torch.fft.fftfreq(h, d=PITCH, dtype=torch.float64)[:, None]
    fx = torch.fft.fftfreq(w, d=PITCH, dtype=torch.float64)[None, :]
    cell = 1.0 / (PITCH * max(h, w))
    band = recon.na / WL + 2.0 * cell
    edge = torch.clamp((band - torch.sqrt(fx * fx + fy * fy)) / cell + 0.5, 0.0, 1.0)
    window = edge * edge * (3.0 - 2.0 * edge)
    return torch.fft.ifft2(torch.fft.fft2(field - background) * window) + background


def test_the_reconstruction_recovers_the_band_limited_field():
    chain, _scope = offaxis_chain()
    out = chain(outputs={"mu": "expected", "E": "field"})
    recon = gx.recon.OffAxis.from_chain(chain)
    field = recon(out.mu)
    truth = band_limited(out.E, recon, 1.0)
    inner = slice(16, 80)
    error = (field - truth)[..., inner, inner].norm() / (truth - 1)[..., inner, inner].norm()
    assert float(error) < 1.5e-3  # the exit criterion is 1e-3; the pixel MTF wraps at the edges


def test_a_point_sampled_hologram_reconstructs_exactly():
    chain, _scope = offaxis_chain()
    field = chain(outputs={"E": gx.out.Field(normalize="none")}).E
    recon = dataclasses.replace(gx.recon.OffAxis.from_chain(chain, normalize=False), mtf=False)
    xs = (torch.arange(96, dtype=torch.float64) + 0.5) * PITCH
    fc = recon.carrier[0] / WL
    reference = 100.0 * torch.exp(1j * 2 * math.pi * fc * (xs[None, :] + xs[:, None]))
    hologram = (field + reference).abs() ** 2 * PITCH**2
    truth = band_limited(field, recon, field[..., :1, :1].mean())
    error = (recon(hologram) - truth).norm() / (truth - truth[..., :1, :1].mean()).norm()
    assert float(error) < 1e-4


def test_the_reference_sets_the_detection_spacing_and_the_warnings():
    chain, scope = offaxis_chain(cycles=28)  # |u_ref| + NA = 2.53: λ/(2·2.53) < the pitch
    static = gx.Pipeline(chain, outputs=("expected",)).static("imaging")
    assert isinstance(static, gx.imaging.CoherentStatic) and static.oversample == 2
    inline = gx.Pipeline(chain.replace(references={}), outputs=("expected",)).static("imaging")
    assert isinstance(inline, gx.imaging.CoherentStatic) and inline.oversample == 1
    small = chain.replace(references={"reference": scope.references["reference"].replace(
        angle=torch.tensor([0.005, 0.005], dtype=torch.float64))})  # fmt: skip
    with pytest.warns(gx.GradixWarning, match="overlaps the autocorrelation"):
        gx.Pipeline(small, outputs=("expected",))
    steep = chain.replace(references={"reference": scope.references["reference"].replace(
        angle=torch.tensor([0.05, 0.05], dtype=torch.float64))})  # fmt: skip
    with pytest.warns(gx.GradixWarning, match="undersamples the off-axis sideband"):
        gx.Pipeline(steep, outputs=("expected",))


def test_phase_stepping_frames_reconstruct_the_same_field():
    steps = torch.tensor([[0.0, 0.5 * math.pi, math.pi, 1.5 * math.pi]], dtype=torch.float64)
    chain, _scope = offaxis_chain(phase=steps)
    frames = chain(outputs=("expected",)).expected  # the phases step on the time axis
    assert frames.shape == (1, 4, 1, 96, 96)
    fields = []
    for t in range(4):
        recon = dataclasses.replace(gx.recon.OffAxis.from_chain(chain.replace(
            references={"reference": chain.references["reference"].replace(
                phase=float(steps[0, t]))})), phase=float(steps[0, t]))  # fmt: skip
        fields.append(recon(frames[:, t]))
    for field in fields[1:]:
        assert torch.allclose(field, fields[0], atol=1e-8)


def test_gradients_reach_the_reference_and_the_reconstruction():
    chain, _scope = offaxis_chain(shape=48, cycles=13)
    table = gx.Pipeline(chain, outputs=("expected",)).gradient_table
    for path in ("reference.irradiance", "reference.angle", "reference.phase", "beads.radius"):
        assert table.quality(path) == "exact", path
    frames = chain(outputs=("expected",)).expected.clone().requires_grad_()
    recon = gx.recon.OffAxis.from_chain(chain)
    (grad,) = torch.autograd.grad(recon(frames).imag.sum(), frames)
    assert torch.isfinite(grad).all() and (grad != 0).any()


def test_the_planner_builds_the_off_axis_microscope():
    chain, scope = offaxis_chain(shape=48, cycles=13)
    sample = gx.Sample({"beads": chain.population("beads")}, environment=chain.environment)
    planned = gx.plan(sample, scope, "standard", outputs=("expected",))
    hand = gx.Pipeline(chain, outputs=("expected",))
    assert torch.equal(planned(sample, scope).expected, hand(chain).expected)
