"""Differentiable field reconstructions (gx.recon, §5.12) against rendered ground truth."""

import pytest
import torch

import gradix as gx


def inline_chain(z=3.0, radius=0.15, shape=(64, 64)):
    camera = gx.Camera(pixel_size=6.5, shape=shape, noise=gx.noise.Ideal())
    objective = gx.Objective(NA=0.8, magnification=60)
    centre = shape[0] // 2 * 6.5 / 60 + 6.5 / 120  # a pixel centre
    beads = gx.Spheres(
        position=torch.tensor([[[centre, centre, z]]], dtype=torch.float64),
        radius=radius,
        material=1.59,
    )
    return gx.Chain(
        light=gx.light.PlaneWave(0.532, irradiance=1e4),
        scatterers={"beads": gx.interact.Mie(beads)},
        imaging=gx.imaging.Coherent(objective, camera),
        environment=gx.env.Homogeneous(1.33),
    )


BACKGROUND = 1e4 * (6.5 / 60) ** 2


def test_an_empty_hologram_reconstructs_to_the_background():
    chain = inline_chain()
    empty = chain.replace(scatterers={})(outputs=("expected",)).expected
    recon = gx.recon.Inline.from_chain(chain, distance=2.0, background=BACKGROUND)
    assert torch.allclose(recon(empty), torch.ones_like(recon(empty)), atol=1e-12)


def test_angular_spectrum_refocuses_the_true_field():
    # propagating the rendered complex field 3 µm reproduces the field rendered in focus there
    chain = inline_chain(z=3.0)
    field = chain(outputs={"E": "field"}).E
    moved = chain.replace(imaging=chain.imaging.replace(
        objective=chain.objective.replace(focus=3.0)))  # fmt: skip
    truth = moved(outputs={"E": "field"}).E
    refocused = 1 + gx.ops.propagation.angular_spectrum(
        field - 1, 6.5 / 60, 0.532, 1.33, 3.0, band=0.9, pad=32
    )
    error = (refocused - truth)[..., 16:48, 16:48].abs().max() / (truth - 1).abs().max()
    assert float(error) < 0.02  # periodic wrap and the pupil's soft edge


def test_the_twin_image_biases_hologram_focus_metrics_slightly():
    # the object term refocuses at the sphere; its defocused twin shifts the magnitude peak
    z = 3.0
    chain = inline_chain(z=z, radius=0.05)
    frames = chain(outputs=("expected",)).expected
    distances = torch.arange(2.0, 4.0, 0.025, dtype=torch.float64)
    peaks = []
    for d in distances.tolist():
        recon = gx.recon.Inline.from_chain(chain, distance=d, background=BACKGROUND)
        peaks.append(float((recon(frames) - 1.0).abs().max()))
    best = float(distances[int(torch.tensor(peaks).argmax())])
    assert abs(best - z) <= 0.25


def test_the_refocused_object_term_is_the_true_field_at_the_spheres_plane():
    z = 3.0
    chain = inline_chain(z=z)
    frames = chain(outputs=("expected",)).expected
    recon = gx.recon.Inline.from_chain(chain, distance=z, background=BACKGROUND)(frames)
    focused = chain.replace(imaging=chain.imaging.replace(
        objective=chain.objective.replace(focus=z)))  # fmt: skip
    truth = focused(outputs={"E": "field"}).E
    centre = 32
    scattered, true = recon[0, 0, centre, centre] - 1, truth[0, 0, centre, centre] - 1
    # the twin image, defocused by 2z, adds a small error at the centre
    assert abs(complex(scattered - true)) < 0.15 * abs(complex(true))


def test_inline_is_affine_and_differentiable():
    chain = inline_chain(z=1.0, shape=(32, 32))
    frames = chain(outputs=("expected",)).expected.clone().requires_grad_()
    recon = gx.recon.Inline.from_chain(chain, distance=1.0, background=BACKGROUND, pad=8)
    g = torch.Generator().manual_seed(0)
    a, b = torch.rand(frames.shape, generator=g, dtype=frames.dtype), 0.3
    lhs = recon(frames + b * a) - recon(frames)
    rhs = recon(BACKGROUND + b * a) - recon(torch.full_like(frames, BACKGROUND))
    assert torch.allclose(lhs, rhs, atol=1e-12)  # R(y + δ) − R(y) depends on δ only
    (grad,) = torch.autograd.grad(recon(frames).abs().sum(), frames)
    assert torch.isfinite(grad).all() and (grad != 0).any()


def test_optics_come_from_the_chain():
    chain = inline_chain()
    optics = gx.recon.optics_of(chain)
    assert optics.pitch == pytest.approx(6.5 / 60) and optics.na == 0.8
    assert optics.wavelength == pytest.approx(0.532) and optics.n == pytest.approx(1.33)
