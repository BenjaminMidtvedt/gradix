"""QWLSI (M3a phase 4): the grating's orders, the camera's fringes and gx.recon.QWLSI."""

import math

import pytest
import torch

import gradix as gx
from gradix.ops import pupil as ops
from gradix.optics.grating import order_indices

WL, MAG, PIXEL = 0.532, 100.0, 6.5
PITCH = PIXEL / MAG
WATER = gx.env.Homogeneous(1.33)


def qwlsi_chain(
    radius=1.0,
    material=1.34,
    shape=64,
    na=0.8,
    irradiance=1.0,
    oversample="auto",
    grating=None,
    distance=500.0,
    period=None,
    kind="hartmann",
):
    camera = gx.Camera(pixel_size=PIXEL, shape=(shape, shape), noise=gx.noise.Ideal(), unit="e")
    c = shape * PITCH / 2 + 0.013  # off the pixel grid
    beads = gx.Spheres(
        position=torch.tensor([[[c, c - 0.021, 0.0]]], dtype=torch.float64),
        radius=radius,
        material=material,
    )
    scope = gx.presets.QWLSI(
        light=gx.light.PlaneWave(WL, irradiance=irradiance),
        objective=gx.Objective(NA=na, magnification=MAG),
        camera=camera,
        distance=distance,
        period=period,
        kind=kind,
    )
    stages = dict(scope.detection_optics) if grating is None else {"grating": grating}
    chain = gx.Chain(
        light=scope.light,
        scatterers={"beads": gx.interact.Mie(beads)},
        imaging=gx.imaging.Coherent(scope.objective, camera, oversample=oversample),
        detection_optics=stages,
        environment=WATER,
    )
    return chain, scope


def expected(chain):
    return chain(outputs=("expected",)).expected


def sampled_mask(field, s, grating):
    """The camera field by brute force: sample the mask, propagate by angular spectrum."""
    h, w = field.shape[-2:]
    dx = PIXEL / s  # image-space sample spacing
    ys = (torch.arange(h, dtype=torch.float64) * dx + PIXEL / 2)[:, None]
    xs = (torch.arange(w, dtype=torch.float64) * dx + PIXEL / 2)[None, :]
    indices, values = order_indices(grating.kind, grating.max_order)
    assert values is not None
    mask = sum(
        c * torch.exp(1j * math.pi * (m * xs + n * ys) / float(grating.period))
        for (m, n), c in zip(indices, values, strict=True)
    )
    fy = torch.fft.fftfreq(h, d=dx, dtype=torch.float64)[:, None]
    fx = torch.fft.fftfreq(w, d=dx, dtype=torch.float64)[None, :]
    kz = torch.sqrt(1 / WL**2 - fx**2 - fy**2)
    propagate = torch.exp(2j * math.pi * float(grating.distance) * (kz - 1 / WL))
    at_grating = field
    if grating.focused == "camera":  # the image plane is the camera: back to the grating
        at_grating = torch.fft.ifft2(torch.fft.fft2(field) / propagate)
    return torch.fft.ifft2(torch.fft.fft2(at_grating * mask) * propagate)


@pytest.mark.parametrize("focused", ["camera", "grating"])
@pytest.mark.parametrize("kind", ["hartmann", "checkerboard"])
def test_the_orders_are_the_sampled_mask_propagated_by_angular_spectrum(kind, focused):
    grating = gx.optics.Grating(period=4 * PIXEL, distance=500.0, kind=kind, focused=focused)
    chain, _scope = qwlsi_chain(radius=0.25, material=1.45, oversample=4, grating=grating)
    s, statics = 4, gx.compose.execute.eager_statics(chain)
    waves = chain.light()
    spectra = gx.compose.execute._contributions(chain, waves, statics)
    field = chain.imaging.image_field(spectra, waves, WATER, static=statics["imaging"])
    camera = sampled_mask(field[0, 0, 0, 0], s, grating)
    frames = ops.pixel_mtf((camera.abs() ** 2)[None, None], s)[..., ::s, ::s] * PITCH**2
    inner = slice(16, 48)  # the brute force wraps at the edges
    rendered = expected(chain)[..., inner, inner]
    assert torch.allclose(rendered, frames[..., inner, inner], rtol=1e-4, atol=0.0)


def test_the_hartmann_fringes_do_not_change_with_distance_and_keep_the_light():
    frames = []
    for distance in (0.0, 300.0, 1000.0):
        chain, _scope = qwlsi_chain(distance=distance)
        frames.append(expected(gx.recon.without_scatterers(chain))[0, 0])
    assert torch.allclose(frames[0], frames[1], rtol=1e-9) and torch.allclose(frames[0], frames[2])
    assert float(frames[0].mean()) == pytest.approx(PITCH**2, rel=1e-9)  # Σ|c_o|² = 1
    # the fringes: the constant and the harmonics (±1, 0), (0, ±1), (±1, ±1)/Λ, nothing else
    power = torch.fft.fft2(frames[0]).abs()
    peaks = {(int(i), int(j)) for i, j in (power > 1e-9 * power.max()).nonzero().tolist()}
    period = 64 // 4
    expected_peaks = {(i % 64, j % 64) for i in (-period, 0, period) for j in (-period, 0, period)}
    assert peaks == expected_peaks


def test_an_order_shears_the_image_by_lambda_d_over_period():
    period, shift = 4 * PIXEL, 2  # a shear of exactly two camera pixels
    g = 1.0 / period  # the order (2, 0): G = (2, 0)/(2Λ)
    distance = shift * PIXEL * math.sqrt(1 / WL**2 - g * g) / g  # exact shear d·G/√(1/λ² − G²)
    single = gx.optics.Grating(
        period=period,
        distance=distance,
        kind="custom",
        orders=((2.0, 0.0),),
        coefficients=torch.ones(1),
    )
    sheared, _scope = qwlsi_chain(radius=0.25, material=1.45, grating=single)
    plain = expected(sheared.replace(detection_optics={}))
    inner = slice(12, 52)
    moved = expected(sheared)[..., inner, slice(12 + shift, 52 + shift)]
    # a shear up to the exact propagation's non-paraxial curvature (as built 4e-5)
    assert torch.allclose(moved, plain[..., inner, inner], rtol=1e-4)
    # ≈ λd/Λ: the exact shear exceeds the paraxial one by λ²G²/2 = 2.1e-4
    assert shift * PIXEL == pytest.approx(WL * distance / period, rel=3e-4)


def band_limited(field, recon, shape):
    fy = torch.fft.fftfreq(shape, d=PITCH, dtype=torch.float64)[:, None]
    fx = torch.fft.fftfreq(shape, d=PITCH, dtype=torch.float64)[None, :]
    cell = 1.0 / (PITCH * shape)
    band = min(recon.na / WL + 2 * cell, 0.5 * MAG / recon.period)
    edge = torch.clamp((band - torch.sqrt(fx * fx + fy * fy)) / cell + 0.5, 0.0, 1.0)
    return torch.fft.ifft2(torch.fft.fft2(field) * edge * edge * (3.0 - 2.0 * edge))


def phase_error(chain, shape):
    recon = gx.recon.QWLSI.from_chain(chain)
    field = recon(expected(chain))[0, 0]
    truth = chain(outputs={"E": gx.out.Field(normalize="background")}).E[0, 0]
    truth = band_limited(truth.to(torch.complex128), recon, shape)
    inner = slice(shape // 8, shape - shape // 8)
    want, got = torch.angle(truth)[inner, inner], torch.angle(field)[inner, inner]
    want, got = want - want.mean(), got - got.mean()
    return float((got - want).norm() / want.norm())


@pytest.mark.parametrize(
    ("radius", "material"),
    [(1.0, 1.34), (0.05, 1.59)],  # a weak phase object (0.23 rad), a 100 nm polystyrene bead
)
def test_qwlsi_recovers_the_phase(radius, material):
    chain, _scope = qwlsi_chain(radius=radius, material=material, shape=128)
    assert phase_error(chain, 128) < 0.02  # as built 8e-4 and 7e-4


def test_the_calibration_renders_at_the_frames_precision():
    chain, _scope = qwlsi_chain()
    recon = gx.recon.QWLSI.from_chain(chain)
    assert recon.background is not None and recon.background.dtype == torch.float64


def test_the_checkerboard_warns_where_its_harmonics_alias_onto_the_fringes():
    aliased, _scope = qwlsi_chain(kind="checkerboard", shape=128)  # Λ = 4 pixels
    with pytest.warns(gx.GradixWarning, match="aliases the grating's harmonics"):
        gx.recon.QWLSI.from_chain(aliased)
    clear, _scope = qwlsi_chain(kind="checkerboard", period=5 * PIXEL, shape=128)
    assert not gx.recon.QWLSI.from_chain(clear).aliased()
    assert phase_error(clear, 128) < 0.02  # as built 1.3e-2


def test_the_grating_is_differentiable():
    chain, _scope = qwlsi_chain(radius=0.25, material=1.45, shape=32, irradiance=1e4)
    pipe = gx.Pipeline(chain, outputs=("expected",))
    for path in ("grating.period", "grating.distance", "grating.rotation", "beads.radius"):
        assert pipe.gradient_table.quality(path) == "exact", path
    distance = torch.tensor(500.0, dtype=torch.float64, requires_grad=True)
    grating = chain.detection_optics["grating"].replace(distance=distance)
    frames = pipe(chain.replace(detection_optics={"grating": grating})).expected
    weights = torch.linspace(0, 1, frames.numel(), dtype=torch.float64).reshape(frames.shape)
    (grad,) = torch.autograd.grad((frames * weights).sum(), distance)

    def loss(d):
        moved = chain.detection_optics["grating"].replace(distance=d)
        frames = pipe(chain.replace(detection_optics={"grating": moved})).expected
        return float((frames * weights).sum())

    numeric = (loss(500.01) - loss(499.99)) / 0.02
    assert float(grad) == pytest.approx(numeric, rel=1e-5)
    bound = gx.crlb(pipe, chain, wrt=("beads.radius",))
    assert torch.isfinite(bound["beads.radius"]).all()


def test_the_reconstruction_is_differentiable_in_the_scene():
    chain, _scope = qwlsi_chain(radius=0.25, material=1.45, shape=32)
    recon = gx.recon.QWLSI.from_chain(chain)
    radius = torch.tensor(0.25, dtype=torch.float64, requires_grad=True)
    beads = chain.population("beads").replace(radius=radius)
    moved = chain.replace(scatterers={"beads": gx.interact.Mie(beads)})
    phase = torch.angle(recon(expected(moved)))
    (grad,) = torch.autograd.grad(phase[..., 16, 16].sum(), radius)
    assert torch.isfinite(grad) and float(grad) != 0.0


def test_the_planner_builds_the_qwlsi_chain_and_refuses_a_reference_with_it():
    chain, scope = qwlsi_chain(shape=32)
    sample = gx.Sample({"beads": chain.population("beads")}, environment=WATER)
    planned = gx.plan(sample, scope, "standard", outputs=("expected",))
    direct = gx.Pipeline(chain, outputs=("expected",))(chain).expected
    assert torch.equal(planned(sample, scope).expected, direct)
    both = scope.replace(references={"reference": gx.light.ReferenceBeam(1.0)})
    with pytest.raises(gx.PlanError, match="grating and an image-side reference"):
        gx.plan(sample, both, "standard", outputs=("expected",))(sample, both)


def test_undersampled_fringes_warn():
    chain, _scope = qwlsi_chain(period=1.5 * PIXEL, shape=32)
    with pytest.warns(gx.GradixWarning, match="undersamples 'grating'"):
        gx.Pipeline(chain, outputs=("expected",))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_a_grating_of_python_numbers_follows_the_scene_to_the_gpu():
    chain, _scope = qwlsi_chain(radius=0.25, material=1.45, shape=32)
    on_gpu = gx.tree.to(chain, "cuda")  # the light and the grating hold numbers: CPU orders
    assert torch.allclose(expected(on_gpu).cpu(), expected(chain), rtol=1e-9)
