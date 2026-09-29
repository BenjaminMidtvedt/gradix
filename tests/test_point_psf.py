"""The pupil PSF of point emitters (M1: emit.pupil_mft): radiometry, Airy, frame, z sign, ladder."""

import math

import pytest
import scipy.special
import torch
from scenes import DEVICES

import gradix as gx
from gradix.ops import pupil as ops
from gradix.testing import gradcheck


def _pupil_power(na, n, wl, samples):
    u, du = ops.pupil_axis(samples, na * 1.03125)
    uy, ux = torch.meshgrid(u, u, indexing="ij")
    u2 = ux**2 + uy**2
    a = ops.soft_aperture(torch.sqrt(u2), torch.tensor(na, dtype=torch.float64), du)
    cos = ops.obliquity(u2, torch.tensor(n, dtype=torch.float64))
    return float((a**2 / cos).sum() * (du / wl) ** 2 / (4 * math.pi * (n / wl) ** 2))


@pytest.mark.parametrize(("na", "n"), [(0.3, 1.0), (0.8, 1.33), (1.2, 1.33), (1.4, 1.518)])
def test_power_is_the_collection_efficiency(na, n):
    eta = float(gx.conventions.collection_efficiency(na, n))
    assert _pupil_power(na, n, 0.6, 64) / eta - 1 < 1.5e-3
    assert abs(_pupil_power(na, n, 0.6, 256) / eta - 1) < 2e-4  # converges with the pupil grid


def _single(na, n, shape, magnification, z=0.0, focus=0.0, pupil=(), dtype=torch.float32, **kw):
    camera = gx.Camera(pixel_size=6.5, shape=shape)
    objective = gx.Objective(NA=na, magnification=magnification, focus=focus, pupil=pupil)
    p = 6.5 / magnification
    c = (shape[1] * p / 2, shape[0] * p / 2)
    beads = gx.Emitters(
        position=torch.tensor([[[c[0], c[1], z]]], dtype=dtype),
        photons=1000.0,
        emission=gx.Spectrum.line(0.6),
    )
    psf = gx.imaging.PointPSF(objective, camera, **kw)
    return psf(gx.lower.emitter_set(beads), gx.env.Homogeneous(n)).data[0, 0, 0], p


def test_low_na_focus_is_an_airy_pattern():
    # 0.65 µm pixels; a fine pupil grid (the default 64 cells leave a 0.4 % edge-sampling error)
    img, p = _single(0.1, 1.0, (65, 65), 10.0, roi=64, oversample=1, pupil_samples=256)
    row = img[32].double()
    x = (torch.arange(65, dtype=torch.float64) - 32) * p
    v = 2 * math.pi * 0.1 * x.abs() / 0.6
    airy = torch.where(
        v > 0, (2 * torch.from_numpy(scipy.special.j1(v.numpy())) / v.clamp(min=1e-12)) ** 2, 1.0
    )
    # pixel integration blurs the pattern: compare pixel-averaged Airy numerically
    fine = torch.linspace(-0.5, 0.5, 21, dtype=torch.float64)[:-1] + 0.025
    xs = x[:, None] + fine[None, :] * p
    vs = 2 * math.pi * 0.1 * xs.abs() / 0.6
    ys = fine * p
    rr = torch.sqrt(xs[:, :, None] ** 2 + ys[None, None, :] ** 2)
    vv = 2 * math.pi * 0.1 * rr / 0.6
    boxed = torch.where(
        vv > 0, (2 * torch.from_numpy(scipy.special.j1(vv.numpy())) / vv.clamp(min=1e-12)) ** 2, 1.0
    ).mean((1, 2))
    assert torch.allclose(row / row.max(), boxed / boxed.max(), atol=2e-3)
    del airy, vs


@pytest.mark.parametrize("device", DEVICES)
def test_energy_matches_the_gaussian_tier(device):
    objective = gx.Objective(NA=0.6, magnification=50)
    camera = gx.Camera(pixel_size=6.5, shape=(64, 64))
    beads = gx.Emitters(
        position=torch.tensor([[[4.1, 4.3, 0.1]]], device=device),
        photons=1000.0,
        emission=gx.Spectrum.line(0.6),
    )
    env = gx.env.Homogeneous(1.33)
    es = gx.lower.emitter_set(beads)
    pupil = gx.imaging.PointPSF(objective, camera)(es, env).data
    gauss = gx.imaging.Sprites(objective, camera)(es, env).data
    assert float(pupil.sum() / gauss.sum()) == pytest.approx(1.0, abs=2e-3)
    assert float(torch.linalg.vector_norm(gauss - pupil) / torch.linalg.vector_norm(pupil)) < 0.20


def test_centroid_follows_to_pixels():
    g = torch.Generator().manual_seed(1)
    b = 200
    p = 6.5 / 50
    x = (26 + 12 * torch.rand(b, 1, generator=g)) * p
    y = (26 + 12 * torch.rand(b, 1, generator=g)) * p
    pos = torch.stack([x, y, torch.zeros_like(x)], -1)
    camera = gx.Camera(pixel_size=6.5, shape=(64, 64))
    objective = gx.Objective(NA=0.7, magnification=50)
    beads = gx.Emitters(position=pos, photons=1000.0, emission=gx.Spectrum.line(0.6))
    psf = gx.imaging.PointPSF(objective, camera)
    img = psf(gx.lower.emitter_set(beads), gx.env.Homogeneous(1.33)).data[:, 0, 0]
    rows = torch.arange(64.0)[:, None]
    cols = torch.arange(64.0)[None, :]
    m = img.sum((-2, -1))
    cx, cy = (img * cols).sum((-2, -1)) / m, (img * rows).sum((-2, -1)) / m
    expected = gx.coords.to_pixels(pos[:, 0], camera, objective)
    # the residual is the ROI's asymmetric truncation of the PSF tail (≤ 0.5 px of placement)
    for got, want in ((cx, expected[:, 0]), (cy, expected[:, 1])):
        assert (got - want).abs().max() < 0.025 and abs(float((got - want).mean())) < 0.005


def test_defocus_is_symmetric_without_aberrations_and_astigmatism_pins_the_z_sign():
    up, _ = _single(0.9, 1.33, (48, 48), 60.0, z=0.4)
    down, _ = _single(0.9, 1.33, (48, 48), 60.0, z=-0.4)
    assert torch.allclose(up, down, atol=1e-5 * float(up.max()))
    astig = (gx.pupil.Zernike(coeffs=torch.tensor([0.08]), indices=(5,)),)  # (n, m) = (2, 2)
    up, _ = _single(0.9, 1.33, (48, 48), 60.0, z=0.4, pupil=astig)
    down, _ = _single(0.9, 1.33, (48, 48), 60.0, z=-0.4, pupil=astig)

    def elongation(img):
        rows = torch.arange(48.0)[:, None]
        cols = torch.arange(48.0)[None, :]
        m = img.sum()
        cx, cy = (img * cols).sum() / m, (img * rows).sum() / m
        return float((img * (cols - cx) ** 2).sum() / m - (img * (rows - cy) ** 2).sum() / m)

    # Paraxially the pupil phase is (π/λ)·[(Δz/n + 2√6·c/NA²)·u_x² + (Δz/n − 2√6·c/NA²)·u_y²]:
    # with c > 0 an emitter above focus (Δz > 0) is more defocused along x: its PSF is wider in x.
    assert elongation(up) > 0 > elongation(down)


def test_gradcheck_with_forward_ad():
    camera = gx.Camera(pixel_size=6.5, shape=(8, 8))
    env = gx.env.Homogeneous(1.33)

    def render(pos, focus, coeffs):
        objective = gx.Objective(
            NA=0.8,
            magnification=50,
            focus=focus,
            pupil=(gx.pupil.Zernike(coeffs=coeffs, indices=(4, 12)),),
        )
        beads = gx.Emitters(position=pos, photons=100.0, emission=gx.Spectrum.line(0.6))
        psf = gx.imaging.PointPSF(objective, camera, roi=8, oversample=1, pupil_samples=16)
        static = psf.eager_static(gx.lower.emitter_set(beads.to(torch.float64)), env)
        return psf(gx.lower.emitter_set(beads), env, static=static).data

    pos = torch.tensor([[[0.5, 0.55, 0.1]]], dtype=torch.float64, requires_grad=True)
    focus = torch.tensor(0.05, dtype=torch.float64, requires_grad=True)
    coeffs = torch.tensor([0.01, 0.02], dtype=torch.float64, requires_grad=True)
    assert gradcheck(render, (pos, focus, coeffs), forward_ad=True, atol=1e-6)


def test_standard_fidelity_now_plans():
    camera = gx.Camera(pixel_size=6.5, shape=(32, 32))
    objective = gx.Objective(NA=0.8, magnification=50)
    beads = gx.Emitters(
        position=torch.tensor([[[2.0, 2.0, 0.0]]]), photons=1000.0, emission=gx.Spectrum.line(0.6)
    )
    sample = gx.Sample({"beads": beads}, environment=gx.env.Homogeneous(1.33))
    plan = gx.plan(
        sample,
        gx.presets.Widefield(objective=objective, camera=camera),
        "standard",
        outputs=("expected",),
    )
    assert plan.routes["beads"].element == "emit.pupil_mft"
    assert (
        plan(sample, gx.presets.Widefield(objective=objective, camera=camera))["expected"].sum() > 0
    )


def _layered_efficiency(na, wavelength, depth, samples, n_s=1.33, n_g=1.518):
    medium = gx.env.LayeredMedium(sample=n_s, coverslip=n_g, immersion=n_g)
    u, du = ops.pupil_axis(samples, na * 1.03125)
    uy, ux = torch.meshgrid(u, u, indexing="ij")
    u2 = ux**2 + uy**2
    aperture = ops.soft_aperture(torch.sqrt(u2), torch.tensor(na, dtype=torch.float64), du)
    wl = torch.tensor(wavelength, dtype=torch.float64)
    amp, _n = medium.pupil_amplitude(u2, wl, torch.tensor(-depth, dtype=torch.float64))
    power = (aperture**2 * (amp.real**2 + amp.imag**2)).sum() * (du / wavelength) ** 2
    return float(power)


@pytest.mark.parametrize(("depth", "reference"), [(0.0, 0.709), (0.1, 0.575)])
def test_layered_collection_efficiency_matches_the_plan(depth, reference):
    """§4.1: NA 1.45, 680 nm, water on glass: 0.709 at the interface, 0.575 at 100 nm depth."""
    coarse = _layered_efficiency(1.45, 0.68, depth, 128)
    fine = _layered_efficiency(1.45, 0.68, depth, 512)
    assert abs(fine - reference) < 1e-3
    assert abs(coarse - fine) < 5e-4  # grid-converged


def test_layered_medium_reduces_to_homogeneous_when_indices_match():
    matched = _layered_efficiency(1.2, 0.6, 0.0, 256, n_s=1.518, n_g=1.518)
    eta = float(gx.conventions.collection_efficiency(1.2, 1.518))
    assert abs(matched - eta) < 2e-4


def test_supercritical_psf_renders_with_a_layered_medium():
    camera = gx.Camera(pixel_size=6.5, shape=(96, 96))
    objective = gx.Objective(NA=1.45, magnification=100)
    beads = gx.Emitters(
        position=torch.tensor([[[3.12, 3.12, 0.0]]]),
        photons=1000.0,
        emission=gx.Spectrum.line(0.68),
    )
    es = gx.lower.emitter_set(beads)
    with pytest.raises(gx.ValidityError, match="layered"):
        gx.imaging.PointPSF(objective, camera, psf="scalar")(es, gx.env.Homogeneous(1.333))
    with pytest.warns(gx.GradixWarning, match="falls back to scalar"):
        gx.imaging.PointPSF(objective, camera)(es, gx.env.WaterOnCoverslip(sample=1.33))
    psf = gx.imaging.PointPSF(objective, camera, psf="scalar")  # NA/n_i = 0.955: accept scalar
    image = psf(es, gx.env.WaterOnCoverslip(sample=1.33)).data
    assert float(image.sum()) == pytest.approx(709.0, rel=3e-3)


def test_pixel_mtf_matches_fine_box_integration():
    """M1 exit: the pixel MTF agrees with brute-force box integration within 1 %."""
    na, wl, pitch, roi = 1.2, 0.6, 0.13, 16  # object-space pixel pitch, µm
    u, du = ops.pupil_axis(64, na * 1.03125)
    uy, ux = torch.meshgrid(u, u, indexing="ij")
    radius = torch.sqrt(ux**2 + uy**2)
    pupil = ops.soft_aperture(radius, torch.tensor(na, dtype=torch.float64), du)
    pupil = pupil.to(torch.complex128)[None]
    wavelength = torch.tensor([wl], dtype=torch.float64)

    def intensity(x):
        e = ops.mft_roi(pupil, u, du, wavelength, x[None], x[None])[0]
        return e.real**2 + e.imag**2

    # reference: 16×16 sub-samples per pixel, averaged
    k = torch.arange(roi * 16, dtype=torch.float64)
    fine = intensity((k + 0.5) * pitch / 16 - roi * pitch / 2)
    box = fine.reshape(roi, 16, roi, 16).mean((1, 3))
    # pixel MTF: s = 2 samples per pixel starting at each pixel centre
    s = 2
    k = torch.arange(roi * s, dtype=torch.float64)
    xs = (k // s + 0.5) * pitch + (k % s) * pitch / s - roi * pitch / 2
    mtf = ops.pixel_mtf(intensity(xs), s)[::s, ::s]
    rel = float(torch.linalg.vector_norm(mtf - box) / torch.linalg.vector_norm(box))
    assert rel < 0.01, rel


def test_bead_stack_zernike_recovery():
    """M1 exit (small): Zernike coefficients recovered from a noisy bead stack ≤ λ/50 RMS."""
    dt = torch.float64
    truth = torch.tensor([0.03, -0.02, 0.015], dtype=dt)
    indices = (5, 7, 12)
    camera = gx.Camera(pixel_size=6.5, shape=(24, 24), noise=gx.noise.Ideal(), unit="e")
    steps = torch.tensor([-0.5, 0.0, 0.5], dtype=dt)
    beads = gx.Emitters(
        position=torch.tensor([[[1.56, 1.56, 0.0]]], dtype=dt),
        photons=2e5,
        emission=gx.Spectrum.line(0.6),
    )

    def render(coeffs):
        zern = gx.pupil.Zernike(coeffs=coeffs, indices=indices)
        objective = gx.Objective(NA=1.2, magnification=100, pupil=(zern,))
        psf = gx.imaging.PointPSF(objective, camera, psf="scalar", roi=24)
        chain = gx.Chain(
            emitters={"b": beads},
            imaging=psf,
            environment=gx.env.Homogeneous(1.33),
            acquisition=gx.acq.FocusStack(focus=steps),
        )
        return chain(outputs=("expected",))["expected"]

    mu = render(truth)
    observed = camera.sample(mu, key=torch.tensor([7])).detach()
    coeffs = torch.zeros(3, dtype=dt, requires_grad=True)
    opt = torch.optim.LBFGS([coeffs], max_iter=60, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = -camera.log_prob(observed, render(coeffs)).sum()
        loss.backward()
        return loss

    opt.step(closure)
    rms = float(((coeffs.detach() - truth) ** 2).mean().sqrt())
    assert rms < 0.6 / 50, rms


def test_memory_planning_counts_the_per_emitter_pupils():
    camera = gx.Camera(pixel_size=6.5, shape=(64, 64))
    beads = gx.Emitters(
        position=torch.rand(8, 20, 3) * torch.tensor([8.0, 8.0, 0.2]),
        photons=1000.0,
        emission=gx.Spectrum.line(0.6),
    )
    psf = gx.imaging.PointPSF(gx.Objective(NA=1.2, magnification=50), camera, psf="scalar")
    chain = gx.Chain(emitters={"b": beads}, imaging=psf, environment=gx.env.Homogeneous(1.33))
    pipe = gx.Pipeline(chain, outputs=("expected",))
    static = pipe.static("imaging")
    assert isinstance(static, gx.imaging.PointPSFStatic) and static.pupil is not None
    samples = static.pupil.shape[0]
    assert pipe.memory.estimate_bytes >= pipe.batch * 32 * 48 * samples * samples  # slots 32


def test_layered_media_take_materials_and_the_plans_tuple_forms():
    medium = gx.env.LayeredMedium(
        immersion=gx.materials.Oil(),
        coverslip=(gx.materials.Glass(), 170.0),
        sample=gx.materials.Water(),
    )
    assert isinstance(medium.sample, gx.Material) and medium.thickness == 170.0
    blue, red = medium.index(wavelength=0.45), medium.index(wavelength=0.7)
    assert float(blue.reshape(())) > float(red.reshape(()))  # normal dispersion
    camera = gx.Camera(pixel_size=6.5, shape=(48, 48))
    objective = gx.Objective(NA=1.4, magnification=100)
    bead = gx.Emitters(
        position=torch.tensor([[[1.56, 1.56, -0.2]]]),
        photons=1000.0,
        emission=gx.Spectrum.band(0.6, 0.1, bins=3),
    )
    psf = gx.imaging.PointPSF(objective, camera, psf="scalar")
    image = psf(gx.lower.emitter_set(bead), medium).data
    assert torch.isfinite(image).all() and 400 < float(image.sum()) < 800
    mismatched = gx.env.LayeredMedium(sample=1.33, coverslip=(1.518, 175.0, (1.518, 170.0)))
    assert mismatched.design_thickness == 170.0
    # the tuple's thickness wins, so replace() can swap the coverslip of any medium
    assert (
        gx.env.LayeredMedium(sample=1.33, coverslip=(1.518, 160.0), thickness=150.0).thickness
        == 160.0
    )
    assert mismatched.replace(coverslip=(1.52, 150.0)).thickness == 150.0
    # a dispersive render is the weighted sum of per-bin renders in constant media
    pinned = gx.imaging.PointPSF(objective, camera, psf="scalar", roi=32, pupil_samples=128)
    total = pinned(gx.lower.emitter_set(bead), medium).data
    wavelengths, weights = bead.emission.arrays()
    parts = torch.zeros_like(total)
    for wl, w in zip(wavelengths.reshape(-1).tolist(), weights.reshape(-1).tolist(), strict=True):
        constant = gx.env.LayeredMedium(
            immersion=float(gx.materials.Oil().index(wl)),
            coverslip=(float(gx.materials.Glass().index(wl)), 170.0),
            sample=float(gx.materials.Water().index(wl)),
        )
        line = bead.replace(emission=gx.Spectrum.line(wl), photons=1000.0 * w)
        parts = parts + pinned(gx.lower.emitter_set(line), constant).data
    assert torch.allclose(total, parts, rtol=1e-4, atol=1e-6 * float(total.max()))


def test_the_global_path_matches_large_rois_in_shape_and_is_exact_on_the_camera():
    # ladder edge sparse ROI ↔ global spectrum (ROI → ∞): the ROI path renormalises each ROI
    # to the pupil's power, so the edge holds per emitter up to that scale
    common = {"pupil_samples": 256, "psf": "scalar", "dtype": torch.float64}
    roi, _ = _single(1.2, 1.33, (64, 64), 65, roi=64, **common)
    full, _ = _single(1.2, 1.33, (64, 64), 65, method="global", **common)
    shape = (roi / roi.sum() - full / full.sum()).norm() / (full / full.sum()).norm()
    assert float(shape) < 1e-4
    # the global path loses the Airy tail beyond the frame; the ROI path folds it in
    assert 0.95 < float(full.sum() / roi.sum()) < 1.0


def test_the_global_path_renders_edges_batches_and_gradients_in_a_pipeline():
    camera = gx.Camera(pixel_size=6.5, shape=(40, 32))
    objective = gx.Objective(NA=1.2, magnification=65)
    position = torch.tensor(
        [
            [[2.0, 1.6, 0.0], [0.1, 0.2, 0.4], [2.5, 0.5, -0.3]],
            [[1.0, 1.0, 0.2], [3.0, 2.0, 0.0], [1.5, 2.5, 0.1]],
        ]
    )
    beads = gx.Emitters(
        position=position, photons=torch.full((2, 3), 800.0), emission=gx.Spectrum.line(0.6)
    )
    chain = gx.Chain(
        emitters={"beads": beads},
        imaging=gx.imaging.PointPSF(objective, camera, method="global", psf="scalar"),
        environment=gx.env.Homogeneous(1.33),
    )
    pipe = gx.Pipeline(chain, outputs=("expected",))
    assert "method = global" in pipe.explain()
    x = position.clone().requires_grad_(True)
    image = pipe({"beads.position": x})["expected"]
    assert image.shape == (2, 1, 40, 32)
    assert bool(torch.isfinite(image).all()) and float(image.detach().min()) >= 0.0
    (image * torch.linspace(0, 1, 32)).sum().backward()
    assert x.grad is not None and bool(torch.isfinite(x.grad).all())
    assert float(x.grad[..., 0].abs().min()) > 0.0  # every emitter moves the image in x
    # per-image results do not depend on the other images of the batch
    single = pipe({"beads.position": position[1:]})["expected"]
    assert torch.allclose(single[0], image[1].detach(), rtol=1e-5, atol=1e-6)


def test_a_deepstorm3d_style_mask_learning_loop_runs():
    # a learnable pixel phase mask and a localisation network trained together (§9, M1 exit)
    g = torch.Generator().manual_seed(0)
    camera = gx.Camera(pixel_size=6.5, shape=(24, 24))
    opd = torch.zeros(16, 16, requires_grad=True)
    objective = gx.Objective(NA=1.2, magnification=100, pupil=(gx.pupil.PixelPupil(opd=opd),))
    b, centre = 16, 24 * 0.065 / 2

    def positions(z):
        xy = torch.full((z.shape[0], 1), centre)
        return torch.stack([xy, xy, z[:, None]], -1)  # [B, 1, 3]

    beads = gx.Emitters(
        position=positions(torch.linspace(-0.8, 0.8, b)),
        photons=torch.full((b, 1), 2000.0),
        emission=gx.Spectrum.line(0.6),
    )
    psf = gx.imaging.PointPSF(objective, camera, psf="scalar")
    chain = gx.Chain(emitters={"b": beads}, imaging=psf, environment=gx.env.Homogeneous(1.33))
    pipe = gx.Pipeline(chain, outputs=("expected",), envelope={"b.position.z": (-1.0, 1.0)})
    net = torch.nn.Sequential(
        torch.nn.Conv2d(1, 8, 3, padding=1),
        torch.nn.ReLU(),
        torch.nn.AdaptiveAvgPool2d(4),
        torch.nn.Flatten(),
        torch.nn.Linear(128, 1),
    )
    optimiser = torch.optim.Adam([*net.parameters(), opd], lr=1e-2)
    losses = []
    for _ in range(25):
        z = 1.6 * torch.rand(b, generator=g) - 0.8
        image = pipe({"b.position": positions(z)})["expected"]  # [B, 1, H, W]
        image = image / image.amax(dim=(-2, -1), keepdim=True)
        loss = ((net(image).squeeze(-1) - z) ** 2).mean()
        optimiser.zero_grad()
        loss.backward()
        assert opd.grad is not None and bool(torch.isfinite(opd.grad).all())
        optimiser.step()
        losses.append(float(loss.detach()))
    assert float(opd.detach().abs().max()) > 0.0  # the mask moved
    assert sum(losses[-5:]) < sum(losses[:5])  # and the pair learned something
