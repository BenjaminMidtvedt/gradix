"""Elements: Gaussian sprites (radiometry, frame, gradients) and the camera (units, noise)."""

import dataclasses
import math

import pytest
import scipy.special
import scipy.stats
import torch
from scenes import DEVICES, emitters, optics, tensor

import gradix as gx
from gradix.testing import conformance, gradcheck

ENV = gx.env.Homogeneous(1.33)


def _render(beads, objective, camera, env=ENV):
    return gx.imaging.Sprites(objective, camera)(gx.lower.emitter_set(beads), env)


@pytest.mark.parametrize("device", DEVICES)
def test_sprite_integral_is_the_collection_efficiency(device):
    objective, camera = optics(shape=(96, 96), na=0.5, magnification=50)
    centre = 96 * 0.13 / 2
    beads = gx.Emitters(
        position=torch.tensor([[[centre, centre, 0.3]]], device=device),
        photons=torch.tensor([[1000.0]], device=device),
        emission=gx.Spectrum.line(0.6),
    )
    total = float(_render(beads, objective, camera).data.sum())
    eta = float(gx.conventions.collection_efficiency(0.5, 1.33))
    assert total == pytest.approx(1000.0 * eta, rel=1e-5)


def test_sprite_centroid_follows_to_pixels_without_transposition():
    objective, camera = optics(shape=(48, 64), na=0.6, magnification=50)
    g = torch.Generator().manual_seed(3)
    b = 1000
    pitch = 6.5 / 50
    x = (20 + 24 * torch.rand(b, 1, generator=g)) * pitch
    y = (16 + 16 * torch.rand(b, 1, generator=g)) * pitch
    pos = torch.stack([x, y, torch.zeros_like(x)], -1).double()
    beads = gx.Emitters(position=pos, photons=1e4, emission=gx.Spectrum.line(0.6))
    img = _render(beads, objective, camera).data[:, 0, 0]  # [B, H, W]
    rows = torch.arange(48, dtype=torch.float64)[:, None]
    cols = torch.arange(64, dtype=torch.float64)[None, :]
    mass = img.sum((-2, -1))
    cy = (img * rows).sum((-2, -1)) / mass
    cx = (img * cols).sum((-2, -1)) / mass
    expected = gx.coords.to_pixels(pos[:, 0], camera, objective)
    assert (cx - expected[:, 0]).abs().max() < 0.02
    assert (cy - expected[:, 1]).abs().max() < 0.02


def test_defocus_widens_symmetrically_and_conserves_photons():
    objective, camera = optics(shape=(64, 64), na=0.6, magnification=50)
    c = 32 * 0.13
    pos = torch.tensor([[[c, c, 0.0], [c, c, 0.8], [c, c, -0.8]]]).double()
    beads = gx.Emitters(
        position=pos.reshape(3, 1, 3), photons=1000.0, emission=gx.Spectrum.line(0.6)
    )
    img = _render(beads, objective, camera).data[:, 0, 0]
    assert torch.allclose(img[1], img[2])
    assert img[1].max() < img[0].max()
    assert torch.allclose(img.sum((-2, -1)), img[0].sum().expand(3), rtol=1e-6)


def test_presence_is_linear_and_populations_have_their_own_spectra():
    objective, camera = optics(shape=(32, 32), na=0.6)
    a = emitters(1, 2, fov=4.0, presence=torch.tensor([[1.0, 0.5]]))
    full = emitters(1, 2, fov=4.0)
    half = _render(a, objective, camera).data
    ones = _render(
        full.select(0).replace(position=a.position[:, :1], photons=full.photons[:, :1]),
        objective,
        camera,
    ).data
    other = _render(
        full.select(0).replace(position=a.position[:, 1:], photons=full.photons[:, 1:]),
        objective,
        camera,
    ).data
    assert torch.allclose(half, ones + 0.5 * other, atol=1e-4)
    red = emitters(1, 2, fov=4.0, seed=4, wavelength=0.7)
    es = gx.lower.emitter_set({"red": red, "a": full})
    assert es.species is not None and es.species.tolist() == [[[0, 0, 1, 1]]]
    assert es.wavelengths.shape == (1, 2, 1)
    both = gx.imaging.Sprites(objective, camera)(es, ENV).data
    assert torch.allclose(
        both,
        _render(full, objective, camera).data + _render(red, objective, camera).data,
        atol=1e-4,
    )


def test_sprites_gradcheck_with_forward_ad():
    objective, camera = optics(shape=(8, 8), na=0.6, magnification=50)
    pos = torch.tensor(
        [[[0.5, 0.6, 0.1], [0.3, 0.7, -0.2]]], dtype=torch.float64, requires_grad=True
    )
    photons = torch.tensor([[100.0, 50.0]], dtype=torch.float64, requires_grad=True)
    na = torch.tensor(0.6, dtype=torch.float64, requires_grad=True)
    focus = torch.tensor(0.05, dtype=torch.float64, requires_grad=True)

    def render(pos, photons, na, focus):
        beads = gx.Emitters(position=pos, photons=photons, emission=gx.Spectrum.line(0.6))
        obj = objective.replace(NA=na, focus=focus)
        sprites = gx.imaging.Sprites(obj, camera)
        es = gx.lower.emitter_set(beads)
        return sprites(es, ENV, static=gx.imaging.SpritesStatic(grid=gx.Grid2D((8, 8), 0.13))).data

    assert gradcheck(render, (pos, photons, na, focus), forward_ad=True)


def test_sprites_conformance():
    objective, camera = optics(shape=(16, 16))
    beads = emitters(2, 3, fov=2.0)
    element = gx.imaging.Sprites(objective, camera)
    probes = {
        "position": beads.position.clone().requires_grad_(),
        "photons": beads.photons.clone().requires_grad_(),
        "focus": torch.tensor(0.1, requires_grad=True),
    }

    def rebuild(p):
        b = beads.replace(position=p["position"], photons=p["photons"])
        return element.replace(objective=objective.replace(focus=p["focus"])), (
            gx.lower.emitter_set(b),
            ENV,
        )

    report = conformance(
        element, (gx.lower.emitter_set(beads), ENV), probes=probes, make_inputs=rebuild
    )
    assert all(v == "ok" for v in report.values()), report


def test_sprites_validity_warns_eagerly():
    objective, camera = optics(shape=(16, 16), na=0.9)
    with pytest.warns(gx.GradixWarning, match="high NA"):
        _render(emitters(1, 2, fov=2.0), objective, camera)


def _irradiance(value, shape=(4, 5)):
    return gx.Irradiance(data=torch.full((2, 1, 1, *shape), value), grid=gx.Grid2D(shape, 0.1))


@pytest.mark.parametrize("unit", ["adu", "e", "photons"])
def test_camera_units(unit):
    camera = gx.Camera(pixel_size=0.5, shape=(4, 5), qe=0.8, gain=2.0, offset=100.0, unit=unit)
    mu = camera.expected(_irradiance(10.0), background=torch.tensor([1.0, 2.0]))
    photons = torch.tensor([11.0, 12.0])[:, None, None, None]
    expected = {"adu": 2.0 * 0.8 * photons + 100.0, "e": 0.8 * photons, "photons": photons}[unit]
    assert mu.shape == (2, 1, 4, 5)
    assert torch.allclose(mu, expected.expand(2, 1, 4, 5))
    assert torch.allclose(camera.electrons(mu), (0.8 * photons).expand(2, 1, 4, 5))


def test_camera_maps_and_background_maps():
    gain = torch.linspace(1.0, 2.0, 20).reshape(4, 5)
    camera = gx.Camera(pixel_size=0.5, shape=(4, 5), gain=gain)
    mu = camera.expected(_irradiance(1.0), background=torch.ones(2, 4, 5))
    assert torch.allclose(mu[0, 0], 2.0 * gain)


def test_noise_free_camera_needs_no_key():
    camera = gx.Camera(pixel_size=0.5, shape=(4, 5))
    mu = camera.expected(_irradiance(3.0))
    assert torch.equal(camera.sample(mu), mu)
    with pytest.raises(gx.StructureError, match="no likelihood"):
        camera.log_prob(mu, mu)


@pytest.mark.parametrize("key", [7, torch.tensor([3, 4])])
def test_noisy_camera_draws_integer_electrons(key):
    camera = gx.Camera(pixel_size=0.5, shape=(4, 5), gain=3.0, offset=10.0, noise=gx.noise.Ideal())
    mu = camera.expected(_irradiance(20.0))
    y = camera.sample(mu, key)
    e = camera.electrons(y)
    assert torch.allclose(e, e.round(), atol=1e-4)
    with pytest.raises(gx.StructureError, match="needs a key"):
        camera.sample(mu)


def test_adc_rounds_and_saturates():
    camera = gx.Camera(pixel_size=0.5, shape=(4, 5), gain=10.0, bit_depth=8)
    mu = camera.expected(_irradiance(30.0))  # 300 ADU > 255
    assert torch.all(camera.sample(mu) == 255.0)
    mu = camera.expected(_irradiance(1.234))
    assert torch.equal(camera.sample(mu), torch.full_like(mu, 12.0))


def test_log_prob_is_the_poisson_pmf_with_the_unit_jacobian():
    camera_e = gx.Camera(pixel_size=0.5, shape=(4, 5), unit="e", noise=gx.noise.Ideal())
    k = torch.arange(20, dtype=torch.float64).reshape(1, 1, 4, 5)
    mu = torch.full_like(k, 7.5)
    lp = camera_e.log_prob(k, mu)
    ref = torch.from_numpy(scipy.stats.poisson.logpmf(k.numpy(), 7.5))
    assert torch.allclose(lp, ref)
    camera_adu = gx.Camera(
        pixel_size=0.5, shape=(4, 5), gain=2.5, offset=100.0, noise=gx.noise.Ideal()
    )
    lp_adu = camera_adu.log_prob(2.5 * k + 100.0, 2.5 * mu + 100.0)
    assert torch.allclose(lp_adu, ref - math.log(2.5))


def test_poisson_gaussian_likelihood_and_read_noise():
    noise = gx.noise.PoissonGaussian(read=2.0)
    camera = gx.Camera(pixel_size=0.5, shape=(4, 5), unit="e", noise=noise)
    mu = camera.expected(_irradiance(50.0))[:1].expand(512, 1, 4, 5).contiguous()
    y = camera.sample(mu, torch.arange(512))
    assert abs(float(y.var()) - (50.0 + 4.0)) < 4.0
    k = torch.linspace(0, 100, 11)
    shifted = noise.log_prob(k, torch.full_like(k, 50.0))
    gaussian = noise.replace(likelihood="gaussian").log_prob(k, torch.full_like(k, 50.0))
    assert torch.isfinite(shifted).all() and torch.isfinite(gaussian).all()
    assert int(shifted.argmax()) == int(gaussian.argmax()) == 5


def test_to_unit_converts_recorded_frames():
    camera = gx.Camera(pixel_size=0.5, shape=(4, 5), qe=0.5, gain=2.0, offset=100.0, unit="photons")
    adu = torch.full((1, 1, 4, 5), 120.0)
    photons = gx.detect.to_unit(adu, camera, from_unit="adu")
    assert torch.allclose(photons, torch.full_like(adu, 20.0))  # (120 − 100)/2 e⁻ / 0.5
    assert torch.equal(gx.detect.to_unit(photons, camera, from_unit="photons"), photons)


def test_plane_wave_direction_is_a_per_image_field():
    pw = gx.light.PlaneWave(torch.full((4,), 0.5), direction=torch.rand(4, 2))
    assert tensor(pw.select(1).direction).shape == (1, 2)
    assert pw().u.shape[0] == 4
    tilts = [gx.light.PlaneWave(0.5, direction=torch.tensor([t, 0.0])) for t in (0.1, 0.2)]
    assert tensor(gx.stack(tilts).direction).shape == (2, 2)
    irradiance = torch.zeros(1, requires_grad=True)
    amplitude = gx.light.PlaneWave(0.5, irradiance=irradiance)().amplitude
    amplitude.abs().sum().backward()
    assert torch.isfinite(tensor(irradiance.grad)).all()  # safe square root at zero irradiance


def test_pixel_maps_must_match_the_camera():
    with pytest.raises(gx.StructureError, match="map"):
        gx.Camera(pixel_size=6.5, shape=(64, 64), gain=torch.ones(32, 32))
    with pytest.raises(gx.StructureError, match=r"\[B, 1, 1\]"):
        gx.Camera(pixel_size=6.5, shape=(64, 64), gain=torch.arange(1.0, 9.0)[:, None])
    gx.Camera(pixel_size=6.5, shape=(64, 64), gain=torch.arange(1.0, 9.0)[:, None, None])
    objective, camera = optics()
    with pytest.raises(gx.StructureError, match="background"):
        gx.Chain(
            emitters={"beads": emitters(1, 2)},
            imaging=gx.imaging.Sprites(objective, camera),
            environment=gx.env.Homogeneous(1.33),
            background=torch.ones(8, 1),
        )


def test_plane_waves_honour_travel_and_have_finite_gradients_at_grazing():
    waves = gx.light.PlaneWave(0.5, direction=torch.tensor([1.6, 0.0]))()  # evanescent in n=1.33
    down = waves.replace(travel=-1)
    points = torch.tensor([[[[0.0, 0.0, -0.5]]]])
    up_field = waves.at(points, 1.33).abs()
    down_field = down.at(points, 1.33).abs()
    assert float(up_field.max()) > 1.0 > float(down_field.max())  # decays along its travel
    n = torch.tensor(1.0, dtype=torch.float64, requires_grad=True)
    grazing = gx.light.PlaneWave(0.5, direction=torch.tensor([1.0, 0.0], dtype=torch.float64))()
    value = grazing.at(points.double(), n).abs().sum()
    (grad,) = torch.autograd.grad(value, n)
    assert torch.isfinite(grad)
    with pytest.raises(gx.StructureError, match="scalar or"):
        grazing.at(points.double(), torch.ones(2))


def test_sprites_accept_per_image_pitch_with_shared_emitters():
    objective, camera = optics(pixel_size=torch.tensor([6.5, 13.0]))
    chain_ = gx.Chain(
        emitters={"beads": emitters(1, 3)},
        imaging=gx.imaging.Sprites(objective, camera),
        environment=gx.env.Homogeneous(1.33),
    )
    mu = chain_(outputs=("expected",))["expected"]
    assert mu.shape[0] == 2 and not torch.equal(mu[0], mu[1])


def _conformance_scene(fov=2.0):
    objective, camera = optics(shape=(16, 16))
    beads = emitters(2, 3, fov=fov)
    return objective, camera, beads


def test_conformance_gradients_see_photon_conserving_parameters():
    objective, camera = optics(shape=(64, 64))
    beads = gx.Emitters(  # centred: the image sum does not depend on position or focus
        position=torch.tensor([[[4.16, 4.16, 0.0]], [[4.0, 4.3, 0.1]]]),
        photons=torch.full((2, 1), 1000.0),
        emission=gx.Spectrum.line(0.6),
    )
    element = gx.imaging.Sprites(objective, camera)
    probes = {
        "position": tensor(beads.position).clone().requires_grad_(),
        "focus": torch.tensor(0.05, requires_grad=True),
    }

    def rebuild(p):
        b = beads.replace(position=p["position"])
        return element.replace(objective=objective.replace(focus=p["focus"])), (
            gx.lower.emitter_set(b),
            ENV,
        )

    report = conformance(
        element, (gx.lower.emitter_set(beads), ENV), probes=probes, make_inputs=rebuild
    )
    assert report["gradients"] == "ok", report


def test_conformance_catches_impure_and_batch_breaking_elements():
    objective, camera, beads = _conformance_scene()

    @dataclasses.dataclass(frozen=True, kw_only=True, eq=False)
    class Noisy(gx.imaging.Sprites):
        """Draws from the global generator (forbidden)."""

        def forward(self, *inputs, static):
            out = super().forward(*inputs, static=static)
            torch.manual_seed(0)
            return out.replace(data=out.data + 0 * torch.rand(1))

    @dataclasses.dataclass(frozen=True, kw_only=True, eq=False)
    class FirstImageOnly(gx.imaging.Sprites):
        """Returns one image for a batch."""

        def forward(self, *inputs, static):
            out = super().forward(*inputs, static=static)
            return out.replace(data=out.data[:1])

    @dataclasses.dataclass(frozen=True, kw_only=True, eq=False)
    class BrokenValidity(gx.imaging.Sprites):
        """Raises in validity."""

        def validity(self, desc, envelope):
            raise AttributeError("oops")

    inputs = (gx.lower.emitter_set(beads), ENV)
    assert "random state" in conformance(Noisy(objective, camera), inputs)["purity"]
    assert conformance(FirstImageOnly(objective, camera), inputs)["batch"] != "ok"
    assert "AttributeError" in conformance(BrokenValidity(objective, camera), inputs)["validity"]


def test_sprites_gradcheck_optics_wavelength_and_camera_fields():
    objective, camera = optics(shape=(8, 8), na=0.6, magnification=50)
    pos = torch.tensor([[[0.5, 0.6, 0.1], [0.3, 0.7, -0.2]]], dtype=torch.float64)
    inputs = (
        torch.tensor(6.5, dtype=torch.float64, requires_grad=True),  # pixel size
        torch.tensor(50.0, dtype=torch.float64, requires_grad=True),  # magnification
        torch.tensor(0.6, dtype=torch.float64, requires_grad=True),  # wavelength
        torch.tensor(2.0, dtype=torch.float64, requires_grad=True),  # gain
        torch.tensor(0.9, dtype=torch.float64, requires_grad=True),  # QE
    )

    def render(size, mag, wavelength, gain, qe):
        beads = gx.Emitters(position=pos, photons=100.0, emission=gx.Spectrum.line(wavelength))
        cam = camera.replace(pixel_size=size, gain=gain, qe=qe)
        sprites = gx.imaging.Sprites(objective.replace(magnification=mag), cam)
        es = gx.lower.emitter_set(beads)
        static = gx.imaging.SpritesStatic(grid=gx.Grid2D((8, 8), 0.13))
        return cam.expected(sprites(es, ENV, static=static))

    assert gradcheck(render, inputs)


def test_sprites_pin_the_gaussian_width_and_its_defocus_law():
    """σ₀ = 0.22·λ/NA and σ(z_R) = √2·σ₀ with z_R = 4π·σ₀²·n/λ (§5.1)."""
    objective, camera = optics(shape=(64, 64), na=0.7, magnification=100)
    n = 1.33
    for dz in (0.0, None):
        sigma0 = 0.22 * 0.6 / 0.7
        z_r = 4 * math.pi * sigma0**2 * n / 0.6
        z = 0.0 if dz is not None else z_r
        beads = gx.Emitters(
            position=torch.tensor([[[2.08, 2.08, z]]], dtype=torch.float64),
            photons=1.0,
            emission=gx.Spectrum.line(0.6),
        )
        if dz is None:
            with pytest.warns(gx.GradixWarning, match="DOF"):
                image = _render(beads, objective, camera).data[0, 0, 0].double()
        else:
            image = _render(beads, objective, camera).data[0, 0, 0].double()
        pitch = 6.5 / 100
        c = (torch.arange(64, dtype=torch.float64) + 0.5) * pitch
        total = image.sum()
        mean_x = (image.sum(0) * c).sum() / total
        var_x = (image.sum(0) * (c - mean_x) ** 2).sum() / total - pitch**2 / 12
        expected = sigma0**2 * (1.0 if dz is not None else 2.0)
        assert float(var_x) == pytest.approx(expected, rel=2e-3)


def test_camera_likelihood_values_match_reference_densities():
    k = torch.tensor([[[[0.0, 3.0], [7.0, 12.0]]]], dtype=torch.float64)  # electrons
    lam = torch.tensor([[[[0.5, 2.0], [6.0, 10.0]]]], dtype=torch.float64)
    read = torch.tensor([[1.0, 1.5], [2.0, 0.5]], dtype=torch.float64)  # a read-noise map
    var = read**2
    gaussian = gx.noise.PoissonGaussian(read=read)
    want = scipy.stats.norm.logpdf(k.numpy(), lam.numpy(), (lam + var).sqrt().numpy())
    assert torch.allclose(gaussian.log_prob(k, lam), torch.from_numpy(want), atol=1e-10)
    shifted = gx.noise.PoissonGaussian(read=read, likelihood="shifted_poisson")
    kv, lv = (k + var).numpy(), (lam + var).numpy()
    want = scipy.special.xlogy(kv, lv) - lv - scipy.special.gammaln(kv + 1.0)
    assert torch.allclose(shifted.log_prob(k, lam), torch.from_numpy(want), atol=1e-10)
    poisson = gx.noise.Ideal()
    want = scipy.stats.poisson.logpmf(k.numpy(), lam.numpy())
    assert torch.allclose(poisson.log_prob(k, lam), torch.from_numpy(want), atol=1e-10)
    # in ADU the density includes the Jacobian 1/gain per pixel
    camera = gx.Camera(pixel_size=6.5, shape=(2, 2), gain=2.0, offset=100.0, noise=poisson)
    adu = k * 2.0 + 100.0
    mu = lam * 2.0 + 100.0
    assert torch.allclose(
        camera.log_prob(adu, mu), torch.from_numpy(want) - math.log(2.0), atol=1e-10
    )
    del want
