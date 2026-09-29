"""Coherent imaging of Mie spheres (§4.1, §4.4, §5.3, §5.12) against direct integrals."""

import math

import numpy as np
import pytest
import scipy.integrate
import scipy.special
import torch

import gradix as gx
from gradix.special import mie

N_WATER, WL, NA = 1.33, 0.532, 1.2


def _scene(z=0.0, radius=0.1, na=NA, shape=(33, 33), dtype=torch.float64, material=1.59):
    camera = gx.Camera(pixel_size=6.5, shape=shape)
    objective = gx.Objective(NA=na, magnification=100)
    centre = shape[1] * 0.065 / 2
    beads = gx.Spheres(
        position=torch.tensor([[[centre, centre, z]]], dtype=dtype),
        radius=torch.tensor([[radius]], dtype=dtype),
        material=material,
    )
    return camera, objective, beads


def _amplitudes(radius, cos_theta, material=1.59, n=N_WATER, wl=WL):
    """S₁ and S₂ of a sphere at scattering angles (numpy, fp64), from gradix.special.mie."""
    k = 2 * math.pi * n / wl
    x = torch.tensor(k * radius, dtype=torch.float64)
    a, b = mie.coefficients(x, material / n, mie.terms(float(x)))
    mu = torch.as_tensor(np.atleast_1d(cos_theta), dtype=torch.float64)
    s1, s2 = mie.amplitudes(a, b, *mie.angular(mu, a.shape[-1]))
    return s1.numpy(), s2.numpy()


def _field_on_axis(radius, na=NA, n=N_WATER, wl=WL):
    """E_s(0) in focus for E_inc = 1: −1/(2k)·∫ q·(S₁ + S₂)/√(k·k_z) dq over the aperture."""
    k, qmax = 2 * math.pi * n / wl, 2 * math.pi * na / wl

    def integrand(q, part):
        kz = math.sqrt(k * k - q * q)
        s1, s2 = _amplitudes(radius, kz / k)
        value = -(q * (s1[0] + s2[0]) / math.sqrt(k * kz)) / (2 * k)
        return value.real if part == 0 else value.imag

    re = scipy.integrate.quad(integrand, 0, qmax, args=(0,), limit=200)[0]
    im = scipy.integrate.quad(integrand, 0, qmax, args=(1,), limit=200)[0]
    return complex(re, im)


def _field_on_x(radius, rho, dz, na=NA, n=N_WATER, wl=WL):
    """E_s(ρ, 0) for E_inc = 1: the S∥ pupil field integrated in φ gives J₀ ∓ J₂ (not the MFT)."""
    k, qmax = 2 * math.pi * n / wl, 2 * math.pi * na / wl

    def integrand(q, part):
        kz = math.sqrt(k * k - q * q)
        s1, s2 = _amplitudes(radius, kz / k)
        j0, j2 = scipy.special.jv(0, q * rho), scipy.special.jv(2, q * rho)
        angular = math.pi * (s2[0] * (j0 - j2) + s1[0] * (j0 + j2))
        value = (1j / (2 * math.pi)) * (1j / k) * q * angular * np.exp(1j * kz * dz)
        value = value * math.sqrt(kz / k) / kz
        return value.real if part == 0 else value.imag

    re = scipy.integrate.quad(integrand, 0, qmax, args=(0,), limit=400)[0]
    im = scipy.integrate.quad(integrand, 0, qmax, args=(1,), limit=400)[0]
    return complex(re, im)


def _scattered(element, spectra, waves, medium):
    static = element.eager_static(spectra, waves, medium)
    total = element.image_field(spectra, waves, medium, static=static)
    background = element.image_field((), waves, medium, static=static)
    return total - background


@pytest.mark.parametrize(("kind", "radius"), [("mie", 0.1), ("dipole", 0.02)])
def test_in_focus_interference_matches_the_analytic_centre_value(kind, radius):
    camera, objective, beads = _scene(radius=radius)
    medium = gx.env.Homogeneous(N_WATER)
    waves = gx.light.PlaneWave(WL, irradiance=1.0)()  # transmitted illumination, E_b = 1
    element_type = gx.interact.Mie if kind == "mie" else gx.interact.Dipole
    contributions = element_type(beads)(waves, medium)
    element = gx.imaging.Coherent(objective, camera, oversample=4, pupil_samples=128)
    image = element(contributions, waves, medium).data[0, 0, 0]
    background = element((), waves, medium).data[0, 0, 0]
    e_sc = _field_on_axis(radius)  # full Mie: the dipole (n = 1) is within 1 % at x = 0.31
    expected = 2 * e_sc.real + abs(e_sc) ** 2
    centre = (image[16, 16] - background[16, 16]) / 0.065**2
    assert float(centre) == pytest.approx(expected, rel=0.03)  # pixel averaging over the peak


def test_background_is_uniform_and_carries_the_irradiance():
    camera, objective, _beads = _scene()
    medium = gx.env.Homogeneous(N_WATER)
    waves = gx.light.PlaneWave(WL, irradiance=2.0)()
    element = gx.imaging.Coherent(objective, camera)
    background = element((), waves, medium).data
    assert torch.allclose(background, torch.full_like(background, 2.0 * 0.065**2), rtol=1e-6)
    # an oblique wave delivers I·cosθ (aplanatic); epi and evanescent waves deliver nothing
    tilted = gx.light.PlaneWave(WL, irradiance=2.0, direction=torch.tensor([0.6, 0.0]))()
    cos = math.sqrt(1 - (0.6 / N_WATER) ** 2)
    oblique = element((), tilted, medium).data
    assert torch.allclose(oblique, torch.full_like(oblique, 2.0 * cos * 0.065**2), rtol=1e-5)
    epi = gx.light.Uniform(irradiance=2.0, wavelength=WL)()  # travels away (−z)
    assert float(element((), epi, medium).data.abs().max()) == 0.0
    evanescent = gx.light.PlaneWave(WL, direction=torch.tensor([1.45, 0.0]))()
    assert float(element((), evanescent, medium).data.abs().max()) == 0.0


@pytest.mark.parametrize(("z", "tolerance"), [(0.0, 5e-3), (0.5, 1.5e-2)])
def test_the_imaged_mie_field_matches_direct_summation(z, tolerance):
    # the pupil-sample evaluation (MFT) against the angular-spectrum integral along a radial line
    camera, objective, beads = _scene(z=z, radius=0.15)
    medium = gx.env.Homogeneous(N_WATER)
    waves = gx.light.PlaneWave(WL)()
    spectra = gx.interact.Mie(beads)(waves, medium)
    e_inc = complex(waves.at(spectra[0].position, torch.tensor([[N_WATER]]))[..., 0, :].reshape(()))
    ref = torch.tensor([e_inc * _field_on_x(0.15, j * 0.065, -z) for j in range(17)])
    errors = []
    for samples in (256, 512):
        element = gx.imaging.Coherent(objective, camera, oversample=1, pupil_samples=samples)
        row = _scattered(element, spectra, waves, medium)[0, 0, 0, 0, 16, 16:]
        errors.append(float((row - ref).abs().max() / ref.abs().max()))
    assert errors[0] < tolerance
    assert errors[1] < 0.6 * errors[0]  # converges with the pupil grid (soft edge: O(du))


def test_the_imaged_backscatter_is_the_power_collected_by_the_aperture():
    # epi illumination: no background reaches the camera, and aplanatic imaging conserves the
    # power of the co-polarised (P = 1) backscatter inside the collection cone
    camera, objective, beads = _scene(radius=0.1, shape=(97, 97))
    medium = gx.env.Homogeneous(N_WATER)
    epi = gx.light.Uniform(irradiance=1.0, wavelength=WL)()
    image = gx.imaging.Coherent(objective, camera, oversample=2, pupil_samples=96)(
        gx.interact.Mie(beads)(epi, medium), epi, medium
    ).data
    k = 2 * math.pi * N_WATER / WL
    theta_max = math.asin(NA / N_WATER)

    def integrand(theta):  # ∫|S∥|² dφ = 3π/4·(|S₁|² + |S₂|²) + π/2·Re(S₁S₂*)
        s1, s2 = _amplitudes(0.1, -math.cos(theta))  # backscatter: Θ = π − θ
        power = 0.75 * math.pi * (abs(s1[0]) ** 2 + abs(s2[0]) ** 2)
        power += 0.5 * math.pi * (s1[0] * np.conj(s2[0])).real
        return power * math.sin(theta) / k**2

    collected = scipy.integrate.quad(integrand, 0, theta_max, limit=200)[0]
    assert float(image.sum()) == pytest.approx(collected, rel=0.04)  # the tail leaves the frame


def test_mie_approaches_the_dipole_for_small_spheres():
    camera, objective, beads = _scene(radius=0.02, z=0.3)
    medium = gx.env.Homogeneous(N_WATER)
    waves = gx.light.PlaneWave(WL)()
    element = gx.imaging.Coherent(objective, camera, pupil_samples=64)
    full = _scattered(element, gx.interact.Mie(beads)(waves, medium), waves, medium)
    dipole = _scattered(element, gx.interact.Dipole(beads)(waves, medium), waves, medium)
    assert float((full - dipole).norm() / full.norm()) < 0.02  # x = 0.31 (§5.2: ≤ 2 %)


def test_pupil_modifiers_act_on_background_and_scattered_light_alike():
    camera, objective, beads = _scene(radius=0.1, z=0.2, shape=(17, 17))
    medium = gx.env.Homogeneous(N_WATER)
    waves = gx.light.PlaneWave(WL)()
    spectra = gx.interact.Mie(beads)(waves, medium)
    plain = gx.imaging.Coherent(objective, camera, pupil_samples=48)(spectra, waves, medium).data
    # a uniform OPD is a global phase: nothing changes; a uniform amplitude 0.5 quarters the image
    flat = gx.pupil.PixelPupil(opd=torch.full((8, 8), 0.3), amplitude=torch.full((8, 8), 0.5))
    masked = objective.replace(pupil=(flat,))
    shaded = gx.imaging.Coherent(masked, camera, pupil_samples=48)(spectra, waves, medium).data
    assert torch.allclose(shaded, 0.25 * plain, rtol=1e-5, atol=1e-12)


def test_mie_images_are_differentiable_in_position_radius_and_index_in_both_modes():
    camera, objective, beads = _scene(z=0.2, shape=(17, 17))
    medium = gx.env.Homogeneous(N_WATER)
    waves = gx.light.PlaneWave(WL)()
    element = gx.imaging.Coherent(objective, camera, pupil_samples=32)
    position = beads.position.clone().requires_grad_()
    radius = torch.tensor([[0.1]], dtype=torch.float64, requires_grad=True)
    material = torch.tensor([[1.59]], dtype=torch.float64, requires_grad=True)

    def render(position, radius, material):
        spheres = beads.replace(position=position, radius=radius, material=material)
        return element(gx.interact.Mie(spheres)(waves, medium), waves, medium).data

    image = render(position, radius, material)
    weights = torch.rand_like(image, generator=torch.Generator().manual_seed(0))
    grads = torch.autograd.grad((image * weights).sum(), (position, radius, material))
    assert all(torch.isfinite(g).all() and (g != 0).any() for g in grads)
    # forward mode (gx.crlb's Jacobians) agrees with the reverse-mode gradient
    tangent = torch.ones_like(radius)
    pushed = torch.func.jvp(
        lambda r: render(position.detach(), r, material.detach()), (radius.detach(),), (tangent,)
    )
    assert float((pushed[1] * weights).sum()) == pytest.approx(float(grads[1].sum()), rel=1e-8)


def test_per_image_and_per_frame_optics_align_with_separate_renders():
    camera, objective, beads = _scene(z=0.1, shape=(17, 17), radius=0.03)
    beads = gx.stack([beads, beads.replace(position=beads.position + 0.05)])
    waves = gx.light.Uniform(wavelength=WL)()
    element = gx.imaging.Coherent(objective, camera, pupil_samples=32)
    indices = torch.tensor([1.33, 1.40], dtype=torch.float64)
    medium = gx.env.Homogeneous(indices)
    focus = torch.tensor([0.0, 0.3], dtype=torch.float64)
    varied = element.replace(objective=objective.replace(focus=focus))
    batch = varied(gx.interact.Dipole(beads)(waves, medium), waves, medium).data
    assert batch.shape == (2, 1, 1, 17, 17)
    for b in range(2):
        one = gx.tree.map(lambda t, b=b: t[b : b + 1], beads)
        med = gx.env.Homogeneous(float(indices[b]))
        single = element.replace(objective=objective.replace(focus=float(focus[b])))
        ref = single(gx.interact.Dipole(one)(waves, med), waves, med).data
        assert torch.allclose(batch[b], ref[0], rtol=1e-10, atol=1e-15)
    # per-frame focus [1, A]: each frame equals the render at that focus
    frames = element.replace(objective=objective.replace(focus=focus[None]))
    one = gx.tree.map(lambda t: t[:1], beads)
    water = gx.env.Homogeneous(1.33)
    out = frames(gx.interact.Dipole(one)(waves, water), waves, water).data
    assert out.shape == (1, 2, 1, 17, 17)
    for a in range(2):
        single = element.replace(objective=objective.replace(focus=float(focus[a])))
        ref = single(gx.interact.Dipole(one)(waves, water), waves, water).data
        assert torch.allclose(out[:, a], ref[:, 0], rtol=1e-10, atol=1e-15)
