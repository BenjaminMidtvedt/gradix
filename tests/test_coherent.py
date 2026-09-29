"""The M1 coherent smoke thread (§4.1, §4.4, §5.3): dipole spectra, pupil imaging, interference."""

import math

import pytest
import torch

import gradix as gx


def test_the_dipole_amplitude_satisfies_the_optical_theorem():
    k = torch.tensor(2 * math.pi * 1.33 / 0.532, dtype=torch.float64)
    f = gx.interact.scattering_amplitude(
        torch.tensor(0.05, dtype=torch.float64), torch.tensor(1.59 / 1.33, dtype=torch.float64), k
    )
    # extinction (4π/k)·Im f equals scattering 4π|f|² for a lossless scatterer
    assert float(f.imag) == pytest.approx(float(k * f.abs() ** 2), rel=1e-12)


def _scene(z=0.0, radius=0.03, na=1.2, shape=(33, 33), dtype=torch.float64):
    camera = gx.Camera(pixel_size=6.5, shape=shape)
    objective = gx.Objective(NA=na, magnification=100)
    centre = shape[1] * 0.065 / 2
    beads = gx.Spheres(
        position=torch.tensor([[[centre, centre, z]]], dtype=dtype),
        radius=torch.tensor([[radius]], dtype=dtype),
        material=1.59,
    )
    return camera, objective, beads


def test_in_focus_interference_matches_the_analytic_centre_value():
    camera, objective, beads = _scene(radius=0.02, na=1.2, shape=(33, 33))
    medium = gx.env.Homogeneous(1.33)
    waves = gx.light.PlaneWave(0.532, irradiance=1.0)()  # transmitted illumination
    contributions = gx.interact.Dipole(beads)(waves, medium)
    element = gx.imaging.Coherent(objective, camera, oversample=4, pupil_samples=128)
    image = element(contributions, waves, medium).data[0, 0, 0]
    background = element((), waves, medium).data[0, 0, 0]
    params = contributions[0].params
    assert isinstance(params, gx.interact.DipoleParams)
    f = params.amplitude.reshape(()).to(torch.complex128)
    n, wl, na = 1.33, 0.532, 1.2
    k, kz = 2 * math.pi * n / wl, 2 * math.pi * math.sqrt(n * n - na * na) / wl
    # E_sc(0) = i·f·∫ q/k_z·√(k_z/k) dq over the aperture (aplanatic), for |a| = 1
    e_sc = 1j * f * (2.0 / 3.0) * (k**1.5 - kz**1.5) / math.sqrt(k)
    expected = 2 * float(e_sc.real) + float(abs(e_sc) ** 2)
    pixel = 0.065**2
    centre = (image[16, 16] - background[16, 16]) / pixel
    assert float(centre) == pytest.approx(expected, rel=0.03)  # pixel averaging over the peak


def test_background_is_uniform_and_carries_the_irradiance():
    camera, objective, _beads = _scene()
    medium = gx.env.Homogeneous(1.33)
    waves = gx.light.PlaneWave(0.532, irradiance=2.0)()
    element = gx.imaging.Coherent(objective, camera)
    background = element((), waves, medium).data
    assert torch.allclose(background, torch.full_like(background, 2.0 * 0.065**2), rtol=1e-6)
    # an oblique wave delivers I·cosθ (aplanatic); epi and evanescent waves deliver nothing
    tilted = gx.light.PlaneWave(0.532, irradiance=2.0, direction=torch.tensor([0.6, 0.0]))()
    cos = math.sqrt(1 - (0.6 / 1.33) ** 2)
    oblique = element((), tilted, medium).data
    assert torch.allclose(oblique, torch.full_like(oblique, 2.0 * cos * 0.065**2), rtol=1e-5)
    epi = gx.light.Uniform(irradiance=2.0, wavelength=0.532)()  # travels away (−z)
    assert float(element((), epi, medium).data.abs().max()) == 0.0
    evanescent = gx.light.PlaneWave(0.532, direction=torch.tensor([1.45, 0.0]))()
    assert float(element((), evanescent, medium).data.abs().max()) == 0.0


def test_coherent_images_are_differentiable_in_position_and_radius():
    camera, objective, beads = _scene(z=0.2, shape=(17, 17))
    position = beads.position.clone().requires_grad_()
    radius = torch.tensor([[0.03]], dtype=torch.float64, requires_grad=True)
    medium = gx.env.Homogeneous(1.33)
    waves = gx.light.Uniform(wavelength=0.532)()
    dipole = gx.interact.Dipole(beads.replace(position=position, radius=radius))
    element = gx.imaging.Coherent(objective, camera, pupil_samples=32)
    image = element(dipole(waves, medium), waves, medium).data
    weights = torch.rand_like(image, generator=torch.Generator().manual_seed(0))
    grads = torch.autograd.grad((image * weights).sum(), (position, radius))
    assert all(torch.isfinite(g).all() and (g != 0).any() for g in grads)


def test_per_image_and_per_frame_optics_align_with_separate_renders():
    camera, objective, beads = _scene(z=0.1, shape=(17, 17))
    beads = gx.stack([beads, beads.replace(position=beads.position + 0.05)])
    waves = gx.light.Uniform(wavelength=0.532)()
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


def _direct_field(rho, dz, n=1.33, na=1.2, wl=0.532):
    """∫₀^{k·NA} J₀(qρ)·exp(i·k_z·Δz)·q/k_z·√(k_z/k) dq by adaptive quadrature (not the MFT)."""
    import numpy as np
    import scipy.integrate
    import scipy.special

    k, qmax = 2 * math.pi * n / wl, 2 * math.pi * na / wl

    def integrand(q, part):
        kz = math.sqrt(k * k - q * q)
        value = scipy.special.j0(q * rho) * np.exp(1j * kz * dz) * q / kz * math.sqrt(kz / k)
        return value.real if part == 0 else value.imag

    re = scipy.integrate.quad(integrand, 0, qmax, args=(0,), limit=400)[0]
    im = scipy.integrate.quad(integrand, 0, qmax, args=(1,), limit=400)[0]
    return complex(re, im)


@pytest.mark.parametrize(("z", "tolerance"), [(0.0, 5e-3), (0.5, 1.5e-2)])
def test_the_imaged_scattered_field_matches_direct_summation(z, tolerance):
    # the pupil-support evaluation (MFT) against the angular-spectrum integral of a point
    # scatterer, E(ρ) = i·f·E_inc·∫ J₀(qρ)·e^{i·k_z·Δz}·q/k_z·√(k_z/k) dq (aplanatic), along a
    # radial line
    camera, objective, beads = _scene(z=z, radius=0.02, na=1.2, shape=(33, 33))
    medium = gx.env.Homogeneous(1.33)
    waves = gx.light.Uniform(wavelength=0.532)()
    spectra = gx.interact.Dipole(beads)(waves, medium)[0]
    params = spectra.params
    assert isinstance(params, gx.interact.DipoleParams)
    f = complex(params.amplitude.reshape(()))
    e_inc = complex(waves.at(spectra.position, torch.tensor([[1.33]]))[..., 0, :].reshape(()))
    xs = ((torch.arange(33, dtype=torch.float64) + 0.5) * 0.065)[None]
    errors = []
    for samples in (256, 512):
        element = gx.imaging.Coherent(objective, camera, oversample=1, pupil_samples=samples)
        field = element._scattered(
            spectra, waves, torch.tensor([1.33]), torch.tensor([1.2]), torch.zeros(1, 1),
            torch.tensor([[0.532]], dtype=torch.float64), xs, xs, 1.2,
        )  # fmt: skip
        row = field[0, 0, 0, 0, 16, 16:]
        ref = torch.tensor([1j * f * e_inc * _direct_field(j * 0.065, -z) for j in range(17)])
        errors.append(float((row - ref).abs().max() / ref.abs().max()))
    assert errors[0] < tolerance
    assert errors[1] < 0.6 * errors[0]  # converges with the pupil grid (soft edge: O(du))


def test_the_imaged_scattered_power_is_the_power_collected_by_the_aperture():
    # aplanatic imaging conserves power: Σ|E_sc|² over the image equals I·|f|²·2π(1 − cosθ_max)
    # (without the √cosθ factor it read ⟨1/cosθ⟩ ≈ 1.48× too high at NA 1.2)
    camera = gx.Camera(pixel_size=6.5, shape=(97, 97))
    objective = gx.Objective(NA=1.2, magnification=100)
    centre = 97 * 0.065 / 2
    beads = gx.Spheres(
        position=torch.tensor([[[centre, centre, 0.0]]], dtype=torch.float64),
        radius=torch.tensor([[0.02]], dtype=torch.float64),
        material=1.59,
    )
    medium = gx.env.Homogeneous(1.33)
    epi = gx.light.Uniform(irradiance=1.0, wavelength=0.532)()  # no background reaches the camera
    contributions = gx.interact.Dipole(beads)(epi, medium)
    image = gx.imaging.Coherent(objective, camera, oversample=2, pupil_samples=96)(
        contributions, epi, medium
    ).data
    params = contributions[0].params
    assert isinstance(params, gx.interact.DipoleParams)
    f = complex(params.amplitude.reshape(()))
    collected = abs(f) ** 2 * 2 * math.pi * (1 - math.sqrt(1 - (1.2 / 1.33) ** 2))
    assert float(image.sum()) == pytest.approx(collected, rel=0.04)  # the tail leaves the frame
