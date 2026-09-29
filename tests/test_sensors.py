"""M1 sensors (§5.5): EMCCD, sCMOS calibration maps and Fisher information."""

import math

import numpy as np
import pytest
import torch
from scenes import emitters, optics, tensor

import gradix as gx


def test_emccd_mean_and_excess_noise():
    camera = gx.Camera(
        pixel_size=6.5, shape=(64, 64), noise=gx.noise.EMCCD(gain=50.0, cic=0.01), unit="e"
    )
    lam = torch.full((8, 1, 64, 64), 4.0)
    irradiance = gx.Irradiance(data=lam[:, :, None], grid=gx.Grid2D((64, 64), 0.1))
    expected = camera.expected(irradiance)
    assert torch.allclose(expected, torch.full_like(expected, 50.0 * 4.01))
    image = camera.sample(expected, key=5)
    mean, var = float(image.mean()), float(image.var())
    assert mean == pytest.approx(50.0 * 4.01, rel=0.02)
    assert var == pytest.approx(2.0 * 50.0**2 * 4.01, rel=0.05)  # excess noise factor √2
    with pytest.raises(gx.StructureError, match="batch-keyed"):
        camera.sample(expected, key=torch.arange(8))


def test_emccd_gradients_are_unbiased_at_low_light():
    camera = gx.Camera(pixel_size=6.5, shape=(64, 64), noise=gx.noise.EMCCD(gain=30.0), unit="e")
    for level in (0.2, 1.0, 3.0):
        lam = torch.tensor(level, requires_grad=True)
        assert camera.noise is not None
        mu = camera.noise.mean(lam.expand(32, 1, 64, 64))
        image = camera.sample(mu, key=11)
        (grad,) = torch.autograd.grad(image.mean(), lam)
        assert float(grad) == pytest.approx(30.0, rel=0.05)  # dE[y]/dλ = G


def test_scmos_camera_from_calibration_maps(tmp_path):
    g = np.random.default_rng(0)
    offset = 100.0 + g.random((32, 40)).astype(np.float32)
    gain = 2.0 + 0.1 * g.random((32, 40)).astype(np.float32)
    variance = (1.5 * gain) ** 2  # read noise 1.5 e in ADU²
    path = tmp_path / "maps.npz"
    np.savez(path, offset=offset, gain=gain, variance=variance)
    camera = gx.Camera.scmos(maps=path, pixel_size=6.5, roi=(4, 6, 16, 20))
    assert camera.shape == (16, 20)
    assert isinstance(camera.noise, gx.noise.SCMOS)
    assert torch.allclose(tensor(camera.noise.read), torch.full((16, 20), 1.5), atol=1e-5)
    corners = torch.tensor([[0, 0], [10, 12]])
    per_image = gx.Camera.scmos(maps=path, pixel_size=6.5, roi=corners, shape=(8, 8))
    assert tuple(tensor(per_image.gain).shape) == (2, 8, 8)
    assert torch.equal(tensor(per_image.offset)[1], torch.as_tensor(offset[10:18, 12:20]))


def test_fisher_information_of_a_position_matches_the_poisson_formula():
    objective, camera = optics(shape=(32, 32), noise=gx.noise.Ideal(), unit="e")
    beads = emitters(1, 1, fov=4.0)

    def render(position):
        b = beads.replace(position=position)
        chain = gx.Chain(
            emitters={"b": b},
            imaging=gx.imaging.Sprites(objective, camera),
            environment=gx.env.Homogeneous(1.33),
            background=2.0,
        )
        return chain(outputs=("expected",))["expected"]

    position = beads.position.double().clone()
    mu = render(position)
    jac = torch.func.jacfwd(render)(position)  # [1, 1, 32, 32, 1, 1, 3]
    jac = jac.reshape(*mu.shape, 3)
    info = gx.detect.fisher(jac, mu, camera)
    assert info.shape == (1, 3, 3)
    by_hand = ((jac[..., 0] ** 2) / mu).sum()
    assert float(info[0, 0, 0]) == pytest.approx(float(by_hand), rel=1e-6)
    crlb_x = 1.0 / math.sqrt(float(info[0, 0, 0]))
    assert 0.0 < crlb_x < 0.2  # µm: tens of nm for ~100 detected photons over a background
