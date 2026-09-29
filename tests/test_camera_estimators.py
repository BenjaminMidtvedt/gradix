"""The camera estimator suite (§11.6) and the M0 exit criterion on learnable photon statistics."""

import math

import pytest
import torch
from scenes import DEVICES, optics

import gradix as gx
from gradix.testing import camera_estimators


@pytest.fixture(scope="module")
def suite():
    return camera_estimators.run(draws=200_000)


def _cell(results, estimator, loss, lam):
    return next(r for r in results if r.estimator == estimator and r.loss == loss and r.lam == lam)


@pytest.mark.parametrize("lam", [0.2, 1.0, 3.0])
def test_plain_straight_through_is_flagged(suite, lam):
    cell = _cell(suite, "straight_through", "quadratic", lam)
    assert abs(cell.estimate) < 0.05 and cell.truth == pytest.approx(1.0, abs=1e-3)
    assert not cell.consistent()


@pytest.mark.parametrize("lam", [0.2, 1.0, 3.0])
@pytest.mark.parametrize("loss", ["mean", "quadratic"])
def test_scaled_straight_through_is_unbiased_up_to_quadratic(suite, lam, loss):
    assert _cell(suite, "scaled_st", loss, lam).consistent()


@pytest.mark.parametrize("loss", ["anscombe", "log1p"])
def test_scaled_straight_through_bias_on_transforms_is_bounded(suite, loss):
    for lam in (1.0, 3.0):
        cell = _cell(suite, "scaled_st", loss, lam)
        assert abs(cell.bias) <= 0.10 * abs(cell.truth) + 5 * cell.stderr  # the declared +2 … +10 %


def test_estimators_are_finite_at_zero_rate(suite):
    for cell in suite:
        if cell.lam == 0.0:
            assert math.isfinite(cell.estimate)


@pytest.mark.parametrize("device", DEVICES)
def test_lognormal_photon_statistics_get_gradients_through_the_image(device):
    """M0 exit: non-zero, finite gradients reach a torch-sampled LogNormal median and σ."""
    b, n = 16, 8
    g = torch.Generator(device="cpu").manual_seed(0)
    log_median = torch.nn.Parameter(torch.tensor(math.log(300.0), device=device))
    sigma = torch.nn.Parameter(torch.tensor(0.4, device=device))
    eps = torch.randn(b, n, generator=g).to(device)
    photons = torch.exp(log_median + sigma * eps)  # reparameterised LogNormal draws
    pos = torch.cat([0.5 + 3.0 * torch.rand(b, n, 2, generator=g), torch.zeros(b, n, 1)], -1).to(
        device
    )
    beads = gx.Emitters(position=pos, photons=photons, emission=gx.Spectrum.line(0.6))
    objective, camera = optics(shape=(32, 32), noise=gx.noise.PoissonGaussian(read=1.0))
    chain = gx.Chain(
        emitters={"beads": beads},
        imaging=gx.imaging.Sprites(objective, camera),
        environment=gx.env.Homogeneous(1.33),
    )
    image = gx.Pipeline(chain)(chain, key=torch.arange(b, device=device))["image"]
    # a variance-sensitive statistic of the images (plain ST would give exactly zero for σ)
    loss = ((image - image.mean((-2, -1), keepdim=True).detach()) ** 2).mean()
    loss.backward()
    for p in (log_median, sigma):
        assert p.grad is not None and torch.isfinite(p.grad) and p.grad != 0
    # λ = 0 pixels (dark background) keep the gradient finite
    assert torch.isfinite(image).all()


# ---- numerics audit regressions: the camera path itself ----------------------------------


def test_scaled_st_through_the_camera_has_gradient_at_zero_rate():
    camera = gx.Camera(pixel_size=6.5, shape=(8, 8), noise=gx.noise.Ideal(), unit="e")
    mu = torch.zeros(4, 1, 8, 8, requires_grad=True)
    image = camera.sample(mu, key=torch.arange(4))
    (grad,) = torch.autograd.grad(image.sum(), mu)
    assert torch.equal(grad, torch.ones_like(grad))  # g = 1 at λ = 0, so dark pixels can turn on


def test_flat_field_variance_separates_scaled_from_plain_straight_through():
    # For a uniform field, plain ST gives d Var/dλ = 0 exactly; scaled-ST recovers dVar/dλ = 1.
    keys = torch.arange(64)
    grads = {}
    for estimator in ("scaled_st", "straight_through"):
        camera = gx.Camera(
            pixel_size=6.5, shape=(32, 32), noise=gx.noise.Ideal(estimator=estimator), unit="e"
        )
        lam = torch.tensor(5.0, requires_grad=True)
        image = camera.sample(lam.expand(64, 1, 32, 32), key=keys)
        (grads[estimator],) = torch.autograd.grad(image.var(), lam)
    assert abs(float(grads["straight_through"])) < 1e-6
    assert abs(float(grads["scaled_st"]) - 1.0) < 0.1


def test_adc_gradient_survives_rounding_onto_zero():
    camera = gx.Camera(
        pixel_size=6.5,
        shape=(64, 64),
        noise=gx.noise.PoissonGaussian(read=1.5),
        bit_depth=12,
    )
    mu = torch.full((16, 1, 64, 64), 0.5, requires_grad=True)
    image = camera.sample(mu, key=torch.arange(16))
    (grad,) = torch.autograd.grad(image.sum(), mu)
    clipped = (image.detach() <= 0).float().mean()
    zero_grad = (grad == 0).float().mean()
    assert zero_grad <= clipped + 1e-6  # only pixels below 0 before rounding lose gradient


def test_low_light_offset_fit_is_unbiased_with_the_default_likelihood():
    camera = gx.Camera(
        pixel_size=6.5, shape=(128, 128), noise=gx.noise.PoissonGaussian(read=1.5), offset=100.0
    )
    mu = torch.full((8, 1, 128, 128), 100.2)  # 0.2 e over the offset
    observed = camera.sample(mu, key=torch.arange(8))
    offsets = torch.linspace(99.0, 101.0, 81)
    scores = [
        float(camera.replace(offset=o).log_prob(observed, mu - 100.0 + o).sum()) for o in offsets
    ]
    best = float(offsets[int(torch.tensor(scores).argmax())])
    assert abs(best - 100.0) <= 0.05


def test_poisson_log_prob_is_minus_infinity_below_zero_rate():
    from gradix.ops.detect import poisson_log_prob

    k = torch.tensor([0.0, 0.0, 1.0, 3.0])
    assert torch.isinf(poisson_log_prob(k, torch.full((4,), -1e4))).all()
    lam = torch.tensor([0.0, 0.0], requires_grad=True)
    value = poisson_log_prob(torch.tensor([0.0, 2.0]), lam)
    assert value[0] == 0 and torch.isinf(value[1])
    (grad,) = torch.autograd.grad(value[0], lam)
    assert torch.isfinite(grad).all()


def test_scaled_st_known_limits_at_low_rates(suite):
    """§6.2's "unbiased up to quadratic" holds for λ > 0; at λ = 0 the quadratic loss reads 0."""
    at_zero = _cell(suite, "scaled_st", "quadratic", 0.0)
    assert abs(at_zero.estimate) < 1e-9 and at_zero.truth == pytest.approx(1.0, abs=1e-3)
    for loss, bound in (("anscombe", 0.14), ("log1p", 0.12)):
        cell = _cell(suite, "scaled_st", loss, 0.2)  # measured +9.7 % and +7.6 %
        assert abs(cell.bias) <= bound * abs(cell.truth) + 5 * cell.stderr


def test_gradient_probe_flags_plain_straight_through():
    from gradix.testing.gradprobe import gradient_probe

    verdicts = {}
    for estimator in ("scaled_st", "straight_through"):
        objective, camera = optics(noise=gx.noise.PoissonGaussian(read=1.0, estimator=estimator))
        photons = torch.full((2, 3), 800.0, requires_grad=True)
        beads = gx.Emitters(
            position=torch.tensor([[[2.0, 2.0, 0.0], [3.0, 5.0, 0.0], [6.0, 6.0, 0.0]]] * 2),
            photons=photons,
            emission=gx.Spectrum.line(0.6),
        )
        chain_ = gx.Chain(
            emitters={"beads": beads},
            imaging=gx.imaging.Sprites(objective, camera),
            environment=gx.env.Homogeneous(1.33),
        )
        report = gradient_probe(gx.Pipeline(chain_, outputs=("image",)), chain_, key=3)
        assert report["linear"]["emitters.beads.photons"] == "ok"
        verdicts[estimator] = report["noise_energy"]["camera.noise"]
    assert verdicts == {"scaled_st": "ok", "straight_through": "zero"}
