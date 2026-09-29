"""gx.crlb: Cramér–Rao bounds from forward-mode Jacobians and the camera's Fisher information."""

import importlib
from typing import Any

import pytest
import torch

import gradix as gx

crlb_module = importlib.import_module("gradix.api.crlb")  # the package attribute is the function


def scene(
    *,
    b=3,
    n=1,
    dtype=torch.float64,
    na=0.5,  # NA/n = 0.38 in water: below the scalar-PSF warning
    photons=40000.0,  # emitted; about 1300 reach the camera at NA 0.5 in water
    background=4.0,
    noise=None,
    pupil=(),
    presence=None,
    material=None,
):
    camera = gx.Camera(pixel_size=6.5, shape=(16, 16), noise=noise or gx.noise.Ideal(), unit="e")
    g = torch.Generator().manual_seed(1)
    xy = 1.0 + 0.6 * torch.rand(b, n, 2, generator=g, dtype=dtype)
    z = 0.8 * torch.rand(b, n, 1, generator=g, dtype=dtype) - 0.4
    beads = gx.Emitters(
        position=torch.cat([xy, z], -1),
        photons=photons,
        presence=presence,
        material=material,
        emission=gx.Spectrum.line(0.6),
    )
    psf = gx.imaging.PointPSF(
        gx.Objective(NA=na, magnification=40, pupil=pupil), camera, psf="scalar", method="global"
    )
    return gx.Chain(
        emitters={"beads": beads},
        imaging=psf,
        environment=gx.env.Homogeneous(1.33),
        background=background,
    )


def expected(pipe, chain, changes):
    """Render the expected frames with values replaced by logical path."""
    tree = {chain.resolve(path)[0]: value for path, value in changes.items()}
    return pipe(gx.tree.replace(chain, tree)).expected


def test_the_bound_inverts_the_poisson_fisher_information_of_one_image():
    chain = scene(b=1)
    pipe = gx.Pipeline(chain, outputs=("expected",))
    position = chain.emitters["beads"].position
    bound = gx.crlb(pipe, chain, wrt="beads.position")
    mu = expected(pipe, chain, {})
    jac = torch.func.jacfwd(lambda p: expected(pipe, chain, {"beads.position": p}))(position)
    jac = jac.reshape(*mu.shape, 3)
    info = torch.einsum("...i,...j->ij", jac / mu[..., None], jac)  # Poisson: Var = μ
    by_hand = torch.linalg.inv(info).diagonal().sqrt()
    assert torch.allclose(bound["beads.position"].reshape(3), by_hand, rtol=1e-8)
    assert bound.method == "forward" and bound.labels[2] == "beads.position[0].z"
    assert 0.0 < float(by_hand[0]) < 0.05  # µm: tens of nm for ~1300 detected photons


def test_every_image_is_its_own_problem():
    chain = scene(b=3, n=2)  # two coupled emitters per image
    pipe = gx.Pipeline(chain, outputs=("expected",))
    together = gx.crlb(pipe, chain, wrt="beads.position")
    assert together["beads.position"].shape == (3, 2, 3)
    assert together.fisher.shape == (3, 6, 6) and together.shared == 0
    for i in range(3):
        alone = gx.crlb(pipe, chain.select(slice(i, i + 1)), wrt="beads.position")
        assert torch.allclose(together["beads.position"][i], alone["beads.position"][0])


def test_shared_values_are_informed_by_every_image():
    # photons and background are numbers, so each is one parameter of the whole batch
    chain = scene(b=3)
    pipe = gx.Pipeline(chain, outputs=("expected",))
    wrt = ("beads.position", "beads.photons", "background")
    bound = gx.crlb(pipe, chain, wrt=wrt)
    assert bound.shared == 2 and bound.labels[-2:] == ("beads.photons", "background")
    assert bound["beads.photons"].shape == () and bound["background"].shape == ()

    def render(theta):  # every parameter of the batch in one vector
        changes = {"beads.position": theta[:9].reshape(3, 1, 3)}
        changes |= {"beads.photons": theta[9], "background": theta[10]}
        return expected(pipe, chain, changes)

    position = chain.emitters["beads"].position
    theta = torch.cat([position.reshape(-1), torch.tensor([40000.0, 4.0], dtype=torch.float64)])
    mu = render(theta)
    jac = torch.func.jacfwd(render)(theta)  # [3, 1, 16, 16, 11]
    info = torch.einsum("...i,...j->ij", jac / mu[..., None], jac)
    covariance = torch.linalg.inv(info)
    std = covariance.diagonal().sqrt()
    assert torch.allclose(bound["beads.position"].reshape(-1), std[:9], rtol=1e-6)
    assert torch.allclose(bound["beads.photons"], std[9], rtol=1e-6)
    assert torch.allclose(bound["background"], std[10], rtol=1e-6)
    for i in range(3):  # each image's marginal: its position with the shared parameters
        rows = [3 * i, 3 * i + 1, 3 * i + 2, 9, 10]
        marginal = covariance[rows][:, rows]
        assert torch.allclose(bound.covariance[i], marginal, rtol=1e-6, atol=1e-12)


def test_only_shared_parameters():
    chain = scene(b=4)
    pipe = gx.Pipeline(chain, outputs=("expected",))
    bound = gx.crlb(pipe, chain, wrt="beads.photons")
    per_image = bound.fisher[:, 0, 0]  # each image's information about the one photon count
    assert torch.allclose(bound["beads.photons"], per_image.sum().rsqrt(), rtol=1e-6)


def test_components_bound_one_coordinate_with_the_others_known():
    chain = scene(b=2)
    pipe = gx.Pipeline(chain, outputs=("expected",))
    full = gx.crlb(pipe, chain, wrt="beads.position")
    z = gx.crlb(pipe, chain, wrt="beads.position.z")
    assert z["beads.position.z"].shape == (2, 1)
    assert torch.allclose(z["beads.position.z"][:, 0], full.fisher[:, 2, 2].rsqrt(), rtol=1e-8)
    assert z["beads.position.z"][0, 0] < full["beads.position"][0, 0, 2]  # knowing x, y helps
    xz = gx.crlb(pipe, chain, wrt=("beads.position.x", "beads.position.z"))
    assert xz.labels == ("beads.position[0].x", "beads.position[0].z")
    with pytest.raises(gx.StructureError, match="overlap"):
        gx.crlb(pipe, chain, wrt=("beads.position", "beads.position.z"))
    with pytest.raises(gx.StructureError, match="component"):
        gx.crlb(pipe, chain, wrt="beads.position.w")


def test_absent_objects_have_no_information():
    presence = torch.tensor([[1.0, 1.0], [1.0, 0.0]], dtype=torch.float64)
    chain = scene(b=2, n=2, presence=presence)
    bound = gx.crlb(chain, wrt="beads.position")
    std = bound["beads.position"]
    assert torch.isinf(std[1, 1]).all() and torch.isfinite(std[0]).all()
    alone = torch.linalg.inv(bound.fisher[1, :3, :3]).diagonal().sqrt()  # the absent one adds 0
    assert torch.allclose(std[1, 0], alone)


def test_central_differences_agree_with_forward_mode():
    chain = scene(b=2)
    pipe = gx.Pipeline(chain, outputs=("expected",))
    wrt = ("beads.position", "beads.photons")
    forward = gx.crlb(pipe, chain, wrt=wrt)
    central = gx.crlb(pipe, chain, wrt=wrt, method="central")
    assert central.method == "central"
    for path in wrt:
        assert torch.allclose(central[path], forward[path], rtol=1e-5)


def test_auto_falls_back_to_central_differences(monkeypatch):
    def unavailable(render, f, k):
        raise NotImplementedError("Trying to use forward AD with an op that does not support it")

    monkeypatch.setattr(crlb_module, "_forward", unavailable)
    chain = scene(b=1)
    with pytest.warns(gx.GradixWarning, match="central differences"):
        bound = gx.crlb(chain, wrt="beads.position")
    assert bound.method == "central"
    with pytest.raises(NotImplementedError):
        gx.crlb(chain, wrt="beads.position", method="forward")


def test_bounds_are_differentiable_for_design():
    def total(coeffs):
        pupil = (gx.pupil.Zernike(coeffs=coeffs, indices=(5,)),)
        return gx.crlb(scene(b=2, pupil=pupil), wrt="beads.position")["beads.position"].sum()

    coeffs = torch.tensor([0.08], dtype=torch.float64, requires_grad=True)
    total(coeffs).backward()  # reverse mode over the forward-mode Jacobian
    assert coeffs.grad is not None
    h = 1e-5
    with torch.no_grad():
        fd = (total(coeffs + h) - total(coeffs - h)) / (2 * h)
    assert float(coeffs.grad) == pytest.approx(float(fd), rel=1e-4)


def test_values_by_path_plans_and_float32():
    chain = scene(b=2, dtype=torch.float32)
    pipe = gx.Pipeline(chain, outputs=("image",))  # the Pipeline's own outputs do not matter
    moved = chain.emitters["beads"].position + torch.tensor([0.1, 0.0, 0.1])
    by_path = gx.crlb(pipe, {"beads.position": moved}, wrt="beads.position")
    direct = gx.crlb(
        gx.tree.replace(chain, {"emitters.beads.position": moved}), wrt="beads.position"
    )
    assert by_path["beads.position"].dtype == torch.float32
    assert torch.allclose(by_path["beads.position"], direct["beads.position"], rtol=1e-4)
    sample = gx.Sample({"beads": chain.emitters["beads"]}, environment=gx.env.Homogeneous(1.33))
    camera = gx.Camera(pixel_size=6.5, shape=(16, 16), noise=gx.noise.Ideal(), unit="e")
    scope = gx.presets.Widefield(objective=gx.Objective(NA=0.5, magnification=40), camera=camera)
    planned = gx.plan(sample, scope, outputs=("expected",))
    bound = gx.crlb(planned, planned.chain(sample, scope), wrt="beads.position")
    assert bound["beads.position"].shape == (2, 1, 3)
    assert torch.isfinite(bound["beads.position"]).all()


def test_the_report_and_the_noise_note():
    chain = scene(b=2, noise=gx.noise.PoissonGaussian(read=1.5))
    bound = gx.crlb(chain, wrt=("beads.position", "background"))
    report = bound.explain()
    assert "1 shared" in report and "forward mode" in report and "beads.position" in report
    assert any("Gaussian" in note for note in bound.notes)
    assert set(bound) == {"beads.position", "background"} and len(bound) == 2
    assert bound.units["beads.position"] == "µm"
    with pytest.raises(KeyError, match="bounds:"):
        bound["beads.photons"]


def test_scalar_psfs_at_high_na_warn():
    chain = scene(b=1, na=1.2)
    with pytest.warns(gx.GradixWarning, match="vectorial"):
        bound = gx.crlb(chain, wrt="beads.position")
    assert any("NA/n = 0.902" in note for note in bound.notes)


def test_refusals():
    untyped: Any = "reverse"  # as an untyped caller would pass it
    chain = scene(b=2)
    with pytest.raises(gx.StructureError, match="no field"):
        gx.crlb(chain, wrt="beads.positon")
    with pytest.raises(gx.StructureError, match="wrt names no field"):
        gx.crlb(chain, wrt=())
    with pytest.raises(gx.StructureError, match="unknown method"):
        gx.crlb(chain, wrt="beads.position", method=untyped)
    with pytest.raises(gx.StructureError, match="alone"):
        gx.crlb(chain, chain, wrt="beads.position")
    with pytest.raises(gx.GradientPathError, match="no image informs it"):
        gx.crlb(scene(b=2, material=1.4), wrt="beads.material")
    noiseless = gx.tree.replace(chain, {"imaging.camera.noise": None})
    with pytest.raises(gx.StructureError, match="noise model"):
        gx.crlb(noiseless, wrt="beads.position")


def test_expanded_values_are_bounded():
    # a value broadcast with .expand() shares memory across images; forward mode needs a copy
    chain = scene(b=3)
    photons = torch.tensor(1000.0).expand(3, 1)
    beads = chain.population("beads").replace(photons=photons)
    chain = chain.replace(emitters={"beads": beads})
    bound = gx.crlb(chain, wrt="beads.photons")
    assert torch.isfinite(bound["beads.photons"]).all()
