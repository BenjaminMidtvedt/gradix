"""Bounds through reconstructions (ADR-44, M3a phase 5): gx.crlb(..., reconstruction=R)."""

import math

import pytest
import torch

import gradix as gx

WL, NA, MAG, PIXEL = 0.532, 0.5, 60.0, 6.5
PITCH = PIXEL / MAG
F64 = torch.float64
WRT = ("beads.radius", "beads.position.x", "beads.position.z")


def offaxis(shape=32, cycles=9, irradiance=1e3):
    """A bead under an off-axis reference; every value that matters is a float64 tensor."""
    u = WL * cycles / (shape * PITCH)  # the carrier on the DFT grid
    angle = math.asin(u / MAG)
    camera = gx.Camera(pixel_size=PIXEL, shape=(shape, shape), noise=gx.noise.Ideal(), unit="e")
    objective = gx.Objective(NA=NA, magnification=MAG)
    c = shape * PITCH / 2
    beads = gx.Spheres(
        position=torch.tensor([[[c + 0.01, c - 0.02, 0.5]]], dtype=F64),
        radius=torch.tensor([[[0.15]]], dtype=F64),
        material=1.59,
    )
    scope = gx.presets.OffAxisHolography(
        light=gx.light.PlaneWave(WL, irradiance=irradiance),
        objective=objective,
        camera=camera,
        angle=torch.tensor([angle, angle], dtype=F64),
        reference_irradiance=torch.tensor(irradiance, dtype=F64),
    )
    return gx.Chain(
        light=scope.light,
        scatterers={"beads": gx.interact.Mie(beads)},
        references=dict(scope.references),
        imaging=gx.imaging.Coherent(objective, camera),
        environment=gx.env.Homogeneous(1.33),
    )


def dense(pipe, chain, recon):
    """The information through R by dense linear algebra: R as a matrix, C⁺ by pinv."""
    frames = pipe(chain).expected.double()
    shape = frames.shape
    count = frames[0].numel()
    zero = recon.linear(torch.zeros_like(frames))
    rows = []
    for k in range(count):
        e = torch.zeros(count, dtype=F64)
        e[k] = 1.0
        rows.append((recon.linear(e.reshape(shape)) - zero).reshape(-1))
    r = torch.stack(rows, -1)  # [M, K]
    r = torch.cat([r.real, r.imag]) if r.is_complex() else r
    variance = pipe.template.camera.variance(frames).reshape(-1)
    columns = []
    for path in WRT:
        name, _, component = path.rpartition(".") if path.endswith((".x", ".z")) else (path, "", "")
        value = gx.tree.get(chain, chain.resolve(name)[0])
        assert isinstance(value, torch.Tensor)
        tangent = torch.zeros_like(value)
        if component:
            tangent[..., "xyz".index(component)] = 1.0
        else:
            tangent.fill_(1.0)

        def render(v, name=name):
            return pipe(gx.tree.replace(chain, {chain.resolve(name)[0]: v})).expected

        columns.append(torch.func.jvp(render, (value,), (tangent,))[1].reshape(-1).double())
    jacobian = torch.stack(columns, -1)
    rj = r @ jacobian
    covariance = (r * variance) @ r.T
    return rj.T @ torch.linalg.pinv(covariance, rtol=1e-12, hermitian=True) @ rj


def test_the_bound_through_offaxis_matches_dense_linear_algebra():
    chain = offaxis()
    pipe = gx.Pipeline(chain, outputs=("expected",))
    recon = gx.recon.OffAxis.from_chain(chain)
    bound = gx.crlb(pipe, chain, wrt=WRT, reconstruction=recon)
    reference = dense(pipe, chain, recon)
    scale = reference.diagonal().sqrt()
    error = (bound.fisher[0] - reference).abs() / (scale[:, None] * scale[None, :])
    assert float(error.max()) < 1e-6  # as built 3e-15
    assert "through OffAxis" in bound.explain()


@pytest.mark.parametrize("kind", ["offaxis", "inline", "qwlsi"])
def test_a_reconstruction_never_adds_information(kind):
    if kind == "offaxis":
        chain = offaxis()
        recon = gx.recon.OffAxis.from_chain(chain)
    else:
        camera = gx.Camera(pixel_size=PIXEL, shape=(32, 32), noise=gx.noise.Ideal(), unit="e")
        c = 32 * PITCH / 2
        beads = gx.Spheres(
            position=torch.tensor([[[c + 0.01, c - 0.02, 0.3]]], dtype=F64),
            radius=torch.tensor([[[0.15]]], dtype=F64),
            material=1.59,
        )
        stages = {}
        if kind == "qwlsi":
            stages = {"grating": gx.optics.Grating(period=4 * PIXEL, distance=300.0)}
        chain = gx.Chain(
            light=gx.light.PlaneWave(WL, irradiance=1e3),
            scatterers={"beads": gx.interact.Mie(beads)},
            imaging=gx.imaging.Coherent(gx.Objective(NA=NA, magnification=MAG), camera),
            detection_optics=stages,
            environment=gx.env.Homogeneous(1.33),
        )
        if kind == "inline":
            # periodic: padding and cropping leave a continuum of small singular values,
            # which conjugate gradients resolve slowly (the bound stays a lower bound)
            recon = gx.recon.Inline.from_chain(chain, distance=0.3, normalize=True, pad=0)
        else:
            recon = gx.recon.QWLSI.from_chain(chain)
    pipe = gx.Pipeline(chain, outputs=("expected",))
    raw = gx.crlb(pipe, chain, wrt=WRT).fisher[0]
    kept = gx.crlb(pipe, chain, wrt=WRT, reconstruction=recon).fisher[0]
    lost = torch.linalg.eigvalsh(raw - kept)
    assert float(lost.min()) > -1e-9 * float(raw.diagonal().max())
    assert bool((kept.diagonal() > 0).all())


def test_an_invertible_reconstruction_keeps_everything():
    chain = offaxis(shape=24, cycles=7)
    pipe = gx.Pipeline(chain, outputs=("expected",))
    raw = gx.crlb(pipe, chain, wrt=WRT).fisher[0]
    reference = pipe(chain).expected.detach()
    for recon in (gx.recon.ISCATContrast(reference=reference), lambda frames: 3.0 * frames):
        kept = gx.crlb(pipe, chain, wrt=WRT, reconstruction=recon).fisher[0]
        scale = raw.diagonal().sqrt()
        error = (kept - raw).abs() / (scale[:, None] * scale[None, :])
        assert float(error.max()) < 1e-8


def test_the_bound_is_differentiable_through_the_solver():
    chain = offaxis()
    pipe = gx.Pipeline(chain, outputs=("expected",))
    recon = gx.recon.OffAxis.from_chain(chain)
    reference = chain.references["reference"]

    def information(irradiance):
        moved = chain.replace(references={"reference": reference.replace(irradiance=irradiance)})
        bound = gx.crlb(pipe, moved, wrt=WRT[:1], reconstruction=recon, tolerance=1e-12)
        return bound.fisher[0, 0, 0]

    irradiance = torch.tensor(1e3, dtype=F64, requires_grad=True)
    (grad,) = torch.autograd.grad(information(irradiance), irradiance)
    with torch.no_grad():
        plus = information(torch.tensor(1e3 + 1.0, dtype=F64))
        minus = information(torch.tensor(1e3 - 1.0, dtype=F64))
    assert float(grad) == pytest.approx(float(plus - minus) / 2.0, rel=1e-4)  # as built 3e-7


def test_a_reconstruction_that_calibrates_on_each_frame_is_refused():
    chain = offaxis(shape=16, cycles=5)
    with pytest.raises(gx.StructureError, match="not affine"):
        gx.crlb(chain, wrt=WRT[:1], reconstruction=gx.recon.ISCATContrast())


def test_the_solver_warns_when_it_stops_early():
    chain = offaxis()
    recon = gx.recon.OffAxis.from_chain(chain)
    with pytest.warns(gx.GradixWarning, match="conjugate gradients stopped"):
        early = gx.crlb(chain, wrt=WRT[:1], reconstruction=recon, iterations=3)
    assert any("underestimated" in note for note in early.notes)
    full = gx.crlb(chain, wrt=WRT[:1], reconstruction=recon)
    assert float(early.fisher[0, 0, 0]) <= float(full.fisher[0, 0, 0])  # a lower bound


def test_nearly_degenerate_radius_and_index_keep_finite_bounds():
    # a 100 nm sphere: radius and index enter mostly through the dipole α ∝ r³(m² − 1)/(m² + 2)
    camera = gx.Camera(pixel_size=PIXEL, shape=(32, 32), noise=gx.noise.Ideal(), unit="e")
    c = 32 * PITCH / 2
    beads = gx.Spheres(
        position=torch.tensor([[[c + 0.01, c - 0.02, 0.0]]], dtype=F64),
        radius=torch.tensor([[[0.05]]], dtype=F64),
        material=torch.tensor([[[1.59]]], dtype=F64),
    )
    chain = gx.Chain(
        light=gx.light.PlaneWave(WL, irradiance=torch.tensor(1e4, dtype=F64)),
        scatterers={"beads": gx.interact.Mie(beads)},
        imaging=gx.imaging.Coherent(gx.Objective(NA=0.8, magnification=100.0), camera),
        environment=gx.env.Homogeneous(1.33),
    )
    pipe = gx.Pipeline(chain, outputs=("expected",))
    wrt = ("beads.radius", "beads.material")
    raw = gx.crlb(pipe, chain, wrt=wrt)
    recon = gx.recon.Inline.from_chain(chain, normalize=True, pad=0)
    kept = gx.crlb(pipe, chain, wrt=wrt, reconstruction=recon)
    for path in wrt:
        assert torch.isfinite(kept[path]).all(), path
        assert bool((kept[path] >= raw[path]).all()), path  # a reconstruction never helps


def test_independent_measurements_add_their_information():
    chain = offaxis(shape=24, cycles=7)
    pipe = gx.Pipeline(chain, outputs=("expected",))
    wrt = ("beads.radius", "beads.material")
    recon = gx.recon.OffAxis.from_chain(chain)
    radius = torch.tensor([[[0.15]]], dtype=F64, requires_grad=True)
    material = torch.tensor([[[1.59]]], dtype=F64)
    beads = chain.population("beads").replace(radius=radius, material=material)
    moved = chain.replace(scatterers={"beads": gx.interact.Mie(beads)})
    once = gx.crlb(pipe, moved, wrt=wrt, reconstruction=recon)
    twice = once + once  # two identical exposures: half the variance
    for path in wrt:
        assert torch.allclose(twice[path], once[path] / math.sqrt(2.0), rtol=1e-10)
    total = sum([once, once])
    assert isinstance(total, gx.CRLB) and total.measurements == 2
    assert "2 measurements" in twice.explain()
    (grad,) = torch.autograd.grad(twice["beads.radius"].sum(), radius)
    assert torch.isfinite(grad).all()
    other = gx.crlb(pipe, moved, wrt=wrt[:1], reconstruction=recon)
    with pytest.raises(gx.StructureError, match="same parameters"):
        _ = once + other
