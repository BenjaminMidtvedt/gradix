"""Dense emission by depth strata (M1: emit.strata_otf): on-plane equivalence, Pipelines, grads."""

import math

import pytest
import torch

import gradix as gx

PITCH = 0.1  # object-space pitch: 6.5 µm pixels at 65×


def _scene(medium, z_planes=(-0.3, 0.0, 0.3), dtype=torch.float64, na=1.2):
    camera = gx.Camera(pixel_size=6.5, shape=(32, 32))
    objective = gx.Objective(NA=na, magnification=65)
    values = torch.zeros(len(z_planes), 20, 20, dtype=dtype)
    values[1, 10, 12] = 1000.0
    values[0, 5, 5] = 500.0
    dz = z_planes[1] - z_planes[0]
    origin = (6.5 * PITCH, 4.5 * PITCH, z_planes[0])  # voxel centres on pixel centres
    cells = gx.Voxels(
        values=values,
        spacing=(dz, PITCH, PITCH),
        origin=origin,
        quantity="density",
        emission=gx.Spectrum.line(0.6),
    )
    points = gx.Emitters(
        position=torch.tensor(
            [
                [
                    [origin[0] + 12 * PITCH, origin[1] + 10 * PITCH, z_planes[1]],
                    [origin[0] + 5 * PITCH, origin[1] + 5 * PITCH, z_planes[0]],
                ]
            ],
            dtype=dtype,
        ),
        photons=torch.tensor([[1000.0, 500.0]], dtype=dtype),
        emission=gx.Spectrum.line(0.6),
    )
    return camera, objective, cells, points


@pytest.mark.parametrize(
    ("medium", "planes"),
    [
        (gx.env.Homogeneous(1.33), (-0.3, 0.0, 0.3)),
        (gx.env.LayeredMedium(sample=1.33), (-0.9, -0.6, -0.3)),
    ],
)
def test_strata_equal_sparse_points_on_their_planes(medium, planes):
    # ladder edge strata OTF ↔ sparse (emitters on strata planes): the same pupil PSFs
    camera, objective, cells, points = _scene(medium, planes, na=1.2)
    strata = gx.imaging.Strata(objective, camera, pupil_samples=128)
    dense = strata(gx.lower.emitter_density(cells), medium).data
    psf = gx.imaging.PointPSF(
        objective, camera, method="global", oversample=1, pupil_samples=128, psf="scalar"
    )
    sparse = psf(gx.lower.emitter_set(points), medium).data
    assert float((dense - sparse).norm() / sparse.norm()) < 1e-10


def test_strata_render_in_chains_and_pipelines_with_gradients():
    medium = gx.env.Homogeneous(1.33)
    camera, objective, cells, _ = _scene(medium, dtype=torch.float32)
    chain = gx.Chain(
        emitters={"cells": cells},
        imaging=gx.imaging.Strata(objective, camera),
        environment=medium,
    )
    eager = chain(outputs=("expected",))["expected"]
    assert eager.shape == (1, 1, 32, 32) and float(eager.sum()) > 0
    pipe = gx.Pipeline(chain, outputs=("expected",), inputs=("cells.values",))
    values = torch.rand(3, 20, 20, requires_grad=True)
    image = pipe({"cells.values": values})["expected"]
    (image * torch.linspace(0, 1, 32)).sum().backward()
    assert values.grad is not None and bool(torch.isfinite(values.grad).all())
    assert float(values.grad.abs().min()) > 0.0  # every voxel reaches the camera
    assert "pitch / voxel spacing" in pipe.explain()


def test_strata_check_their_inputs():
    medium = gx.env.Homogeneous(1.33)
    camera, objective, cells, points = _scene(medium)
    strata = gx.imaging.Strata(objective, camera)
    shifted = cells.replace(origin=(0.63, 0.45, -0.3))  # between fine samples
    with pytest.raises(gx.StructureError, match="between the camera's fine samples"):
        strata(gx.lower.emitter_density(shifted), medium)
    mixed = gx.Chain(emitters={"p": points}, imaging=strata, environment=medium)
    with pytest.raises(gx.PlanError, match=r"renders gx.Voxels"):
        mixed(outputs=("expected",))
    sparse_planes = cells.replace(spacing=(2.0, PITCH, PITCH))
    chain = gx.Chain(emitters={"cells": sparse_planes}, imaging=strata, environment=medium)
    with pytest.warns(gx.GradixWarning, match="axial resolution"):
        gx.Pipeline(chain, outputs=("expected",))
    with pytest.raises(gx.StructureError, match="emission spectrum"):
        gx.Voxels(values=torch.zeros(1, 2, 2), spacing=(1.0, 0.1, 0.1), quantity="density")


def test_a_turbid_slab_attenuates_and_blurs_with_depth_and_keeps_the_power():
    medium = gx.env.Homogeneous(1.33)
    camera, objective, cells, _ = _scene(medium)
    clear = gx.imaging.Strata(objective, camera, pupil_samples=128)
    density = gx.lower.emitter_density(cells)
    reference = clear(density, medium).data
    null = clear.replace(turbid=gx.env.TurbidSlab(attenuation=0.0))
    assert torch.allclose(null(density, medium).data, reference, rtol=1e-10, atol=1e-12)
    mu = torch.tensor(2.0, dtype=torch.float64, requires_grad=True)
    slab = gx.env.TurbidSlab(attenuation=mu, blur=0.2, diffuse=0.2, surface=0.3)
    turbid = clear.replace(turbid=slab)(density, medium).data
    # the scattered light is redistributed, not lost (a narrow halo stays on the camera)
    assert float(turbid.detach().sum() / reference.sum()) == pytest.approx(1.0, abs=5e-3)
    assert float(turbid.detach().max()) < float(reference.max())  # a dimmer ballistic core
    turbid.max().backward()
    assert mu.grad is not None and float(mu.grad) < 0.0  # more scattering, dimmer core


def test_plans_route_densities_to_strata():
    medium = gx.env.Homogeneous(1.33)
    camera, objective, cells, points = _scene(medium, dtype=torch.float32)
    sample = gx.Sample({"cells": cells}, environment=medium)
    scope = gx.presets.Widefield(objective=objective, camera=camera)
    plan = gx.plan(sample, scope, "standard", outputs=("expected",))
    assert plan.routes["cells"].element == "emit.strata_otf"
    chain = plan.chain(sample, scope)
    assert isinstance(chain.imaging, gx.imaging.Strata)
    assert torch.equal(plan(sample, scope)["expected"], plan.pipeline(chain)["expected"])
    mixed = gx.Sample({"cells": cells, "beads": points}, environment=medium)
    with pytest.raises(gx.PlanError, match="different elements"):
        gx.plan(mixed, scope, "standard")


def test_dense_sim_equals_point_sim_on_the_same_labels():
    # ladder edge dense-density SIM (product grid) ↔ EmitterSet SIM: exact for on-grid labels
    medium = gx.env.Homogeneous(1.33)
    camera, objective, cells, points = _scene(medium)
    phase = torch.tensor(0.4, dtype=torch.float64, requires_grad=True)
    light = gx.light.SIMBeams(period=0.3, phase=phase, wavelength=0.488)
    dense = gx.Chain(
        light=light,
        emitters={"cells": cells},
        excite=gx.excite.Linear(),
        imaging=gx.imaging.Strata(objective, camera, pupil_samples=128),
        environment=medium,
    )
    sparse = gx.Chain(
        light=light,
        emitters={"beads": points},
        excite=gx.excite.Linear(),
        imaging=gx.imaging.PointPSF(
            objective, camera, method="global", oversample=1, pupil_samples=128, psf="scalar"
        ),
        environment=medium,
    )
    a = dense(outputs=("expected",))["expected"]
    b = sparse(outputs=("expected",))["expected"]
    assert float((a - b).detach().norm() / b.detach().norm()) < 1e-10
    a.sum().backward()
    assert phase.grad is not None and float(phase.grad) != 0.0  # the pattern moves the image


def test_strata_refuse_supercritical_collection_in_homogeneous_media():
    medium = gx.env.Homogeneous(1.33)
    camera, objective, cells, _ = _scene(medium)
    strata = gx.imaging.Strata(objective.replace(NA=1.4), camera)
    chain = gx.Chain(emitters={"cells": cells}, imaging=strata, environment=medium)
    with pytest.raises(gx.ValidityError, match="supercritical"):
        gx.Pipeline(chain, outputs=("expected",))


def test_sources_far_outside_the_frame_do_not_alias_into_it():
    # the pupil's period is sized from where sources actually are (audit: ghost PSFs)
    medium = gx.env.Homogeneous(1.33)
    camera = gx.Camera(pixel_size=6.5, shape=(32, 32))
    objective = gx.Objective(NA=1.2, magnification=65)
    far = gx.Emitters(
        position=torch.tensor([[[17.1, 1.6, 0.0]]], dtype=torch.float64),
        photons=torch.tensor([[1000.0]], dtype=torch.float64),
        emission=gx.Spectrum.line(0.6),
    )
    psf = gx.imaging.PointPSF(objective, camera, psf="scalar", method="global")
    # its own tail puts 0.010 photons into the frame (converged); ~275 for an emitter inside
    assert float(psf(gx.lower.emitter_set(far), medium).data.sum()) < 0.03
    values = torch.zeros(1, 4, 4, dtype=torch.float64)
    values[0, 0, 0] = 1000.0
    voxel = gx.Voxels(
        values=values,
        spacing=(0.2, PITCH, PITCH),
        origin=(17.05, 1.65, 0.0),
        quantity="density",
        emission=gx.Spectrum.line(0.6),
    )
    strata = gx.imaging.Strata(objective, camera)
    assert float(strata(gx.lower.emitter_density(voxel), medium).data.sum()) < 0.03


def test_densities_render_repeatedly_through_gx_render():
    medium = gx.env.Homogeneous(1.33)
    camera, objective, cells, _ = _scene(medium, dtype=torch.float32)
    sample = gx.Sample({"cells": cells}, environment=medium)
    scope = gx.presets.Widefield(objective=objective, camera=camera)
    for _ in range(3):
        out = gx.render(sample, scope, "standard", outputs=("expected",))
        assert out["expected"].shape == (1, 1, 32, 32)


def test_density_chains_are_checked_before_rendering():
    medium = gx.env.Homogeneous(1.33)
    camera, objective, cells, _ = _scene(medium)
    strata = gx.imaging.Strata(objective, camera)
    index = cells.replace(quantity="n")
    with pytest.raises(gx.PlanError, match="not an emitter density"):
        gx.Pipeline(gx.Chain(emitters={"c": index}, imaging=strata, environment=medium))
    other = cells.replace(values=torch.zeros(2, 20, 20, dtype=torch.float64))
    two = gx.Chain(emitters={"a": cells, "b": other}, imaging=strata, environment=medium)
    with pytest.raises(gx.PlanError, match="share one grid"):
        gx.Pipeline(two)
    layered = gx.env.LayeredMedium(sample=1.33)
    raised = gx.Chain(emitters={"c": cells}, imaging=strata, environment=layered)
    with pytest.raises(gx.ValidityError, match="above the coverslip"):
        gx.Pipeline(raised, outputs=("expected",))


def test_periodic_strata_agree_with_linear_ones_inside_the_frame():
    # the margin absorbs the wrap of in-focus light: densities inside the frame agree closely
    medium = gx.env.Homogeneous(1.33)
    camera, objective, cells, _ = _scene(medium)
    density = gx.lower.emitter_density(cells)
    linear = gx.imaging.Strata(objective, camera)(density, medium).data
    periodic = gx.imaging.Strata(objective, camera, boundary="periodic")
    wrapped = periodic(density, medium).data
    assert float((wrapped - linear).norm() / linear.norm()) < 1e-3
    chain = gx.Chain(emitters={"cells": cells}, imaging=periodic, environment=medium)
    pipe = gx.Pipeline(chain, outputs=("expected",))
    assert any("periodic boundary" in v.message for v in pipe.violations)
    again = gx.Pipeline.from_json(pipe.to_json(), chain)
    assert again.hash == pipe.hash
    assert torch.equal(again(chain)["expected"], pipe(chain)["expected"])


@pytest.mark.parametrize("shape", [(31, 45), (45, 31), (33, 27)])
def test_periodic_strata_place_voxels_like_linear_ones_on_any_frame(shape):
    # at equal pupil sampling the two boundaries differ only by the wrap: in-focus light of a
    # density inside an odd or non-square frame lands on the same pixels
    medium = gx.env.Homogeneous(1.33)
    camera, objective, cells, _ = _scene(medium)
    camera = gx.Camera(pixel_size=6.5, shape=shape)
    density = gx.lower.emitter_density(cells)
    linear = gx.imaging.Strata(objective, camera, pupil_samples=96)(density, medium).data
    periodic = gx.imaging.Strata(objective, camera, pupil_samples=96, boundary="periodic")
    wrapped = periodic(density, medium).data
    assert float((wrapped - linear).norm() / linear.norm()) < 1e-5


def test_periodic_strata_drop_voxels_outside_their_period_and_keep_gradients():
    medium = gx.env.Homogeneous(1.33)
    camera, objective, cells, _ = _scene(medium)
    periodic = gx.imaging.Strata(objective, camera, boundary="periodic")
    values = torch.zeros(3, 20, 120, dtype=torch.float64)
    values[:, :, :20] = cells.values
    far = values.clone()
    far[1, 10, 110] = 1e6  # 100 px right of the density's start: outside frame and margin
    shifted = cells.replace(values=values)
    near = periodic(gx.lower.emitter_density(shifted), medium).data
    dropped = periodic(gx.lower.emitter_density(shifted.replace(values=far)), medium).data
    assert torch.equal(near, dropped)
    grad_values = cells.values.clone().requires_grad_()
    na = torch.tensor(1.2, dtype=torch.float64, requires_grad=True)
    strata = gx.imaging.Strata(objective.replace(NA=na), camera, boundary="periodic")
    image = strata(gx.lower.emitter_density(cells.replace(values=grad_values)), medium).data
    weights = torch.linspace(0, 1, 32, dtype=torch.float64)
    g_values, g_na = torch.autograd.grad((image * weights).sum(), (grad_values, na))
    assert bool(torch.isfinite(g_values).all()) and float(g_values.abs().min()) > 0.0
    assert math.isfinite(float(g_na)) and float(g_na) != 0.0
