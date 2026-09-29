"""Solids, labelings and the raster lowering (cap-06, ADR-41): exact integrals, Strata renders."""

import dataclasses
import math
from collections.abc import Mapping
from typing import ClassVar

import pytest
import torch
from scenes import DEVICES

import gradix as gx
from gradix._core.grid import Grid2D, VolumeGrid


def _grid(dx=0.05, dz=0.05, extent=3.0, zext=2.4):
    n, nz = round(extent / dx), round(zext / dz)
    xy = Grid2D((n, n), dx, (-extent / 2, -extent / 2))
    return VolumeGrid(xy=xy, z0=-zext / 2 + dz / 2, dz=dz, nz=nz)


def _integral(objects, grid, surface=False, reach=1.2):
    vals = gx.lower.rasterise(
        objects, grid, sigma=0.3 * grid.xy.spacing, reach=reach, surface=surface
    )
    return float(vals.detach().sum()) * grid.xy.spacing**2 * grid.dz


ORIGIN = torch.zeros(1, 1, 3, dtype=torch.float64)
TURN = torch.tensor([[[0.924, 0.115, 0.0, 0.364]]], dtype=torch.float64)  # an oblique rotation


def _f64(*values):
    return torch.tensor(values, dtype=torch.float64)


@pytest.mark.parametrize(
    ("objects", "volume", "area"),
    [
        (gx.Spheres(position=ORIGIN, radius=_f64([0.5])), 4 / 3 * math.pi * 0.125, math.pi),
        (
            gx.Ellipsoids(position=ORIGIN, rotation=TURN, semi_axes=_f64([[1.0, 0.5, 0.3]])),
            4 / 3 * math.pi * 0.15,
            None,
        ),
        (
            gx.Boxes(position=ORIGIN, rotation=TURN, size=_f64([[1.0, 0.6, 0.4]])),
            0.24,
            2 * (0.6 + 0.24 + 0.4),
        ),
        (
            gx.Capsules(position=ORIGIN, length=_f64([2.0]), radius=_f64([0.4])),
            math.pi * 0.16 * 1.2 + 4 / 3 * math.pi * 0.064,
            2 * math.pi * 0.4 * 1.2 + 4 * math.pi * 0.16,
        ),
        (
            gx.Cylinders(position=ORIGIN, rotation=TURN, length=_f64([1.2]), radius=_f64([0.4])),
            math.pi * 0.16 * 1.2,
            2 * math.pi * 0.4 * 1.2 + 2 * math.pi * 0.16,
        ),
        (
            gx.Gaussians(position=ORIGIN, sigma=_f64([[0.2, 0.3, 0.25]])),
            (2 * math.pi) ** 1.5 * 0.015,
            None,
        ),
    ],
    ids=["sphere", "ellipsoid", "box", "capsule", "cylinder", "gaussian"],
)
def test_rasterised_solids_hold_their_volume_and_area(objects, volume, area):
    for grid in (_grid(), _grid(dz=0.2)):  # coarse planes are spread by the axial kernel
        assert _integral(objects, grid) == pytest.approx(volume, rel=8e-3)
        if area is not None:  # edges of boxes blur their area by O(σ/size)
            assert _integral(objects, grid, surface=True) == pytest.approx(area, rel=2.5e-2)


def test_the_blurred_ball_gives_exact_volume_gradients():
    radius = torch.tensor([[0.5]], dtype=torch.float64, requires_grad=True)
    position = ORIGIN.clone().requires_grad_()
    volume = _integral(gx.Spheres(position=position, radius=radius), _grid())
    values = gx.lower.rasterise(
        gx.Spheres(position=position, radius=radius), _grid(), sigma=0.015, reach=0.6
    )
    d_radius, d_position = torch.autograd.grad(values.sum() * 0.05**3, (radius, position))
    assert volume == pytest.approx(4 / 3 * math.pi * 0.125, rel=2e-3)
    assert float(d_radius) == pytest.approx(4 * math.pi * 0.25, rel=3e-3)  # 4πR²
    assert float(d_position.abs().max()) < 1e-9  # moving a sphere keeps its volume


def test_shape_fields_reach_the_capability_methods_by_name():
    # a field may be called anything (here "x" and "blur", the methods' own argument names):
    # capability methods take the shape as one mapping (ADR-42)
    @dataclasses.dataclass(frozen=True, kw_only=True, eq=False)
    class Slabs(gx.Solid):
        shape_fields: ClassVar[tuple[str, ...]] = ("x", "blur")
        x: torch.Tensor = gx.field(quantity="length", role="object", shape_affecting=True)
        blur: torch.Tensor = gx.field(quantity="length", role="object", shape_affecting=True)

        def sdf(self, x: torch.Tensor, shape: Mapping[str, torch.Tensor]) -> torch.Tensor:
            return torch.maximum(
                x[..., 0].abs() - shape["x"], x[..., 1:].abs().amax(-1) - shape["blur"]
            )

        def reach(self, shape: Mapping[str, torch.Tensor]) -> torch.Tensor:
            return math.sqrt(3.0) * torch.maximum(shape["x"], shape["blur"])

    slabs = Slabs(position=ORIGIN, x=_f64([0.3]), blur=_f64([0.2]))
    assert _integral(slabs, _grid()) == pytest.approx(0.6 * 0.4 * 0.4, rel=1e-2)


def test_labelings_put_photons_where_they_say():
    spectrum = gx.Spectrum.line(0.6)
    grid = _grid()
    photons = gx.Labeling(emission=spectrum, photons=_f64([1000.0, 500.0]))
    two = gx.Capsules(
        position=torch.tensor([[[0.0, 0.0, 0.0], [0.6, 0.6, 0.2]]], dtype=torch.float64),
        length=_f64([1.2, 0.8]),
        radius=_f64([0.3, 0.2]),
        labeling=photons,
    )
    assert float(gx.lower.label_density(two, grid, sigma=0.015, reach=0.8).sum()) == pytest.approx(
        1500.0
    )
    surface = gx.Labeling(emission=spectrum, surface_density=10.0)
    ball = gx.Spheres(position=ORIGIN, radius=_f64([0.5]), labeling=surface)
    total = float(gx.lower.label_density(ball, grid, sigma=0.015, reach=0.6).sum())
    assert total == pytest.approx(10.0 * math.pi, rel=1e-3)  # 10 photons/µm² over 4πR²
    with pytest.raises(gx.StructureError, match="exactly one"):
        gx.Labeling(emission=spectrum, density=1.0, photons=10.0)
    with pytest.raises(gx.StructureError, match="surface_density"):
        gx.Labeling(emission=spectrum, density=1.0, where="surface")
    blob = gx.Gaussians(position=ORIGIN, sigma=_f64([[0.2, 0.2, 0.2]]), labeling=surface)
    with pytest.raises(gx.StructureError, match="no surface"):
        gx.lower.label_density(blob, grid, sigma=0.015, reach=0.8)


@pytest.mark.parametrize("device", DEVICES)
def test_rasterising_is_deterministic_and_device_independent(device):
    g = torch.Generator().manual_seed(3)
    many = gx.Spheres(
        position=torch.rand(2, 30, 3, generator=g, dtype=torch.float64) * 2 - 1,
        radius=0.1 + 0.2 * torch.rand(2, 30, generator=g, dtype=torch.float64),
    )
    grid = _grid(dx=0.08, dz=0.16)
    reference = gx.lower.rasterise(many, grid, sigma=0.024, reach=0.3, deterministic=True)
    moved = gx.tree.to(many, device)
    atomic = gx.lower.rasterise(moved, grid, sigma=0.024, reach=0.3).cpu()
    fixed = gx.lower.rasterise(moved, grid, sigma=0.024, reach=0.3, deterministic=True).cpu()
    assert torch.allclose(atomic, reference, rtol=0, atol=1e-12)
    assert torch.allclose(fixed, reference, rtol=0, atol=1e-12)


def _cells(dtype=torch.float64):
    spectrum = gx.Spectrum.line(0.6)
    return gx.Capsules(
        position=torch.tensor([[[1.0, 1.2, -0.2], [2.0, 1.8, 0.3]]], dtype=dtype),
        rotation=torch.tensor([[[0.92, 0.0, 0.39, 0.0], [0.7, 0.0, 0.0, 0.71]]], dtype=dtype),
        length=torch.tensor([[1.2, 0.9]], dtype=dtype),
        radius=torch.tensor([[0.25, 0.2]], dtype=dtype),
        labeling=gx.Labeling(emission=spectrum, surface_density=20.0),
    )


def test_labelled_solids_render_through_strata_on_the_grid_it_asks_for():
    camera = gx.Camera(pixel_size=6.5, shape=(32, 32))
    objective = gx.Objective(NA=1.2, magnification=65)
    medium = gx.env.Homogeneous(1.33)
    strata = gx.imaging.Strata(objective, camera)
    cells = _cells()
    chain = gx.Chain(emitters={"cells": cells}, imaging=strata, environment=medium)
    pipe = gx.Pipeline(chain, outputs=("expected",))
    request = strata.density_request(pipe.static("imaging"))
    assert request is not None and "raster.planes" in pipe.explain()
    # the Pipeline renders exactly what the same density, given as voxels on that grid, renders
    grid = request.grid
    density = gx.lower.label_density(cells, grid, sigma=request.blur, reach=request.reach["cells"])
    spacing = grid.xy.spacing
    voxels = gx.Voxels(
        values=density[:, 0],
        spacing=(grid.dz, spacing, spacing),
        origin=(grid.xy.origin[0] + spacing / 2, grid.xy.origin[1] + spacing / 2, grid.z0),
        quantity="density",
        emission=gx.Spectrum.line(0.6),
    )
    as_voxels = strata(gx.lower.emitter_density(voxels), medium, static=pipe.static("imaging"))
    rendered = pipe(chain).expected
    assert torch.equal(rendered, as_voxels.data[:, :, 0])  # frames [B, C, H, W]
    eager = chain(outputs=("expected",)).expected  # its own grid, without headroom
    assert float((eager - rendered).norm() / rendered.norm()) < 1e-2
    # gradients reach every geometry and label field
    radius = cells.radius.clone().requires_grad_()  # type: ignore[union-attr]
    rotation = cells.rotation.clone().requires_grad_()  # type: ignore[union-attr]
    image = pipe({"cells.radius": radius, "cells.rotation": rotation}).expected
    (image * torch.linspace(0, 1, 32, dtype=image.dtype)).sum().backward()
    assert radius.grad is not None and bool(radius.grad.abs().min() > 0)
    assert rotation.grad is not None and bool(torch.isfinite(rotation.grad).all())
    assert pipe.gradient_table.quality("cells.labeling.surface_density") == "exact"
    assert pipe.gradient_table.quality("cells.material") == "zero"


def test_a_labelled_sphere_images_like_the_point_cloud_it_contains():
    # the dense path against its exact reference: point emitters on a lattice inside the sphere
    camera = gx.Camera(pixel_size=6.5, shape=(24, 24))
    objective = gx.Objective(NA=1.2, magnification=65)
    medium = gx.env.Homogeneous(1.33)
    spectrum = gx.Spectrum.line(0.6)
    centre, radius, photons = _f64(1.23, 1.18, 0.2), 0.3, 2000.0
    axis = torch.arange(-radius, radius + 1e-9, 0.04, dtype=torch.float64)
    lattice = torch.stack(torch.meshgrid(axis, axis, axis, indexing="ij"), -1).reshape(-1, 3)
    points = lattice[lattice.norm(dim=-1) <= radius] + centre
    cloud = gx.Emitters(
        position=points[None],
        photons=torch.full((1, len(points)), photons / len(points), dtype=torch.float64),
        emission=spectrum,
    )
    psf = gx.imaging.PointPSF(
        objective, camera, psf="scalar", method="global", oversample=3, pupil_samples=96
    )
    reference = psf(gx.lower.emitter_set(cloud), medium).data
    sphere = gx.Spheres(
        position=centre[None, None],
        radius=_f64([radius]),
        labeling=gx.Labeling(emission=spectrum, photons=_f64([photons])),
    )
    strata = gx.imaging.Strata(objective, camera, pupil_samples=96)
    dense = gx.Chain(emitters={"s": sphere}, imaging=strata, environment=medium)
    image = dense(outputs=("expected",)).expected
    assert float((image - reference).norm() / reference.norm()) < 3e-2


def test_solids_need_a_labeling_and_a_density_consumer():
    camera = gx.Camera(pixel_size=6.5, shape=(16, 16))
    objective = gx.Objective(NA=1.2, magnification=65)
    medium = gx.env.Homogeneous(1.33)
    plain = gx.Spheres(position=ORIGIN + 0.8, radius=_f64([0.2]))
    chain = gx.Chain(
        emitters={"s": plain}, imaging=gx.imaging.Strata(objective, camera), environment=medium
    )
    with pytest.raises(gx.PlanError, match="without a labeling"):
        chain(outputs=("expected",))
    labelled = plain.replace(labeling=gx.Labeling(emission=gx.Spectrum.line(0.6), photons=1e3))
    sparse = chain.replace(
        emitters={"s": labelled}, imaging=gx.imaging.PointPSF(objective, camera, psf="scalar")
    )
    with pytest.raises(gx.PlanError, match=r"renders gx.Emitters"):
        sparse(outputs=("expected",))


def test_plans_route_labelled_solids_to_strata():
    camera = gx.Camera(pixel_size=6.5, shape=(32, 32))
    objective = gx.Objective(NA=1.2, magnification=65)
    sample = gx.Sample({"cells": _cells(torch.float32)}, environment=gx.env.Homogeneous(1.33))
    scope = gx.presets.Widefield(objective=objective, camera=camera)
    planned = gx.plan(sample, scope, gx.Fidelity("standard", psf="scalar"), outputs=("expected",))
    assert planned.routes["cells"].element == "emit.strata_otf"
    chain = planned.chain(sample, scope)
    assert isinstance(chain.imaging, gx.imaging.Strata) and chain.imaging.boundary == "periodic"
    assert torch.equal(planned(sample, scope).expected, planned.pipeline(chain).expected)
