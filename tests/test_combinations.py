"""Every imaging element with every acquisition, medium, light and camera noise (M1 sweep).

Each combination renders eagerly and through a Pipeline (same shapes, finite values), explains
itself, and passes gradients to its population. Validity warnings are expected in some
combinations and ignored here.
"""

import itertools
import warnings

import pytest
import torch

import gradix as gx

P = 0.1
SHAPE = (24, 24)


def camera(noise):
    if noise == "none":
        return gx.Camera(pixel_size=6.5, shape=SHAPE)
    if noise == "pg":
        return gx.Camera(pixel_size=6.5, shape=SHAPE, noise=gx.noise.PoissonGaussian(read=1.2))
    if noise == "scmos":
        return gx.Camera.scmos(pixel_size=6.5, shape=SHAPE, gain=2.0, offset=100.0, read=1.2)
    return gx.Camera(pixel_size=6.5, shape=SHAPE, noise=gx.noise.EMCCD(gain=100.0, read=10.0))


def medium(kind):
    return gx.env.Homogeneous(1.33) if kind == "homog" else gx.env.LayeredMedium(sample=1.33)


def light(kind):
    return {
        "none": None,
        "uniform": gx.light.Uniform(irradiance=2.0, wavelength=0.488),
        "sim": gx.light.SIMBeams(period=0.3, wavelength=0.488),
        "sheet": gx.light.Sheet(waist=0.6, center=-0.2, focus=1.2, wavelength=0.488),
        "tirf": gx.light.Evanescent(angle=1.2, wavelength=0.488),
    }[kind]


def population(imaging, acq, med):
    z = -0.3 if med == "layered" else 0.0
    if imaging in ("strata", "periodic"):
        vals = torch.rand(3, 16, 16)
        return gx.Voxels(
            values=vals,
            spacing=(0.2, P, P),
            origin=(0.45, 0.45, z - 0.2),
            quantity="density",
            emission=gx.Spectrum.line(0.6),
        )
    b, n = 2, 3
    g = torch.Generator().manual_seed(1)
    pos = torch.cat([0.4 + 1.6 * torch.rand(b, n, 2, generator=g), torch.full((b, n, 1), z)], -1)
    if acq == "frames":
        pos = pos[:, None].expand(b, 2, n, 3).clone()
        pos[:, 1, :, 0] += 0.05
    return gx.Emitters(
        position=pos, photons=torch.full((b, n), 800.0), emission=gx.Spectrum.line(0.6)
    )


def element(imaging, objective, cam):
    if imaging == "sprites":
        return gx.imaging.Sprites(objective, cam)
    if imaging == "roi":
        return gx.imaging.PointPSF(objective, cam, psf="scalar")
    if imaging == "global":
        return gx.imaging.PointPSF(objective, cam, psf="scalar", method="global")
    if imaging == "periodic":
        return gx.imaging.Strata(objective, cam, boundary="periodic")
    return gx.imaging.Strata(objective, cam)


def acquisition(kind):
    return {
        "none": None,
        "frames": gx.acq.Frames(2),
        "stack": gx.acq.FocusStack(focus=torch.tensor([-0.2, 0.2])),
    }[kind]


def _combinations():
    for imaging, acq, med, lt, noise in itertools.product(
        ("sprites", "roi", "global", "strata", "periodic"),
        ("none", "frames", "stack"),
        ("homog", "layered"),
        ("none", "uniform", "sim", "sheet", "tirf"),
        ("none", "pg", "scmos", "emccd"),
    ):
        if imaging == "sprites" and med == "layered":
            continue
        if lt != "none" and noise not in ("none", "pg"):
            continue
        if noise != "none" and acq != "none":
            continue
        yield imaging, acq, med, lt, noise


def _run(imaging, acq, med, lt, noise):
    na = 1.2 if imaging != "sprites" else 0.7
    objective = gx.Objective(NA=na, magnification=65)
    cam = camera(noise)
    pop = population(imaging, acq, med)
    chain = gx.Chain(
        light=light(lt),
        emitters={"p": pop},
        excite=gx.excite.Linear() if lt != "none" else None,
        imaging=element(imaging, objective, cam),
        environment=medium(med),
        acquisition=acquisition(acq),
    )
    outs = ("image", "expected") if noise != "none" else ("expected",)
    key = 3 if noise != "none" else None
    eager = chain(outputs=outs, key=key)
    out = gx.Pipeline(chain, outputs=outs)(chain, key=key)
    assert out["expected"].shape == eager["expected"].shape
    assert bool(torch.isfinite(out["expected"]).all())
    name = "p.values" if isinstance(pop, gx.Voxels) else "p.photons"
    value = (pop.values if isinstance(pop, gx.Voxels) else pop.photons).clone().requires_grad_()
    pipe = gx.Pipeline(chain, outputs=("expected",), inputs=(name,))
    assert "imaging" in pipe.explain()
    pipe({name: value})["expected"].sum().backward()
    assert value.grad is not None and bool(torch.isfinite(value.grad).all())


def test_every_element_renders_with_every_acquisition_medium_light_and_noise():
    failures = []
    for combination in _combinations():
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                _run(*combination)
        except Exception as err:
            failures.append(f"{'/'.join(combination)}: {type(err).__name__}: {err}")
    assert not failures, chr(10).join(failures)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_every_element_renders_on_cuda():
    # a third of the combinations, with Chains built on the CPU and moved (gx.tree.to)
    failures = []
    for i, (imaging, acq, med, lt, noise) in enumerate(_combinations()):
        if i % 3:
            continue
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                na = 1.2 if imaging != "sprites" else 0.7
                chain = gx.Chain(
                    light=light(lt),
                    emitters={"p": population(imaging, acq, med)},
                    excite=gx.excite.Linear() if lt != "none" else None,
                    imaging=element(imaging, gx.Objective(NA=na, magnification=65), camera(noise)),
                    environment=medium(med),
                    acquisition=acquisition(acq),
                )
                chain = gx.tree.to(chain, "cuda")
                outs = ("image", "expected") if noise != "none" else ("expected",)
                out = gx.Pipeline(chain, outputs=outs)(chain, key=3 if noise != "none" else None)
                assert out["expected"].device.type == "cuda"
                assert bool(torch.isfinite(out["expected"]).all())
        except Exception as err:
            failures.append(f"{imaging}/{acq}/{med}/{lt}/{noise}: {type(err).__name__}: {err}")
    assert not failures, chr(10).join(failures)
