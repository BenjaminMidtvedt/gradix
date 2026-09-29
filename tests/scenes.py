"""Scene builders shared by the tests (plain torch sampling, as a caller would write it)."""

import torch

import gradix as gx
from gradix.units import nm, um

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def tensor(value: object) -> torch.Tensor:
    """Narrow a field value to a tensor (for the type checker) and return it."""
    assert isinstance(value, torch.Tensor), type(value)
    return value


def emitters(
    b=4,
    n=6,
    *,
    device="cpu",
    seed=0,
    dtype=torch.float32,
    fov=8.0,
    z=0.2,
    photons=None,
    presence=None,
    wavelength=600 * nm,
):
    g = torch.Generator(device="cpu").manual_seed(seed)
    xy = 0.5 + (fov - 1.0) * torch.rand(b, n, 2, generator=g, dtype=dtype)
    zz = (2 * torch.rand(b, n, 1, generator=g, dtype=dtype) - 1) * z
    pos = torch.cat([xy, zz], -1).to(device)
    if photons is None:
        photons = (500 + 1500 * torch.rand(b, n, generator=g, dtype=dtype)).to(device)
    return gx.Emitters(
        position=pos, photons=photons, presence=presence, emission=gx.Spectrum.line(wavelength)
    )


def optics(
    *,
    shape=(64, 64),
    na=0.7,
    magnification=50.0,
    noise=None,
    gain=1.0,
    offset=0.0,
    bit_depth=None,
    unit="adu",
    pixel_size=6.5 * um,
):
    camera = gx.Camera(
        pixel_size=pixel_size,
        shape=shape,
        gain=gain,
        offset=offset,
        noise=noise,
        bit_depth=bit_depth,
        unit=unit,
    )
    objective = gx.Objective(NA=na, magnification=magnification)
    return objective, camera


def chain(b=4, n=6, *, device="cpu", seed=0, noise=None, **kw):
    objective, camera = optics(noise=noise, **kw)
    beads = emitters(b, n, device=device, seed=seed)
    return gx.Chain(
        emitters={"beads": beads},
        imaging=gx.imaging.Sprites(objective, camera),
        environment=gx.env.Homogeneous(1.33),
    )
