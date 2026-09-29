"""Randomised Pipelines (fixed seed): finite outputs, chunk-size invariance, per-image independence.

Each configuration draws a batch size, an object count (including 0), per-frame fields, presence,
per-image NA and focus, a dtype, an imaging path (Sprites, PointPSF ROI or global) and camera
noise. Its Pipeline must give finite outputs, the same outputs at another chunk size (within
rounding: batched GEMMs round differently), and the same result for an image rendered alone
(with its own image key) as for its slice of the batch.
"""

import random
import warnings

import torch

import gradix as gx


def _scene(rng: random.Random, seed: int):
    g = torch.Generator().manual_seed(seed)
    dtype = torch.float64 if rng.random() < 0.3 else torch.float32
    b, n, t = rng.choice([1, 2, 3]), rng.choice([0, 1, 4]), rng.choice([1, 1, 2])
    shape = (rng.choice([16, 20]), rng.choice([16, 20]))
    imaging = rng.choice(["sprites", "roi", "global"])
    na = 0.6 if imaging == "sprites" else 1.0
    if rng.random() < 0.3:
        na = na * (1 - 0.1 * torch.rand(b, generator=g, dtype=dtype))
    focus = (0.2 * torch.rand(b, generator=g, dtype=dtype) - 0.1) if rng.random() < 0.3 else 0.0
    objective = gx.Objective(NA=na, magnification=65, focus=focus)
    noisy = rng.random() < 0.5
    camera = gx.Camera(
        pixel_size=6.5, shape=shape, noise=gx.noise.PoissonGaussian(read=1.2) if noisy else None
    )
    lead = (b, t) if t > 1 else (b,)
    fov = min(shape) * 0.1
    xy = 0.2 + (fov - 0.4) * torch.rand(*lead, n, 2, generator=g, dtype=dtype)
    z = 0.3 * torch.rand(*lead, n, 1, generator=g, dtype=dtype) - 0.15
    presence = None
    if rng.random() < 0.5:
        presence = (torch.rand(*lead, n, generator=g) > 0.3).to(dtype)
    beads = gx.Emitters(
        position=torch.cat([xy, z], -1),
        photons=500 + 1000 * torch.rand(*lead, n, generator=g, dtype=dtype),
        presence=presence,
        emission=gx.Spectrum.line(0.6),
    )
    element = {
        "sprites": lambda: gx.imaging.Sprites(objective, camera),
        "roi": lambda: gx.imaging.PointPSF(objective, camera, psf="scalar"),
        "global": lambda: gx.imaging.PointPSF(objective, camera, psf="scalar", method="global"),
    }[imaging]()
    chain = gx.Chain(
        emitters={"p": beads},
        imaging=element,
        environment=gx.env.Homogeneous(1.33),
        acquisition=gx.acq.Frames(t) if t > 1 else None,
    )
    return chain, b, noisy


def _close(a: torch.Tensor, b: torch.Tensor, rel: float) -> bool:
    if a.shape != b.shape:
        return False
    if a.numel() == 0:
        return True
    scale = float(a.double().abs().max()) or 1.0
    return float((a.double() - b.double()).abs().max()) <= rel * scale


def test_random_pipelines_are_finite_chunk_invariant_and_per_image():
    rng = random.Random(2026)
    failures = []
    for seed in range(24):
        chain, b, noisy = _scene(rng, seed)
        outputs = {"mu": "expected", "tab": gx.labels.EmitterTable("p")}
        if noisy:
            outputs["img"] = "image"
        key = torch.arange(b) + 3 if noisy else None
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                whole = gx.Pipeline(chain, outputs=outputs, deterministic=True)
                out = whole(chain, key=key)
                chunked = gx.Pipeline(
                    chain, outputs=outputs, deterministic=True, recorded_chunks={"batch": 1}
                )(chain, key=key)
                alone = whole(chain.select(b - 1), key=None if key is None else key[-1:])
            for name, value in out.items():
                rel = 1e-5 if value.dtype == torch.float32 else 1e-12
                assert bool(torch.isfinite(value).all()), f"non-finite {name}"
                assert _close(value, chunked[name], rel), f"chunked {name}"
                assert _close(value[-1:], alone[name], 1e3 * rel), f"image alone {name}"
        except AssertionError as err:
            failures.append(f"seed {seed}: {err}")
    assert not failures, chr(10).join(failures)
