# gradix

A standalone, differentiable, GPU-accelerated renderer for light-microscopy images, built on PyTorch.

> **Status: pre-alpha. M0 (walking skeleton) is complete; M1 (pupil & fluorescence) is in
> progress.** The three API levels run end to end for point emitters, as Gaussian sprites (the
> `draft` tier) or scalar pupil PSFs (`standard`: sparse ROIs or the global-spectrum path, Zernike
> and pixel pupils, layered media with supercritical collection), and for emitter densities
> (`gx.Voxels` imaged plane by plane with `gx.imaging.Strata`), with keyed camera noise
> (Poisson–Gaussian, sCMOS maps, EMCCD; Gaussian, shifted-Poisson or exact likelihoods), frames and
> focus stacks, uniform, TIRF, SIM and light-sheet excitation, value binding, envelopes, gradients
> and Fisher information. A scalar coherent smoke thread (point dipoles, coherent pupil imaging)
> exercises the coherent carriers. The plan is in [`docs/architecture.md`](docs/architecture.md);
> start with §0, which fits on two pages.

## A first render (M0)

```python
import torch
import gradix as gx
from gradix.units import nm, um

beads = gx.Emitters(
    position=torch.tensor([[[2.6, 2.6, 0.0], [5.2, 3.9, 0.1]]]),  # [B=1, N=2, 3] µm
    photons=torch.tensor([[2000.0, 800.0]]),
    emission=gx.Spectrum.line(600 * nm),
)
camera = gx.Camera(pixel_size=6.5 * um, shape=(64, 64), noise=gx.noise.PoissonGaussian(read=1.5))
objective = gx.Objective(NA=0.7, magnification=50)
env = gx.env.Homogeneous(1.33)

# L2: call elements directly
sprites = gx.imaging.Sprites(objective, camera)
mu = camera.expected(sprites(gx.lower.emitter_set(beads), env))
image = camera.sample(mu, key=torch.arange(1))  # image-keyed noise

# L3: a Chain in a Pipeline (static grids, value binding, explain())
chain = gx.Chain(emitters={"beads": beads}, imaging=sprites, environment=env)
pipe = gx.Pipeline(chain, outputs=("image", "expected"))
out = pipe(chain, key=torch.arange(1))  # bit-identical to the L2 calls

# L4: a plan chooses the elements from a Fidelity
sample = gx.Sample({"beads": beads}, environment=env)
scope = gx.presets.Widefield(objective=objective, camera=camera)
plan = gx.plan(sample, scope, "draft", outputs=("image",))
print(plan.explain())
```

## Pupil PSFs, layered media and excitation (M1)

```python
import math

import torch
import gradix as gx

beads = gx.Emitters(
    position=torch.tensor([[[2.0, 2.0, -0.1], [3.1, 2.6, -0.4]]]),  # z < 0: below the coverslip
    photons=torch.tensor([[3000.0, 3000.0]]),
    emission=gx.Spectrum.line(0.68),
)
astigmatism = gx.pupil.Zernike(coeffs=torch.tensor([0.05]), indices=(5,))  # µm of OPD
objective = gx.Objective(NA=1.45, magnification=100, pupil=(astigmatism,))
camera = gx.Camera.scmos(pixel_size=6.5, shape=(80, 80), gain=2.0, offset=100.0, read=1.2)
sample = gx.Sample({"beads": beads}, environment=gx.env.WaterOnCoverslip())
scope = gx.presets.TIRF(objective=objective, camera=camera, angle=math.radians(68.0))
plan = gx.plan(sample, scope, gx.Fidelity("standard", psf="scalar"), outputs=("image", "expected"))
out = plan(sample, scope, key=torch.arange(1))  # evanescent excitation, supercritical PSFs

# dense fluorescence: an emitter density on 16 planes, imaged plane by plane with pupil PSFs
cells = gx.Voxels(
    values=torch.rand(16, 64, 64),  # photons per voxel, [Z, Y, X]
    spacing=(0.2, 0.065, 0.065),  # µm: the camera pitch in object space (6.5 µm / 100)
    origin=(0.5 * 0.065, 0.5 * 0.065, -1.6),  # first voxel centre on a pixel centre
    quantity="density",
    emission=gx.Spectrum.line(0.6),
)
dense = gx.Chain(
    emitters={"cells": cells},
    imaging=gx.imaging.Strata(gx.Objective(NA=1.2, magnification=100), camera),
    environment=gx.env.Homogeneous(1.33),
)
image = dense(outputs=("expected",))["expected"]  # [1, 1, 80, 80]
```

## What it is for

gradix does the physics; you decide what to render. It takes batched tensors describing matter, light,
optics and camera, and returns images, fields and geometry-derived labels.

- **Bring your own sampling.** Randomising scenes is a few lines of plain torch, and the docs ship a tested
  cookbook. DeepTrack2 integration is planned, timed to DeepTrack2's own optics-pipeline revision.
- **Gradients to every input.** Every tensor you pass in can receive gradients: pupils, NA, sizes, positions,
  and values sampled with reparameterised draws, so distribution parameters become learnable. This covers
  sim-to-real calibration, pupil and aberration fitting, end-to-end optical design including optical
  neural networks, and inverse problems solved with the same forward model.
- **Tunable fidelity.** Choose methods yourself, or let a `Fidelity` policy (`draft`, `standard`,
  `accurate`, `reference`) choose them and explain why.

## Three levels, each usable on its own

1. **Building blocks.** Data objects (`Spheres`, `Emitters`, `SDF`, `Voxels`, …) and elements (light
   sources, Mie/Born/multislice interactions, coherent and point-emitter imaging, polarisation optics,
   cameras). Elements exchange typed carriers, so they compose like functions:
   `camera.expected(gx.imaging.Coherent(objective, camera)(gx.interact.Mie(beads)(light(), env), light(), env))`.
2. **Composition.** A `gx.Chain` wires elements along the physical slots:
   - Source → Interact → Objective → Detect;
   - for fluorescence, Source → Transduce → Emit → Detect.

   A `gx.Pipeline` turns a Chain into a pure, static-shape function that is built once and fed data: new
   Chains, or values by field path, for datasets of any size and object counts that vary per image. It adds
   grids from a value envelope, validity checks, gradient tables, memory planning, keys and CUDA graphs.
3. **Convenience.** `gx.plan(sample, microscope, fidelity="standard")` chooses the elements for you and
   builds the Chain. `plan.chain(...)` shows it, so you can edit it and run it yourself.

All coherent light meets in one representation: the scattered plane-wave spectrum, carried alongside exact
analytic plane waves. See [`docs/architecture.md`](docs/architecture.md) for the full design, the roadmap and
the reasoning behind each decision. Prior-art surveys and measurement scripts are in
[`docs/background/`](docs/background/).

## Development

```bash
uv sync
```

```bash
uv run pytest
```

```bash
uv run ruff check .
```

```bash
uv run ruff format --check .
```

```bash
uv run ty check
```

`pytest` also runs the numpydoc docstring gate, the docstring examples, the tutorial notebooks in
[`docs/tutorials/`](docs/tutorials/) and a schema snapshot: a change to a registered class's fields must
bump its `schema_version` and regenerate the snapshot (`GRADIX_UPDATE_SNAPSHOTS=1 uv run pytest
tests/test_schema_snapshot.py`). Benchmarks live in [`benchmarks/`](benchmarks/). `AGENTS.md` summarises
the project's hygiene for coding agents.

On Linux and Windows, `uv sync` installs CUDA 13.0 builds of PyTorch from the PyTorch wheel index; see
`[tool.uv.sources]` in `pyproject.toml`.

## License

MIT; see [`LICENSE`](LICENSE).
