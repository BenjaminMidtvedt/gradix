"""M1 benchmarks (docs/architecture.md §8.3) on the reference GPU.

- sparse fluorescence: scalar pupil PSFs of 64 × 128² images with 50 emitters (targets ≥ 20k img/s
  forward, ≥ 5k img/s forward + backward);
- SMLM: 256 noisy 64² frames with 20 emitters, NA 1.2 in water (≥ 30k frames/s) and NA 1.45
  through a coverslip;
- dense fluorescence: 16 × 256² images of 32-plane densities with Strata, linear and periodic
  boundaries (≥ 5k img/s).

Run with ``uv run python benchmarks/m1.py``; results are appended to ``benchmarks/results.json``.
"""

from pathlib import Path

import torch

import gradix as gx
from gradix.testing.bench import BenchResult, benchmark, record


def main() -> None:
    """Time PointPSF expected frames and a forward + backward pass at two NAs."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    b, n = 64, 50
    g = torch.Generator().manual_seed(0)
    xy = 1.0 + 14.6 * torch.rand(b, n, 2, generator=g)
    z = 0.4 * torch.rand(b, n, 1, generator=g) - 0.2
    pos = torch.cat([xy, z], -1).to(device)
    camera = gx.Camera(pixel_size=6.5, shape=(128, 128))
    results = []
    for na in (0.7, 1.2):
        results.extend(_workloads(na, pos, camera, b))
    results.extend(_smlm(device))
    results.extend(_dense(device))
    for r in results:
        print(f"{r.name:34s} {r.median_ms:8.2f} ms  {r.throughput:10.0f} img/s  ({r.device})")
    record(results, Path(__file__).with_name("results.json"))


def _workloads(na: float, pos: torch.Tensor, camera: gx.Camera, b: int) -> list[BenchResult]:
    """Benchmark the expected frames and a forward + backward pass at one NA."""
    beads = gx.Emitters(position=pos, photons=1000.0, emission=gx.Spectrum.line(0.6))
    psf = gx.imaging.PointPSF(gx.Objective(NA=na, magnification=50), camera, psf="scalar")
    chain = gx.Chain(emitters={"beads": beads}, imaging=psf, environment=gx.env.Homogeneous(1.33))
    pipe = gx.Pipeline(chain, outputs=("expected",))
    position = pos.clone().requires_grad_()
    grad_chain = chain.replace(emitters={"beads": beads.replace(position=position)})

    def forward_backward() -> None:
        pipe(grad_chain)["expected"].sum().backward()

    return [
        benchmark(f"m1/pointpsf/na{na}/expected", lambda: pipe(chain), items=b),
        benchmark(f"m1/pointpsf/na{na}/fwd+bwd", forward_backward, items=b),
    ]


def _smlm(device: str) -> list[BenchResult]:
    """Benchmark noisy SMLM frames: NA 1.2 in water, and NA 1.45 through a coverslip."""
    b, n = 256, 20
    g = torch.Generator().manual_seed(0)
    xy = 0.5 + 5.4 * torch.rand(b, n, 2, generator=g)
    z = 0.6 * torch.rand(b, n, 1, generator=g) - 0.3
    camera = gx.Camera(pixel_size=6.5, shape=(64, 64), noise=gx.noise.PoissonGaussian(read=1.5))
    out = []
    for na, medium, depth in (
        (1.2, gx.env.Homogeneous(1.33), z),
        (1.45, gx.env.LayeredMedium(sample=1.33), -z.abs() - 0.05),  # below the coverslip
    ):
        pos = torch.cat([xy, depth], -1).to(device)
        beads = gx.Emitters(position=pos, photons=2000.0, emission=gx.Spectrum.line(0.68))
        psf = gx.imaging.PointPSF(gx.Objective(NA=na, magnification=65), camera, psf="scalar")
        chain = gx.Chain(emitters={"b": beads}, imaging=psf, environment=medium)
        pipe = gx.Pipeline(chain, outputs=("image",))
        name = f"m1/smlm/na{na}/{type(medium).__name__}/image"
        out.append(benchmark(name, lambda pipe=pipe, chain=chain: pipe(chain, key=3), items=b))
    return out


def _dense(device: str) -> list[BenchResult]:
    """Benchmark dense fluorescence: 32-plane densities imaged with Strata (both boundaries)."""
    b, z = 16, 32
    g = torch.Generator().manual_seed(0)
    values = torch.rand(b, z, 256, 256, generator=g).to(device)
    cells = gx.Voxels(
        values=values,
        spacing=(0.25, 0.1, 0.1),  # the camera pitch in object space: 6.5 µm / 65
        origin=(0.05, 0.05, -4.0),  # voxel centres on pixel centres
        quantity="density",
        emission=gx.Spectrum.line(0.6),
    )
    camera = gx.Camera(pixel_size=6.5, shape=(256, 256))
    objective = gx.Objective(NA=0.8, magnification=65)
    medium = gx.env.Homogeneous(1.33)
    out = []
    for boundary in ("linear", "periodic"):
        strata = gx.imaging.Strata(objective, camera, boundary=boundary)
        chain = gx.Chain(emitters={"c": cells}, imaging=strata, environment=medium)
        pipe = gx.Pipeline(chain, outputs=("expected",))
        name = f"m1/strata/{boundary}/expected"
        out.append(benchmark(name, lambda pipe=pipe, chain=chain: pipe(chain), items=b))
    return out


if __name__ == "__main__":
    main()
