"""M0 benchmark: Gaussian-sprite rendering of 64 × 128² images (docs/architecture.md §8.3).

Run with ``uv run python benchmarks/m0.py``; results are appended to ``benchmarks/results.json``.
"""

from pathlib import Path

import torch

import gradix as gx
from gradix.testing.bench import benchmark, record


def main() -> None:
    """Time the expected image, keyed noisy images, and a forward + backward pass."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    b, n = 64, 50
    g = torch.Generator().manual_seed(0)
    pos = torch.cat([0.5 + 7.3 * torch.rand(b, n, 2, generator=g), torch.zeros(b, n, 1)], -1)
    beads = gx.Emitters(position=pos.to(device), photons=1000.0, emission=gx.Spectrum.line(0.6))
    camera = gx.Camera(pixel_size=6.5, shape=(128, 128), noise=gx.noise.PoissonGaussian(read=1.5))
    chain = gx.Chain(
        emitters={"beads": beads},
        imaging=gx.imaging.Sprites(gx.Objective(NA=0.7, magnification=100), camera),
        environment=gx.env.Homogeneous(1.33),
    )
    expected = gx.Pipeline(chain, outputs=("expected",))
    noisy = gx.Pipeline(chain, outputs=("image",))
    keys = torch.arange(b, device=device)
    photons = torch.full((b, n), 1000.0, device=device, requires_grad=True)
    grad_chain = chain.replace(emitters={"beads": beads.replace(photons=photons)})
    grad_pipe = gx.Pipeline(grad_chain, outputs=("expected",))

    def forward_backward() -> None:
        grad_pipe(grad_chain)["expected"].sum().backward()

    results = [
        benchmark("m0/sprites/expected", lambda: expected(chain), items=b),
        benchmark("m0/sprites/image-batch-keyed", lambda: noisy(chain, key=7), items=b),
        benchmark("m0/sprites/image-image-keyed", lambda: noisy(chain, key=keys), items=b),
        benchmark("m0/sprites/expected-fwd+bwd", forward_backward, items=b),
    ]
    for r in results:
        print(f"{r.name:34s} {r.median_ms:8.2f} ms  {r.throughput:10.0f} img/s  ({r.device})")
    record(results, Path(__file__).with_name("results.json"))


if __name__ == "__main__":
    main()
