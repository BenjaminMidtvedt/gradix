import json
from importlib.metadata import version

import torch

import gradix


def test_version_matches_package_metadata():
    assert gradix.__version__ == version("gradix")


def test_complex_fft_autograd():
    x = torch.randn(8, 8, dtype=torch.complex64, requires_grad=True)
    torch.fft.ifft2(torch.fft.fft2(x)).abs().square().sum().backward()
    assert x.grad is not None


def test_benchmark_and_ladder_harnesses(tmp_path):
    from gradix.testing import bench, ladders

    result = bench.benchmark("noop", lambda: torch.zeros(4).sum(), items=4, device="cpu", repeat=3)
    assert result.device == "cpu" and result.throughput > 0
    bench.record([result], tmp_path / "bench.json")
    record = json.loads((tmp_path / "bench.json").read_text())[0]
    assert {"gradix", "torch", "commit"} <= set(record)
    ladders.register_ladder(
        ladders.Ladder(
            name="toy_test",
            lower="emit.gaussian",
            higher="detect.camera",  # any two registered elements: the toy never renders
            sweep={"x": (0.1, 0.5)},
            evaluate=lambda x: x,
            tolerance=0.2,
        )
    )
    try:
        rows = ladders.run("toy_test")
        assert rows is not None and [r.within for r in rows] == [True, False]
        edge = ladders.run("gaussian_pupil")  # the Sprites → PointPSF approximation edge
        assert edge is not None and all(row.within for row in edge)
    finally:
        del ladders.LADDERS["toy_test"]
