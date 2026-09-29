"""The benchmark runner: median-of-k synchronised timings, recorded as JSON (§11.12)."""

from __future__ import annotations

import dataclasses
import json
import statistics
import subprocess
import time
from collections.abc import Callable, Sequence
from pathlib import Path

import torch

from gradix._version import __version__

__all__ = ["BenchResult", "benchmark", "record"]


@dataclasses.dataclass(frozen=True)
class BenchResult:
    """A timing.

    Parameters
    ----------
    name : str
        Workload name.
    median_ms : float
        Median wall time per call, ms.
    times_ms : tuple of float
        Every timed call, ms.
    items : int
        Items (images) per call, for throughput.
    device : str
        Device description.
    """

    name: str
    median_ms: float
    times_ms: tuple[float, ...]
    items: int
    device: str

    @property
    def throughput(self) -> float:
        """Items per second at the median time.

        Returns
        -------
        float
            Items per second.
        """
        return self.items / (self.median_ms * 1e-3)


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def benchmark(
    name: str,
    fn: Callable[[], object],
    *,
    items: int = 1,
    warmup: int = 3,
    repeat: int = 5,
    device: torch.device | str | None = None,
) -> BenchResult:
    """Time a callable: warm up, then take the median of ``repeat`` synchronised calls.

    Parameters
    ----------
    name : str
        Workload name.
    fn : callable
        The workload.
    items : int, default 1
        Items per call.
    warmup : int, default 3
        Untimed calls first.
    repeat : int, default 5
        Timed calls.
    device : torch.device or str, optional
        Device the workload runs on: it is synchronised and names the result. Defaults to the
        current CUDA device when available, so pass ``"cpu"`` for CPU workloads.

    Returns
    -------
    BenchResult
        The timing.
    """
    if device is None:
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    dev = torch.device(device)
    for _ in range(warmup):
        fn()
    _sync(dev)
    times: list[float] = []
    for _ in range(repeat):
        start = time.perf_counter()
        fn()
        _sync(dev)
        times.append(1e3 * (time.perf_counter() - start))
    label = torch.cuda.get_device_name(dev) if dev.type == "cuda" else dev.type
    return BenchResult(name, statistics.median(times), tuple(times), items, label)


def record(results: Sequence[BenchResult], path: str | Path) -> None:
    """Append results to a JSON file (a list of records).

    Parameters
    ----------
    results : sequence of BenchResult
        The timings.
    path : str or Path
        The JSON file.
    """
    target = Path(path)
    existing = json.loads(target.read_text()) if target.exists() else []
    stamp = time.strftime("%Y-%m-%dT%H:%M:%S")
    for r in results:
        entry = dataclasses.asdict(r)
        entry["throughput"] = r.throughput
        entry["torch"] = torch.__version__
        entry["gradix"] = __version__
        entry["commit"] = _commit()
        entry["time"] = stamp
        existing.append(entry)
    target.write_text(json.dumps(existing, indent=2))


def _commit() -> str | None:
    """Return the git commit of the working tree, if there is one."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None
