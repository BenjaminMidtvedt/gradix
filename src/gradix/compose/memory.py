"""The memory strategy (Pipeline pass P6): dominant-tensor estimates and recorded chunk sizes.

Chunk sizes are chosen when a Pipeline is built, quantised to powers of two, recorded in its
JSON and ``out.meta``, and part of its hash, so the summation order is reproducible (§4.3). A
frozen Pipeline (``from_json``) reuses the recorded values.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from typing import Any

import torch

from gradix._core.errors import PlanError

__all__ = ["MemoryPlan", "plan_memory"]


@dataclasses.dataclass(frozen=True)
class MemoryPlan:
    """Chunk sizes and the peak-memory estimate of a Pipeline.

    Parameters
    ----------
    chunks : Mapping[str, int]
        Chunk size per reduction (``"batch"``, ``"emitters"``, …).
    estimate_bytes : int
        Estimated peak bytes of one call at capacity, forward and backward.
    strategy : str
        ``"autograd"`` (M0), ``"checkpoint"`` or ``"no_grad"``.
    """

    chunks: Mapping[str, int]
    estimate_bytes: int
    strategy: str = "autograd"

    def to_json(self) -> dict[str, Any]:
        """Return a JSON-serialisable description.

        Returns
        -------
        dict
            The fields.
        """
        return {
            "chunks": dict(sorted(self.chunks.items())),
            "estimate_bytes": self.estimate_bytes,
            "strategy": self.strategy,
        }


def _device_bytes(device: torch.device) -> int | None:
    """Total memory of a CUDA device: a fixed quantity, so chunk sizes are reproducible."""
    if device.type != "cuda":
        return None
    return int(torch.cuda.get_device_properties(device).total_memory)


def plan_memory(
    *,
    batch: int,
    frames: int,
    slots: Mapping[str, int],
    bins: int,
    shape: tuple[int, int],
    itemsize: int,
    device: torch.device,
    budget: float,
    recorded: Mapping[str, int] | None = None,
    per_image_hint: int | None = None,
) -> MemoryPlan:
    """Estimate the dominant tensors of the emission path and choose chunk sizes.

    Parameters
    ----------
    batch : int
        Batch capacity.
    frames : int
        Acquisition size A.
    slots : Mapping[str, int]
        Slot capacity of each population.
    bins : int
        Largest number of wavelength bins.
    shape : tuple of int
        Camera ``(H, W)``.
    itemsize : int
        Bytes per real element.
    device : torch.device
        Device the Pipeline renders on.
    budget : float
        Fraction of the device's total memory a call may use. Total, not free, memory keeps
        chunk sizes (and so the Pipeline hash) independent of what else runs on the device.
    recorded : Mapping[str, int], optional
        Chunk sizes from a frozen Pipeline; reused as they are.
    per_image_hint : int, optional
        The largest per-image estimate the elements declare (:meth:`Element.memory`); the
        plan uses the larger of it and its own dominant-tensor estimate.

    Returns
    -------
    MemoryPlan
        The plan.

    Raises
    ------
    PlanError
        If one image at capacity cannot fit the budget (pre-flight refusal).
    """
    n = sum(slots.values()) * bins
    height, width = shape
    per_image = frames * itemsize * (3 * n * (height + width + 2) + 4 * height * width)
    if per_image_hint is not None:
        per_image = max(per_image, int(per_image_hint))
    estimate = batch * per_image
    total = _device_bytes(device)
    if recorded is not None:
        return MemoryPlan(dict(recorded), estimate)
    chunk = batch
    if total is not None:
        allowed = budget * total
        if per_image > allowed:
            need, have = per_image / 2**20, allowed / 2**20
            msg = f"one image needs ≈{need:.0f} MiB but the budget allows {have:.0f} MiB"
            raise PlanError(msg, fix="reduce the slot capacity or the camera shape")
        while chunk > 1 and chunk * per_image > allowed:
            chunk //= 2
    return MemoryPlan({"batch": chunk}, estimate)
