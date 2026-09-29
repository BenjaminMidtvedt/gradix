"""The dynamic gradient probe (§6.2): which input fields receive gradients through a render.

Two checks run. A random linear functional of all outputs is backpropagated to every field
that requires gradients. The camera's noise estimator is probed on a flat field: the sample
variance ``Σ(yᵢ − ȳ)²/N`` of a uniform rate λ has gradient exactly 0 under a plain
straight-through estimator (every pixel's derivative is 1, and the deviations sum to 0) and
≈ 1 under scaled straight-through, so it flags cameras that cannot learn photon statistics.
Taps that name the first stage where the vector-Jacobian product vanishes arrive with
``Pipeline.run`` (M2).
"""

from __future__ import annotations

import torch
from torch import Tensor

from gradix._core import keys as _keys
from gradix.compose.chain import Chain
from gradix.compose.pipeline import Pipeline
from gradix.detect.camera import Camera
from gradix.tree import tensors

__all__ = ["gradient_probe"]


def gradient_probe(
    pipe: Pipeline, chain: Chain, *, key: int | Tensor | None = None, seed: int = 0
) -> dict[str, dict[str, str]]:
    """Report, per functional, whether each field requiring gradients gets a usable gradient.

    Parameters
    ----------
    pipe : Pipeline
        A Pipeline whose outputs include ``"image"`` or ``"expected"``.
    chain : Chain
        Inputs whose fields that require gradients are probed.
    key : int or Tensor, optional
        Detector-noise key for the image.
    seed : int, default 0
        Key of the random linear functional.

    Returns
    -------
    dict of str to dict of str to str
        ``{"linear": {...}, "noise_energy": {...}}``. ``linear`` maps each field path to
        ``"ok"``, ``"zero"``, ``"none"`` or ``"non-finite"``; ``noise_energy`` maps
        ``"camera.noise"`` to the flat-field verdict (absent for noise-free cameras).
    """
    fields = {p: t for p, t in tensors(chain).items() if t.requires_grad}
    report: dict[str, dict[str, str]] = {}
    out = pipe(chain, key=key)
    frames = [out[name] for name in out if out[name].is_floating_point()]
    if not frames:
        raise ValueError("the Pipeline has no floating outputs to probe")
    total = torch.zeros((), dtype=frames[0].dtype, device=frames[0].device)
    for i, y in enumerate(frames):
        weights = _keys.uniform(torch.tensor([seed + i], device=y.device), "probe", y.shape)
        total = total + (y * weights.reshape(y.shape).to(y.dtype)).sum()
    report["linear"] = _status(total, fields)
    report["noise_energy"] = {}
    camera = chain.camera
    if isinstance(camera, Camera) and camera.noise is not None:
        level = torch.tensor(5.0, dtype=frames[0].dtype, device=frames[0].device)
        level.requires_grad_(True)
        flat = camera.from_electrons(camera.noise.mean(level.expand(64, 1, 32, 32)))
        keys = torch.arange(64, device=level.device) + 1000 * (seed + 1)
        electrons = camera.electrons(camera.sample(flat, keys))
        (grad,) = torch.autograd.grad(electrons.var(), level)
        # d Var/dλ is 1 for Poisson statistics; plain ST gives 0 up to rounding
        verdict = "non-finite" if not bool(torch.isfinite(grad)) else "ok"
        if verdict == "ok" and abs(float(grad)) < 0.1:
            verdict = "zero"
        report["noise_energy"] = {"camera.noise": verdict}
    return report


def _status(loss: Tensor, fields: dict[str, Tensor]) -> dict[str, str]:
    names = list(fields)
    if not loss.requires_grad:
        return dict.fromkeys(names, "none")
    grads = torch.autograd.grad(
        loss, [fields[n] for n in names], allow_unused=True, retain_graph=True
    )
    out: dict[str, str] = {}
    for name, g in zip(names, grads, strict=True):
        if g is None:
            out[name] = "none"
        elif not bool(torch.isfinite(g).all()):
            out[name] = "non-finite"
        elif not bool((g != 0).any()):
            out[name] = "zero"
        else:
            out[name] = "ok"
    return out
