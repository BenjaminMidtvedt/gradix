"""Camera functions usable outside a Pipeline: ``sample``, ``log_prob``, ``fisher``, ``to_unit``."""

from __future__ import annotations

import torch
from torch import Tensor

from gradix._core.errors import StructureError
from gradix.detect.camera import UNITS, Camera

__all__ = ["fisher", "log_prob", "sample", "to_unit"]


def sample(mu: Tensor, camera: Camera, key: int | Tensor | None = None) -> Tensor:
    """Draw noisy frames from expected frames: re-draw only the noise of a fixed batch.

    Parameters
    ----------
    mu : Tensor
        Expected frames in the camera's unit, ``[B, …, H, W]``.
    camera : Camera
        The camera.
    key : int or Tensor, optional
        A batch key or image keys; required when the camera has noise.

    Returns
    -------
    Tensor
        Noisy frames in the camera's unit.
    """
    return camera.sample(mu, key)


def log_prob(observed: Tensor, mu: Tensor, camera: Camera) -> Tensor:
    """Return the per-pixel log-likelihood of observed frames (§5.5).

    Parameters
    ----------
    observed : Tensor
        Observed frames in the camera's unit.
    mu : Tensor
        Expected frames in the camera's unit.
    camera : Camera
        The camera, with a noise model.

    Returns
    -------
    Tensor
        Log-likelihood per pixel; sum it for a total.
    """
    return camera.log_prob(observed, mu)


def to_unit(frames: Tensor, camera: Camera, from_unit: str = "adu") -> Tensor:
    """Convert recorded frames into the camera's declared unit.

    Parameters
    ----------
    frames : Tensor
        Frames ``[B, …, H, W]`` recorded in ``from_unit``.
    camera : Camera
        The camera, whose ``qe``, ``gain`` and ``offset`` define the conversion.
    from_unit : {"adu", "e", "photons"}, default "adu"
        Unit of ``frames``.

    Returns
    -------
    Tensor
        Frames in ``camera.unit``.
    """
    if from_unit not in UNITS:
        raise StructureError(f"unknown unit {from_unit!r}; units are {UNITS}")
    if from_unit == camera.unit:
        return frames
    electrons = camera.replace(unit=from_unit).electrons(frames)
    return camera.from_electrons(electrons)


def fisher(jacobian: Tensor, mu: Tensor, camera: Camera) -> Tensor:
    """Return the per-image Fisher information of parameters under the camera's noise (§5.5).

    ``F_ij = Σ_pixels (∂μ/∂θ_i)(∂μ/∂θ_j) / Var(μ)``: exact for Poisson noise, and the Gaussian
    approximation otherwise (Poisson ⊛ Gaussian, EMCCD), which underestimates the information
    at a few photoelectrons per pixel (about half at λ ≈ 0.05, within 1 % for λ ≳ 10).
    Compute the Jacobian with ``torch.func.jacfwd`` through a render (every element supports
    forward mode).

    Parameters
    ----------
    jacobian : Tensor
        ``∂μ/∂θ`` in the camera's unit per parameter, ``[B, …, P]`` (the frame axes of ``mu``,
        then P parameters).
    mu : Tensor
        Expected frames in the camera's unit, ``[B, …]``.
    camera : Camera
        The camera (its noise model gives the variance).

    Returns
    -------
    Tensor
        Fisher matrices ``[B, P, P]``; their inverses bound the parameters' covariance (CRLB).

    Raises
    ------
    StructureError
        If the Jacobian does not match the frames, or the camera has no noise model.
    """
    if tuple(jacobian.shape[:-1]) != tuple(mu.shape):
        got, want = list(jacobian.shape), [*mu.shape, "P"]
        raise StructureError(f"the Jacobian has shape {got}, expected {want}")
    var = torch.clamp(camera.variance(mu), min=1e-12)
    var = var.expand(torch.broadcast_shapes(var.shape, mu.shape))
    b, p = var.shape[0], jacobian.shape[-1]
    j = jacobian.expand(b, *jacobian.shape[1:]).reshape(b, -1, p)
    w = (1.0 / var).reshape(b, -1, 1)
    return torch.einsum("bki,bkj->bij", j * w, j)
