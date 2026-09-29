"""Precision policy: dtypes, IEEE matmuls, autocast control and fp64 phase construction.

Kernels run in float32/complex64 by default, special functions (Mie, Bessel, phases of long
paths) in float64. Two numerical rules are enforced here:

- **IEEE matmuls.** Matrix Fourier transforms and separable sums are matmuls. TF32 would cut
  their precision to ≈1e-3, so kernels that matmul wrap it in :func:`ieee_matmul`.
- **fp64 phases with the carrier subtracted** (§4.1): long phase arguments are built in float64,
  a reference (carrier) phase is subtracted, the remainder is reduced modulo 2π and only then cast.
"""

from __future__ import annotations

import contextlib
import dataclasses
import math
from collections.abc import Iterator

import torch
from torch import Tensor

__all__ = [
    "DEFAULT",
    "PrecisionPolicy",
    "ieee_matmul",
    "no_autocast",
    "result_dtype",
    "unit_phasor",
    "wrap_phase",
]


@dataclasses.dataclass(frozen=True)
class PrecisionPolicy:
    """The dtypes used by kernels.

    Parameters
    ----------
    real : torch.dtype, default torch.float32
        Dtype of real intermediates.
    special : torch.dtype, default torch.float64
        Dtype of special functions and phase construction.
    matmul : {"ieee", "tf32"}, default "ieee"
        Float32 matmul precision inside kernels.
    """

    real: torch.dtype = torch.float32
    special: torch.dtype = torch.float64
    matmul: str = "ieee"

    @property
    def complex(self) -> torch.dtype:
        """Complex dtype matching :attr:`real`.

        Returns
        -------
        torch.dtype
            ``complex64`` for float32, ``complex128`` for float64.
        """
        return torch.complex128 if self.real == torch.float64 else torch.complex64

    def describe(self) -> str:
        """Return a short description for reports and cache keys.

        Returns
        -------
        str
            Such as ``"float32/complex64/fp64-special/ieee"``.
        """
        real = str(self.real).removeprefix("torch.")
        cplx = str(self.complex).removeprefix("torch.")
        special = "fp64" if self.special == torch.float64 else str(self.special)
        return f"{real}/{cplx}/{special}-special/{self.matmul}"


DEFAULT = PrecisionPolicy()
"""The default policy: float32 kernels, float64 special functions, IEEE matmuls."""


def result_dtype(*values: object, default: torch.dtype = torch.float32) -> torch.dtype:
    """Return the real floating dtype that a set of inputs computes in.

    float64 if any floating tensor is float64 (so gradchecks in float64 stay in float64),
    otherwise ``default``. Complex tensors count by their real counterpart.

    Parameters
    ----------
    *values : object
        Tensors or other values; non-tensors are ignored.
    default : torch.dtype, default torch.float32
        Dtype when no tensor is float64.

    Returns
    -------
    torch.dtype
        ``torch.float64`` or ``default``.
    """
    for v in values:
        if isinstance(v, Tensor) and v.dtype in (torch.float64, torch.complex128):
            return torch.float64
    return default


@contextlib.contextmanager
def ieee_matmul() -> Iterator[None]:
    """Run float32 matmuls and convolutions at IEEE precision inside the block.

    Scoped and re-entrant: the previous settings are restored on exit, also when blocks nest.
    The settings are process-wide (torch has no thread-local switch), so concurrent threads
    share them.

    Yields
    ------
    None
        Control returns to the block.
    """
    matmul = torch.backends.cuda.matmul
    conv = getattr(torch.backends.cudnn, "conv", None)
    if hasattr(matmul, "fp32_precision") and conv is not None:
        saved = (matmul.fp32_precision, conv.fp32_precision)
        matmul.fp32_precision = "ieee"
        conv.fp32_precision = "ieee"
        try:
            yield
        finally:
            matmul.fp32_precision, conv.fp32_precision = saved
    else:  # torch < 2.9: the legacy flags
        legacy = (matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
        matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        try:
            yield
        finally:
            matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = legacy


@contextlib.contextmanager
def no_autocast() -> Iterator[None]:
    """Disable autocast inside the block, so kernels keep their own dtypes.

    Yields
    ------
    None
        Control returns to the block.
    """
    with contextlib.ExitStack() as stack:
        stack.enter_context(torch.autocast("cpu", enabled=False))
        if torch.cuda.is_available():
            stack.enter_context(torch.autocast("cuda", enabled=False))
        yield


def wrap_phase(phase: Tensor, carrier: Tensor | float = 0.0) -> Tensor:
    """Subtract a carrier phase and reduce the remainder to [-π, π), in the input's dtype.

    Build ``phase`` in float64 (§4.1): the reduction is only as accurate as the phase it
    receives.

    Parameters
    ----------
    phase : Tensor
        Phase in rad, ideally float64.
    carrier : Tensor or float, default 0.0
        Reference phase to subtract, in rad.

    Returns
    -------
    Tensor
        ``phase − carrier`` reduced to [-π, π).
    """
    rest = phase - carrier
    return torch.remainder(rest + math.pi, 2.0 * math.pi) - math.pi


def unit_phasor(
    phase: Tensor,
    carrier: Tensor | float = 0.0,
    *,
    dtype: torch.dtype = torch.complex64,
) -> Tensor:
    """Return ``exp(i·(phase − carrier))`` built from a float64 phase and cast at the end.

    Parameters
    ----------
    phase : Tensor
        Phase in rad (float64 recommended).
    carrier : Tensor or float, default 0.0
        Reference phase to subtract, in rad.
    dtype : torch.dtype, default torch.complex64
        Complex dtype of the result.

    Returns
    -------
    Tensor
        Unit phasors of dtype ``dtype``.
    """
    wrapped = wrap_phase(phase, carrier)
    return torch.polar(torch.ones_like(wrapped), wrapped).to(dtype)
