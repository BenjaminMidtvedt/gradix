"""Gradient checks with forward-mode AD, for kernels, elements and plugins (§7.4, §11.5)."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import torch
from torch import Tensor

__all__ = ["gradcheck", "nonzero_gradients"]


def gradcheck(
    fn: Callable[..., Tensor],
    inputs: Sequence[Tensor],
    *,
    forward_ad: bool = True,
    batched: bool = True,
    second_order: bool = True,
    eps: float = 1e-6,
    atol: float = 1e-5,
    rtol: float = 1e-3,
) -> bool:
    """Check reverse-mode (and forward-mode) gradients of ``fn`` against finite differences.

    Run it in float64 or complex128 on tiny shapes that include removable singular points
    (q = 0, x → 0, θ = 0).

    Parameters
    ----------
    fn : callable
        Function of the inputs returning a tensor.
    inputs : sequence of Tensor
        Inputs; those with ``requires_grad`` are checked.
    forward_ad : bool, default True
        Also check forward-mode AD (needed by ``jacfwd`` and ``gx.crlb``).
    batched : bool, default True
        Also check batched gradients (vmap over the VJP), which ``torch.func`` users rely on.
    second_order : bool, default True
        Also run ``gradgradcheck`` (Hessian-vector products, Gauss-Newton fits).
    eps : float, default 1e-6
        Finite-difference step.
    atol : float, default 1e-5
        Absolute tolerance.
    rtol : float, default 1e-3
        Relative tolerance.

    Returns
    -------
    bool
        True when every check passes; failures raise with torch's report.
    """
    ok = torch.autograd.gradcheck(
        fn,
        tuple(inputs),
        eps=eps,
        atol=atol,
        rtol=rtol,
        check_forward_ad=forward_ad,
        check_batched_grad=batched,
    )
    if second_order:
        ok = ok and torch.autograd.gradgradcheck(fn, tuple(inputs), eps=eps, atol=atol, rtol=rtol)
    return bool(ok)


def nonzero_gradients(loss: Tensor, tensors: dict[str, Tensor]) -> dict[str, str]:
    """Return which tensors receive a finite, non-zero gradient from a loss.

    Parameters
    ----------
    loss : Tensor
        A scalar loss.
    tensors : dict of str to Tensor
        Tensors that require gradients, by name.

    Returns
    -------
    dict of str to str
        ``"ok"``, ``"none"`` (not reached), ``"zero"`` or ``"non-finite"`` per name.
    """
    names = list(tensors)
    grads = torch.autograd.grad(loss, [tensors[n] for n in names], allow_unused=True)
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
