"""The conformance suite (§7.4): the bar every registered element meets, built-in or plugin.

M0 checks the contract: schema, capabilities, validity and configuration on the eager
envelope (exceptions become report entries), purity (equal results and untouched global random
state), batch handling (each image rendered alone equals its slice of the batched render),
CPU/CUDA parity when a GPU is present, and finite non-zero gradients of a keyed random linear
functional of the output to every probed field. A plain sum of the output would be blind to
photon-conserving elements, whose total does not depend on positions or focus. The ``declared``
check probes every floating field of the element itself (numbers become tensors) and compares
whether a gradient flows with the quality the element declares for it
(:meth:`~gradix._core.contract.Element.field_quality`), so the gradient table cannot report
``zero`` for a field that trains, or ``exact`` for one that never reaches the output.

The full suite (gradcheck at removable singular points, linearity, energy, tilt equivariance,
convergence to the declared edges) grows with the elements.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import torch
from torch import Tensor

from gradix._core import keys as _keys
from gradix._core.contract import Element, Static, Violation
from gradix.schema.base import Node, iter_leaves, tree_axis_sizes
from gradix.schema.fields import specs
from gradix.tree import replace as tree_replace

__all__ = ["conformance", "linear_functional", "output_tensors"]


def output_tensors(out: object) -> list[Tensor]:
    """Return the tensors of an element's output (a tensor, a carrier, or a tuple of them).

    Parameters
    ----------
    out : object
        The output.

    Returns
    -------
    list of Tensor
        Its floating or complex tensors.
    """
    if isinstance(out, Tensor):
        return [out]
    if isinstance(out, (tuple, list)):
        return [t for item in out for t in output_tensors(item)]
    return [
        v
        for _p, _s, v in iter_leaves(out)
        if isinstance(v, Tensor) and (v.is_floating_point() or v.is_complex())
    ]


def linear_functional(out: object, *, seed: int = 0) -> Tensor:
    """Return a keyed random linear functional of an output: ``Σ w·Re(out)`` with ``w ~ U(0, 1)``.

    Parameters
    ----------
    out : object
        An element's output.
    seed : int, default 0
        Key of the weights; the same seed gives the same weights for the same shapes.

    Returns
    -------
    Tensor
        A scalar whose gradient is non-zero for any parameter the output depends on (almost
        surely), unlike the plain sum.

    Raises
    ------
    ValueError
        If the output holds no floating tensors.
    """
    total: Tensor | None = None
    for i, t in enumerate(output_tensors(out)):
        values = t.real if t.is_complex() else t
        key = torch.tensor([seed * 7919 + i], device=values.device)
        w = _keys.uniform(key, "conformance", tuple(values.shape))[0].to(values.dtype)
        term = (w * values).sum()
        total = term if total is None else total + term
    if total is None:
        raise ValueError("the output holds no floating tensors")
    return total


def _rng_state() -> list[Tensor]:
    states = [torch.get_rng_state()]
    if torch.cuda.is_available() and torch.cuda.is_initialized():
        states.extend(torch.cuda.get_rng_state_all())
    return states


def _select(value: object, i: int) -> object:
    if isinstance(value, (tuple, list)):  # several carriers (an interaction's contributions)
        return type(value)(_select(v, i) for v in value)
    return value.select(i) if isinstance(value, Node) else value


def _move(value: object, device: str) -> object:
    if isinstance(value, (tuple, list)):
        return type(value)(_move(v, device) for v in value)
    return value.to(device) if isinstance(value, Node) else value


def _close(a: Tensor, b: Tensor, *, rtol: float, atol: float) -> bool:
    return a.shape == b.shape and bool(torch.allclose(a.cpu(), b.cpu(), rtol=rtol, atol=atol))


def conformance(
    element: Element,
    inputs: Sequence[object],
    *,
    probes: dict[str, Tensor] | None = None,
    make_inputs: Callable[[dict[str, Tensor]], tuple[Element, Sequence[object]]] | None = None,
    parity_rtol: float = 1e-4,
) -> dict[str, str]:
    """Run the conformance checks on an element with example inputs.

    Parameters
    ----------
    element : Element
        The element.
    inputs : sequence of object
        Its inputs (carriers, then the environment); use at least two images to exercise the
        batch check.
    probes : dict of str to Tensor, optional
        Tensors (requiring gradients) whose gradients must be finite and non-zero; each must be
        used by ``make_inputs``.
    make_inputs : callable, optional
        ``make_inputs(probes) -> (element, inputs)`` rebuilding the call around the probes.
    parity_rtol : float, default 1e-4
        Relative tolerance of the CPU/CUDA comparison.

    Returns
    -------
    dict of str to str
        ``"ok"`` (or ``"skipped: …"``) or a failure description per check: ``schema``,
        ``capabilities``, ``validity``, ``output``, ``purity``, ``batch``, ``parity``,
        ``declared`` and ``gradients``.
    """
    report: dict[str, str] = {}
    cls = type(element)
    try:
        specs(cls)
        report["schema"] = "ok"
    except TypeError as err:
        report["schema"] = str(err)
    caps = cls.caps
    report["capabilities"] = "ok" if getattr(cls, "slot", None) is not None else "no slot"
    try:
        envelope = element.eager_envelope(*inputs)
        found = element.validity(element.eager_description(*inputs), envelope)
        ok = isinstance(found, list) and all(isinstance(v, Violation) for v in found)
        report["validity"] = "ok" if ok else "validity must return a list of Violation"
        static = element.configure(element.eager_description(*inputs), envelope)
    except Exception as err:
        report["validity"] = f"{type(err).__name__}: {err}"
        return report
    before = _rng_state()
    first = element.forward(*inputs, static=static)
    second = element.forward(*inputs, static=static)
    after = _rng_state()
    produces = caps.produces if isinstance(caps.produces, tuple) else (caps.produces,)
    report["output"] = (
        "ok"
        if isinstance(first, produces)
        else f"produced {type(first).__name__}, declared {produces}"
    )
    same = all(
        torch.equal(a, b)
        for a, b in zip(output_tensors(first), output_tensors(second), strict=True)
    )
    untouched = len(before) == len(after) and all(
        torch.equal(a, b) for a, b in zip(before, after, strict=True)
    )
    if not same:
        report["purity"] = "two identical calls differ"
    elif not untouched:
        report["purity"] = "forward changed the global random state"
    else:
        report["purity"] = "ok"
    report["batch"] = _batch_check(element, inputs, static, first)
    report["parity"] = _parity_check(element, inputs, static, first, parity_rtol)
    report["declared"] = _declared_check(element, inputs, static, first)
    if probes:
        if make_inputs is None:
            raise ValueError("probes need make_inputs")
        probe_element, probe_inputs = make_inputs(probes)
        out = probe_element.forward(*probe_inputs, static=probe_element.eager_static(*probe_inputs))
        functional = linear_functional(out)
        if not functional.requires_grad:
            report["gradients"] = f"no gradient path to any probe: {list(probes)}"
            return report
        grads = torch.autograd.grad(functional, list(probes.values()), allow_unused=True)
        bad = [
            name
            for name, g in zip(probes, grads, strict=True)
            if g is None or not bool(torch.isfinite(g).all()) or not bool((g != 0).any())
        ]
        report["gradients"] = "ok" if not bad else f"no finite non-zero gradient: {bad}"
    return report


def _declared_check(
    element: Element, inputs: Sequence[object], static: Static, first: object
) -> str:
    """Compare where gradients flow with the qualities the element declares for its fields."""
    outputs = output_tensors(first)
    dtype = next((t.dtype for t in outputs if t.is_floating_point()), torch.float64)
    device = next((t.device for t in outputs), None)
    problems = []
    for path, spec, value in iter_leaves(element):
        if spec is None or spec.kind != "tensor" or value is None or isinstance(value, bool):
            continue
        if isinstance(value, Tensor):
            if not (value.is_floating_point() or value.is_complex()):
                continue
            leaf = value.detach().clone().requires_grad_()
        elif isinstance(value, (int, float)):
            leaf = torch.tensor(float(value), dtype=dtype, device=device, requires_grad=True)
        else:
            continue
        declared = element.field_quality(path)
        try:
            probe = tree_replace(element, {path: leaf})
            functional = linear_functional(probe.forward(*inputs, static=static))
            grad = None
            if functional.requires_grad:
                grad = torch.autograd.grad(functional, leaf, allow_unused=True)[0]
        except Exception as err:
            problems.append(f"{path}: probing raised {type(err).__name__}: {err}")
            continue
        if grad is not None and not bool(torch.isfinite(grad).all()):
            problems.append(f"{path}: non-finite gradient")
            continue
        flows = grad is not None and bool((grad != 0).any())
        if flows and declared == "zero":
            problems.append(f"{path}: declared zero, but its gradient is non-zero")
        elif not flows and declared != "zero":
            problems.append(f"{path}: declared {declared}, but no gradient reaches the output")
    return "ok" if not problems else "; ".join(problems)


def _batch_check(element: Element, inputs: Sequence[object], static: Static, full: object) -> str:
    """Each image rendered alone must equal its slice of the batched render."""
    full_tensors = output_tensors(full)
    sizes = tree_axis_sizes((element, *[x for x in inputs if isinstance(x, Node)])).get("B", 1)
    batch = max([sizes, *(t.shape[0] for t in full_tensors if t.ndim > 0)])
    if batch < 2:
        return "skipped: the inputs have one image"
    for i in (0, batch - 1):
        alone = element.select(i).forward(*[_select(x, i) for x in inputs], static=static)
        for whole, one in zip(full_tensors, output_tensors(alone), strict=True):
            if whole.ndim == 0 or whole.shape[0] != batch:
                return f"an output has leading size {tuple(whole.shape)[:1]} for {batch} images"
            want = whole[i : i + 1]
            got = one if one.shape[0] == 1 else one[:1]
            if not _close(want, got.expand_as(want), rtol=1e-5, atol=1e-6):
                return f"image {i} rendered alone differs from its slice of the batch"
    return "ok"


def _parity_check(
    element: Element, inputs: Sequence[object], static: Static, cpu_out: object, rtol: float
) -> str:
    if not torch.cuda.is_available():
        return "skipped: no CUDA device"
    moved = element.to("cuda")
    try:
        out = moved.forward(*[_move(x, "cuda") for x in inputs], static=static)
    except Exception as err:
        return f"CUDA forward raised {type(err).__name__}: {err}"
    for a, b in zip(output_tensors(cpu_out), output_tensors(out), strict=True):
        scale = float(a.abs().max()) if a.numel() else 0.0
        if not _close(a, b, rtol=rtol, atol=rtol * scale):
            return "CPU and CUDA outputs differ beyond the tolerance"
    return "ok"
