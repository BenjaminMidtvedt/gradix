"""Level equivalence (§11.10): L2 direct calls, an L3 Pipeline and an L4 plan give the same images.

Given the same inputs, key and grids, the three levels must be bit-identical: they share the
element code, and L2 calls here use the Pipeline's static configuration (``static=``). Every
level accumulates deterministically (``deterministic=True``), since atomic scatter-adds on CUDA
are not bit-reproducible. The L2 and L3 renders are built by hand from the user's objects, not
from the planner's Chain, so a planner that builds the wrong element (or changes its knobs)
fails the comparison.
"""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import Tensor

from gradix._core.carriers import Irradiance
from gradix._core.contract import Element
from gradix.api.plan import plan
from gradix.compose.chain import Chain
from gradix.compose.pipeline import Pipeline
from gradix.containers.containers import Microscope, Sample
from gradix.imaging.sprites import Sprites
from gradix.lower.emitters import emitter_set
from gradix.objects.objectset import Emitters
from gradix.planner.fidelity import Fidelity
from gradix.schema.signature import signature

__all__ = ["identical", "level_outputs", "max_differences"]


def level_outputs(
    sample: Sample,
    microscope: Microscope,
    *,
    key: int | Tensor | None,
    imaging: Element | None = None,
    fidelity: Fidelity | str = "draft",
) -> dict[str, dict[str, Tensor]]:
    """Render the same inputs at L2, L3 and L4.

    Parameters
    ----------
    sample : Sample
        The sample (emitter populations).
    microscope : Microscope
        The optics.
    key : int or Tensor, optional
        Detector-noise key.
    imaging : Element, optional
        The imaging element of the hand-built L2 and L3 renders; by default
        ``Sprites(microscope.objective, microscope.camera)``. The L4 plan is pinned to its
        registry name for every population.
    fidelity : Fidelity or str, default "draft"
        Fidelity of the L4 plan (its other knobs).

    Returns
    -------
    dict of str to dict of str to Tensor
        ``{"L2": {...}, "L3": {...}, "L4": {...}}`` with ``"expected"`` and, with a key,
        ``"image"``.

    Raises
    ------
    ValueError
        If the plan built a different imaging element than the hand-built one (another type,
        other knobs or another objective).
    """
    outputs = ("expected", "image") if key is not None else ("expected",)
    element = imaging if imaging is not None else Sprites(microscope.objective, microscope.camera)
    name = type(element).__dict__.get("registry_name")
    methods = {pop: name for pop in sample.populations} if isinstance(name, str) else None
    planned = plan(
        sample, microscope, fidelity, outputs=outputs, methods=methods, deterministic=True
    )
    built = planned.chain(sample, microscope).imaging
    differs = signature(element).incompatibility(signature(built))
    if (
        type(built) is not type(element)
        or differs is not None
        or getattr(built, "objective", None) is not getattr(element, "objective", None)
    ):
        detail = f" ({differs})" if differs else ""
        msg = f"the plan built {built!r}, not the hand-built {type(element).__name__}{detail}"
        raise ValueError(msg)
    l4 = planned(sample, microscope, key=key)
    chain = Chain(
        emitters=dict(sample.populations),
        imaging=element,
        background=microscope.background,
        environment=sample.environment,
    )
    pipe = Pipeline(chain, outputs=outputs, deterministic=True)
    l3 = pipe(chain, key=key)
    camera = microscope.camera
    pops = {k: v for k, v in sample.populations.items() if isinstance(v, Emitters)}
    irradiance = element(emitter_set(pops), sample.environment, static=pipe.static("imaging"))
    if not isinstance(irradiance, Irradiance):
        raise TypeError(f"the imaging element returned {type(irradiance).__name__}")
    mu = camera.expected(irradiance, microscope.background)
    l2 = {"expected": mu}
    if key is not None:
        l2["image"] = camera.sample(mu, key)
    return {
        "L2": l2,
        "L3": {k: l3[k] for k in outputs},
        "L4": {k: l4[k] for k in outputs},
    }


def identical(outputs: Mapping[str, Mapping[str, Tensor]]) -> dict[str, bool]:
    """Return whether the levels agree bit for bit (same shape, dtype and values), per output.

    Parameters
    ----------
    outputs : Mapping
        The result of :func:`level_outputs`.

    Returns
    -------
    dict of str to bool
        ``{"L2-L3/expected": …, "L3-L4/expected": …, …}``.
    """
    out: dict[str, bool] = {}
    for a, b in (("L2", "L3"), ("L3", "L4")):
        for name, value in outputs[a].items():
            other = outputs[b][name]
            same = value.shape == other.shape and value.dtype == other.dtype
            out[f"{a}-{b}/{name}"] = same and bool(torch.equal(value, other))
    return out


def max_differences(outputs: Mapping[str, Mapping[str, Tensor]]) -> dict[str, float]:
    """Return the largest absolute difference between levels, per output.

    Parameters
    ----------
    outputs : Mapping
        The result of :func:`level_outputs`.

    Returns
    -------
    dict of str to float
        ``{"L2-L3/expected": …, "L3-L4/expected": …, …}``; 0.0 means equal values, and
        ``inf`` marks a shape or dtype mismatch (which broadcasting would otherwise hide).
    """
    out: dict[str, float] = {}
    for a, b in (("L2", "L3"), ("L3", "L4")):
        for name, value in outputs[a].items():
            other = outputs[b][name]
            if value.shape != other.shape or value.dtype != other.dtype:
                out[f"{a}-{b}/{name}"] = float("inf")
            else:
                out[f"{a}-{b}/{name}"] = float(torch.max(torch.abs(value - other)))
    return out
