"""The plan-example checker (§11.13): every ``gx.*`` name the documented examples use exists.

The examples of ``docs/architecture.md`` (§0.3, §9) are the API contract users review. Until
each becomes an executed notebook (§12.7), this checker parses every Python block and resolves
every ``gx.`` attribute chain: a name must exist in the package, or be listed in :data:`PLANNED`
with the milestone that delivers it. Drift between the plan and the code then fails CI.
"""

from __future__ import annotations

import ast
import inspect
import re
from pathlib import Path

import gradix

__all__ = [
    "CURRENT_MILESTONE",
    "PLANNED",
    "PLANNED_KEYWORDS",
    "check_document",
    "gx_names",
    "keyword_problems",
    "milestone_problems",
    "python_blocks",
    "resolve",
]

PLANNED: dict[str, str] = {
    # data objects and helpers
    "LayeredSpheres": "M3",
    "Ellipsoids": "M1",
    "Capsules": "M1",
    "Cylinders": "M1",
    "Boxes": "M1",
    "Gaussians": "M1",
    "SDF": "M1",
    "Occupancy": "M1",
    "JonesScreen": "M4",
    "NeuralField": "M4",
    "Texture": "M4",
    "Parent": "M4",
    "geometry": "M1",
    "geom.local_box": "M1",
    "geom.quat_z": "M1",
    "Labeling": "M1",
    "Material.uniaxial": "M4",
    "lower.ri_volume": "M4",
    "lower.projected": "M4",
    "lower.form_factor": "M3",
    # light
    "light.Koehler": "M3",
    "light.LEDArray": "M3",
    "light.ReferenceBeam": "M3",
    "light.GaussianSheet": "M1",
    "light.GaussianBeam": "M3",
    "light.Shaped": "M3",
    "light.IlluminationNuisance": "M3",
    "light.IlluminationEnvelope": "M3",
    "Annulus": "M3",
    "Disk": "M3",
    "Quadrature": "M3",
    "OnSupport": "M3",
    "GridAbbe": "M3",
    "Polarization.stokes": "M4",
    # elements
    "interact.Mie": "M3",
    "interact.Multislice": "M3",
    "interact.Projection": "M3",
    "pupil.PhaseRing": "M4",
    "optics.Aperture": "M3",
    "optics.DiffractiveStack": "M3",
    "optics.Mask": "M3",
    "devices.HeightMap": "M3",
    "devices.PhaseSLM": "M3",
    "acq.LEDSequence": "M3",
    # labels and outputs
    "labels.Phase": "M3",
    "labels.OPL": "M4",
    "labels.PerObjectImages": "M1",
    "out.Field": "M3",
    "out.Tap": "M2",
    "out.Regions": "M3",
    # L4 and tools
    "presets.InlineHolography": "M3",
    "presets.OffAxisHolography": "M3",
    "presets.Brightfield": "M3",
    "presets.Darkfield": "M3",
    "presets.PhaseContrast": "M4",
    "presets.DIC": "M4",
    "presets.DPC": "M4",
    "presets.QPI": "M4",
    "presets.ISCAT": "M3",
    "presets.FPM": "M4",
    "presets.Lensless": "M4",
    "presets.PolarizedLight": "M4",
    "presets.PolScope": "M4",
    "Pipeline.diff": "M2",
    "compare": "M5",
    "crlb": "M5",
    "save_inputs": "M2",
    "load_inputs": "M2",
    "views.OccupancyChannels": "M1",
    "planning.slice_dz": "M1",
}
"""Names the examples use that later milestones deliver, spelled exactly (a planned class covers its
methods). Wildcards are not accepted: a typo in a family must fail."""

PLANNED_KEYWORDS: dict[str, dict[str, str]] = {
    "light.PlaneWave": {"tilt": "M1"},
    "labels.InstanceMask": {"parts": "M4"},
    # positioned volumes, resampled onto element grids, come with the raster lowering (cap-06)
    "Voxels": {"position": "M1"},
}
"""Keywords of implemented callables that later milestones add, by callable and milestone."""

CURRENT_MILESTONE = "M0"
"""The milestone the package implements; a name planned for it (or earlier) must exist."""

_FENCE = re.compile(r"^```python\n(.*?)^```", re.S | re.M)


def python_blocks(markdown: str) -> list[str]:
    """Return the Python code blocks of a Markdown document.

    Parameters
    ----------
    markdown : str
        The document.

    Returns
    -------
    list of str
        The blocks' sources.
    """
    return [m.group(1) for m in _FENCE.finditer(markdown)]


def gx_names(source: str) -> set[str]:
    """Return every dotted name used as ``gx.<name>`` in Python source.

    Parameters
    ----------
    source : str
        Python source (it must parse).

    Returns
    -------
    set of str
        Names such as ``"imaging.Sprites"`` (without the ``gx.`` prefix).
    """
    names: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Attribute):
            continue
        parts: list[str] = []
        current: ast.expr = node
        while isinstance(current, ast.Attribute):
            parts.append(current.attr)
            current = current.value
        if isinstance(current, ast.Name) and current.id == "gx":
            names.add(".".join(reversed(parts)))
    # keep only maximal chains: "a.b.c" makes "a.b" redundant
    return {n for n in names if not any(o != n and o.startswith(f"{n}.") for o in names)}


def resolve(name: str) -> str:
    """Resolve a ``gx.`` name: ``"implemented"``, or the milestone that plans it.

    Parameters
    ----------
    name : str
        Dotted name without the ``gx.`` prefix.

    Returns
    -------
    str
        ``"implemented"`` or a milestone.

    Raises
    ------
    KeyError
        If the name is neither implemented nor planned.
    """
    obj: object = gradix
    parts = name.split(".")
    for part in parts:
        if hasattr(obj, part):
            obj = getattr(obj, part)
            continue
        # a planned name covers its attributes (a planned class covers its methods)
        for j in range(len(parts), 0, -1):
            prefix = ".".join(parts[:j])
            if prefix in PLANNED:
                return PLANNED[prefix]
        raise KeyError(name)
    return "implemented"


def _target(name: str) -> object:
    """Return the implemented object of a ``gx.`` name, or None."""
    obj: object = gradix
    for part in name.split("."):
        if not hasattr(obj, part):
            return None
        obj = getattr(obj, part)
    return obj


def _call_name(func: ast.expr) -> str | None:
    parts: list[str] = []
    while isinstance(func, ast.Attribute):
        parts.append(func.attr)
        func = func.value
    if isinstance(func, ast.Name) and func.id == "gx":
        return ".".join(reversed(parts))
    return None


def keyword_problems(source: str) -> list[str]:
    """Return calls of implemented ``gx.`` callables with keywords their signatures lack.

    Parameters
    ----------
    source : str
        Python source (it must parse).

    Returns
    -------
    list of str
        One line per unknown keyword, such as ``"gx.light.PlaneWave(tilt=…)"``.
    """
    problems: list[str] = []
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Call):
            continue
        name = _call_name(node.func)
        target = _target(name) if name else None
        if target is None or not callable(target):
            continue
        try:
            params = inspect.signature(target).parameters
        except (TypeError, ValueError):
            continue
        if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
            continue
        planned = PLANNED_KEYWORDS.get(name or "", {})
        for kw in node.keywords:
            if kw.arg is not None and kw.arg not in params and kw.arg not in planned:
                problems.append(f"gx.{name}({kw.arg}=…) has no parameter {kw.arg!r}")
    return problems


def milestone_problems() -> list[str]:
    """Return planned names whose milestone has passed but that are still missing.

    Returns
    -------
    list of str
        One line per overdue name.
    """
    order = ["M0", "M1", "M2", "M3", "M4", "M5", "M6"]
    current = order.index(CURRENT_MILESTONE)
    overdue = []
    for name, milestone in PLANNED.items():
        if milestone in order and order.index(milestone) <= current and _target(name) is None:
            overdue.append(f"gx.{name} is planned for {milestone} but missing")
    return overdue


def check_document(path: str | Path) -> list[str]:
    """Check every Python block of a document: it parses, and every ``gx.`` name resolves.

    Parameters
    ----------
    path : str or Path
        The Markdown document.

    Returns
    -------
    list of str
        Problems; empty when the document is consistent with the package.
    """
    problems: list[str] = []
    for i, block in enumerate(python_blocks(Path(path).read_text(encoding="utf-8"))):
        try:
            names = gx_names(block)
        except SyntaxError as err:
            problems.append(f"block {i}: does not parse: {err}")
            continue
        for name in sorted(names):
            try:
                resolve(name)
            except KeyError:
                problems.append(f"block {i}: gx.{name} is neither implemented nor planned")
        problems.extend(f"block {i}: {p}" for p in keyword_problems(block))
    return problems
