"""Schema freeze (§11.10): every registered class's schema matches its snapshot.

A schema change must bump the class's ``schema_version`` and regenerate the snapshot::

    GRADIX_UPDATE_SNAPSHOTS=1 uv run pytest tests/test_schema_snapshot.py

Regeneration refuses a changed schema whose version was not bumped.
"""

import json
import os
from pathlib import Path

import gradix
from gradix._core.registry import REGISTRIES
from gradix.schema.base import Node
from gradix.schema.export import schema_of

SNAPSHOT = Path(__file__).parent / "snapshots" / "schemas.json"


CORE = (
    gradix.Chain,
    gradix.Sample,
    gradix.Microscope,
    gradix.Objective,
    gradix.Spectrum,
    gradix.Polarization,
    gradix.env.Homogeneous,
    gradix.env.LayeredMedium,
    gradix.EmitterSet,
    gradix.EmitterDensity,
    gradix.Irradiance,
    gradix.PlaneWaves,
    gradix.Field,
    gradix.ObjectSpectra,
)
"""Unregistered classes whose schemas are part of the frozen structure (containers, carriers)."""


def _strip_docs(value):
    if isinstance(value, dict):
        return {k: _strip_docs(v) for k, v in value.items() if k != "doc"}
    if isinstance(value, list):
        return [_strip_docs(v) for v in value]
    return value


def _schema(cls):
    return _strip_docs(json.loads(json.dumps(schema_of(cls), default=str)))


def _current():
    out = {f"core:{cls.__qualname__}": _schema(cls) for cls in CORE}
    for kind, registry in REGISTRIES.items():
        for name in registry.names():
            obj = registry.get(name)
            if (
                isinstance(obj, type)
                and issubclass(obj, Node)
                and obj.__module__.startswith("gradix")
            ):
                out[f"{kind}:{name}"] = _schema(obj)
    return out


def _version(schema):
    return schema.get("schema_version", schema.get("version"))


def test_schemas_match_the_snapshot():
    current = _current()
    stored = json.loads(SNAPSHOT.read_text(encoding="utf-8")) if SNAPSHOT.exists() else {}
    unbumped = [
        key
        for key, schema in current.items()
        if key in stored and schema != stored[key] and _version(schema) == _version(stored[key])
    ]
    if os.environ.get("GRADIX_UPDATE_SNAPSHOTS"):
        assert not unbumped, f"bump schema_version before regenerating: {unbumped}"
        SNAPSHOT.write_text(json.dumps(current, indent=1, sort_keys=True) + "\n", encoding="utf-8")
        return
    assert stored, f"no snapshot: run with GRADIX_UPDATE_SNAPSHOTS=1 to create {SNAPSHOT.name}"
    changed = sorted(k for k in current if k in stored and current[k] != stored[k])
    new = sorted(set(current) - set(stored))
    removed = sorted(set(stored) - set(current))
    assert not unbumped, f"schemas changed without a schema_version bump: {unbumped}"
    assert not (changed or new or removed), (
        f"schemas changed {changed}, added {new}, removed {removed}: "
        "regenerate the snapshot with GRADIX_UPDATE_SNAPSHOTS=1"
    )
