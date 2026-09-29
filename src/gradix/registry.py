"""The registries (``gx.registry``), for introspection; register with :mod:`gradix.register`."""

from gradix._core.registry import (
    REGISTRIES,
    Registry,
    data_objects,
    elements,
    labels,
    lowerings,
    noise_models,
    object_sets,
    presets,
    pupil_modifiers,
    views,
)

__all__ = [
    "REGISTRIES",
    "Registry",
    "data_objects",
    "elements",
    "labels",
    "lowerings",
    "noise_models",
    "object_sets",
    "presets",
    "pupil_modifiers",
    "views",
]
