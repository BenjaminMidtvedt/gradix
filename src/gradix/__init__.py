"""Differentiable, GPU-accelerated simulation of light-microscopy images.

gradix renders batches of microscope images from tensors that describe matter, light, optics
and the camera, and differentiates the images with respect to every one of those tensors. The
architecture plan is ``docs/architecture.md``.

The public API has four levels above a foundation layer (§3.1):

- **L2 building blocks**: data objects (``gx.Emitters``, ``gx.Spheres``, ``gx.Spectrum``,
  ``gx.env``), elements (``gx.light``, ``gx.imaging``, ``gx.Objective``, ``gx.Camera``), lowering
  (``gx.lower``), labels and coordinates. Elements run eagerly.
- **L3 composition**: ``gx.Chain`` wires elements; ``gx.Pipeline`` fixes grids from an envelope
  and is fed data by field path.
- **L4 convenience**: ``gx.plan`` chooses elements by ``gx.Fidelity`` for a ``gx.Sample`` and a
  ``gx.Microscope``; ``gx.render`` caches plans.
- **L0/L1 foundations**: ``gx.units``, ``gx.conventions``, schemas (``gx.field``), ``gx.tree``,
  envelopes, grids, keys, registries (``gx.register``) and kernels (``gx.ops``).

Examples
--------
>>> import torch, gradix as gx
>>> from gradix.units import nm, um
>>> beads = gx.Emitters(
...     position=torch.tensor([[[2.6, 2.6, 0.0]]]),
...     photons=torch.tensor([[1000.0]]),
...     emission=gx.Spectrum.line(600 * nm),
... )
>>> camera = gx.Camera(pixel_size=6.5 * um, shape=(32, 32))
>>> sprites = gx.imaging.Sprites(gx.Objective(NA=0.7, magnification=40), camera)
>>> mu = camera.expected(sprites(gx.lower.emitter_set(beads), gx.env.Homogeneous(1.33)))
>>> tuple(mu.shape)
(1, 1, 32, 32)
"""

from gradix import (
    acq,
    conventions,
    coords,
    detect,
    dipole,
    env,
    excite,
    imaging,
    interact,
    labels,
    light,
    lower,
    materials,
    noise,
    ops,
    out,
    presets,
    pupil,
    recon,
    register,
    registry,
    sampling,
    special,
    tree,
    units,
)
from gradix._core.axes import AcqIndex
from gradix._core.carriers import (
    DiffractionOrders,
    EmitterDensity,
    EmitterSet,
    Field,
    GaussianSheet,
    Irradiance,
    ObjectSpectra,
    PlaneWaves,
)
from gradix._core.contract import (
    Capabilities,
    Cost,
    Description,
    Edge,
    Element,
    Slot,
    Static,
    Violation,
)
from gradix._core.envelope import Entry, Envelope, envelope_of
from gradix._core.errors import (
    BindingError,
    CapacityError,
    EnvelopeError,
    GradientPathError,
    GradixError,
    GradixWarning,
    PlanError,
    RegistryError,
    StructureError,
    ValidityError,
)
from gradix._core.grid import FreqGrid, Grid2D, PatchGrid, PupilGrid, VolumeGrid
from gradix._version import __version__
from gradix.api import CRLB, Plan, crlb, plan, render
from gradix.compose import Chain, Output, Pipeline, SamplingPlan
from gradix.containers import Microscope, Sample
from gradix.detect import Camera
from gradix.light import Polarization
from gradix.objects import (
    Boxes,
    Capsules,
    Cylinders,
    Ellipsoids,
    Emitters,
    Gaussians,
    Labeling,
    ObjectSet,
    Solid,
    Spectrum,
    Spheres,
    Voxels,
)
from gradix.objects.materials import Material
from gradix.optics import Objective, PupilModifier
from gradix.planner import Fidelity
from gradix.schema.base import DataObject, Node
from gradix.schema.export import schema_of
from gradix.schema.fields import child, field, knob
from gradix.schema.signature import signature
from gradix.schema.stack import pad, stack
from gradix.schema.units import from_si, to_si

__all__ = [
    "CRLB",
    "AcqIndex",
    "BindingError",
    "Boxes",
    "Camera",
    "Capabilities",
    "CapacityError",
    "Capsules",
    "Chain",
    "Cost",
    "Cylinders",
    "DataObject",
    "Description",
    "DiffractionOrders",
    "Edge",
    "Element",
    "Ellipsoids",
    "EmitterDensity",
    "EmitterSet",
    "Emitters",
    "Entry",
    "Envelope",
    "EnvelopeError",
    "Fidelity",
    "Field",
    "FreqGrid",
    "GaussianSheet",
    "Gaussians",
    "GradientPathError",
    "GradixError",
    "GradixWarning",
    "Grid2D",
    "Irradiance",
    "Labeling",
    "Material",
    "Microscope",
    "Node",
    "ObjectSet",
    "ObjectSpectra",
    "Objective",
    "Output",
    "PatchGrid",
    "Pipeline",
    "Plan",
    "PlanError",
    "PlaneWaves",
    "Polarization",
    "PupilGrid",
    "PupilModifier",
    "RegistryError",
    "Sample",
    "SamplingPlan",
    "Slot",
    "Solid",
    "Spectrum",
    "Spheres",
    "Static",
    "StructureError",
    "ValidityError",
    "Violation",
    "VolumeGrid",
    "Voxels",
    "__version__",
    "acq",
    "child",
    "conventions",
    "coords",
    "crlb",
    "detect",
    "dipole",
    "env",
    "envelope_of",
    "excite",
    "field",
    "from_si",
    "imaging",
    "interact",
    "knob",
    "labels",
    "light",
    "lower",
    "materials",
    "noise",
    "ops",
    "out",
    "pad",
    "plan",
    "presets",
    "pupil",
    "recon",
    "register",
    "registry",
    "render",
    "sampling",
    "schema_of",
    "signature",
    "special",
    "stack",
    "to_si",
    "tree",
    "units",
]


def _register_core() -> None:
    """Register the core data objects, containers and carriers under their stable names."""
    from gradix._core import registry as _registry
    from gradix.interact.mie import MieParams
    from gradix.objects.acquisition import FocusStack, Frames
    from gradix.objects.environment import Homogeneous, LayeredMedium
    from gradix.objects.labeling import Isotropic
    from gradix.objects.materials import Cauchy, Constant, Sellmeier

    core = (
        Objective, Spectrum, Homogeneous, LayeredMedium, Frames, FocusStack, Constant, Cauchy,
        Sellmeier, Isotropic, Polarization, Chain, Sample, Microscope, EmitterSet, EmitterDensity,
        Irradiance, PlaneWaves, GaussianSheet, Field, ObjectSpectra, MieParams,
    )  # fmt: skip
    for cls in core:
        name = cls.__dict__.get("registry_name")
        if isinstance(name, str) and name not in _registry.data_objects:
            _registry.data_objects.add(name, cls)


_register_core()


def __getattr__(name: str) -> object:
    """Import ``gx.testing`` on first use, so ``import gradix`` stays light.

    Parameters
    ----------
    name : str
        The attribute.

    Returns
    -------
    object
        The ``gradix.testing`` module.
    """
    if name == "testing":
        import importlib

        return importlib.import_module("gradix.testing")
    msg = f"module 'gradix' has no attribute {name!r}"
    raise AttributeError(msg)
