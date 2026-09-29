"""Test harnesses shipped with gradix (``gx.testing``).

Conformance, gradient checks and probes, the camera estimator suite, ladders, level
equivalence, the plan-example checker and benchmarks.

Plugins use the same bar as core code: ``gx.testing.conformance(element, inputs)``.
"""

from gradix.testing import bench, camera_estimators, examples, ladders, levels
from gradix.testing.conformance import conformance, output_tensors
from gradix.testing.gradcheck import gradcheck, nonzero_gradients
from gradix.testing.gradprobe import gradient_probe

__all__ = [
    "bench",
    "camera_estimators",
    "conformance",
    "examples",
    "gradcheck",
    "gradient_probe",
    "ladders",
    "levels",
    "nonzero_gradients",
    "output_tensors",
]
