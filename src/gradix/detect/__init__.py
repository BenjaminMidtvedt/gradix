"""L2 detection (``gx.detect``): the camera element, noise models and camera functions (§5.5)."""

from gradix.detect.camera import UNITS, Camera, format_frames, pixel_map
from gradix.detect.functions import fisher, log_prob, sample, to_unit
from gradix.detect.noise import EMCCD, SCMOS, Ideal, NoiseModel, PoissonGaussian

__all__ = [
    "EMCCD",
    "SCMOS",
    "UNITS",
    "Camera",
    "Ideal",
    "NoiseModel",
    "PoissonGaussian",
    "fisher",
    "format_frames",
    "log_prob",
    "pixel_map",
    "sample",
    "to_unit",
]
