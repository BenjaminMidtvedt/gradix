"""L1 kernels (``gx.ops``): pure tensor functions with no knowledge of data objects or elements.

Kernels never draw from global or stateful random state: they take explicit keys or generators.
The public surface is unstable until 1.0 (no SemVer guarantee).
"""

from gradix.ops import detect, emitters, fourier, propagation, pupil

__all__ = ["detect", "emitters", "fourier", "propagation", "pupil"]
