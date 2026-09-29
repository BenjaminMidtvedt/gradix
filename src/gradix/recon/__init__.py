"""Differentiable field reconstructions (``gx.recon``, §5.12, ADR-44).

Each reconstruction maps camera frames ``[B, C, H, W]`` to reconstructed data. Its ``linear``
part is affine in the frames (windows, crops, propagation, demodulation), which is where
information can be lost; ``__call__`` may add invertible steps (pointwise ones, or QWLSI's
integration of the phase gradients). They apply to measured and rendered frames alike, and
:func:`gradix.crlb` measures what their linear part keeps.
"""

from gradix.recon.base import ChainOptics, Reconstruction, optics_of, without_scatterers
from gradix.recon.inline import Inline
from gradix.recon.iscat import ISCATContrast
from gradix.recon.offaxis import OffAxis
from gradix.recon.qwlsi import QWLSI

__all__ = [
    "QWLSI",
    "ChainOptics",
    "ISCATContrast",
    "Inline",
    "OffAxis",
    "Reconstruction",
    "optics_of",
    "without_scatterers",
]
