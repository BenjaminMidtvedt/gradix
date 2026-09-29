"""Differentiable field reconstructions (``gx.recon``, §5.12, ADR-44).

Each reconstruction maps camera frames ``[B, C, H, W]`` to reconstructed data. Its ``linear``
part is affine in the frames (windows, crops, propagation), which is where information can be
lost; ``__call__`` may add an invertible pointwise step. They apply to measured and rendered
frames alike, and :func:`gradix.crlb` measures what their linear part keeps.
"""

from gradix.recon.base import ChainOptics, Reconstruction, optics_of
from gradix.recon.inline import Inline

__all__ = ["ChainOptics", "Inline", "Reconstruction", "optics_of"]
