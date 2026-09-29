"""Interferometric scattering (iSCAT) contrast: frames over the reference, minus one (§5.12)."""

from __future__ import annotations

import dataclasses

import torch
from torch import Tensor

from gradix._core.errors import StructureError
from gradix.compose.chain import Chain
from gradix.recon.base import Reconstruction, without_scatterers

__all__ = ["ISCATContrast"]


@dataclasses.dataclass(frozen=True)
class ISCATContrast(Reconstruction):
    """The iSCAT contrast ``I/I_ref − 1``: 2|E_s/E_r|·cos Δφ plus |E_s/E_r|² for weak scatterers.

    Parameters
    ----------
    reference : float or Tensor, optional
        The reference intensity in the frames' unit: a number, a frame ``[H, W]`` or per image
        ``[B, C, H, W]`` (a frame without the particles, as ratiometric iSCAT records); None
        uses each frame's median.

    Examples
    --------
    >>> import torch
    >>> ISCATContrast(reference=4.0)(torch.full((1, 1, 2, 2), 5.0)).max().item()
    0.25
    """

    reference: float | Tensor | None = None

    @classmethod
    def from_chain(cls, chain: Chain) -> ISCATContrast:
        """Use the Chain rendered without its scatterers as the reference frame.

        Parameters
        ----------
        chain : Chain
            A coherent Chain.

        Returns
        -------
        ISCATContrast
            The reconstruction.
        """
        empty = without_scatterers(chain)
        return cls(reference=empty(outputs=("expected",))["expected"].detach())

    @property
    def is_linear(self) -> bool:
        """Whether the reconstruction is affine: only with a fixed reference.

        Returns
        -------
        bool
            False when the reference is estimated from each frame (its median).
        """
        return self.reference is not None

    def linear(self, frames: Tensor) -> Tensor:
        """Return the contrast (affine in the frames for a given reference).

        Parameters
        ----------
        frames : Tensor
            Frames ``[B, C, H, W]``.

        Returns
        -------
        Tensor
            The contrast, same shape.
        """
        if frames.ndim != 4:
            raise StructureError(f"ISCATContrast takes [B, C, H, W], got {list(frames.shape)}")
        if self.reference is None:
            reference = frames.flatten(-2).median(-1).values[..., None, None]
        else:
            reference = torch.as_tensor(self.reference, dtype=frames.dtype, device=frames.device)
        return frames / reference - 1.0
