"""In-line (Gabor) holography: background normalisation and angular-spectrum refocusing (§5.12)."""

from __future__ import annotations

import dataclasses

import torch
from torch import Tensor

from gradix._core.errors import StructureError
from gradix.compose.chain import Chain
from gradix.ops.propagation import angular_spectrum
from gradix.recon.base import Reconstruction, optics_of

__all__ = ["Inline"]


@dataclasses.dataclass(frozen=True)
class Inline(Reconstruction):
    """Refocus in-line holograms: the background-normalised hologram, back-propagated to a plane.

    A hologram of weak scatterers is I = I_b·|1 + E_s/E_b|² ≈ I_b·(1 + 2Re(E_s/E_b)), so
    ``h = I/I_b − 1 ≈ E_s/E_b + (E_s/E_b)*``. The reconstruction propagates ``h`` (angular
    spectrum, band-limited to the objective's NA) from the focal plane to the plane
    ``distance`` above it and adds the background: the object term of a sphere at that depth
    comes into focus with its full scattered field, while its conjugate twin is defocused by
    twice the distance (Gabor's twin image). The map is affine in the frames.

    Parameters
    ----------
    pitch : float
        Object-space pixel pitch, µm (camera pixel size / magnification).
    wavelength : float
        Vacuum wavelength, µm.
    n : float
        Refractive index of the medium.
    na : float
        Numerical aperture: the band kept.
    distance : float or Tensor, default 0.0
        Reconstruction plane relative to the focal plane, µm (+z toward the objective, §4.1), a
        number or per image ``[B]``.
    background : float or Tensor, optional
        Background intensity in the frames' unit: a number, a frame ``[H, W]`` or per image
        ``[B, 1, H, W]``; None uses each frame's median.
    pad : int, default 16
        Zero padding of ``h`` against periodic wrap, pixels.

    Examples
    --------
    >>> import torch
    >>> recon = Inline(pitch=0.1, wavelength=0.532, n=1.33, na=0.8, background=100.0)
    >>> recon(torch.full((1, 1, 16, 16), 100.0)).abs().max().item()  # an empty hologram
    1.0
    """

    pitch: float
    wavelength: float
    n: float
    na: float
    distance: float | Tensor = 0.0
    background: float | Tensor | None = None
    pad: int = 16

    @classmethod
    def from_chain(
        cls,
        chain: Chain,
        distance: float | Tensor = 0.0,
        background: float | Tensor | None = None,
        pad: int = 16,
    ) -> Inline:
        """Read the pitch, wavelength, index and NA from a coherent Chain.

        Parameters
        ----------
        chain : Chain
            A coherent Chain (plane-wave light, a coherent imaging element, a medium).
        distance : float or Tensor, default 0.0
            Reconstruction plane relative to the focal plane, µm.
        background : float or Tensor, optional
            Background intensity in the camera's unit; None uses each frame's median.
        pad : int, default 16
            Zero padding, pixels.

        Returns
        -------
        Inline
            The reconstruction.
        """
        o = optics_of(chain)
        return cls(o.pitch, o.wavelength, o.n, o.na, distance, background, pad)

    def linear(self, frames: Tensor) -> Tensor:
        """Return the refocused field: affine in the frames (the part a bound measures).

        Parameters
        ----------
        frames : Tensor
            Holograms ``[B, C, H, W]`` in the camera's unit.

        Returns
        -------
        Tensor
            Complex fields ``[B, C, H, W]`` normalised to the background (1 without
            scatterers).
        """
        if frames.ndim != 4:
            raise StructureError(f"Inline takes frames [B, C, H, W], got {list(frames.shape)}")
        frames = frames.to(torch.float64) if frames.dtype == torch.float64 else frames.float()
        if self.background is None:
            background = frames.flatten(-2).median(-1).values[..., None, None]
        else:
            background = torch.as_tensor(self.background, dtype=frames.dtype, device=frames.device)
        h = frames / background - 1.0  # ≈ E_s/E_b + its conjugate
        cdtype = torch.complex128 if frames.dtype == torch.float64 else torch.complex64
        distance = torch.as_tensor(self.distance, dtype=frames.dtype, device=frames.device)
        scattered = angular_spectrum(
            h.to(cdtype),
            self.pitch,
            self.wavelength,
            self.n,
            distance,
            band=self.na,
            pad=self.pad,
        )
        return 1.0 + scattered
