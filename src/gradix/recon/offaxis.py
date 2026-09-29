"""Off-axis digital holography: demodulate the +1 sideband into the complex field (§5.12)."""

from __future__ import annotations

import dataclasses
import math

import torch
from torch import Tensor

from gradix._core.errors import StructureError
from gradix.compose.chain import Chain
from gradix.light.reference import ReferenceBeam
from gradix.recon.base import Reconstruction, optics_of

__all__ = ["OffAxis"]


@dataclasses.dataclass(frozen=True)
class OffAxis(Reconstruction):
    """Recover the complex image field from off-axis holograms.

    A hologram is I = |E_o|² + |E_r|² + E_o·E_r* + E_o*·E_r, with the reference
    E_r = a·e^{i(2πf_c·r + φ)} tilted by f_c = u_ref/λ. The term E_o·E_r* sits at −f_c:
    multiplying the frames by e^{+i2πf_c·r} (exactly, in real space, so the carrier need not lie
    on the FFT grid) brings it to the origin, a soft circular window of radius ``radius``/λ
    keeps it, the pixel MTF it passed through (sinc((f − f_c)·p) at baseband f) is undone, and
    E_o = window(…)·e^{iφ}/(a·p²). The frame mean is removed first, so the strong constant
    |E_r|² + |E_b|² cannot leak into the window. Everything before the optional normalisation is
    linear in the frames.

    Parameters
    ----------
    pitch : float
        Object-space pixel pitch, µm.
    wavelength : float
        Vacuum wavelength, µm.
    na : float
        Numerical aperture of the objective.
    carrier : tuple of float
        The reference's object-space direction ``(u_x, u_y)``, NA units (Mag·sin θ).
    amplitude : float, default 1.0
        Reference amplitude √I_ref, √(photons/µm²), for frames in photons per pixel.
    phase : float, default 0.0
        The reference's phase at the camera's origin, rad.
    background : complex or Tensor, optional
        Field to divide by: the unscattered field (a number, or ``[B|1, C|1, H, W]``), which
        normalises an empty field to 1; None keeps √(photons/µm²).
    radius : float, optional
        Window radius in NA units; None is NA plus two frequency cells, which keeps the whole
        pupil band.
    mtf : bool, default True
        Undo the pixel MTF on the sideband.

    Examples
    --------
    >>> import torch
    >>> recon = OffAxis(pitch=0.1, wavelength=0.532, na=0.4, carrier=(1.3, 1.3))
    >>> recon(torch.full((1, 1, 32, 32), 5.0)).abs().max().item() < 1e-5  # no fringes, no field
    True
    """

    pitch: float
    wavelength: float
    na: float
    carrier: tuple[float, float]
    amplitude: float = 1.0
    phase: float = 0.0
    background: complex | Tensor | None = None
    radius: float | None = None
    mtf: bool = True

    @classmethod
    def from_chain(
        cls,
        chain: Chain,
        reference: str = "reference",
        *,
        normalize: bool = True,
        radius: float | None = None,
        mtf: bool = True,
    ) -> OffAxis:
        """Read the optics and the reference from a coherent Chain.

        Parameters
        ----------
        chain : Chain
            A coherent Chain with an image-side :class:`~gradix.light.ReferenceBeam`.
        reference : str, default "reference"
            The reference's name in ``chain.references``.
        normalize : bool, default True
            Divide by the Chain's unscattered field at the pixel centres (rendered without
            scatterers), as ``gx.out.Field(normalize="background")`` does.
        radius : float, optional
            Window radius in NA units.
        mtf : bool, default True
            Undo the pixel MTF.

        Returns
        -------
        OffAxis
            The reconstruction.

        Raises
        ------
        StructureError
            If the Chain has no such reference, or its values differ between images.
        """
        o = optics_of(chain)
        beam = chain.references.get(reference)
        if not isinstance(beam, ReferenceBeam):
            known = sorted(chain.references)
            raise StructureError(f"no reference beam {reference!r}", fix=f"references: {known}")
        angle = torch.zeros(2, dtype=torch.float64) if beam.angle is None else beam.angle
        angle = torch.as_tensor(angle, dtype=torch.float64).detach().reshape(-1, 2)
        if angle.shape[0] > 1 and not bool((angle == angle[0]).all()):
            raise StructureError("the reference's angle differs between images")
        carrier = tuple(float(o.magnification * math.sin(float(a))) for a in angle[0])
        irradiance = torch.as_tensor(beam.irradiance, dtype=torch.float64).reshape(-1)
        phase = torch.as_tensor(beam.phase, dtype=torch.float64).reshape(-1)
        if bool((irradiance != irradiance[0]).any()) or bool((phase != phase[0]).any()):
            raise StructureError("the reference's irradiance and phase must be one number")
        background = None
        if normalize:
            empty = chain.replace(scatterers={}, references={})
            field = empty(outputs={"E": _field(normalize="none")})
            background = field["E"].detach()
        return cls(
            pitch=o.pitch,
            wavelength=o.wavelength,
            na=o.na,
            carrier=(carrier[0], carrier[1]),
            amplitude=float(irradiance[0].sqrt()),
            phase=float(phase[0]),
            background=background,
            radius=radius,
            mtf=mtf,
        )

    def linear(self, frames: Tensor) -> Tensor:
        """Return the complex field at the pixel centres (linear up to the normalisation).

        Parameters
        ----------
        frames : Tensor
            Holograms ``[B, C, H, W]`` in photons per pixel (offsets cancel).

        Returns
        -------
        Tensor
            Complex ``[B, C, H, W]``: the field in √(photons/µm²), or divided by
            ``background``.
        """
        if frames.ndim != 4:
            raise StructureError(f"OffAxis takes frames [B, C, H, W], got {list(frames.shape)}")
        real = torch.float64 if frames.dtype == torch.float64 else torch.float32
        cdtype = torch.complex128 if real == torch.float64 else torch.complex64
        frames = frames.to(real)
        frames = frames - frames.mean((-2, -1), keepdim=True)  # the constant cannot leak
        height, width = frames.shape[-2:]
        device, p, wl = frames.device, self.pitch, self.wavelength
        fcx, fcy = (u / wl for u in self.carrier)  # cycles/µm
        ys = (torch.arange(height, dtype=torch.float64, device=device) + 0.5) * p
        xs = (torch.arange(width, dtype=torch.float64, device=device) + 0.5) * p
        # the carrier phase in fp64, reduced mod 2π before the cast (§4.1)
        phase = torch.remainder(
            2.0 * math.pi * (fcx * xs[None, :] + fcy * ys[:, None]), 2 * math.pi
        )
        demod = frames * torch.polar(torch.ones_like(phase), phase).to(cdtype)
        spectrum = torch.fft.fft2(demod)
        fy = torch.fft.fftfreq(height, d=p, dtype=torch.float64, device=device)[:, None]
        fx = torch.fft.fftfreq(width, d=p, dtype=torch.float64, device=device)[None, :]
        cell = 1.0 / (p * max(height, width))  # one frequency cell, cycles/µm
        band = (self.na / wl + 2.0 * cell) if self.radius is None else self.radius / wl
        edge = torch.clamp((band - torch.sqrt(fx * fx + fy * fy)) / cell + 0.5, 0.0, 1.0)
        window = edge * edge * (3.0 - 2.0 * edge)
        if self.mtf:  # the sideband passed the pixel MTF at its original frequencies f − f_c
            mtf = torch.sinc((fx - fcx) * p) * torch.sinc((fy - fcy) * p)
            window = torch.where(window > 0, window / mtf, torch.zeros_like(window))
        field = torch.fft.ifft2(spectrum * window.to(cdtype))
        field = field * (
            complex(math.cos(self.phase), math.sin(self.phase)) / (self.amplitude * p * p)
        )
        if self.background is not None:
            background = torch.as_tensor(self.background, device=device)
            field = field / background.to(cdtype)
        return field


def _field(normalize: str) -> object:
    from gradix.compose.outputs import Field

    return Field(normalize=normalize, sampling="centre")
