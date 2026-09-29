"""Spectra: wavelength bins with power weights (§4.5)."""

from __future__ import annotations

import dataclasses
import math
from typing import ClassVar

import numpy as np
import torch
from torch import Tensor

from gradix.schema.base import DataObject
from gradix.schema.fields import field
from gradix.schema.layout import canonical

__all__ = ["Spectrum"]


@dataclasses.dataclass(frozen=True, eq=False)
class Spectrum(DataObject):
    """A spectrum: vacuum wavelengths and the power weight of each bin.

    Weights are power per bin, including the bin width; emission and source spectra are
    normalised so the weights sum to 1. A per-image spectrum is a ``[B, L]`` tensor.

    Parameters
    ----------
    wavelengths : Tensor or float
        Vacuum wavelengths in µm: a scalar (one line), ``[L]`` or ``[B, L]``.
    weights : Tensor or float, optional
        Power weights, same form as ``wavelengths``; None gives every bin ``1/L``.

    Examples
    --------
    >>> from gradix.units import nm
    >>> Spectrum.line(600 * nm).bins
    1
    >>> Spectrum.band(680 * nm, 30 * nm, bins=5).bins
    5
    """

    registry_name: ClassVar[str | None] = "spectrum"
    """Stable name for signatures and saved inputs."""

    wavelengths: Tensor | float = field(
        quantity="wavelength",
        role="wavelength",
        shape_affecting=True,
        constraint="positive",
        doc="vacuum wavelengths",
    )
    _: dataclasses.KW_ONLY
    weights: Tensor | float | None = field(
        quantity="dimensionless",
        role="wavelength",
        constraint="nonnegative",
        default=None,
        doc="power weight of each bin",
    )

    @classmethod
    def line(cls, wavelength: Tensor | float) -> Spectrum:
        """Return a single spectral line.

        Parameters
        ----------
        wavelength : Tensor or float
            Vacuum wavelength in µm, or a per-image ``[B]`` tensor.

        Returns
        -------
        Spectrum
            One bin with weight 1.
        """
        if isinstance(wavelength, Tensor) and wavelength.ndim == 1:
            return cls(wavelength[:, None])
        return cls(wavelength)

    @classmethod
    def band(cls, center: Tensor | float, fwhm: Tensor | float, *, bins: int = 5) -> Spectrum:
        """Return a Gaussian band sampled at Gauss–Hermite nodes.

        The nodes integrate polynomials of degree ``2·bins − 1`` exactly against the Gaussian
        profile.

        Parameters
        ----------
        center : Tensor or float
            Centre wavelength in µm, or a per-image ``[B]`` tensor.
        fwhm : Tensor or float
            Full width at half maximum in µm, or a per-image ``[B]`` tensor.
        bins : int, default 5
            Number of wavelength bins L.

        Returns
        -------
        Spectrum
            ``bins`` bins whose weights sum to 1. Tensor centres and widths keep their
            gradients (the wavelengths are differentiable functions of them).
        """
        x, w = np.polynomial.hermite_e.hermegauss(bins)
        c = torch.as_tensor(center)
        width = torch.as_tensor(fwhm)
        floating = [t.dtype for t in (c, width) if t.is_floating_point()]
        dtype = (
            torch.promote_types(*floating)
            if len(floating) == 2
            else (floating[0] if floating else torch.get_default_dtype())
        )
        device = c.device if c.device.type != "cpu" else width.device
        c = c.to(dtype=dtype, device=device)
        sigma = width.to(dtype=dtype, device=device) / (2.0 * math.sqrt(2.0 * math.log(2.0)))
        nodes = torch.as_tensor(x, dtype=dtype, device=device)
        weights = torch.as_tensor(w / w.sum(), dtype=dtype, device=device)
        if c.ndim == 1 or sigma.ndim == 1:
            wl = c.reshape(-1, 1) + sigma.reshape(-1, 1) * nodes
            return cls(wl, weights=weights.expand_as(wl).clone())
        return cls(c + sigma * nodes, weights=weights)

    @classmethod
    def table(cls, wavelengths: Tensor, spectrum: Tensor, *, normalize: bool = True) -> Spectrum:
        """Return a tabulated spectrum.

        Parameters
        ----------
        wavelengths : Tensor
            Bin wavelengths in µm, ``[L]`` or ``[B, L]``.
        spectrum : Tensor
            Power in each bin (including the bin width), same shape.
        normalize : bool, default True
            Scale the weights to sum to 1 along L.

        Returns
        -------
        Spectrum
            The spectrum.
        """
        weights = spectrum / spectrum.sum(dim=-1, keepdim=True) if normalize else spectrum
        return cls(wavelengths, weights=weights)

    @property
    def bins(self) -> int:
        """The number of wavelength bins L.

        Returns
        -------
        int
            L.
        """
        wl = self.wavelengths
        return int(wl.shape[-1]) if isinstance(wl, Tensor) and wl.ndim > 0 else 1

    def arrays(
        self, *, dtype: torch.dtype = torch.float32, device: torch.device | str | None = None
    ) -> tuple[Tensor, Tensor]:
        """Return wavelengths and weights in canonical layout, ``[B|1, L]`` each.

        Parameters
        ----------
        dtype : torch.dtype, default torch.float32
            Dtype of the results.
        device : torch.device or str, optional
            Device of the results.

        Returns
        -------
        tuple of Tensor
            Wavelengths (µm) and power weights; missing weights are ``1/L``.
        """
        spec = self.schema()
        wl = canonical(self.wavelengths, spec["wavelengths"], dtype=dtype, device=device)
        if self.weights is None:
            w = torch.full_like(wl, 1.0 / wl.shape[-1])
        else:
            w = canonical(self.weights, spec["weights"], dtype=dtype, device=device)
            w = torch.broadcast_to(w, torch.broadcast_shapes(w.shape, wl.shape))
        return wl, w
