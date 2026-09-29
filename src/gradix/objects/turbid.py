"""A phenomenological turbid slab (``gx.env.TurbidSlab``; cap-07, §5.3).

Light emitted at depth ``d`` below the slab's objective-side surface reaches the objective partly
ballistic, attenuated by Beer–Lambert, ``a = exp(−μ·d)``, and blurred by small-angle scattering
(a Gaussian of width ``β·d``), and partly scattered, ``1 − a``, into a broad diffuse halo (a
Gaussian of width ``w``; incoherent). The collected power is kept. It is the minimum fidelity light
sheets in tissue need, not radiative transfer (§2).
"""

from __future__ import annotations

import dataclasses
import math

import torch
from torch import Tensor

from gradix import register
from gradix.schema.base import DataObject
from gradix.schema.fields import field
from gradix.schema.layout import canonical

__all__ = ["TurbidSlab"]


@register.data_object("env.turbid_slab")
@dataclasses.dataclass(frozen=True, kw_only=True, eq=False)
class TurbidSlab(DataObject):
    """Beer–Lambert attenuation × depth blur, plus a diffuse background, for dense emission.

    Parameters
    ----------
    attenuation : Tensor or float
        Scattering coefficient μ, µm⁻¹: the ballistic fraction from depth d is ``exp(−μ·d)``.
    blur : Tensor or float, default 0.0
        Width of the ballistic blur per unit depth, β (µm per µm).
    diffuse : Tensor or float, default 5.0
        Width of the diffuse halo, w, µm.
    surface : Tensor or float, default 0.0
        Height of the slab's objective-side surface, µm; depth is ``surface − z`` (≥ 0).

    Examples
    --------
    >>> slab = TurbidSlab(attenuation=0.01, blur=0.05)
    >>> float(slab.ballistic(torch.tensor([-100.0]))[0, 0])  # e^{-1} at 100 µm
    0.3678794503211975
    """

    attenuation: Tensor | float = field(
        quantity="attenuation", role="image", constraint="nonnegative", doc="μ, 1/µm"
    )
    blur: Tensor | float = field(
        quantity="dimensionless",
        role="image",
        constraint="nonnegative",
        default=0.0,
        doc="blur width per depth",
    )
    diffuse: Tensor | float = field(
        quantity="length", role="image", constraint="positive", default=5.0, doc="halo width"
    )
    surface: Tensor | float = field(
        quantity="length", role="image", default=0.0, doc="objective-side surface height"
    )

    def _get(self, name: str, like: Tensor) -> Tensor:
        value = canonical(getattr(self, name), self.schema()[name], dtype=like.dtype)
        return value.to(like.device).reshape(-1, 1)  # [B|1, 1]

    def depth(self, z: Tensor) -> Tensor:
        """Return the depth of planes below the surface, µm, ``[B|1, Z]``.

        Parameters
        ----------
        z : Tensor
            Plane heights ``[Z]``, µm.

        Returns
        -------
        Tensor
            ``max(surface − z, 0)``.
        """
        return torch.clamp(self._get("surface", z) - z, min=0.0)

    def ballistic(self, z: Tensor) -> Tensor:
        """Return the ballistic fraction ``exp(−μ·d)`` of planes, ``[B|1, Z]``.

        Parameters
        ----------
        z : Tensor
            Plane heights ``[Z]``, µm.

        Returns
        -------
        Tensor
            Fractions in (0, 1].
        """
        return torch.exp(-self._get("attenuation", z) * self.depth(z))

    def transfer(self, z: Tensor, fy: Tensor, fx: Tensor) -> Tensor:
        """Return the slab's transfer function per plane on a frequency grid.

        Parameters
        ----------
        z : Tensor
            Plane heights ``[Z]``, µm.
        fy : Tensor
            Spatial frequencies along y ``[Fy]``, cycles/µm.
        fx : Tensor
            Spatial frequencies along x ``[Fx]``, cycles/µm.

        Returns
        -------
        Tensor
            ``a·G(β·d) + (1 − a)·G(w)``, ``[B|1, 1, 1, Z, Fy, Fx]``, with ``G(σ)`` the transfer of a
            unit Gaussian, ``exp(−2π²σ²|f|²)``: 1 at zero frequency (power is kept).
        """
        f2 = (fy[:, None] ** 2 + fx[None, :] ** 2).to(z.dtype)  # [Fy, Fx]
        a = self.ballistic(z)[..., None, None]  # [B|1, Z, 1, 1]
        sigma = (self._get("blur", z) * self.depth(z))[..., None, None]
        halo = self._get("diffuse", z)[..., None, None]
        ballistic = torch.exp(-2.0 * math.pi**2 * sigma**2 * f2)
        diffuse = torch.exp(-2.0 * math.pi**2 * halo**2 * f2)
        out = a * ballistic + (1.0 - a) * diffuse  # [B|1, Z, Fy, Fx]
        return out[:, None, None]
