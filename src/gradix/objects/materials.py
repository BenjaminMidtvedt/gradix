"""Materials (``gx.Material``, ``gx.materials``): refractive-index models n(λ) (§4.6).

A material maps vacuum wavelengths to (possibly complex) refractive indices. Media and samples
read indices only through materials, so dispersion is modelled once: a :class:`Constant` holds a
per-image index tensor (learnable), :class:`Cauchy` and :class:`Sellmeier` hold dispersion
coefficients. The library functions (:func:`Water`, :func:`Glass`, :func:`Oil`, :func:`IP_Dip`)
return common materials.
"""

from __future__ import annotations

import dataclasses
from typing import ClassVar

import torch
from torch import Tensor

from gradix.schema.base import DataObject
from gradix.schema.fields import field, knob
from gradix.schema.layout import canonical

__all__ = ["Cauchy", "Constant", "Glass", "IP_Dip", "Material", "Oil", "Sellmeier", "Water"]

D_LINE = 0.5876
"""The helium d line, µm: where a dispersive material's nominal index is quoted."""


@dataclasses.dataclass(frozen=True, kw_only=True, eq=False)
class Material(DataObject):
    """Base of materials: :meth:`index` returns n(λ)."""

    def index(
        self,
        wavelength: Tensor | float = D_LINE,
        *,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str | None = None,
    ) -> Tensor:
        """Return the refractive index at vacuum wavelengths.

        Parameters
        ----------
        wavelength : Tensor or float, default 0.5876
            Vacuum wavelengths, µm; the result broadcasts ``[B|1]`` (per-image values) against
            them as ``[B|1, 1, …]``.
        dtype : torch.dtype, default torch.float32
            Real dtype of the result (complex indices keep a complex dtype).
        device : torch.device or str, optional
            Device of the result.

        Returns
        -------
        Tensor
            The index, ``[B|1, *wavelength.shape[1:]]`` for a ``[B|1, …]`` wavelength tensor, or
            ``[B|1]`` for a number.
        """
        raise NotImplementedError(type(self).__name__)

    def _wavelength(
        self, wavelength: Tensor | float, dtype: torch.dtype, device: torch.device | str | None
    ) -> Tensor:
        return torch.as_tensor(wavelength, dtype=dtype, device=device)


@dataclasses.dataclass(frozen=True, eq=False)
class Constant(Material):
    """A non-dispersive material: the same (possibly complex) index at every wavelength.

    Parameters
    ----------
    n : Tensor or complex
        The index, or per image ``[B]``.

    Examples
    --------
    >>> float(Constant(1.33).index())
    1.3300000429153442
    """

    registry_name: ClassVar[str | None] = "material.constant"
    """Stable name for signatures and saved inputs."""

    n: Tensor | complex = field(
        quantity="index",
        role="image",
        dtype="number",
        shape_affecting=True,
        constraint="positive",
        doc="refractive index",
    )

    def index(
        self,
        wavelength: Tensor | float = D_LINE,
        *,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str | None = None,
    ) -> Tensor:
        """Return the index, broadcast against the wavelengths.

        Parameters
        ----------
        wavelength : Tensor or float, default 0.5876
            Vacuum wavelengths, µm (their values are not used).
        dtype : torch.dtype, default torch.float32
            Real dtype of the result.
        device : torch.device or str, optional
            Device of the result.

        Returns
        -------
        Tensor
            ``[B|1, 1, …]`` matching the wavelengths' trailing dimensions.
        """
        n = self.n
        is_complex = isinstance(n, complex) or (isinstance(n, Tensor) and n.is_complex())
        kind = (
            (torch.complex128 if dtype == torch.float64 else torch.complex64)
            if is_complex
            else dtype
        )
        value = canonical(n, self.schema()["n"], dtype=kind, device=device)  # [B|1]
        wl = torch.as_tensor(wavelength)
        return value.reshape(-1, *([1] * max(wl.ndim - 1, 0)))


@dataclasses.dataclass(frozen=True, kw_only=True, eq=False)
class Cauchy(Material):
    """Cauchy dispersion ``n(λ) = a + b/λ² + c/λ⁴`` (λ in µm).

    Parameters
    ----------
    a : float
        Constant term.
    b : float, default 0.0
        Coefficient of 1/λ², µm².
    c : float, default 0.0
        Coefficient of 1/λ⁴, µm⁴.

    Examples
    --------
    >>> round(float(Cauchy(a=1.5, b=0.004).index(0.5)), 4)
    1.516
    """

    registry_name: ClassVar[str | None] = "material.cauchy"
    """Stable name for signatures and saved inputs."""

    a: float = knob(doc="constant term")
    b: float = knob(default=0.0, doc="µm² coefficient")
    c: float = knob(default=0.0, doc="µm⁴ coefficient")

    def index(
        self,
        wavelength: Tensor | float = D_LINE,
        *,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str | None = None,
    ) -> Tensor:
        """Return ``a + b/λ² + c/λ⁴``.

        Parameters
        ----------
        wavelength : Tensor or float, default 0.5876
            Vacuum wavelengths, µm.
        dtype : torch.dtype, default torch.float32
            Dtype of the result.
        device : torch.device or str, optional
            Device of the result.

        Returns
        -------
        Tensor
            The index, same shape as the wavelengths.
        """
        wl2 = self._wavelength(wavelength, dtype, device) ** 2
        return self.a + self.b / wl2 + self.c / (wl2 * wl2)


@dataclasses.dataclass(frozen=True, kw_only=True, eq=False)
class Sellmeier(Material):
    """Sellmeier dispersion ``n² = 1 + Σ Bᵢλ²/(λ² − Cᵢ)`` (λ in µm, Cᵢ in µm²).

    Parameters
    ----------
    b : tuple of float
        The Bᵢ coefficients.
    c : tuple of float
        The Cᵢ coefficients, µm².

    Examples
    --------
    >>> round(float(Glass().index()), 4)
    1.5168
    """

    registry_name: ClassVar[str | None] = "material.sellmeier"
    """Stable name for signatures and saved inputs."""

    b: tuple[float, ...] = knob(doc="B coefficients")
    c: tuple[float, ...] = knob(doc="C coefficients, µm²")

    def index(
        self,
        wavelength: Tensor | float = D_LINE,
        *,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str | None = None,
    ) -> Tensor:
        """Return the Sellmeier index.

        Parameters
        ----------
        wavelength : Tensor or float, default 0.5876
            Vacuum wavelengths, µm.
        dtype : torch.dtype, default torch.float32
            Dtype of the result.
        device : torch.device or str, optional
            Device of the result.

        Returns
        -------
        Tensor
            The index, same shape as the wavelengths.
        """
        wl2 = self._wavelength(wavelength, dtype, device) ** 2
        n2 = torch.ones_like(wl2)
        for bi, ci in zip(self.b, self.c, strict=True):
            n2 = n2 + bi * wl2 / (wl2 - ci)
        return torch.sqrt(n2)


def Water() -> Cauchy:
    """Return water at room temperature (n_D = 1.333, Abbe number ≈ 55.7).

    Returns
    -------
    Cauchy
        The material.
    """
    return Cauchy(a=1.32399, b=0.00313)


def Glass() -> Sellmeier:
    """Return Schott N-BK7, the usual coverslip glass (n_d = 1.5168).

    Returns
    -------
    Sellmeier
        The material.
    """
    return Sellmeier(
        b=(1.03961212, 0.231792344, 1.01046945),
        c=(0.00600069867, 0.0200179144, 103.560653),
    )


def Oil() -> Cauchy:
    """Return a type-F immersion oil (n_e = 1.518 at 546.1 nm, Abbe number ν_e = 41).

    Returns
    -------
    Cauchy
        The material.
    """
    return Cauchy(a=1.49602, b=0.006554)


def IP_Dip() -> Cauchy:
    """Return polymerised IP-Dip photoresist (two-photon lithography; n ≈ 1.548 at 589 nm).

    Returns
    -------
    Cauchy
        The material.
    """
    return Cauchy(a=1.5273, b=0.0065456, c=0.00025345)
