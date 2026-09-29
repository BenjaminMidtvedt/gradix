"""Static grids (§4.3, Appendix D.2).

Grids are made of Python numbers only, so they are hashable, JSON-serialisable and fixed for the
life of a Pipeline. All wavelengths share one spatial grid and one frequency grid; only the
per-wavelength pupil radius NA/λ changes. Sizes are rounded up to 2·3·5·7-smooth numbers because
prime FFT sizes are slow.

Coordinates follow §4.1: x runs along columns and y along rows, arrays are ``[..., Y, X]``, and
``shape`` is ``(Y, X)``. ``origin`` is the ``(x, y)`` position of the grid's top-left corner in
µm, so pixel ``(i, j)`` has its centre at ``(origin_x + (j + ½)·d, origin_y + (i + ½)·d)``.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Any

import torch
from torch import Tensor

__all__ = [
    "FreqGrid",
    "Grid2D",
    "PatchGrid",
    "PupilGrid",
    "PupilSupport",
    "VolumeGrid",
    "is_smooth",
    "nice_size",
]


def is_smooth(n: int) -> bool:
    """Return whether ``n`` has no prime factor above 7.

    Parameters
    ----------
    n : int
        A positive integer.

    Returns
    -------
    bool
        True for 2·3·5·7-smooth numbers.
    """
    if n < 1:
        return False
    for p in (2, 3, 5, 7):
        while n % p == 0:
            n //= p
    return n == 1


def nice_size(n: int) -> int:
    """Return the smallest 2·3·5·7-smooth integer at least ``n``.

    Parameters
    ----------
    n : int
        Minimum size.

    Returns
    -------
    int
        The smooth size (1 for ``n <= 1``).
    """
    m = max(int(n), 1)
    while not is_smooth(m):
        m += 1
    return m


def _check_shape(shape: tuple[int, ...], owner: str) -> None:
    if not all(isinstance(s, int) and s > 0 for s in shape):
        msg = f"{owner}.shape must hold positive ints, got {shape}"
        raise ValueError(msg)


def _check_positive(value: float, name: str) -> None:
    if not (math.isfinite(value) and value > 0):
        msg = f"{name} must be positive and finite, got {value}"
        raise ValueError(msg)


@dataclasses.dataclass(frozen=True)
class Grid2D:
    """A sampled plane in object space.

    Parameters
    ----------
    shape : tuple of int
        ``(Y, X)`` sample counts.
    spacing : float
        Sample spacing in µm (isotropic in v1).
    origin : tuple of float, default (0.0, 0.0)
        ``(x, y)`` of the grid's top-left corner in µm.
    """

    shape: tuple[int, int]
    spacing: float
    origin: tuple[float, float] = (0.0, 0.0)

    def __post_init__(self) -> None:
        object.__setattr__(self, "shape", tuple(int(s) for s in self.shape))
        object.__setattr__(self, "spacing", float(self.spacing))
        object.__setattr__(self, "origin", tuple(float(o) for o in self.origin))
        _check_shape(self.shape, "Grid2D")
        _check_positive(self.spacing, "Grid2D.spacing")
        if len(self.shape) != 2 or len(self.origin) != 2:
            msg = "Grid2D.shape and Grid2D.origin need two entries"
            raise ValueError(msg)

    @property
    def extent(self) -> tuple[float, float]:
        """Width and height ``(x, y)`` in µm.

        Returns
        -------
        tuple of float
            ``(X·d, Y·d)``.
        """
        return (self.shape[1] * self.spacing, self.shape[0] * self.spacing)

    def x(
        self, *, dtype: torch.dtype = torch.float32, device: torch.device | str | None = None
    ) -> Tensor:
        """Return the x coordinates of the sample centres.

        Parameters
        ----------
        dtype : torch.dtype, default torch.float32
            Dtype of the result.
        device : torch.device or str, optional
            Device of the result.

        Returns
        -------
        Tensor
            Shape [X], in µm.
        """
        j = torch.arange(self.shape[1], dtype=torch.float64, device=device)
        return (self.origin[0] + (j + 0.5) * self.spacing).to(dtype)

    def y(
        self, *, dtype: torch.dtype = torch.float32, device: torch.device | str | None = None
    ) -> Tensor:
        """Return the y coordinates of the sample centres.

        Parameters
        ----------
        dtype : torch.dtype, default torch.float32
            Dtype of the result.
        device : torch.device or str, optional
            Device of the result.

        Returns
        -------
        Tensor
            Shape [Y], in µm.
        """
        i = torch.arange(self.shape[0], dtype=torch.float64, device=device)
        return (self.origin[1] + (i + 0.5) * self.spacing).to(dtype)

    def freq(self, support: PupilSupport | None = None) -> FreqGrid:
        """Return the frequency grid of this grid.

        Parameters
        ----------
        support : PupilSupport, optional
            Pupil support to attach.

        Returns
        -------
        FreqGrid
            The FFT dual grid.
        """
        return FreqGrid(of=self, support=support)

    def to_json(self) -> dict[str, Any]:
        """Return a JSON-serialisable description.

        Returns
        -------
        dict
            The fields, with tuples as lists.
        """
        return {
            "type": "Grid2D",
            "shape": list(self.shape),
            "spacing": self.spacing,
            "origin": list(self.origin),
        }


@dataclasses.dataclass(frozen=True)
class PupilSupport:
    """The static set of frequency samples inside the largest pupil, ``|f| ≤ cutoff``.

    Parameters
    ----------
    cutoff : float
        Radius in cycles/µm, ``max over λ of NA/λ``.
    """

    cutoff: float

    def mask(self, grid: FreqGrid) -> Tensor:
        """Return the boolean support mask on a frequency grid.

        Parameters
        ----------
        grid : FreqGrid
            The frequency grid.

        Returns
        -------
        Tensor
            Shape [Ky, Kx], True inside the support.
        """
        fy = grid.fy(dtype=torch.float64)[:, None]
        fx = grid.fx(dtype=torch.float64)[None, :]
        return fx * fx + fy * fy <= self.cutoff * self.cutoff

    def count(self, grid: FreqGrid) -> int:
        """Return K, the number of samples in the support.

        Parameters
        ----------
        grid : FreqGrid
            The frequency grid.

        Returns
        -------
        int
            The count.
        """
        return int(self.mask(grid).sum())


@dataclasses.dataclass(frozen=True)
class FreqGrid:
    """The FFT dual of a :class:`Grid2D`; frequencies in cycles/µm, independent of λ.

    Parameters
    ----------
    of : Grid2D
        The spatial grid.
    support : PupilSupport, optional
        The pupil support, whose sample count is K.
    """

    of: Grid2D
    support: PupilSupport | None = None

    @property
    def shape(self) -> tuple[int, int]:
        """``(Ky, Kx)``, equal to the spatial shape.

        Returns
        -------
        tuple of int
            The shape.
        """
        return self.of.shape

    def fx(
        self, *, dtype: torch.dtype = torch.float32, device: torch.device | str | None = None
    ) -> Tensor:
        """Return the x frequencies in FFT order.

        Parameters
        ----------
        dtype : torch.dtype, default torch.float32
            Dtype of the result.
        device : torch.device or str, optional
            Device of the result.

        Returns
        -------
        Tensor
            Shape [Kx], cycles/µm.
        """
        return torch.fft.fftfreq(self.of.shape[1], d=self.of.spacing, dtype=dtype, device=device)

    def fy(
        self, *, dtype: torch.dtype = torch.float32, device: torch.device | str | None = None
    ) -> Tensor:
        """Return the y frequencies in FFT order.

        Parameters
        ----------
        dtype : torch.dtype, default torch.float32
            Dtype of the result.
        device : torch.device or str, optional
            Device of the result.

        Returns
        -------
        Tensor
            Shape [Ky], cycles/µm.
        """
        return torch.fft.fftfreq(self.of.shape[0], d=self.of.spacing, dtype=dtype, device=device)

    def to_json(self) -> dict[str, Any]:
        """Return a JSON-serialisable description.

        Returns
        -------
        dict
            The fields.
        """
        support = None if self.support is None else {"cutoff": self.support.cutoff}
        return {"type": "FreqGrid", "of": self.of.to_json(), "support": support}


@dataclasses.dataclass(frozen=True)
class PupilGrid:
    """Direction space u = λ₀·f⊥ (NA units): a disc ``|u| ≤ u_max`` on a square grid.

    Parameters
    ----------
    shape : tuple of int
        ``(Ny, Nx)`` samples across ``[-u_max, u_max]``.
    u_max : float
        Radius of the disc, in NA units.
    """

    shape: tuple[int, int]
    u_max: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "shape", tuple(int(s) for s in self.shape))
        _check_shape(self.shape, "PupilGrid")
        _check_positive(float(self.u_max), "PupilGrid.u_max")

    def to_json(self) -> dict[str, Any]:
        """Return a JSON-serialisable description.

        Returns
        -------
        dict
            The fields.
        """
        return {"type": "PupilGrid", "shape": list(self.shape), "u_max": self.u_max}


@dataclasses.dataclass(frozen=True)
class PatchGrid:
    """R×R regions of interest with a shared spacing; per-object origins are tensors.

    Parameters
    ----------
    shape : tuple of int
        ``(R, R)`` samples per patch.
    spacing : float
        Sample spacing in µm.
    """

    shape: tuple[int, int]
    spacing: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "shape", tuple(int(s) for s in self.shape))
        _check_shape(self.shape, "PatchGrid")
        _check_positive(float(self.spacing), "PatchGrid.spacing")

    def to_json(self) -> dict[str, Any]:
        """Return a JSON-serialisable description.

        Returns
        -------
        dict
            The fields.
        """
        return {"type": "PatchGrid", "shape": list(self.shape), "spacing": self.spacing}


@dataclasses.dataclass(frozen=True)
class VolumeGrid:
    """A dense interaction or product grid: a stack of planes.

    Parameters
    ----------
    xy : Grid2D
        The lateral grid.
    z0 : float
        Axial position of the first plane, in µm.
    dz : float
        Plane spacing, in µm.
    nz : int
        Number of planes.
    """

    xy: Grid2D
    z0: float
    dz: float
    nz: int

    def __post_init__(self) -> None:
        _check_positive(float(self.dz), "VolumeGrid.dz")
        if int(self.nz) < 1:
            msg = f"VolumeGrid.nz must be positive, got {self.nz}"
            raise ValueError(msg)

    def z(
        self, *, dtype: torch.dtype = torch.float32, device: torch.device | str | None = None
    ) -> Tensor:
        """Return the z of every plane.

        Parameters
        ----------
        dtype : torch.dtype, default torch.float32
            Dtype of the result.
        device : torch.device or str, optional
            Device of the result.

        Returns
        -------
        Tensor
            Shape [Z], in µm.
        """
        k = torch.arange(self.nz, dtype=torch.float64, device=device)
        return (self.z0 + k * self.dz).to(dtype)

    def to_json(self) -> dict[str, Any]:
        """Return a JSON-serialisable description.

        Returns
        -------
        dict
            The fields.
        """
        return {
            "type": "VolumeGrid",
            "xy": self.xy.to_json(),
            "z0": self.z0,
            "dz": self.dz,
            "nz": self.nz,
        }
