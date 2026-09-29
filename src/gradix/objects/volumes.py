"""Volumes (``gx.Voxels``; §4.6): values sampled on a regular grid.

A volume is one data object per image: a refractive index or index contrast for the coherent
paths (M4), or an emitter density, photons per voxel per exposure, for dense fluorescence
(``gx.imaging.Strata``, M1). The grid (shape, spacing, origin) is structure; the values are
tensors that may require gradients.
"""

from __future__ import annotations

import dataclasses
from typing import ClassVar

from torch import Tensor

from gradix import register
from gradix._core.errors import StructureError
from gradix._core.grid import Grid2D, VolumeGrid
from gradix.objects.spectrum import Spectrum
from gradix.schema.base import DataObject
from gradix.schema.fields import child, field, knob

__all__ = ["Voxels"]


@register.object_set("voxels")
@dataclasses.dataclass(frozen=True, kw_only=True, eq=False)
class Voxels(DataObject):
    """Values on a regular 3-D grid: a tomogram, a segmentation or an emitter density.

    Voxel ``(k, j, i)`` (plane, row, column) has its centre at ``origin + (i·dx, j·dy, k·dz)``.
    The grid is structure: the elements that read a volume use it as it is (``Strata`` needs its
    lateral samples on the camera's fine grid). Volumes placed by a tensor ``position`` and
    resampled onto an element's grid arrive with the raster lowering (cap-06).

    Parameters
    ----------
    values : Tensor
        ``[Z, Y, X]``, or per image ``[B, Z, Y, X]``: refractive indices (``"n"``), index
        contrast (``"dn"``) or photons per voxel per exposure (``"density"``).
    spacing : tuple of float
        ``(dz, dy, dx)`` voxel spacing, µm; dense fluorescence needs ``dy == dx``.
    origin : tuple of float, default (0.0, 0.0, 0.0)
        ``(x, y, z)`` of the centre of voxel ``(0, 0, 0)``, µm.
    quantity : {"n", "dn", "density"}, default "n"
        What the values are.
    emission : Spectrum, optional
        The emission spectrum of a density (required for ``"density"``).

    Examples
    --------
    >>> import torch, gradix as gx
    >>> cell = Voxels(values=torch.zeros(4, 8, 8), spacing=(0.5, 0.1, 0.1), quantity="density",
    ...               emission=gx.Spectrum.line(0.6))
    >>> cell.grid().nz
    4
    """

    kind: ClassVar[str] = "voxels"
    """The population kind the planner routes on."""

    values: Tensor = field(
        quantity="dimensionless", role="image", event=(-1, -1, -1), doc="values on the grid"
    )
    spacing: tuple[float, float, float] = knob(doc="(dz, dy, dx) voxel spacing, µm")
    origin: tuple[float, float, float] = knob(
        default=(0.0, 0.0, 0.0), doc="(x, y, z) of the first voxel's centre, µm"
    )
    quantity: str = knob(default="n", choices=("n", "dn", "density"), doc="what the values are")
    emission: Spectrum | None = child(default=None, doc="emission spectrum of a density")

    def __post_init__(self) -> None:
        object.__setattr__(self, "spacing", tuple(float(s) for s in self.spacing))
        object.__setattr__(self, "origin", tuple(float(o) for o in self.origin))
        if len(self.spacing) != 3 or len(self.origin) != 3 or min(self.spacing) <= 0:
            raise StructureError("Voxels needs spacing=(dz, dy, dx) > 0 and origin=(x, y, z)")
        if self.quantity == "density" and self.emission is None:
            msg = "an emitter density needs its emission spectrum"
            raise StructureError(msg, fix="pass emission=gx.Spectrum.line(λ)")
        super().__post_init__()

    @property
    def contrast(self) -> str:
        """``"emission"`` for an emitter density, else ``"coherent"`` (what the planner routes on).

        Returns
        -------
        str
            The contrast.
        """
        return "emission" if self.quantity == "density" else "coherent"

    def grid(self) -> VolumeGrid:
        """Return the volume grid of the values.

        Returns
        -------
        VolumeGrid
            Planes at ``origin.z + k·dz``, lateral samples on a :class:`~gradix.Grid2D` whose
            corner lies half a voxel before the first centre.

        Raises
        ------
        StructureError
            If the lateral spacing is anisotropic (``dy != dx``).
        """
        dz, dy, dx = self.spacing
        if abs(dy - dx) > 1e-9 * dx:
            raise StructureError(f"lateral voxel spacing must be isotropic, got dy={dy}, dx={dx}")
        shape = tuple(self.values.shape[-3:]) if isinstance(self.values, Tensor) else (1, 1, 1)
        x0, y0, z0 = self.origin
        xy = Grid2D((shape[1], shape[2]), dx, (x0 - 0.5 * dx, y0 - 0.5 * dx))
        return VolumeGrid(xy=xy, z0=z0, dz=dz, nz=shape[0])
