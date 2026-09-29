"""Polarisation states of sources (§4.5).

A state is defined after the condenser optics. Unpolarised light is two incoherent modes on M.
"""

from __future__ import annotations

import dataclasses
import math
from typing import ClassVar

import torch
from torch import Tensor

from gradix.schema.base import DataObject
from gradix.schema.fields import field, knob

__all__ = ["Polarization"]


@dataclasses.dataclass(frozen=True, kw_only=True, eq=False)
class Polarization(DataObject):
    """A polarisation state: a Jones vector, or unpolarised light.

    Build states with :meth:`linear`, :meth:`circular`, :meth:`jones` or :meth:`unpolarized`.

    Parameters
    ----------
    vector : Tensor, optional
        Complex Jones vector ``[2]`` or per-image ``[B, 2]`` (x, y components); normalised on
        use. None with ``incoherent=True``.
    incoherent : bool, default False
        Unpolarised light: two incoherent modes on M with half the power each (see
        :meth:`unpolarized`).
    """

    registry_name: ClassVar[str | None] = "light.polarization"
    """Stable name for signatures and saved inputs."""

    vector: Tensor | None = field(
        quantity="dimensionless",
        role="image",
        event=(2,),
        dtype="complex",
        default=None,
        doc="Jones vector (x, y)",
    )
    incoherent: bool = knob(default=False, doc="two incoherent modes")

    @classmethod
    def linear(cls, angle: Tensor | float) -> Polarization:
        """Return linear polarisation at ``angle`` rad from +x toward +y.

        Parameters
        ----------
        angle : Tensor or float
            Angle of the electric field, rad; a ``[B]`` tensor gives one state per image.

        Returns
        -------
        Polarization
            The state.
        """
        a = torch.as_tensor(angle, dtype=torch.float64)
        vec = torch.stack([torch.cos(a), torch.sin(a)], dim=-1).to(torch.complex64)
        return cls(vector=vec)

    @classmethod
    def circular(cls, handedness: int = 1) -> Polarization:
        """Return circular polarisation, ``(1, ±i)/√2``.

        Parameters
        ----------
        handedness : {1, -1}, default 1
            +1 for ``(1, i)/√2``, −1 for ``(1, −i)/√2``.

        Returns
        -------
        Polarization
            The state.
        """
        s = 1.0 / math.sqrt(2.0)
        return cls(vector=torch.tensor([s, 1j * s * handedness], dtype=torch.complex64))

    @classmethod
    def jones(cls, vector: Tensor) -> Polarization:
        """Return an arbitrary Jones vector.

        Parameters
        ----------
        vector : Tensor
            Complex ``[2]`` or ``[B, 2]``.

        Returns
        -------
        Polarization
            The state.
        """
        return cls(vector=vector)

    @classmethod
    def unpolarized(cls) -> Polarization:
        """Return unpolarised light.

        Returns
        -------
        Polarization
            Two incoherent modes on M.
        """
        return cls(incoherent=True)

    def modes(
        self, *, dtype: torch.dtype = torch.complex64, device: torch.device | str | None = None
    ) -> Tensor:
        """Return the Jones vectors of the incoherent modes, with power weights folded in.

        Parameters
        ----------
        dtype : torch.dtype, default torch.complex64
            Complex dtype of the result.
        device : torch.device or str, optional
            Device of the result.

        Returns
        -------
        Tensor
            ``[B|1, M, 2]``: M = 1 for a Jones state (unit norm), M = 2 for unpolarised light
            (x and y, each of norm 1/√2).
        """
        if self.incoherent or self.vector is None:
            s = 1.0 / math.sqrt(2.0)
            return torch.tensor([[[s, 0.0], [0.0, s]]], dtype=dtype, device=device)
        v = self.vector.to(dtype=dtype, device=device)
        v = v.reshape(-1, 2)
        v = v / torch.linalg.vector_norm(v, dim=-1, keepdim=True)
        return v[:, None, :]
