"""Spatial light modulators (``gx.devices``; §5.11): transmission maps from device parameters."""

from __future__ import annotations

import dataclasses

import torch
from torch import Tensor

from gradix import register
from gradix._core.errors import StructureError
from gradix.schema.base import DataObject
from gradix.schema.fields import field, knob
from gradix.schema.layout import canonical

__all__ = ["PhaseSLM"]


@register.data_object("device.phase_slm")
@dataclasses.dataclass(frozen=True, kw_only=True, eq=False)
class PhaseSLM(DataObject):
    """A phase-only spatial light modulator: the transmission ``e^{iφ}`` on its pixel grid.

    The device is placed by whatever consumes it: ``gx.light.Shaped`` reads it as the
    illumination pupil, its pixels spanning the condenser's NA. Its phase is a tensor, so a
    design can train it. A liquid-crystal modulator sets a retardance, so at another wavelength
    the phase scales as λ_design/λ; an unmodulated fraction η of the field (the zero order of
    fill factor and flicker) passes as t = (1 − η)·e^{iφ} + η.

    Parameters
    ----------
    phase : Tensor, optional
        Phase map in rad, ``[Q_y, Q_x]`` or per image ``[B, Q_y, Q_x]``; None is flat.
    shape : tuple of int, optional
        The pixel grid ``(Q_y, Q_x)``; needed without ``phase``, which must match it.
    pitch : float, optional
        Pixel pitch on the device, µm: its size when a ``Mask`` places it at a field plane
        (M3). A pupil device spans the NA and needs none.
    wavelength : float, optional
        Design wavelength (µm) at which ``phase`` holds; None is achromatic.
    zero_order : Tensor or float, default 0.0
        Unmodulated fraction η of the field, per image ``[B]``.

    Examples
    --------
    >>> import torch
    >>> slm = PhaseSLM(phase=torch.full((4, 4), torch.pi / 2))
    >>> t = slm.transmission(torch.tensor([[0.532]]))
    >>> tuple(t.shape), round(float(t[0, 0, 0, 0].imag), 6)
    ((1, 1, 4, 4), 1.0)
    """

    phase: Tensor | None = field(
        quantity="angle", role="image", event=(-1, -1), default=None, doc="phase map"
    )
    shape: tuple[int, int] | None = knob(default=None, doc="pixel grid (rows, columns)")
    pitch: float | None = knob(default=None, doc="pixel pitch on the device, µm")
    wavelength: float | None = knob(default=None, doc="design wavelength, µm")
    zero_order: Tensor | float = field(
        quantity="dimensionless",
        role="image",
        constraint="unit_interval",
        default=0.0,
        doc="unmodulated fraction of the field",
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.phase is None and self.shape is None:
            raise StructureError("a PhaseSLM needs phase= or shape=")
        if self.phase is not None and self.shape is not None:
            rows, cols = torch.as_tensor(self.phase).shape[-2:]
            if (int(rows), int(cols)) != tuple(self.shape):
                msg = f"phase has {int(rows)}×{int(cols)} pixels, shape= says {self.shape}"
                raise StructureError(msg)

    @property
    def grid(self) -> tuple[int, int]:
        """The pixel grid ``(Q_y, Q_x)``.

        Returns
        -------
        tuple of int
            Rows and columns.
        """
        if self.phase is None:
            assert self.shape is not None
            return int(self.shape[0]), int(self.shape[1])
        rows, cols = torch.as_tensor(self.phase).shape[-2:]
        return int(rows), int(cols)

    def transmission(self, wavelengths: Tensor) -> Tensor:
        """Return the complex transmission at vacuum wavelengths.

        Parameters
        ----------
        wavelengths : Tensor
            ``[B|1, L]`` vacuum wavelengths, µm.

        Returns
        -------
        Tensor
            Complex ``[B|1, L, Q_y, Q_x]``.
        """
        spec = self.schema()
        dtype, device = wavelengths.dtype, wavelengths.device
        if not dtype.is_floating_point:
            dtype = torch.float32
        if self.phase is None:
            phase = torch.zeros(1, *self.grid, dtype=dtype, device=device)
        else:
            phase = canonical(self.phase, spec["phase"], dtype=dtype, device=device)  # [B|1,Q,Q]
        phase = phase.to(torch.promote_types(phase.dtype, wavelengths.dtype))
        eta = canonical(self.zero_order, spec["zero_order"], dtype=phase.dtype, device=device)
        phase = phase[:, None]  # [B|1, 1, Q, Q]
        if self.wavelength is not None:
            phase = phase * (self.wavelength / wavelengths)[..., None, None]
        modulated = torch.polar(torch.ones_like(phase), phase)
        eta = eta.reshape(-1, 1, 1, 1)
        return (1.0 - eta) * modulated + eta
