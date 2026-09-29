"""Detection gratings for shearing interferometry (``stage.grating``; ADR-45, §5.12).

A periodic mask a short distance before the camera: a quadriwave lateral shearing
interferometer (QWLSI) or any other periodic wavefront sensor. The mask is expanded in its
diffraction orders, which the coherent imaging element forms analytically.
"""

from __future__ import annotations

import dataclasses
import math
from typing import ClassVar

import torch
from torch import Tensor

from gradix import register
from gradix._core.carriers import DiffractionOrders, PlaneWaves
from gradix._core.contract import Capabilities, Element, Slot, Static
from gradix._core.errors import StructureError
from gradix.schema.fields import field, knob
from gradix.schema.layout import canonical

__all__ = ["Grating", "order_indices"]

_HARTMANN = ((1.0, 1.0), (-1.0, 1.0), (1.0, -1.0), (-1.0, -1.0))


def _square_wave(m: int) -> float:
    """Return the Fourier coefficient of order m (odd) of a ±1 square wave of period 2Λ."""
    return 2.0 / (math.pi * abs(m)) * (-1.0) ** ((abs(m) - 1) // 2)


def order_indices(
    kind: str, max_order: int = 3, orders: tuple[tuple[float, float], ...] | None = None
) -> tuple[tuple[tuple[float, float], ...], tuple[complex, ...] | None]:
    """Return a grating kind's order indices and, for the built-in kinds, their coefficients.

    Parameters
    ----------
    kind : {"hartmann", "checkerboard", "custom"}
        The mask.
    max_order : int, default 3
        The checkerboard's largest odd index per axis.
    orders : tuple of (float, float), optional
        The custom kind's indices.

    Returns
    -------
    tuple
        The indices ``(m, n)``, orders at ``(m, n)/(2Λ)``, and the coefficients (None for the
        custom kind, whose coefficients are a tensor field).

    Examples
    --------
    >>> indices, coefficients = order_indices("checkerboard", max_order=1)
    >>> len(indices), round(abs(coefficients[0]), 4)  # (2/π)² each
    (4, 0.4053)
    """
    if kind == "hartmann":
        return _HARTMANN, (0.5, 0.5, 0.5, 0.5)
    if kind == "checkerboard":
        odd = [m for m in range(-max_order, max_order + 1) if m % 2]
        pairs = tuple((float(m), float(n)) for n in odd for m in odd)
        values = tuple(complex(_square_wave(int(m)) * _square_wave(int(n))) for m, n in pairs)
        return pairs, values
    if orders is None:
        raise StructureError("a custom grating needs orders=((m, n), ...) and coefficients=")
    return tuple((float(m), float(n)) for m, n in orders), None


@register.element("stage.grating")
@dataclasses.dataclass(frozen=True, eq=False)
class Grating(Element[DiffractionOrders]):
    """A periodic mask a distance d before the camera, expanded in diffraction orders (ADR-45).

    The mask's orders sit at ``G = R(rotation)·(m, n)/(2Λ)`` on the camera side. Each carries
    a copy of the image field, tilted by G and sheared by λd·G. Two orders whose indices differ
    by (2, 0) interfere at the fringe period Λ along the grating's x axis, and their shear is
    λd/Λ. The coherent imaging element forms every order analytically: a pupil phase for the
    shear and the propagation, and a carrier on the camera grid (§5.12). The kinds:

    - ``"hartmann"``: Primot's modified Hartmann mask, idealised as four equal orders
      (±1, ±1) of coefficient 1/2 that carry all the light. Their fringes do not change with d.
    - ``"checkerboard"``: a 0–π phase checkerboard of squares of side Λ, whose orders
      (m, n), m and n odd, have coefficients (2/π)²/(mn) up to sign. The orders up to
      ``max_order`` are kept; the rest of the light is dropped (19 % at 3, 13 % at 5).
    - ``"custom"``: the indices ``orders`` and complex ``coefficients`` of any periodic mask.

    Parameters
    ----------
    period : Tensor or float
        Fringe period Λ on the camera, µm (image space), per image ``[B]``: the Hartmann hole
        pitch, or the checkerboard's square side. Four camera pixels is typical.
    distance : Tensor or float
        Grating-to-camera distance d, µm, per image ``[B]``. The shear λd/Λ is typically a
        few camera pixels.
    rotation : Tensor or float, default 0.0
        Rotation of the grating's axes from the camera's, rad, per image ``[B]``.
    kind : {"hartmann", "checkerboard", "custom"}, default "hartmann"
        The mask.
    max_order : int, default 3
        The checkerboard's largest odd index per axis.
    orders : tuple of (float, float), optional
        Order indices ``(m, n)`` of a custom mask, orders at ``(m, n)/(2Λ)``.
    coefficients : Tensor, optional
        Complex ``[O]`` coefficients of a custom mask's orders.
    focused : {"camera", "grating"}, default "camera"
        The plane conjugate to the objective's focal plane: the camera, where each order is an
        in-focus sheared copy of the image, or the grating.

    Examples
    --------
    >>> import gradix as gx
    >>> orders = Grating(period=26.0, distance=500.0)(gx.light.PlaneWave(0.532)())
    >>> tuple(orders.frequencies.shape), round(float(orders.frequencies[0, 0, 0]), 5)
    ((1, 4, 2), 0.01923)
    """

    slot: ClassVar[Slot] = Slot.STAGE
    caps: ClassVar[Capabilities] = Capabilities(
        accepts=frozenset({PlaneWaves}), produces=DiffractionOrders
    )

    period: Tensor | float = field(
        quantity="length",
        role="image",
        constraint="positive",
        shape_affecting=True,
        doc="fringe period on the camera",
    )
    distance: Tensor | float = field(
        quantity="length", role="image", constraint="nonnegative", doc="grating to camera"
    )
    _: dataclasses.KW_ONLY
    rotation: Tensor | float = field(
        quantity="angle", role="image", default=0.0, doc="rotation of the grating's axes"
    )
    kind: str = knob(
        default="hartmann", choices=("hartmann", "checkerboard", "custom"), doc="the mask"
    )
    max_order: int = knob(default=3, doc="the checkerboard's largest odd index")
    orders: tuple[tuple[float, float], ...] | None = knob(
        default=None, doc="order indices of a custom mask"
    )
    coefficients: Tensor | None = field(
        quantity="dimensionless",
        role="shared",
        event=(-1,),
        dtype="number",
        default=None,
        doc="coefficients of a custom mask's orders",
    )
    focused: str = knob(default="camera", choices=("camera", "grating"), doc="the plane in focus")

    def __post_init__(self) -> None:
        super().__post_init__()
        custom = self.kind == "custom"
        if custom and (self.orders is None or self.coefficients is None):
            raise StructureError("a custom grating needs orders= and coefficients=")
        if not custom and (self.orders is not None or self.coefficients is not None):
            raise StructureError(
                f"orders= and coefficients= define a custom mask, not kind={self.kind!r}",
                fix='pass kind="custom"',
            )
        if custom and torch.as_tensor(self.coefficients).reshape(-1).shape[0] != len(
            self.orders or ()
        ):
            raise StructureError("coefficients= needs one value per order")
        if self.kind == "checkerboard" and (self.max_order < 1 or self.max_order % 2 == 0):
            raise StructureError("max_order must be a positive odd integer")

    def indices(self) -> tuple[tuple[float, float], ...]:
        """Return the order indices ``(m, n)``; the orders sit at ``R·(m, n)/(2Λ)``.

        Returns
        -------
        tuple of (float, float)
            The indices.
        """
        return order_indices(self.kind, self.max_order, self.orders)[0]

    def forward(self, *inputs: object, static: Static) -> DiffractionOrders:
        """Return the orders at the light's wavelengths.

        Parameters
        ----------
        *inputs : object
            The illumination's :class:`~gradix.PlaneWaves` (its wavelengths and precision).
        static : Static
            The (empty) configuration.

        Returns
        -------
        DiffractionOrders
            ``frequencies [B|1, O, 2]`` in cycles/µm on the camera side, ``coefficients
            [B|1, O, L]`` and ``distance [B|1]``.
        """
        if not inputs or not isinstance(inputs[0], PlaneWaves):
            raise StructureError("Grating takes the illumination's PlaneWaves")
        light = inputs[0]
        spec = self.schema()
        # geometry in fp64 whatever the scene's precision: the orders' carriers reach
        # hundreds of radians across the frame (the consumer forms phases in fp64 too)
        dtype, device = torch.float64, light.wavelengths.device
        cdtype = torch.complex128
        period = canonical(self.period, spec["period"], dtype=dtype, device=device)  # [B|1]
        distance = canonical(self.distance, spec["distance"], dtype=dtype, device=device)
        rotation = canonical(self.rotation, spec["rotation"], dtype=dtype, device=device)
        indices, values = order_indices(self.kind, self.max_order, self.orders)
        mn = torch.tensor(indices, dtype=dtype, device=device)  # [O, 2]
        cos, sin = torch.cos(rotation)[:, None], torch.sin(rotation)[:, None]  # [B|1, 1]
        scale = 1.0 / (2.0 * period)[:, None]  # [B|1, 1]
        gx = (cos * mn[:, 0] - sin * mn[:, 1]) * scale  # [B|1, O]
        gy = (sin * mn[:, 0] + cos * mn[:, 1]) * scale
        frequencies = torch.stack([gx, gy], dim=-1)  # [B|1, O, 2]
        if values is None:
            c = torch.as_tensor(self.coefficients, device=device).reshape(-1).to(cdtype)
        else:
            c = torch.tensor(values, dtype=cdtype, device=device)
        bins = light.wavelengths.shape[1]
        coefficients = c[None, :, None].expand(1, c.shape[0], bins)  # [1, O, L]
        return DiffractionOrders(
            frequencies=frequencies,
            coefficients=coefficients,
            distance=distance,
            focused=self.focused,
        )

    def __call__(
        self, *inputs: object, grid: object = None, static: Static | None = None
    ) -> DiffractionOrders:
        """Return the orders eagerly.

        Parameters
        ----------
        *inputs : object
            The illumination's :class:`~gradix.PlaneWaves`.
        grid : object, optional
            Unused; the orders are analytic.
        static : Static, optional
            Unused.

        Returns
        -------
        DiffractionOrders
            The orders.
        """
        return self.forward(*inputs, static=static or Static())
