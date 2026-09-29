"""Units and physical quantities.

gradix works in micrometres: every length, vacuum wavelength and optical path difference is in
µm, time is in seconds, angles are in radians and counts are in photons (§4.1 of the plan). The
constants below convert other units into these, so ``532 * nm`` is 0.532 and ``6.5 * um`` is 6.5.

Every schema field names a :class:`Quantity`. A quantity records its internal unit and its power
of length, which is how :func:`gradix.from_si` and :func:`gradix.to_si` convert a whole tree
between SI units and the internal ones.

Examples
--------
>>> from gradix.units import nm, um
>>> 532 * nm
0.532
>>> QUANTITIES["irradiance"].si_factor
1e-12
"""

from __future__ import annotations

import dataclasses
import math
from typing import Final

__all__ = [
    "QUANTITIES",
    "Quantity",
    "deg",
    "m",
    "mm",
    "ms",
    "nm",
    "rad",
    "register_quantity",
    "s",
    "um",
    "us",
]

um: Final = 1.0
"""One micrometre, the internal length unit."""
nm: Final = 1e-3
"""One nanometre in µm."""
mm: Final = 1e3
"""One millimetre in µm."""
m: Final = 1e6
"""One metre in µm."""
s: Final = 1.0
"""One second, the internal time unit."""
ms: Final = 1e-3
"""One millisecond in s."""
us: Final = 1e-6
"""One microsecond in s."""
rad: Final = 1.0
"""One radian, the internal angle unit."""
deg: Final = math.pi / 180.0
"""One degree in rad."""


@dataclasses.dataclass(frozen=True)
class Quantity:
    """A physical quantity that schema fields declare.

    Parameters
    ----------
    name : str
        Registry name, such as ``"length"``.
    unit : str
        The internal unit, for documentation (``"µm"``, ``"photons/µm²"``).
    length_power : int
        Power of length in the internal unit. A value in SI units is multiplied by
        ``1e6 ** length_power`` to reach the internal unit.
    doc : str
        One-line description.
    """

    name: str
    unit: str
    length_power: int
    doc: str

    @property
    def si_factor(self) -> float:
        """Factor that converts a value in SI units into the internal unit.

        Returns
        -------
        float
            ``1e6 ** length_power``.
        """
        return 1e6**self.length_power


QUANTITIES: dict[str, Quantity] = {}
"""Registered quantities by name. Plugins add theirs with :func:`register_quantity`."""


def register_quantity(quantity: Quantity, *, override: bool = False) -> Quantity:
    """Register a quantity so that schema fields may declare it.

    Parameters
    ----------
    quantity : Quantity
        The quantity to register.
    override : bool, default False
        Replace an existing quantity of the same name instead of raising.

    Returns
    -------
    Quantity
        The registered quantity.

    Raises
    ------
    ValueError
        If the name is taken and ``override`` is False.
    """
    if quantity.name in QUANTITIES and not override:
        msg = f"quantity {quantity.name!r} is already registered; pass override=True to replace it"
        raise ValueError(msg)
    QUANTITIES[quantity.name] = quantity
    return quantity


for _q in (
    Quantity("length", "µm", 1, "positions, sizes, distances, pixel pitches"),
    Quantity("wavelength", "µm", 1, "vacuum wavelengths"),
    Quantity("opd", "µm", 1, "optical path differences (aberrations)"),
    Quantity("frequency", "cycles/µm", -1, "spatial frequencies"),
    Quantity("attenuation", "1/µm", -1, "attenuation and scattering coefficients"),
    Quantity("irradiance", "photons/µm²", -2, "photons per µm² per exposure"),
    Quantity("density", "photons/µm³", -3, "photon densities per µm³ per exposure"),
    Quantity("angle", "rad", 0, "angles"),
    Quantity("index", "1", 0, "refractive indices, possibly complex"),
    Quantity("photons", "photons", 0, "photon counts per exposure"),
    Quantity("electrons", "e⁻", 0, "photo-electron counts"),
    Quantity("adu", "ADU", 0, "camera output units"),
    Quantity("gain", "ADU/e⁻", 0, "camera conversion gain"),
    Quantity("rate", "1/s", 0, "rates"),
    Quantity("time", "s", 0, "times"),
    Quantity("dimensionless", "1", 0, "pure numbers: weights, fractions, NA, magnification"),
    Quantity("key", "int64", 0, "random keys (never converted)"),
    Quantity("index_map", "int64", 0, "integer indices such as species or identities"),
):
    register_quantity(_q)
