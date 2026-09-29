"""L4 presets (``gx.presets``): recipes that build Microscopes."""

from gradix.presets.holography import ISCAT, QWLSI, InlineHolography, OffAxisHolography
from gradix.presets.tirf import SIM2D, TIRF
from gradix.presets.widefield import Widefield

__all__ = [
    "ISCAT",
    "QWLSI",
    "SIM2D",
    "TIRF",
    "InlineHolography",
    "OffAxisHolography",
    "Widefield",
]
