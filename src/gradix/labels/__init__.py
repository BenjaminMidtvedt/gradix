"""L2 labels (``gx.labels``): geometry-derived labels rendered in the image's frame (§6.6).

Value labels (the positions, radii and identities a caller sampled) are the caller's; gradix
renders only labels that need its frame, lowering or camera grid.
"""

from gradix.labels.maps import DistanceMap, EmitterTable, Heatmap, InstanceMask, SemanticMask
from gradix.labels.positions import Label, Positions, render

__all__ = [
    "DistanceMap",
    "EmitterTable",
    "Heatmap",
    "InstanceMask",
    "Label",
    "Positions",
    "SemanticMask",
    "render",
]
