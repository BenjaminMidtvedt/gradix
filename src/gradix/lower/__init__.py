"""L2 lowering (``gx.lower``): data objects to the views that elements consume (§4.6).

The exact emitter lowering, the density lowering of volumes and labelled solids, and the raster
lowering behind it (ADR-41); projection and form-factor lowerings arrive with M3–M4.
"""

from gradix.lower.density import emitter_density
from gradix.lower.emitters import emitter_set, populations
from gradix.lower.raster import label_density, quaternion_matrix, rasterise

__all__ = [
    "emitter_density",
    "emitter_set",
    "label_density",
    "populations",
    "quaternion_matrix",
    "rasterise",
]
