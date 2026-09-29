"""L2 lowering (``gx.lower``): data objects to the views that elements consume (§4.6).

M0 ships the exact emitter lowering; the raster, projection and form-factor lowerings arrive
with M1–M4.
"""

from gradix.lower.density import emitter_density
from gradix.lower.emitters import emitter_set, populations

__all__ = ["emitter_density", "emitter_set", "populations"]
