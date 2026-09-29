"""Grid rules (``gx.sampling``): pure functions of an envelope that size grids (§4.3)."""

from gradix._core.grid import is_smooth, nice_size
from gradix._core.rules import Decision, batch_bucket, detection_spacing, slot_bucket

__all__ = ["Decision", "batch_bucket", "detection_spacing", "is_smooth", "nice_size", "slot_bucket"]
