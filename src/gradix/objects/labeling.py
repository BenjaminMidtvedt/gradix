"""Emitter dipole models (``gx.dipole``) and, from M1, labelings of volumes and surfaces."""

from __future__ import annotations

import dataclasses
from typing import ClassVar

from gradix.schema.base import DataObject

__all__ = ["Isotropic"]


@dataclasses.dataclass(frozen=True, kw_only=True, eq=False)
class Isotropic(DataObject):
    """A freely rotating (isotropic) emitter dipole: the default of :class:`gradix.Emitters`.

    Fixed and wobbling dipoles, which need the vectorial PSF, arrive with M4b.
    """

    registry_name: ClassVar[str | None] = "dipole.isotropic"
    """Stable name for signatures and saved inputs."""
