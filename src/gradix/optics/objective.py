"""The objective: aperture, magnification, focus and pupil modifiers (§5.3)."""

from __future__ import annotations

import dataclasses
from typing import ClassVar

from torch import Tensor

from gradix.schema.base import DataObject, Node
from gradix.schema.fields import child, field

__all__ = ["Objective"]


@dataclasses.dataclass(frozen=True, kw_only=True, eq=False)
class Objective(DataObject):
    """An objective: every modifier of every conjugate pupil plane is fused into its pupil.

    Parameters
    ----------
    NA : Tensor or float
        Numerical aperture, or a per-image ``[B]`` tensor.
    magnification : Tensor or float, default 1.0
        Lateral magnification; the object-space pixel pitch is ``pixel_size / magnification``.
    focus : Tensor or float, default 0.0
        Axial position of the focal plane in µm (§4.1); defocus is ``z − focus``. Per image
        ``[B]``, or per image and frame ``[B, T]`` (a focus stack or focus drift).
    pupil : tuple of Node, default ()
        Pupil modifiers, applied in order (M1: aperture, Zernike, pixel pupil, …).

    Examples
    --------
    >>> Objective(NA=1.4, magnification=100).focus
    0.0
    """

    registry_name: ClassVar[str | None] = "optics.objective"
    schema_version: ClassVar[int] = 2
    """Stable name for signatures and saved inputs."""

    NA: Tensor | float = field(
        quantity="dimensionless",
        role="image",
        shape_affecting=True,
        constraint="positive",
        doc="numerical aperture",
    )
    magnification: Tensor | float = field(
        quantity="dimensionless",
        role="image",
        shape_affecting=True,
        constraint="positive",
        default=1.0,
        doc="lateral magnification",
    )
    focus: Tensor | float = field(
        quantity="length",
        role="setting",
        shape_affecting=True,
        default=0.0,
        doc="z of the focal plane",
    )
    pupil: tuple[Node, ...] = child(container="tuple", default=(), doc="pupil modifiers")

    def with_pupil(self, *modifiers: Node) -> Objective:
        """Return a copy with pupil modifiers appended.

        Parameters
        ----------
        *modifiers : Node
            Modifiers to append.

        Returns
        -------
        Objective
            The copy.
        """
        return self.replace(pupil=(*self.pupil, *modifiers))
