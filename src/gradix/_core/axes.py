"""Canonical axes (§4.2) and the acquisition index.

Sampled coherent tensors have fixed rank 7, ``[B, A, M, L, P, Y, X]``; the other carriers have
fixed documented shapes that broadcast over B and A. Adding physics never adds an axis.

The acquisition axis A flattens every *separate-frame* axis of an acquisition (time frames,
focus steps, SIM angle·phase, LED index, pre-split channels). :class:`AcqIndex` records how, so
outputs can be unflattened into ``[B, (T,) C, H, W]``.
"""

from __future__ import annotations

import dataclasses
import math

__all__ = [
    "COHERENT",
    "EMITTER_SET",
    "IRRADIANCE",
    "OUTPUT",
    "PLANE_WAVES",
    "AcqIndex",
]

COHERENT: tuple[str, ...] = ("B", "A", "M", "L", "P", "Y", "X")
"""Axes of a sampled coherent field (``Field.scattered``)."""
PLANE_WAVES: tuple[str, ...] = ("B", "A", "M", "J", "L", "P")
"""Axes of ``PlaneWaves.amplitude``."""
EMITTER_SET: tuple[str, ...] = ("B", "A", "N")
"""Leading axes of every ``EmitterSet`` field."""
IRRADIANCE: tuple[str, ...] = ("B", "A", "L", "Y", "X")
"""Axes of ``Irradiance.data``."""
OUTPUT: tuple[str, ...] = ("B", "T", "C", "H", "W")
"""Axes of output frames; T is present only when the acquisition has a time axis."""


@dataclasses.dataclass(frozen=True)
class AcqIndex:
    """How the acquisition axis A flattens separate-frame axes.

    Parameters
    ----------
    axes : tuple of (str, int)
        ``(name, size)`` of each flattened axis, slowest first, such as
        ``(("time", 100), ("focus", 3))``. Empty means A = 1 and no named axis.
    """

    axes: tuple[tuple[str, int], ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "axes", tuple((str(n), int(s)) for n, s in self.axes))
        names = [n for n, _ in self.axes]
        if len(set(names)) != len(names):
            msg = f"acquisition axis names must be unique, got {names}"
            raise ValueError(msg)
        if any(s < 1 for _, s in self.axes):
            msg = f"acquisition axis sizes must be positive, got {self.axes}"
            raise ValueError(msg)
        if "time" in names and names[0] != "time":
            msg = "the time axis must be the slowest acquisition axis"
            raise ValueError(msg)

    @property
    def size(self) -> int:
        """The flattened size A.

        Returns
        -------
        int
            Product of the axis sizes (1 when empty).
        """
        return math.prod(s for _, s in self.axes)

    @property
    def frames(self) -> int | None:
        """The number of time frames T, or None without a time axis.

        Returns
        -------
        int or None
            T.
        """
        for name, size in self.axes:
            if name == "time":
                return size
        return None

    def output_layout(self, channels: int) -> tuple[int | None, int]:
        """Return ``(T, C)`` of output frames for a camera with ``channels`` channels.

        Non-time acquisition axes are flattened into C, slowest first (§4.2).

        Parameters
        ----------
        channels : int
            Detector channels per acquisition frame.

        Returns
        -------
        tuple of (int or None, int)
            T (None without a time axis) and the output channel count.
        """
        frames = self.frames
        rest = self.size // frames if frames else self.size
        return frames, rest * channels
