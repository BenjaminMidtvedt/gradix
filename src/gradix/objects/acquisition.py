"""Acquisitions (``gx.acq``): separate frames on the acquisition axis A (§4.2, §6.7).

An acquisition declares how it flattens into A (its :class:`~gradix.AcqIndex`): the frame count
is structure, set here rather than guessed from value shapes, so a static scene (T = 1) can be
imaged over many frames and per-frame fields (``[B, T, …]``) are checked against it. An
acquisition that changes the instrument between frames returns the per-frame values of the
settings it drives from :meth:`Acquisition.bindings`, by logical path (a focus stack binds
``objective.focus``); fields of role ``setting`` accept them as ``[B, A]``. M1 ships
:class:`Frames` and :class:`FocusStack`; LED and SIM sequences, channels and polarisation states
follow the same pattern.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from typing import ClassVar

from torch import Tensor

from gradix._core.axes import AcqIndex
from gradix._core.errors import StructureError
from gradix.schema.base import DataObject
from gradix.schema.fields import child, field, knob
from gradix.schema.layout import canonical

__all__ = ["Acquisition", "Bound", "FocusStack", "Frames"]


@dataclasses.dataclass(frozen=True, kw_only=True, eq=False)
class Acquisition(DataObject):
    """Base of acquisitions: each declares how it flattens into the acquisition axis A."""

    drives: ClassVar[Mapping[str, str]] = {}
    """Own field → the logical path of the setting it drives through :meth:`bindings` (the
    gradient table routes those fields there; the others reach no output)."""

    def index(self) -> AcqIndex:
        """Return the acquisition index.

        Returns
        -------
        AcqIndex
            The flattened axes.
        """
        raise NotImplementedError(type(self).__name__)

    def bindings(self, chain: object) -> dict[str, object]:
        """Return the per-frame values of the settings this acquisition drives.

        Parameters
        ----------
        chain : object
            The Chain being rendered (its current settings, such as ``chain.objective``).

        Returns
        -------
        dict of str to object
            Values by logical path, ``[B|1, A]`` along the acquisition axis; empty by default.
        """
        return {}


@dataclasses.dataclass(frozen=True, eq=False)
class Frames(Acquisition):
    """A time series of ``n`` frames, rendered at mid-exposure (motion blur arrives in v1.x).

    Per-frame fields (positions, presence and photons as ``[B, T, N]``) must have T = n or 1.

    Parameters
    ----------
    n : int
        Number of frames T.
    interval : Tensor or float, optional
        Time between frame starts, s.
    exposure : Tensor or float, optional
        Exposure time per frame, s (at most ``interval``).

    Examples
    --------
    >>> Frames(100, interval=0.02).index().frames
    100
    """

    registry_name: ClassVar[str | None] = "acq.frames"
    """Stable name for signatures and saved inputs."""

    n: int = knob(doc="number of frames")
    _: dataclasses.KW_ONLY
    interval: Tensor | float | None = field(
        quantity="time", role="image", constraint="positive", default=None, doc="frame interval"
    )
    exposure: Tensor | float | None = field(
        quantity="time", role="image", constraint="positive", default=None, doc="exposure time"
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        if not isinstance(self.n, int) or isinstance(self.n, bool) or self.n < 1:
            raise StructureError(f"Frames.n must be a positive int, got {self.n!r}")
        interval, exposure = self.interval, self.exposure
        if isinstance(interval, (int, float)) and isinstance(exposure, (int, float)):
            if exposure > interval:
                raise StructureError(f"exposure {exposure} s exceeds the interval {interval} s")

    def index(self) -> AcqIndex:
        """Return the acquisition index: one time axis of ``n`` frames.

        Returns
        -------
        AcqIndex
            ``AcqIndex((("time", n),))``.
        """
        return AcqIndex((("time", self.n),))


@dataclasses.dataclass(frozen=True, kw_only=True, eq=False)
class FocusStack(Acquisition):
    """A focus series: the objective's focal plane stepped through ``focus`` (µm).

    The steps flatten into A as a ``"focus"`` axis, which output frames place in C. Each step's
    focal plane is ``Objective.focus + focus[a]``; the step values are tensors, so they can be
    learned (a focus-offset calibration) or differ per image.

    Parameters
    ----------
    focus : Tensor
        Focal-plane steps in µm: ``[A]``, or per image ``[B, A]``.

    Examples
    --------
    >>> import torch
    >>> FocusStack(focus=torch.linspace(-1.0, 1.0, 41)).index().size
    41
    """

    drives: ClassVar[Mapping[str, str]] = {"focus": "objective.focus"}
    """The steps move the objective's focal plane."""
    registry_name: ClassVar[str | None] = "acq.focus_stack"
    """Stable name for signatures and saved inputs."""

    focus: Tensor = field(
        quantity="length",
        role="image",
        event=(-1,),
        shape_affecting=True,
        doc="focal-plane steps",
    )

    def index(self) -> AcqIndex:
        """Return the acquisition index: one focus axis of A steps.

        Returns
        -------
        AcqIndex
            ``AcqIndex((("focus", A),))``.
        """
        return AcqIndex((("focus", int(self.focus.shape[-1])),))

    def bindings(self, chain: object) -> dict[str, object]:
        """Return the focal plane of every step: ``objective.focus + focus``, ``[B|1, A]``.

        Parameters
        ----------
        chain : object
            The Chain; its objective's focus is the stack's origin.

        Returns
        -------
        dict of str to object
            ``{"objective.focus": [B|1, A] tensor}``.
        """
        objective = getattr(chain, "objective", None)
        steps = canonical(self.focus, self.schema()["focus"])  # [B|1, A]
        if objective is None:
            return {}
        spec = objective.schema()["focus"]
        base = canonical(objective.focus, spec, dtype=steps.dtype, device=steps.device)
        base = base.reshape(base.shape[0], -1)  # [B|1, T|1]
        if base.shape[1] not in (1, steps.shape[1]):
            msg = (
                f"objective.focus varies over {base.shape[1]} frames but the focus stack has "
                f"{steps.shape[1]} steps"
            )
            raise StructureError(msg, fix="give objective.focus one value per step, or per image")
        return {"objective.focus": base + steps}


@dataclasses.dataclass(frozen=True, kw_only=True, eq=False)
class Bound(Acquisition):
    """An acquisition whose settings are already bound into the Chain (``Chain.acquired``).

    It keeps the acquisition's axis and binds nothing more, so binding twice is binding once.

    Parameters
    ----------
    source : Acquisition
        The acquisition that was bound.
    """

    source: Acquisition = child(doc="the bound acquisition")

    def index(self) -> AcqIndex:
        """Return the bound acquisition's index.

        Returns
        -------
        AcqIndex
            The flattened axes.
        """
        return self.source.index()
