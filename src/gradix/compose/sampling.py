"""The SamplingPlan (Pipeline pass P5): every element's static configuration, with provenance."""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from typing import Any

from gradix._core.contract import Description, Element, Static
from gradix._core.envelope import Envelope
from gradix._core.grid import Grid2D
from gradix._core.rules import Decision

__all__ = ["SamplingPlan", "sampling_plan"]


@dataclasses.dataclass(frozen=True)
class SamplingPlan:
    """Static configurations of a Pipeline's elements, and the decisions that set them.

    Parameters
    ----------
    statics : Mapping[str, Static]
        Configuration of each element, by element path.
    """

    statics: Mapping[str, Static]

    @property
    def decisions(self) -> tuple[tuple[str, Decision], ...]:
        """Every decision, with the path of the element that made it.

        Returns
        -------
        tuple of (str, Decision)
            ``(element path, decision)`` pairs.
        """
        return tuple((path, d) for path, static in self.statics.items() for d in static.decisions)

    def grid(self, path: str = "imaging") -> Grid2D | None:
        """Return the grid of an element's configuration, if it has one.

        Parameters
        ----------
        path : str, default "imaging"
            Element path.

        Returns
        -------
        Grid2D or None
            The grid.
        """
        grid = getattr(self.statics.get(path), "grid", None)
        return grid if isinstance(grid, Grid2D) else None

    def to_json(self) -> dict[str, Any]:
        """Return a JSON-serialisable description.

        Returns
        -------
        dict
            Each element's configuration by path.
        """
        return {path: static.to_json() for path, static in self.statics.items()}

    def lines(self) -> list[str]:
        """Return one explain() line per decision.

        Returns
        -------
        list of str
            ``path: name = value  ← rule``.
        """
        return [f"{path}: {d.name} = {d.value}  ← {d.rule}" for path, d in self.decisions]


def sampling_plan(
    elements: Mapping[str, Element], envelope: Envelope, descs: Mapping[str, Description]
) -> SamplingPlan:
    """Configure every element on the envelope (pass P5).

    Parameters
    ----------
    elements : Mapping[str, Element]
        Elements by path.
    envelope : Envelope
        The Pipeline's envelope.
    descs : Mapping[str, Description]
        Each element's static description.

    Returns
    -------
    SamplingPlan
        The configurations. Elements are configured in the given (skeleton) order; each sees
        the configurations of the elements before it in ``desc.upstream``.
    """
    statics: dict[str, Static] = {}
    for path, element in elements.items():
        desc = dataclasses.replace(descs[path], upstream=dict(statics))
        statics[path] = element.configure(desc, envelope)
    return SamplingPlan(statics)
