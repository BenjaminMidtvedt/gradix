"""Grid rules: pure functions of an envelope that size grids (§4.3; ``gx.sampling``).

Every rule returns plain Python numbers. Elements call them from ``configure`` on the envelope
they are given (a Pipeline's, or in an eager call the envelope of the call's own inputs), and
record the outcome as a :class:`Decision`, so ``explain()`` can show each decision with the
criterion and the envelope entries that drove it.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Mapping
from typing import Any

from gradix._core.grid import nice_size

__all__ = [
    "Decision",
    "batch_bucket",
    "detection_spacing",
    "nice_size",
    "slot_bucket",
]


@dataclasses.dataclass(frozen=True)
class Decision:
    """One sizing decision with its provenance.

    Parameters
    ----------
    name : str
        What was decided, such as ``"detection.spacing"``.
    value : object
        The decided value (a Python number, tuple or string).
    rule : str
        The criterion, in words or as a formula.
    inputs : Mapping[str, object]
        The envelope entries and other quantities the rule read, by path.
    """

    name: str
    value: object
    rule: str
    inputs: Mapping[str, object] = dataclasses.field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        """Return a JSON-serialisable description.

        Returns
        -------
        dict
            The fields; tuples become lists.
        """
        value = list(self.value) if isinstance(self.value, tuple) else self.value
        return {"name": self.name, "value": value, "rule": self.rule, "inputs": dict(self.inputs)}


def slot_bucket(n: int) -> int:
    """Return the slot capacity of a population with ``n`` objects: 8, 16, 32, ….

    Parameters
    ----------
    n : int
        Largest object count.

    Returns
    -------
    int
        The smallest power of two that is at least 8 and at least ``n``.
    """
    return max(8, 1 << max(int(n) - 1, 0).bit_length())


def batch_bucket(b: int) -> int:
    """Return the batch capacity for ``b`` images: the next power of two.

    Parameters
    ----------
    b : int
        Batch size.

    Returns
    -------
    int
        The smallest power of two at least ``b`` (1 for ``b <= 1``).
    """
    return 1 << max(int(b) - 1, 0).bit_length()


def detection_spacing(
    pitch: float,
    wavelength_min: float,
    na: float,
    *,
    reference_u: float | None = None,
    oversample: int | str = "auto",
) -> tuple[float, int, Decision]:
    """Return the detection spacing: the pitch over the smallest integer that meets Nyquist.

    The intensity band is ``max|u_a − u_b|/λ`` over interfering pairs: 2NA for scattered light,
    ``|u_ref| + NA`` with an image-side reference (§4.3). The spacing must be at most
    ``λ_min / (2·band)``, which is λ/(4NA) without a reference.

    Parameters
    ----------
    pitch : float
        Camera pixel pitch in object space (µm).
    wavelength_min : float
        Shortest vacuum wavelength (µm).
    na : float
        Detection numerical aperture.
    reference_u : float, optional
        ``|u_ref|`` of an image-side reference.
    oversample : int or "auto", default "auto"
        A pinned factor s, or ``"auto"`` for the smallest s that meets the rule.

    Returns
    -------
    tuple of (float, int, Decision)
        The spacing (µm), the factor s and the decision record.
    """
    band = 2.0 * na if reference_u is None else max(2.0 * na, reference_u + na)
    limit = wavelength_min / (2.0 * band)
    if oversample == "auto":
        s = max(1, math.ceil(pitch / limit - 1e-9))
        how = "auto"
    else:
        s = int(oversample)
        how = "pinned"
    spacing = pitch / s
    rule = (
        f"pitch/s, s the smallest integer with spacing <= λ_min/(2·{band:.4g}) "
        f"= {limit:.4g} µm ({how})"
    )
    return (
        spacing,
        s,
        Decision(
            "detection.spacing", spacing, rule, {"pitch": pitch, "λ_min": wavelength_min, "NA": na}
        ),
    )
