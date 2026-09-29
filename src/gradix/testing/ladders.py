"""Ladders (§11.4): regime sweeps of approximation edges into versioned error tables.

Each approximation edge of §5.9 gets a :class:`Ladder`: a sweep over its regime, a metric, and
the two elements it compares. Running it produces a table (the "fidelity atlas") whose entries
calibrate validity thresholds and the tolerances ``explain()`` prints. A ladder whose elements
are not registered yet is reported as pending instead of failing.
"""

from __future__ import annotations

import dataclasses
import itertools
from collections.abc import Callable, Mapping, Sequence

import torch
from torch import Tensor

from gradix._core.registry import elements

__all__ = ["LADDERS", "Ladder", "LadderRow", "register_ladder", "rel_l2", "run"]


def rel_l2(approx: Tensor, reference: Tensor) -> float:
    """Return the relative L2 error ``‖a − r‖ / ‖r‖``.

    Parameters
    ----------
    approx : Tensor
        The approximation.
    reference : Tensor
        The reference.

    Returns
    -------
    float
        The relative error.
    """
    return float(torch.linalg.vector_norm(approx - reference) / torch.linalg.vector_norm(reference))


@dataclasses.dataclass(frozen=True)
class LadderRow:
    """One point of a ladder's sweep.

    Parameters
    ----------
    point : Mapping[str, object]
        The regime parameters.
    error : float
        The metric's value.
    within : bool, default True
        Whether the error is within the ladder's declared tolerance.
    """

    point: Mapping[str, object]
    error: float
    within: bool = True


@dataclasses.dataclass(frozen=True)
class Ladder:
    """An approximation edge with a regime sweep.

    Parameters
    ----------
    name : str
        Table identifier, such as ``"gaussian_pupil"``.
    lower : str
        Registry name of the approximating element.
    higher : str
        Registry name of the reference element.
    sweep : Mapping[str, Sequence[object]]
        Regime parameters and their values; the sweep is their product.
    evaluate : callable
        ``evaluate(**point) -> float``: the metric at one point.
    tolerance : float
        The initial target of the edge.
    """

    name: str
    lower: str
    higher: str
    sweep: Mapping[str, Sequence[object]]
    evaluate: Callable[..., float]
    tolerance: float

    def pending(self) -> bool:
        """Return whether an element of the edge is not registered yet.

        Returns
        -------
        bool
            True when the ladder cannot run in this version.
        """
        return self.lower not in elements or self.higher not in elements

    def points(self) -> list[dict[str, object]]:
        """Return every point of the sweep.

        Returns
        -------
        list of dict
            The product of the sweep's values.
        """
        keys = list(self.sweep)
        return [
            dict(zip(keys, values, strict=True))
            for values in itertools.product(*self.sweep.values())
        ]


LADDERS: dict[str, Ladder] = {}
"""Registered ladders by name."""


def register_ladder(ladder: Ladder) -> Ladder:
    """Register a ladder.

    Parameters
    ----------
    ladder : Ladder
        The ladder.

    Returns
    -------
    Ladder
        The same ladder.
    """
    LADDERS[ladder.name] = ladder
    return ladder


def run(name: str) -> list[LadderRow] | None:
    """Run a ladder's sweep.

    Parameters
    ----------
    name : str
        The ladder's name.

    Returns
    -------
    list of LadderRow or None
        One row per point, each marked against the ladder's tolerance, or None when the ladder
        is pending.
    """
    ladder = LADDERS[name]
    if ladder.pending():
        return None
    rows = []
    for point in ladder.points():
        error = float(ladder.evaluate(**point))
        rows.append(LadderRow(point, error, error <= ladder.tolerance))
    return rows


def _gaussian_vs_pupil(**point: object) -> float:
    """Relative L2 error of Gaussian sprites against the scalar pupil PSF (one emitter)."""
    from gradix.detect.camera import Camera
    from gradix.imaging.point_psf import PointPSF
    from gradix.imaging.sprites import Sprites
    from gradix.lower.emitters import emitter_set
    from gradix.objects.environment import Homogeneous
    from gradix.objects.objectset import Emitters
    from gradix.objects.spectrum import Spectrum
    from gradix.optics.objective import Objective

    na = float(str(point["na"]))
    defocus = float(str(point["defocus"]))
    magnification = 100.0 * na  # keeps ~2.3 pixels per λ/(2NA) at every NA
    camera = Camera(pixel_size=6.5, shape=(48, 48))
    objective = Objective(NA=na, magnification=magnification)
    centre = 24.3 * 6.5 / magnification
    beads = Emitters(
        position=torch.tensor([[[centre, centre, defocus]]], dtype=torch.float64),
        photons=1.0,
        emission=Spectrum.line(0.6),
    )
    lowered = emitter_set(beads)
    medium = Homogeneous(1.33)
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # the Gaussian tier warns outside its regime
        gauss = Sprites(objective, camera)(lowered, medium).data
        pupil = PointPSF(objective, camera, pupil_samples=128)(lowered, medium).data
    return float(torch.linalg.vector_norm(gauss - pupil) / torch.linalg.vector_norm(pupil))


register_ladder(
    Ladder(
        name="gaussian_pupil",
        lower="emit.gaussian",
        higher="emit.pupil_mft",
        sweep={"na": (0.3, 0.5, 0.7), "defocus": (0.0, 0.1, 0.2)},
        evaluate=_gaussian_vs_pupil,
        tolerance=0.20,
    )
)
