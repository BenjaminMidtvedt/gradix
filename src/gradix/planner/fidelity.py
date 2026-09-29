"""Fidelity: a frozen policy whose named presets expand into explicit, overridable knobs (§5.6).

A Fidelity acts only at L4: :func:`gradix.plan` uses it to choose an element for every
population without an entry in ``methods=``, and writes its knob values into the elements it
builds. It is discrete and never differentiated. Knob values double as registry names, so a
plugin is selected exactly like a built-in.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from typing import Any, Literal, cast

from gradix._core.errors import PlanError

__all__ = ["PRESETS", "Fidelity"]

PRESETS: dict[str, dict[str, object]] = {
    "draft": {
        "emitters": "gaussian",
        "emitter_path": "auto",
        "psf": "scalar",
        "polarization": "auto",
        "illumination_coupling": "local",
        "spheres": "auto",
        "compact": "projection",
        "volumes": "projection",
        "coupling": "C0",
        "wavelength_bins": 1,
        "oversample": 1,
        "interaction_band": "optical",
        "raster_sigma": 0.5,
        "raster_oversample": 1,
    },
    "standard": {
        "emitters": "pupil",
        "emitter_path": "auto",
        "psf": "auto",
        "polarization": "auto",
        "illumination_coupling": "auto",
        "spheres": "mie",
        "compact": "auto",
        "volumes": "projection",
        "coupling": "C0",
        "wavelength_bins": 1,
        "oversample": "auto",
        "interaction_band": "optical",
        "raster_sigma": 0.3,
        "raster_oversample": 1,
    },
    "accurate": {
        "emitters": "pupil",
        "emitter_path": "auto",
        "psf": "auto",
        "polarization": "vector",
        "illumination_coupling": "exact",
        "spheres": "mie",
        "compact": "voxel",
        "volumes": "multislice",
        "coupling": "C0",
        "wavelength_bins": "auto",
        "oversample": "auto",
        "interaction_band": "full",
        "raster_sigma": 0.3,
        "raster_oversample": 2,
    },
    "reference": {
        "emitters": "pupil",
        "emitter_path": "sparse",
        "psf": "vectorial",
        "polarization": "vector",
        "illumination_coupling": "exact",
        "spheres": "mie",
        "compact": "voxel",
        "volumes": "multislice",
        "coupling": "C0",
        "wavelength_bins": "auto",
        "oversample": "auto",
        "interaction_band": "full",
        "raster_sigma": 0.3,
        "raster_oversample": 2,
    },
}
"""Knob values of each preset (the table of §5.6)."""

PER_POPULATION_KEYS = frozenset(
    {"emitters", "emitter_path", "psf", "spheres", "compact", "volumes", "raster_sigma"}
)
"""Knobs that ``per_population`` may override (§5.6)."""


@dataclasses.dataclass(frozen=True)
class Fidelity:
    """A fidelity policy: a named preset expanded into knobs, with overrides.

    Parameters
    ----------
    preset : {"draft", "standard", "accurate", "reference"}, default "standard"
        The preset.
    emitters : str, optional
        ``"gaussian"`` or ``"pupil"`` (or a plugin's registry name).
    emitter_path : str, optional
        ``"auto"``, ``"sparse"``, ``"global"`` or ``"strata"``.
    psf : str, optional
        ``"scalar"``, ``"vectorial"`` or ``"auto"``.
    polarization : str, optional
        ``"scalar"``, ``"vector"`` or ``"auto"`` (coherent paths).
    illumination_coupling : str, optional
        ``"exact"``, ``"local"`` or ``"auto"``.
    spheres : str, optional
        ``"dipole"``, ``"born"``, ``"mie"``, ``"voxel"`` or ``"auto"``.
    compact : str, optional
        ``"born"``, ``"projection"``, ``"voxel"`` or ``"auto"``.
    volumes : str, optional
        ``"projection"`` or ``"multislice"``.
    coupling : str, optional
        ``"C0"`` (``"C1"`` from v1.x).
    source : object, optional
        Source-node strategy for Köhler and LED densities.
    wavelength_bins : int or str, optional
        Wavelength bins, or ``"auto"``.
    oversample : int or str, optional
        Detection oversampling, or ``"auto"``.
    interaction_band : str or float, optional
        ``"optical"`` or ``"full"``.
    raster_sigma : float, optional
        Raster prefilter width in voxel units.
    raster_oversample : int or str, optional
        Raster oversampling factor.
    dz : float or str, default "auto"
        Slice thickness in µm.
    pad : float or str, default "auto"
        Padding in µm.
    roi : int or str, default "auto"
        ROI size in pixels.
    per_population : Mapping[str, Mapping[str, object]], optional
        Per-population overrides of the model-choice knobs.
    on_invalid : {"warn", "raise", "ignore"}, default "warn"
        Passed to the Pipeline.
    memory_budget : float, default 0.5
        Passed to the Pipeline.
    backward : str, optional
        Cross-fidelity surrogate backward (v1.x); must be None in v1.

    Examples
    --------
    >>> Fidelity("draft").knob("emitters")
    'gaussian'
    >>> Fidelity("standard", per_population={"beads": {"emitters": "gaussian"}}).knob(
    ...     "emitters", population="beads"
    ... )
    'gaussian'
    """

    preset: Literal["draft", "standard", "accurate", "reference"] = "standard"
    _: dataclasses.KW_ONLY
    emitters: str | None = None
    emitter_path: str | None = None
    psf: str | None = None
    polarization: str | None = None
    illumination_coupling: str | None = None
    spheres: str | None = None
    compact: str | None = None
    volumes: str | None = None
    coupling: str | None = None
    source: object = None
    wavelength_bins: int | str | None = None
    oversample: int | str | None = None
    interaction_band: str | float | None = None
    raster_sigma: float | None = None
    raster_oversample: int | str | None = None
    dz: float | str = "auto"
    pad: float | str = "auto"
    roi: int | str = "auto"
    per_population: Mapping[str, Mapping[str, object]] = dataclasses.field(default_factory=dict)
    on_invalid: Literal["warn", "raise", "ignore"] = "warn"
    memory_budget: float = 0.5
    backward: str | None = None

    def __post_init__(self) -> None:
        if self.preset not in PRESETS:
            raise PlanError(
                f"unknown fidelity preset {self.preset!r}", fix=f"use one of {list(PRESETS)}"
            )
        if self.backward is not None:
            raise NotImplementedError("cross-fidelity surrogate backwards arrive in v1.x")
        for pop, overrides in self.per_population.items():
            bad = set(overrides) - PER_POPULATION_KEYS
            if bad:
                msg = f"per_population[{pop!r}] sets {sorted(bad)}, which are global knobs"
                raise PlanError(msg, fix=f"per-population knobs: {sorted(PER_POPULATION_KEYS)}")

    @classmethod
    def coerce(cls, fidelity: Fidelity | str) -> Fidelity:
        """Return a Fidelity from a Fidelity or a preset name.

        Parameters
        ----------
        fidelity : Fidelity or str
            A policy or a preset name.

        Returns
        -------
        Fidelity
            The policy.
        """
        if isinstance(fidelity, Fidelity):
            return fidelity
        if fidelity not in PRESETS:
            raise PlanError(
                f"unknown fidelity preset {fidelity!r}", fix=f"use one of {list(PRESETS)}"
            )
        return cls(cast("Literal['draft', 'standard', 'accurate', 'reference']", fidelity))

    def __hash__(self) -> int:
        return hash(repr(sorted(self.to_json().items())))

    def knob(self, name: str, *, population: str | None = None) -> object:
        """Return a knob's value: a per-population override, an explicit value, or the preset's.

        Parameters
        ----------
        name : str
            Knob name, such as ``"emitters"``.
        population : str, optional
            Population whose overrides apply.

        Returns
        -------
        object
            The value.
        """
        if population is not None and name in self.per_population.get(population, {}):
            return self.per_population[population][name]
        explicit = getattr(self, name, None)
        if explicit is not None:
            return explicit
        return PRESETS[self.preset].get(name)

    def to_json(self) -> dict[str, Any]:
        """Return a JSON-serialisable description.

        Returns
        -------
        dict
            Every field (``source`` by its repr).
        """
        out = dataclasses.asdict(self)
        out["source"] = None if self.source is None else repr(self.source)
        out["per_population"] = {k: dict(v) for k, v in self.per_population.items()}
        return out
