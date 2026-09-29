"""The Chain: elements wired along the slot skeleton (§3.1, §5.1).

A Chain is a pytree like any data object: it is edited by copy (:meth:`Chain.replace`,
:func:`gradix.tree.replace`), stacked with :func:`gradix.stack`, and fed to a
:class:`~gradix.Pipeline`. Called directly, it renders eagerly, sizing grids from its own inputs.

Field paths (§6.4) name parts of a Chain logically: a population by its name (``beads``,
``beads.position``), and ``light``, ``objective`` (the imaging element's objective),
``camera``, ``environment``, ``acquisition``, ``background`` and named stages. :meth:`Chain.resolve`
maps a logical path to the tree paths it stands for.

The camera has one home. An imaging element that samples on the camera grid (Sprites, PointPSF)
holds it, and ``chain.camera`` reads it from there; the Chain stores a camera itself (in
``detector``) only when its imaging element holds none. Passing ``camera=`` is accepted when it
is that same object, so a tree never holds two copies that edits could separate.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from typing import ClassVar

from torch import Tensor

from gradix._core.axes import AcqIndex
from gradix._core.contract import Element
from gradix._core.errors import BindingError, StructureError
from gradix.compose.outputs import Output
from gradix.detect.camera import Camera, check_pixel_map
from gradix.objects.acquisition import Acquisition, Bound
from gradix.objects.environment import Medium
from gradix.objects.objectset import ObjectSet
from gradix.objects.volumes import Voxels
from gradix.schema.base import Node
from gradix.schema.fields import child, field

__all__ = ["PARTS", "Chain"]

PARTS: tuple[str, ...] = (
    "light",
    "objective",
    "camera",
    "environment",
    "acquisition",
    "background",
    "imaging",
    "excite",
)
"""Reserved part names; populations and stages may not use them."""


@dataclasses.dataclass(frozen=True, kw_only=True, eq=False)
class Chain(Node):
    """Elements wired along the slot skeleton.

    Chains render emitter populations (``emitters=``: points or densities, optionally excited
    by ``light`` through ``excite``). Coherent slots (``light`` driving ``scatterers``, stages,
    references) render from M3; until then the coherent elements run at L2 (called directly).

    Parameters
    ----------
    light : Element, optional
        The source element.
    illumination_optics : Mapping[str, Element], optional
        Ordered stage elements before the sample (M3).
    scatterers : Mapping[str, Element], optional
        Interaction elements by population name (M3).
    emitters : Mapping[str, ObjectSet or Voxels], optional
        Emitting populations by name: point sets (``gx.Emitters``) for point-emitter imaging
        elements, emitter densities (``gx.Voxels``) for ``gx.imaging.Strata``.
    excite : Element, optional
        The transduction element (M1).
    imaging : Element, optional
        The imaging element, such as :class:`gradix.imaging.Sprites`.
    references : Mapping[str, Element], optional
        Reference beams by name (M3).
    detection_optics : Mapping[str, Element], optional
        Ordered stage elements after the objective (M3).
    camera : Camera, optional
        The camera. When the imaging element holds a camera this may only repeat that same
        object; otherwise it is stored in ``detector``. After construction ``chain.camera``
        returns the camera in use.
    detector : Camera, optional
        The stored camera of a Chain whose imaging element holds none (set through
        ``camera=``).
    background : Tensor or float, optional
        Background photons per pixel per exposure, added before the sensor: a constant, per
        image ``[B]``, a map ``[H, W]`` or maps ``[B, H, W]``.
    acquisition : Node, optional
        Acquisition axes (M1: frames, focus stacks, …).
    environment : Medium, optional
        The medium around the sample.

    Examples
    --------
    >>> import torch, gradix as gx
    >>> from gradix.units import nm, um
    >>> camera = gx.Camera(pixel_size=6.5 * um, shape=(32, 32))
    >>> chain = Chain(
    ...     emitters={"beads": gx.Emitters(
    ...         position=torch.tensor([[[2.6, 2.6, 0.0]]]),
    ...         photons=torch.tensor([[1000.0]]),
    ...         emission=gx.Spectrum.line(600 * nm),
    ...     )},
    ...     imaging=gx.imaging.Sprites(gx.Objective(NA=0.7, magnification=40), camera),
    ...     environment=gx.env.Homogeneous(1.33),
    ... )
    >>> chain.resolve("objective.focus")
    ['imaging.objective.focus']
    """

    registry_name: ClassVar[str | None] = "chain"
    """Stable name for signatures and saved inputs."""

    light: Element | None = child(default=None, doc="source element")
    illumination_optics: Mapping[str, Element] = child(
        container="mapping",
        default_factory=dict,
        doc="stages before the sample",
        ordered=True,
    )
    scatterers: Mapping[str, Element] = child(
        container="mapping", default_factory=dict, doc="interaction elements by population"
    )
    emitters: Mapping[str, ObjectSet | Voxels] = child(
        container="mapping", default_factory=dict, doc="emitting populations"
    )
    excite: Element | None = child(default=None, doc="transduction element")
    imaging: Element | None = child(default=None, doc="imaging element")
    references: Mapping[str, Element] = child(
        container="mapping", default_factory=dict, ordered=True, doc="reference beams by name"
    )
    detection_optics: Mapping[str, Element] = child(
        container="mapping",
        default_factory=dict,
        doc="stages after the objective",
        ordered=True,
    )
    init_only: ClassVar[tuple[str, ...]] = ("camera",)
    camera: dataclasses.InitVar[Camera | None] = None
    detector: Camera | None = child(default=None, doc="camera, when the imaging element has none")
    background: Tensor | float | None = field(
        quantity="photons", role="pixels", constraint="nonnegative", default=None, doc="background"
    )
    acquisition: Node | None = child(default=None, doc="acquisition axes")
    environment: Medium | None = child(default=None, doc="medium")

    def __post_init__(self, camera: Camera | None) -> None:
        imaging_camera = getattr(self.imaging, "camera", None)
        held = imaging_camera if isinstance(imaging_camera, Camera) else None
        if camera is not None:
            if not isinstance(camera, Camera):
                raise StructureError(f"camera must be a gx.Camera, got {type(camera).__name__}")
            if held is not None and camera is not held:
                msg = "camera= is not the imaging element's camera"
                fix = "give the camera once, to the imaging element; camera= may only repeat it"
                raise StructureError(msg, fix=fix)
            if held is None:
                if self.detector is not None and self.detector is not camera:
                    raise StructureError("camera= and detector= name different cameras")
                object.__setattr__(self, "detector", camera)
        if held is not None and self.detector is not None:
            msg = "the imaging element holds the camera, so the Chain cannot hold another"
            raise StructureError(msg, fix="drop detector=, or edit the imaging element's camera")
        super().__post_init__()
        camera = held if held is not None else self.detector
        if camera is not None and isinstance(self.background, Tensor):
            spec = type(self).schema()["background"]
            check_pixel_map(self.background, spec, camera.shape, where="background")
        names: dict[str, str] = {}
        groups = (
            ("emitters", self.emitters),
            ("scatterers", self.scatterers),
            ("illumination_optics", self.illumination_optics),
            ("references", self.references),
            ("detection_optics", self.detection_optics),
        )
        for group, mapping in groups:
            for name in mapping:
                if name in PARTS:
                    msg = f"{group}[{name!r}]: {name!r} is a reserved part name"
                    raise StructureError(msg, fix="rename the population or stage")
                if name in names:
                    msg = f"{name!r} names both {names[name]}[{name!r}] and {group}[{name!r}]"
                    raise StructureError(msg, fix="give every population and stage its own name")
                names[name] = group
        if self.environment is not None and not isinstance(self.environment, Medium):
            raise StructureError("environment must be a medium such as gx.env.Homogeneous")
        if self.acquisition is not None and not isinstance(self.acquisition, Acquisition):
            msg = f"acquisition must be a gx.acq object, got {type(self.acquisition).__name__}"
            raise StructureError(msg, fix="use gx.acq.Frames(n) or gx.acq.FocusStack(offsets)")

    @property
    def objective(self) -> Node | None:
        """The imaging element's objective, if it has one.

        Returns
        -------
        Node or None
            The objective.
        """
        objective = getattr(self.imaging, "objective", None)
        return objective if isinstance(objective, Node) else None

    def population(self, name: str) -> ObjectSet:
        """Return a population's data object by name.

        Parameters
        ----------
        name : str
            Population name.

        Returns
        -------
        ObjectSet
            The data object (for scatterers, the one their element renders).

        Raises
        ------
        StructureError
            If there is no such population.
        """
        if name in self.emitters:
            population = self.emitters[name]
            if isinstance(population, ObjectSet):
                return population
            raise StructureError(f"{name!r} is a volume, not an object set")
        if name in self.scatterers:
            element = self.scatterers[name]
            objects = getattr(element, getattr(element, "population_field", "objects"), None)
            if isinstance(objects, ObjectSet):
                return objects
        known = sorted({*self.emitters, *self.scatterers})
        raise StructureError(f"no population {name!r}", fix=f"populations: {known}")

    def paths(self) -> dict[str, str]:
        """Return the path table: each logical path prefix and the tree path it stands for.

        This one table (§6.4) is behind :meth:`resolve` (logical → tree), :meth:`logical`
        (tree → logical), :meth:`parts`, :meth:`own_parts` and the names in the gradient
        table and ``explain()``, so they cannot disagree.

        Returns
        -------
        dict of str to str
            Tree path by logical prefix: populations by name (``beads`` →
            ``emitters.beads``, or the scatterer element's population), ``objective`` and
            ``camera`` where they are held, the Chain's own parts (``light``, ``environment``,
            ``acquisition``, ``background``, ``imaging``, ``excite``), named stages and
            references by name, and each scatterer element at ``scatterers.<name>``.
        """
        table: dict[str, str] = {}
        for name in self.emitters:
            table[name] = f"emitters.{name}"
        for name, element in self.scatterers.items():
            table[name] = f"scatterers.{name}.{getattr(element, 'population_field', 'objects')}"
            table[f"scatterers.{name}"] = f"scatterers.{name}"
        if self.objective is not None:
            table["objective"] = "imaging.objective"
        if isinstance(getattr(self.imaging, "camera", None), Camera):
            table["camera"] = "imaging.camera"
        elif self.detector is not None:
            table["camera"] = "detector"
        for name in ("light", "environment", "acquisition", "background", "imaging", "excite"):
            table[name] = name
        for group in ("illumination_optics", "references", "detection_optics"):
            for name in getattr(self, group):
                table[name] = f"{group}.{name}"
        return table

    def logical(self, path: str) -> str:
        """Return the logical path of a tree path, the inverse of :meth:`resolve`.

        Parameters
        ----------
        path : str
            A tree path such as ``"emitters.beads.photons"``.

        Returns
        -------
        str
            The logical path (``"beads.photons"``); a tree path outside the table is
            returned unchanged.

        Examples
        --------
        >>> import gradix as gx
        >>> Chain(imaging=gx.imaging.Sprites(gx.Objective(NA=0.7, magnification=40),
        ...     gx.Camera(pixel_size=6.5, shape=(8, 8)))).logical("imaging.objective.NA")
        'objective.NA'
        """
        best = ""
        head = ""
        for name, tree in self.paths().items():
            if (path == tree or path.startswith(tree + ".")) and len(tree) > len(best):
                best, head = tree, name
        return head + path[len(best) :] if best else path

    def parts(self) -> dict[str, Node]:
        """Return the named parts whose fields have envelope and binding paths.

        Returns
        -------
        dict of str to Node
            Populations, ``light``, ``objective``, ``camera``, ``environment``,
            ``acquisition`` and stages, by path prefix (from :meth:`paths`).
        """
        out: dict[str, Node] = {}
        for name, tree in self.paths().items():
            if name in ("background", "imaging", "excite") or name.startswith("scatterers."):
                continue  # elements' own fields are own parts; the background is a tensor
            node = _get(self, tree)
            if isinstance(node, Node):
                out[name] = node
        return out

    def own_parts(self) -> dict[str, Element]:
        """Return the elements whose own fields have envelope paths (their children do not).

        Returns
        -------
        dict of str to Element
            ``imaging``, ``excite`` and ``scatterers.<name>`` when present: an element's own
            shape-affecting tensor fields are enveloped as ``imaging.<field>``; its
            children (objective, camera, population) are parts of their own.
        """
        out: dict[str, Element] = {}
        for name, tree in self.paths().items():
            if name in ("imaging", "excite") or name.startswith("scatterers."):
                element = _get(self, tree)
                if isinstance(element, Element):
                    out[name] = element
        return out

    def elements(self) -> dict[str, Element]:
        """Return the Chain's elements in skeleton order, by element path.

        Returns
        -------
        dict of str to Element
            ``light``, ``illumination_optics.<name>``, ``scatterers.<name>`` (sorted),
            ``excite``, ``imaging``, ``references.<name>``, ``detection_optics.<name>`` and
            ``camera``, for those present. Validity, the SamplingPlan, ``explain()`` and the
            gradient table walk this order.
        """
        out: dict[str, Element] = {}
        if isinstance(self.light, Element):
            out["light"] = self.light
        for name, element in self.illumination_optics.items():
            out[f"illumination_optics.{name}"] = element
        for name in sorted(self.scatterers):
            out[f"scatterers.{name}"] = self.scatterers[name]
        if isinstance(self.excite, Element):
            out["excite"] = self.excite
        if isinstance(self.imaging, Element):
            out["imaging"] = self.imaging
        for name, element in self.references.items():
            out[f"references.{name}"] = element
        for name, element in self.detection_optics.items():
            out[f"detection_optics.{name}"] = element
        camera = self.camera
        if isinstance(camera, Element):
            out["camera"] = camera
        return out

    def acquired(self, *, keep: bool = False) -> Chain:
        """Return the Chain with its acquisition's per-frame settings bound (§6.7).

        Envelopes, validity and rendering all read the bound Chain, so a setting an
        acquisition drives (a focus stack's focal planes, a plugin's biplane offsets) sizes
        grids like any other value.

        Parameters
        ----------
        keep : bool, default False
            Keep the acquisition itself (its fields keep their paths, as envelopes need);
            otherwise it is wrapped in :class:`~gradix.objects.acquisition.Bound`, so binding
            the result again changes nothing.

        Returns
        -------
        Chain
            The Chain whose ``setting`` fields carry the acquisition's per-frame values, or the
            Chain itself when the acquisition binds nothing.
        """
        from gradix.tree import replace as tree_replace

        acquisition = self.acquisition
        if not isinstance(acquisition, Acquisition):
            return self
        values = acquisition.bindings(self)
        if not values:
            return self
        changes = {tree: value for path, value in values.items() for tree in self.resolve(path)}
        if not keep:
            changes["acquisition"] = Bound(source=acquisition)  # binding twice is binding once
        return tree_replace(self, changes)

    def acq_index(self) -> AcqIndex:
        """Return the acquisition axis A: the declared acquisition's, else time frames from T.

        Returns
        -------
        AcqIndex
            The acquisition's index; without one, a time axis of the T that per-frame fields
            and settings share (none when T = 1).
        """
        from gradix.schema.base import tree_axis_sizes

        if isinstance(self.acquisition, Acquisition):
            return self.acquisition.index()
        frames = tree_axis_sizes(self).get("T", 1)
        return AcqIndex((("time", frames),)) if frames > 1 else AcqIndex()

    def __call__(self, *, key: int | Tensor | None = None, outputs: object = ("image",)) -> Output:
        """Render eagerly: grids sized from the Chain's own inputs (L2 semantics).

        Parameters
        ----------
        key : int or Tensor, optional
            Detector-noise key: a batch key or image keys ``Tensor[B]``. Required for images
            from a noisy camera.
        outputs : str, tuple of str, or Mapping, default ("image",)
            What to compute, as for :class:`~gradix.Pipeline`.

        Returns
        -------
        Output
            The outputs; ``meta`` records the key kind.

        Raises
        ------
        StructureError
            If a noisy image is requested without a key.
        """
        from gradix._core.keys import check_key
        from gradix.compose import execute
        from gradix.compose.outputs import Image, normalize_outputs
        from gradix.schema.base import tree_axis_sizes

        specs = normalize_outputs(outputs)
        statics = execute.eager_statics(self)
        camera = self.camera
        noisy = isinstance(camera, Camera) and camera.noise is not None
        if key is None and noisy and any(isinstance(s, Image) for s in specs.values()):
            raise StructureError(
                "the image output needs a key (the camera has noise)",
                fix="pass key=..., or request outputs=('expected',)",
            )
        batch = tree_axis_sizes(self).get("B", 1)
        if isinstance(key, Tensor):
            check_key(key, batch)
        elif key is not None:
            check_key(key)
        values = execute.run(self, statics, key=key, outputs=specs, batch=batch)
        kind = "none" if key is None else ("image" if isinstance(key, Tensor) else "batch")
        return Output(values, {"key": kind, "level": "eager"})

    def resolve(self, path: str) -> list[str]:
        """Map a logical field path to the tree paths it stands for.

        Parameters
        ----------
        path : str
            A logical path such as ``"beads.position"``, ``"objective.focus"`` or ``"camera"``.

        Returns
        -------
        list of str
            Tree paths; ``camera`` resolves to the imaging element's camera, or to
            ``detector`` when the imaging element holds none.

        Raises
        ------
        BindingError
            If the path names nothing in the Chain.
        """
        table = self.paths()
        matches = [name for name in table if path == name or path.startswith(name + ".")]
        if not matches:
            head = path.partition(".")[0]
            if head == "objective":
                raise BindingError("the Chain has no objective (its imaging element has none)")
            if head == "camera":
                raise BindingError("the Chain has no camera")
            known = sorted(name for name in table if "." not in name)
            raise BindingError(f"{path!r} names nothing in the Chain", fix=f"known names: {known}")
        name = max(matches, key=len)
        return [table[name] + path[len(name) :]]


def _get(chain: Chain, path: str) -> object:
    """Return the value at a tree path of a Chain (attributes and mapping keys)."""
    value: object = chain
    for name in path.split("."):
        value = value.get(name) if isinstance(value, Mapping) else getattr(value, name, None)
        if value is None:
            return None
    return value


def _camera(self: Chain) -> Camera | None:
    """Return the camera in use: the imaging element's, else the stored ``detector``."""
    held = getattr(self.imaging, "camera", None)
    return held if isinstance(held, Camera) else self.detector


# ``camera`` is an init-only argument; after construction it reads the camera in use.
if "camera" in Chain.__dict__:
    delattr(Chain, "camera")
type.__setattr__(Chain, "camera", property(_camera, doc="The camera in use (read-only)."))
