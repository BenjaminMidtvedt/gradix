"""Registration decorators (``gx.register``): the plugin API of every extension point (§7.1).

Examples
--------
>>> import dataclasses, gradix as gx
>>> @gx.register.object_set("torus_example", override=True)
... @dataclasses.dataclass(frozen=True, kw_only=True, eq=False)
... class Tori(gx.ObjectSet):
...     kind = "compact"
...     major: float = gx.field(quantity="length", role="object", shape_affecting=True)
>>> gx.registry.object_sets.get("torus_example") is Tori
True
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

from gradix._core import registry as _registry
from gradix._core.contract import Element, Slot
from gradix._core.errors import RegistryError
from gradix.schema.base import Node
from gradix.schema.fields import specs

__all__ = [
    "data_object",
    "element",
    "label",
    "lowering",
    "noise",
    "object_set",
    "preset",
    "pupil_modifier",
    "view",
]

C = TypeVar("C", bound=type[Node])
V = TypeVar("V", bound=type)
F = TypeVar("F", bound=Callable[..., object])


def _set_name(cls: type[Node], name: str) -> None:
    cls.registry_name = name


def _node_class(cls: type, kind: str) -> None:
    if not isinstance(cls, type) or not issubclass(cls, Node):
        msg = f"only gradix nodes can be registered as a {kind}, got {cls!r}"
        raise RegistryError(msg, fix="subclass gx.DataObject or gx.Element")
    specs(cls)  # raises when a field lacks its schema declaration


def element(
    name: str, *, slot: Slot | str | None = None, override: bool = False
) -> Callable[[C], C]:
    """Register an element class under ``"<slot family>.<name>"``.

    Parameters
    ----------
    name : str
        The name; a dotted name such as ``"emit.gaussian"`` is used as given, a bare name is
        prefixed with the slot, as in ``"interact.wpm"``.
    slot : Slot or str, optional
        The slot; the class's ``slot`` attribute by default.
    override : bool, default False
        Replace an existing registration.

    Returns
    -------
    callable
        The class decorator.
    """

    def register(cls: C) -> C:
        _node_class(cls, "element")
        if not issubclass(cls, Element):
            raise RegistryError(f"{cls.__qualname__} is not a gx.Element")
        the_slot = Slot(slot) if slot is not None else getattr(cls, "slot", None)
        if the_slot is None:
            raise RegistryError(f"{cls.__qualname__} declares no slot", fix="set slot = Slot.<X>")
        full = name if "." in name else f"{Slot(the_slot).value}.{name}"
        _set_name(cls, full)
        _registry.elements.add(full, cls, override=override)
        return cls

    return register


def object_set(name: str, *, override: bool = False) -> Callable[[C], C]:
    """Register an object-set class.

    Parameters
    ----------
    name : str
        Registry name, such as ``"spheres"``.
    override : bool, default False
        Replace an existing registration.

    Returns
    -------
    callable
        The class decorator.
    """

    def register(cls: C) -> C:
        _node_class(cls, "object set")
        _set_name(cls, name)
        _registry.object_sets.add(name, cls, override=override)
        return cls

    return register


def pupil_modifier(name: str, *, override: bool = False) -> Callable[[C], C]:
    """Register a pupil-modifier class.

    Parameters
    ----------
    name : str
        Registry name, such as ``"zernike"``.
    override : bool, default False
        Replace an existing registration.

    Returns
    -------
    callable
        The class decorator.
    """

    def register(cls: C) -> C:
        _node_class(cls, "pupil modifier")
        _set_name(cls, name)
        _registry.pupil_modifiers.add(name, cls, override=override)
        return cls

    return register


def data_object(name: str, *, override: bool = False) -> Callable[[C], C]:
    """Register a data object, container, acquisition, material or carrier under a stable name.

    Signatures (cache keys, Pipeline hashes) and saved inputs name registered classes by this
    name, so moving a class between modules does not change them.

    Parameters
    ----------
    name : str
        Registry name, such as ``"optics.objective"``.
    override : bool, default False
        Replace an existing registration.

    Returns
    -------
    callable
        The class decorator.
    """

    def register(cls: C) -> C:
        _node_class(cls, "data object")
        _set_name(cls, name)
        _registry.data_objects.add(name, cls, override=override)
        return cls

    return register


def noise(name: str, *, override: bool = False) -> Callable[[C], C]:
    """Register a camera noise-model class.

    Parameters
    ----------
    name : str
        Registry name, such as ``"sensor.poisson_gaussian"``.
    override : bool, default False
        Replace an existing registration.

    Returns
    -------
    callable
        The class decorator.
    """

    def register(cls: C) -> C:
        _node_class(cls, "noise model")
        _set_name(cls, name)
        _registry.noise_models.add(name, cls, override=override)
        return cls

    return register


def label(name: str, *, override: bool = False) -> Callable[[C], C]:
    """Register a label renderer class.

    Parameters
    ----------
    name : str
        Registry name, such as ``"positions"``.
    override : bool, default False
        Replace an existing registration.

    Returns
    -------
    callable
        The class decorator.
    """

    def register(cls: C) -> C:
        _node_class(cls, "label")
        _set_name(cls, name)
        _registry.labels.add(name, cls, override=override)
        return cls

    return register


def view(name: str, *, override: bool = False) -> Callable[[V], V]:
    """Register a lowered view type.

    Parameters
    ----------
    name : str
        Registry name, such as ``"emitter_set"``.
    override : bool, default False
        Replace an existing registration.

    Returns
    -------
    callable
        The class decorator.
    """

    def register(cls: V) -> V:
        _registry.views.add(name, cls, override=override)
        return cls

    return register


def lowering(name: str, *, override: bool = False) -> Callable[[F], F]:
    """Register a lowering function under ``"<object kind>-><view>"``.

    Parameters
    ----------
    name : str
        Registry name, such as ``"point->emitter_set"``.
    override : bool, default False
        Replace an existing registration.

    Returns
    -------
    callable
        The function decorator.
    """

    def register(fn: F) -> F:
        _registry.lowerings.add(name, fn, override=override)
        return fn

    return register


def preset(name: str, *, override: bool = False) -> Callable[[F], F]:
    """Register a microscope preset (a function that builds a ``gx.Microscope``).

    Parameters
    ----------
    name : str
        Registry name, such as ``"widefield"``.
    override : bool, default False
        Replace an existing registration.

    Returns
    -------
    callable
        The function decorator.
    """

    def register(fn: F) -> F:
        _registry.presets.add(name, fn, override=override)
        return fn

    return register
