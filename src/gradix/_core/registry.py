"""Typed registries and the decorators that fill them (§7.1).

There is one registry per extension point. Saved data objects, elements and Pipelines refer to
registry names plus a schema version, never to import paths. Duplicate names are errors unless
``override=True``.
"""

from __future__ import annotations

import difflib
from collections.abc import Callable, Iterator
from typing import TYPE_CHECKING, Any, Generic, TypeVar

from gradix._core.errors import RegistryError
from gradix.schema.base import Node

if TYPE_CHECKING:
    from gradix._core.contract import Element

__all__ = [
    "REGISTRIES",
    "Registry",
    "elements",
    "labels",
    "lowerings",
    "noise_models",
    "object_sets",
    "presets",
    "pupil_modifiers",
    "views",
]

T = TypeVar("T")


class Registry(Generic[T]):
    """A mapping from registry names to registered objects.

    Parameters
    ----------
    kind : str
        What the registry holds, such as ``"element"``; used in messages.
    """

    def __init__(self, kind: str) -> None:
        self.kind = kind
        self._items: dict[str, T] = {}

    def add(self, name: str, obj: T, *, override: bool = False) -> T:
        """Register an object under a name.

        Parameters
        ----------
        name : str
            Registry name, such as ``"emit.gaussian"``.
        obj : T
            The object to register.
        override : bool, default False
            Replace an existing entry instead of raising.

        Returns
        -------
        T
            ``obj``, so the method can end a decorator.

        Raises
        ------
        RegistryError
            If the name is taken and ``override`` is False.
        """
        if not name or not isinstance(name, str):
            msg = f"{self.kind} names must be non-empty strings, got {name!r}"
            raise RegistryError(msg)
        if name in self._items and not override and self._items[name] is not obj:
            msg = f"{self.kind} {name!r} is already registered ({self._items[name]!r})"
            raise RegistryError(msg, fix="choose another name, or pass override=True")
        self._items[name] = obj
        return obj

    def get(self, name: str) -> T:
        """Return the object registered under a name.

        Parameters
        ----------
        name : str
            Registry name.

        Returns
        -------
        T
            The registered object.

        Raises
        ------
        RegistryError
            If nothing is registered under ``name``; the message suggests close names.
        """
        try:
            return self._items[name]
        except KeyError:
            close = difflib.get_close_matches(name, list(self._items), n=3)
            hint = f"; did you mean {', '.join(map(repr, close))}?" if close else ""
            msg = f"no {self.kind} is registered as {name!r}{hint}"
            raise RegistryError(msg, fix=f"registered: {sorted(self._items)}") from None

    def find(self, obj: object) -> str | None:
        """Return the name an object is registered under, or None.

        Parameters
        ----------
        obj : object
            A registered object (compared by identity).

        Returns
        -------
        str or None
            Its registry name.
        """
        for name, item in self._items.items():
            if item is obj:
                return name
        return None

    def unregister(self, name: str) -> T:
        """Remove a registration (for tests and plugin reloads).

        Parameters
        ----------
        name : str
            The registered name.

        Returns
        -------
        object
            The object that was registered.

        Raises
        ------
        RegistryError
            If nothing is registered under ``name``.
        """
        obj = self.get(name)
        del self._items[name]
        return obj

    def names(self) -> list[str]:
        """Return the registered names, sorted.

        Returns
        -------
        list of str
            The names.
        """
        return sorted(self._items)

    def __contains__(self, name: object) -> bool:
        return name in self._items

    def __iter__(self) -> Iterator[str]:
        return iter(sorted(self._items))

    def __len__(self) -> int:
        return len(self._items)

    def __repr__(self) -> str:
        return f"Registry({self.kind!r}, {len(self._items)} entries)"

    def decorator(self, name: str, *, override: bool = False) -> Callable[[T], T]:
        """Return a decorator that registers its argument under ``name``.

        Parameters
        ----------
        name : str
            Registry name.
        override : bool, default False
            Replace an existing entry instead of raising.

        Returns
        -------
        callable
            The decorator; it returns its argument unchanged.
        """

        def register(obj: T) -> T:
            return self.add(name, obj, override=override)

        return register


object_sets: Registry[type[Node]] = Registry("object set")
"""Object-set classes (``Emitters``, ``Spheres``, …) by kind name."""
views: Registry[type] = Registry("view")
"""Lowered view types (``EmitterSet``, ``SphereList``, …)."""
lowerings: Registry[Callable[..., object]] = Registry("lowering")
"""Lowering functions by ``"<object kind>-><view>"`` name."""
elements: Registry[type[Element]] = Registry("element")
"""Element classes by ``"<slot family>.<name>"``, such as ``"emit.gaussian"``."""
pupil_modifiers: Registry[type[Node]] = Registry("pupil modifier")
"""Pupil-modifier classes."""
labels: Registry[type[Node]] = Registry("label")
"""Label renderers."""
presets: Registry[Callable[..., object]] = Registry("preset")
"""Microscope recipes."""
noise_models: Registry[type[Node]] = Registry("noise model")
"""Camera noise models by name, such as ``"sensor.poisson_gaussian"``."""
data_objects: Registry[type[Node]] = Registry("data object")
"""Other data objects, containers, acquisitions, materials and carriers, by stable name: their
signatures and saved inputs use the name, not the import path."""

REGISTRIES: dict[str, Registry[Any]] = {
    "object_set": object_sets,
    "view": views,
    "lowering": lowerings,
    "element": elements,
    "pupil_modifier": pupil_modifiers,
    "label": labels,
    "preset": presets,
    "noise": noise_models,
    "data_object": data_objects,
}
"""Every registry by extension-point name."""
