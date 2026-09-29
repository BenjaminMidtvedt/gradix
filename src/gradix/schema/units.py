"""Schema-driven SI conversion: :func:`gx.from_si <from_si>` and :func:`gx.to_si <to_si>` (§6.1).

Each tensor field declares a quantity, and each quantity its power of length, so a whole tree of
data objects, elements, Chains or containers converts between SI units (metres, photons/m², …)
and the internal µm-based units in one call. Integer fields (keys, indices) never change;
integer values of a dimensional quantity are converted to floating point.

The result is a converted copy: a learnable tensor (a leaf that requires grad) becomes a derived
tensor, so an optimiser holding the original would update a tensor the converted tree no longer
reads. Both functions warn when that happens; convert first, then make the converted tensors
learnable.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from typing import TypeVar

import torch
from torch import Tensor

from gradix._core.errors import GradixWarning
from gradix.schema.base import Leaf, map_leaves
from gradix.schema.fields import FieldSpec
from gradix.units import QUANTITIES

__all__ = ["from_si", "to_si"]

T = TypeVar("T")


def _scale(power: int, name: str) -> Callable[[str, FieldSpec | None, Leaf], Leaf]:
    def apply(path: str, spec: FieldSpec | None, value: Leaf) -> Leaf:
        if spec is None or spec.quantity is None or value is None or isinstance(value, bool):
            return value
        exponent = QUANTITIES[spec.quantity].length_power * power
        if exponent == 0 or spec.dtype in ("integer", "bool"):
            return value
        if isinstance(value, Tensor):
            if value.dtype == torch.bool:
                return value
            if not (value.is_floating_point() or value.is_complex()):
                value = value.to(torch.get_default_dtype())
            elif value.requires_grad and value.is_leaf:
                msg = (
                    f"gx.{name}: {path} is learnable; the converted tree holds a derived copy, "
                    "which an optimiser of the original tensor would not update"
                )
                warnings.warn(msg, GradixWarning, stacklevel=4)
        return value * 1e6**exponent

    return apply


def from_si(tree: T) -> T:
    """Convert every field of a tree from SI units into gradix's internal units.

    Lengths, wavelengths and OPDs are multiplied by 10⁶ (m → µm), irradiances by 10⁻¹²
    (photons/m² → photons/µm²), and so on by each quantity's power of length. Other quantities
    are unchanged.

    Parameters
    ----------
    tree : T
        A data object, element, Chain, container, or a mapping or sequence of them.

    Returns
    -------
    T
        A converted copy; the input is unchanged.
    """
    return map_leaves(_scale(1, "from_si"), tree)


def to_si(tree: T) -> T:
    """Convert every field of a tree from gradix's internal units into SI units.

    The inverse of :func:`from_si`.

    Parameters
    ----------
    tree : T
        A data object, element, Chain, container, or a mapping or sequence of them.

    Returns
    -------
    T
        A converted copy; the input is unchanged.
    """
    return map_leaves(_scale(-1, "to_si"), tree)
