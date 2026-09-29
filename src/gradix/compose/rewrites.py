"""Rewrites (Pipeline pass P7): pupil fusion, propagation merging, strata collapse, hoisting.

Rewrites never change results beyond round-off; each applied rewrite is recorded by name. None
is implemented yet: the imaging elements fuse their pupil modifiers themselves
(``gradix.imaging._shared.fused_modifiers``), and strata collapse and hoisting of grad-free
invariants (the dense path's OTFs, for example) are still to come.
"""

from __future__ import annotations

from collections.abc import Mapping

from gradix._core.contract import Element

__all__ = ["rewrites"]


def rewrites(elements: Mapping[str, Element]) -> tuple[str, ...]:
    """Return the rewrites that apply to a Pipeline's elements.

    Parameters
    ----------
    elements : Mapping[str, Element]
        Elements by path.

    Returns
    -------
    tuple of str
        Names of the applied rewrites (none yet).
    """
    return ()
