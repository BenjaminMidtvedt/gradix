"""Validity aggregation (pass P3): element predicates on the envelope, plus Chain checks."""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping

from gradix._core.contract import Description, Element, Violation
from gradix._core.envelope import Envelope
from gradix.compose.chain import Chain

__all__ = ["check"]


def check(
    chain: Chain,
    elements: Mapping[str, Element],
    envelope: Envelope,
    descs: Mapping[str, Description],
) -> list[Violation]:
    """Collect validity findings for a Chain on an envelope.

    Parameters
    ----------
    chain : Chain
        The Chain.
    elements : Mapping[str, Element]
        Its elements by path.
    envelope : Envelope
        The envelope.
    descs : Mapping[str, Description]
        Each element's static description.

    Returns
    -------
    list of Violation
        Findings, each tagged with the path of the element that reported it.
    """
    out: list[Violation] = []
    for path, element in elements.items():
        for v in element.validity(descs[path], envelope):
            out.append(v if v.element else dataclasses.replace(v, element=path))
    if chain.light is not None and chain.excite is None and not chain.scatterers:
        out.append(
            Violation(
                "info",
                "light is unused: without an excitation element, emitters render from their "
                "own photons",
                element="light",
                fix="add excite=gx.excite.Linear() to let the light drive the emitters",
            )
        )
    return out
