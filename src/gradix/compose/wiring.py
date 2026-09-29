"""Carrier compatibility along a Chain's wiring (§7.1): declared ``accepts`` meet ``produces``.

Every link of the incoherent path is checked when a Chain is rendered or planned: the light's
carrier must be one the excitation accepts, the excitation's output one the imaging element
accepts, and the imaging element must produce :class:`~gradix.Irradiance` for the camera. A
plugin that declares otherwise is refused with the link named, before any forward runs.
"""

from __future__ import annotations

from gradix._core.carriers import Irradiance
from gradix._core.contract import Element
from gradix._core.errors import PlanError
from gradix.compose.chain import Chain

__all__ = ["check_wiring"]


def _produces(element: Element) -> tuple[type, ...]:
    produced = element.caps.produces
    return produced if isinstance(produced, tuple) else (produced,)


def _link(upstream: str, producer: Element, downstream: str, consumer: Element) -> None:
    accepts = consumer.caps.accepts
    if not any(issubclass(p, a) for p in _produces(producer) for a in accepts):
        made = ", ".join(p.__name__ for p in _produces(producer))
        want = ", ".join(sorted(a.__name__ for a in accepts)) or "nothing"
        msg = f"{upstream} produces {made}, but {downstream} accepts {want}"
        raise PlanError(msg, fix=f"use a {downstream} element that accepts {made}")


def check_wiring(chain: Chain) -> None:
    """Raise if two linked elements of a Chain disagree on the carrier between them.

    Parameters
    ----------
    chain : Chain
        The Chain.

    Raises
    ------
    PlanError
        Naming the link whose producer's carrier the consumer does not accept.
    """
    light, excite, imaging = chain.light, chain.excite, chain.imaging
    if isinstance(light, Element) and isinstance(excite, Element):
        _link("light", light, "excite", excite)
    if isinstance(excite, Element) and isinstance(imaging, Element):
        _link("excite", excite, "imaging", imaging)
    if isinstance(imaging, Element):
        produced = _produces(imaging)
        if not any(issubclass(p, Irradiance) for p in produced):
            made = ", ".join(p.__name__ for p in produced)
            msg = f"imaging produces {made}, but the camera takes an Irradiance"
            raise PlanError(msg, fix="declare produces=Irradiance on the imaging element")
