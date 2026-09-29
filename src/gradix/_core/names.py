"""User-chosen names and the rule that keeps them apart from the framework's (ADR-42).

Names users choose key into several namespaces next to names gradix reserves: population and
stage names start logical paths (``beads.position``), output names become keys and attributes
(``out.pos``), and shape fields reach capability methods. Rather than asking users to avoid
framework words wherever a name travels, every user name is checked once, where it is
introduced: it must be a Python identifier that is not a keyword (so it works as a path segment
and as an attribute), and it must not be a reserved word of that namespace. Framework code never
merges user names into a namespace of its own (shape fields travel as one mapping, fidelity knobs
only reach the fields an element declares), so the reserved lists stay short and fixed.
"""

from __future__ import annotations

import keyword
from collections.abc import Collection

from gradix._core.errors import StructureError

__all__ = ["PATH_NAMES", "check_name"]

PATH_NAMES: frozenset[str] = frozenset(
    {
        # a Chain's own parts
        "light",
        "objective",
        "camera",
        "environment",
        "acquisition",
        "background",
        "imaging",
        "excite",
        "detector",
        # the groups whose members are named
        "emitters",
        "scatterers",
        "illumination_optics",
        "references",
        "detection_optics",
    }
)
"""Names that start a Chain's logical paths themselves; populations and stages may not use them."""


def check_name(name: object, *, what: str, reserved: Collection[str] = ()) -> str:
    """Return a user-chosen name after checking it: an identifier, not a keyword, not reserved.

    Parameters
    ----------
    name : object
        The name.
    what : str
        What the name names, for the message (``"population"``, ``"output"``, …).
    reserved : Collection[str], optional
        The reserved words of the namespace the name enters.

    Returns
    -------
    str
        The name.

    Raises
    ------
    StructureError
        If the name is not an identifier, is a Python keyword or is reserved.

    Examples
    --------
    >>> check_name("beads", what="population", reserved=PATH_NAMES)
    'beads'
    """
    if not isinstance(name, str) or not name.isidentifier() or keyword.iskeyword(name):
        msg = f"{what} name {name!r} is not an identifier"
        fix = "use letters, digits and underscores, not starting with a digit (it becomes a path"
        fix += " segment such as 'beads.position' and an attribute such as out.beads)"
        raise StructureError(msg, fix=fix)
    if name in reserved:
        msg = f"{what} name {name!r} is reserved"
        raise StructureError(msg, fix=f"choose another name; reserved here: {sorted(reserved)}")
    return name
