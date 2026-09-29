"""The gradient-quality table (Pipeline pass P4) and the call-time zero-path refusal (§6.2).

Every tensor field of a Chain gets a quality per output: ``exact``, ``exact-a.e.``,
``biased`` or ``zero``. The qualities come from the elements, not from field names: each
element reports its output's quality with respect to its own fields
(:meth:`~gradix._core.contract.Element.field_quality`, from ``caps.grad_quality``) and to the
input fields it consumes (:meth:`~gradix._core.contract.Element.input_quality`, from
``caps.reads``); labels report theirs (``Label.gradient_quality``). A route's quality is the
worst along it, a field's quality the best over its routes. A field that requires gradients
when all its routes to the requested outputs are ``zero`` raises
:class:`~gradix._core.errors.GradientPathError` at call time.
"""

from __future__ import annotations

import dataclasses
import difflib
import warnings
from collections.abc import Mapping

from torch import Tensor

from gradix._core.contract import GRAD_QUALITIES, Element, worst_quality
from gradix._core.errors import GradientPathError, GradixWarning
from gradix.compose.chain import Chain
from gradix.compose.outputs import Expected, Image, OutputSpec
from gradix.detect.camera import Camera
from gradix.labels.positions import Label
from gradix.schema.base import iter_leaves

__all__ = ["GradientTable", "gradient_table"]


def _best(qualities: list[str]) -> str:
    """Return the best non-zero quality of several routes, or ``"zero"``."""
    live = [q for q in qualities if q != "zero"]
    if not live:
        return "zero"
    order = [q for q in GRAD_QUALITIES if q != "zero"]
    return min(live, key=order.index)


def _through(quality: str, *later: str) -> str:
    """Return the quality of a route that passes through further stages."""
    if quality == "zero":
        return "zero"
    for q in later:
        if q == "zero":
            return "zero"
        quality = worst_quality(quality, q)
    return quality


def _route_quality(chain: Chain, path: str, output: str) -> str:
    """Return the quality of a field's routes to the camera's frames (expected or image)."""
    head, _, rest = path.partition(".")
    imaging = chain.imaging if isinstance(chain.imaging, Element) else None
    camera = chain.camera if isinstance(chain.camera, Camera) else None
    sensor = camera.input_quality("irradiance", output) if camera is not None else "exact"
    routes: list[str] = []
    if head == "emitters" and imaging is not None:
        _population, _, field = rest.partition(".")
        if field:
            routes.append(_through(imaging.input_quality(field, output), sensor))
    elif head == "imaging" and imaging is not None:
        routes.append(_through(imaging.field_quality(rest, output), sensor))
        held = getattr(imaging, "camera", None)
        if rest.startswith("camera.") and camera is not None and held is camera:
            routes.append(camera.field_quality(rest[len("camera.") :], output))
    elif head == "detector" and camera is not None:
        routes.append(camera.field_quality(rest, output))
    elif head == "environment" and imaging is not None:
        routes.append(_through(imaging.input_quality("environment", output), sensor))
        excite = chain.excite if isinstance(chain.excite, Element) else None
        if excite is not None:  # the medium also shapes the excitation (evanescent decay)
            emitted = imaging.input_quality("photons", output)
            routes.append(_through(excite.input_quality("environment"), emitted, sensor))
    elif head == "acquisition":
        # an acquisition's fields reach the image through the settings they drive (a focus
        # stack's steps move objective.focus); the others (frame timing) reach nothing
        target = getattr(chain.acquisition, "drives", {}).get(rest.partition(".")[0])
        if target is not None:
            routes.extend(_route_quality(chain, tree, output) for tree in chain.resolve(target))
    elif head in ("light", "excite") and imaging is not None:
        excite = chain.excite if isinstance(chain.excite, Element) else None
        if excite is not None:  # light → excitation → emitted photons → image
            own = excite.input_quality("light") if head == "light" else excite.field_quality(rest)
            emitted = imaging.input_quality("photons", output)
            routes.append(_through(own, emitted, sensor))
    elif head == "background" and camera is not None:
        routes.append(camera.input_quality("background", output))
    return _best(routes)


def _label_quality(chain: Chain, label: Label, path: str, logical: str) -> str:
    """Return a label's quality: its own, and through the excitation for rendered photons."""
    routes = [label.gradient_quality(logical)]
    excite = chain.excite if isinstance(chain.excite, Element) else None
    population = getattr(label, "population", None)
    if excite is not None and "photons" in label.rendered and isinstance(population, str):
        # light → excitation → rendered photons → label (the medium shapes the excitation too)
        head, _, rest = path.partition(".")
        through = label.gradient_quality(f"{population}.photons")
        if head == "light":
            routes.append(_through(excite.input_quality("light"), through))
        elif head == "excite":
            routes.append(_through(excite.field_quality(rest), through))
        elif head == "environment":
            routes.append(_through(excite.input_quality("environment"), through))
    return _best(routes)


@dataclasses.dataclass(frozen=True)
class GradientTable:
    """Gradient quality of every tensor field of a Chain, per requested output.

    Parameters
    ----------
    rows : Mapping[str, Mapping[str, str]]
        Quality per output name, by tree path.
    names : Mapping[str, str], optional
        Logical path by tree path (:meth:`gradix.Chain.logical`), for messages.
    """

    rows: Mapping[str, Mapping[str, str]]
    names: Mapping[str, str] = dataclasses.field(default_factory=dict)

    def name(self, path: str) -> str:
        """Return the logical path of a tree path.

        Parameters
        ----------
        path : str
            Tree path.

        Returns
        -------
        str
            The logical path, as envelopes, binding and ``explain()`` spell it.
        """
        return self.names.get(path, path)

    def _row(self, path: str) -> Mapping[str, str] | None:
        """Return a field's row by tree path or logical path, or None."""
        if path in self.rows:
            return self.rows[path]
        for tree, name in self.names.items():
            if name == path and tree in self.rows:
                return self.rows[tree]
        return None

    def quality(self, path: str) -> str:
        """Return a field's overall quality: the best over outputs; ``"partial"`` if some are zero.

        Parameters
        ----------
        path : str
            Logical path (``"beads.position"``) or tree path (``"emitters.beads.position"``).

        Returns
        -------
        str
            ``exact``, ``exact-a.e.``, ``biased``, ``partial`` or ``zero``.

        Raises
        ------
        KeyError
            If the path names no field of the table's Chain.
        """
        row = self._row(path)
        if row is None:
            known = sorted(self.names.get(p, p) for p in self.rows)
            close = difflib.get_close_matches(path, known, n=3)
            hint = f"; did you mean {close}?" if close else ""
            raise KeyError(f"{path!r} names no field of the Chain{hint}")
        qualities = list(row.values())
        best = _best(qualities)
        if best == "zero":
            return "zero"
        return "partial" if "zero" in qualities else best

    def check(self, chain: Chain) -> None:
        """Refuse fields that require gradients along only zero paths.

        Parameters
        ----------
        chain : Chain
            The Chain of a call; only its ``requires_grad`` flags are read.

        Raises
        ------
        GradientPathError
            If a field requires gradients but reaches no output with a non-zero gradient.
        """
        for path, _spec, value in iter_leaves(chain):
            if not isinstance(value, Tensor) or not value.requires_grad or path not in self.rows:
                continue
            if self.quality(path) == "zero":
                outputs = sorted(self.rows.get(path, {}))
                name = self.name(path)
                msg = (
                    f"{name} requires gradients, but none of the requested outputs {outputs} "
                    "depends on it (the elements on its routes report zero gradient)"
                )
                fix = (
                    "request an output that depends on it, or detach it; "
                    "pipe.explain() lists every field's routes"
                )
                raise GradientPathError(msg, fix=fix)

    def to_json(self) -> dict[str, dict[str, str]]:
        """Return a JSON-serialisable description.

        Returns
        -------
        dict
            ``{path: {output: quality}}``, sorted.
        """
        return {p: dict(sorted(q.items())) for p, q in sorted(self.rows.items())}

    def lines(self, chain: Chain | None = None) -> list[str]:
        """Return one explain() line per field (only fields requiring gradients with ``chain``).

        Parameters
        ----------
        chain : Chain, optional
            Restrict to the fields of this Chain that require gradients.

        Returns
        -------
        list of str
            ``logical path  output: quality, …``.
        """
        wanted = None
        if chain is not None:
            wanted = {
                p for p, _s, v in iter_leaves(chain) if isinstance(v, Tensor) and v.requires_grad
            }
        out = []
        for path, qualities in sorted(self.rows.items()):
            if wanted is not None and path not in wanted:
                continue
            detail = ", ".join(f"{name}: {q}" for name, q in sorted(qualities.items()))
            out.append(f"{self.name(path)}  {detail}")
        return out


def gradient_table(chain: Chain, outputs: Mapping[str, OutputSpec]) -> GradientTable:
    """Build the gradient table of a Chain for requested outputs (pass P4).

    Parameters
    ----------
    chain : Chain
        The Chain.
    outputs : Mapping[str, OutputSpec]
        Requested outputs.

    Returns
    -------
    GradientTable
        Quality per field and output.
    """
    rows: dict[str, dict[str, str]] = {}
    names: dict[str, str] = {}
    for path, spec, _value in iter_leaves(chain):
        names[path] = chain.logical(path)
        integer = spec is not None and spec.dtype in ("integer", "bool")
        row: dict[str, str] = {}
        for name, out in outputs.items():
            if integer:
                row[name] = "zero"
            elif isinstance(out, Expected):
                row[name] = _route_quality(chain, path, "expected")
            elif isinstance(out, Image):
                row[name] = _route_quality(chain, path, "image")
            elif isinstance(out, Label):
                row[name] = _label_quality(chain, out, path, names[path])
            else:
                row[name] = "zero"
        rows[path] = row
    return GradientTable(rows, names)


def warn_partial(table: GradientTable, chain: Chain) -> None:
    """Warn about fields that require gradients and reach only some outputs.

    Parameters
    ----------
    table : GradientTable
        The table.
    chain : Chain
        The Chain of a call.
    """
    for path, _spec, value in iter_leaves(chain):
        if not isinstance(value, Tensor) or not value.requires_grad or path not in table.rows:
            continue
        if table.quality(path) == "partial":
            zero = sorted(o for o, q in table.rows.get(path, {}).items() if q == "zero")
            msg = f"{table.name(path)} requires gradients but has none through outputs {zero}"
            warnings.warn(msg, GradixWarning, stacklevel=3)
