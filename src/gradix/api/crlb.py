"""``gx.crlb``: Cramér–Rao lower bounds from the camera's Fisher information (§5.5, cap-19).

The bound needs the Jacobian ``∂μ/∂θ`` of the expected frames over every pixel, for a few
parameters. Forward mode computes it cheaply, and images are independent: a tangent that moves
the same parameter in every image gives every image's column at once, so the cost is one
forward-mode pass per parameter *of an image*, whatever the batch size. A value shared by all
images (a Python number, or a tensor without a batch axis) is one parameter that every image
informs; a Schur complement combines those with the per-image parameters exactly.

Through a reconstruction R (ADR-44, §5.12) the information is that of R's output under the
camera noise it propagates: F_R = (RJ)ᵀ(RΣRᵀ)⁺(RJ), solved by conjugate gradients on
matrix-free products of R and its adjoint, and differentiated through a form stationary in the
solver's output.
"""

from __future__ import annotations

import dataclasses
import math
import warnings
from collections.abc import Callable, Iterator, Mapping, Sequence
from typing import Literal, cast

import torch
from torch import Tensor

from gradix._core.envelope import in_envelope
from gradix._core.errors import GradientPathError, GradixWarning, StructureError
from gradix.api.plan import Plan
from gradix.compose.chain import Chain
from gradix.compose.gradients import gradient_table
from gradix.compose.outputs import Expected
from gradix.compose.pipeline import Pipeline
from gradix.detect.camera import Camera
from gradix.detect.functions import fisher
from gradix.detect.noise import Ideal
from gradix.imaging._shared import ratio_per_image
from gradix.imaging.point_psf import PointPSF
from gradix.imaging.strata import Strata
from gradix.optics.objective import Objective
from gradix.recon.base import Reconstruction
from gradix.schema.base import Node, tree_axis_sizes
from gradix.schema.fields import FieldSpec
from gradix.schema.layout import ROLE_RULES, leading_axes
from gradix.tree import get as tree_get
from gradix.tree import replace as tree_replace
from gradix.units import QUANTITIES

__all__ = ["CRLB", "QUANTITATIVE_NA_N", "crlb"]

QUANTITATIVE_NA_N = 0.45
"""NA/n above which quantitative bounds need the vectorial PSF model (§5.3)."""

METHODS = ("auto", "forward", "central")
"""Ways to compute the Jacobian: forward mode, falling back to central differences (auto)."""


@dataclasses.dataclass(frozen=True)
class _Slot:
    """Where one ``wrt`` entry's bounds sit in the parameter vector, and their shape."""

    path: str
    per_image: bool
    count: int
    shape: tuple[int, ...]
    dtype: torch.dtype


@dataclasses.dataclass(frozen=True, eq=False)
class CRLB(Mapping[str, Tensor]):
    """Cramér–Rao lower bounds on fields of a Chain, read as a mapping from path to bound.

    ``bound["beads.position"]`` is the smallest standard deviation that any unbiased estimator
    of that field can reach from the images, in the field's unit and shaped like its value (a
    component path such as ``"beads.position.z"`` drops the component axis). Parameters that
    no image depends on, such as the positions of absent objects, have an infinite bound.

    Independent measurements of the same parameters add their Fisher information:
    ``bound_a + bound_b`` (or ``sum(bounds)``) is the bound from both, such as frames taken under
    two states of a spatial light modulator, or two modalities imaging the same sample.

    Parameters
    ----------
    std : Mapping[str, Tensor]
        The bounds by ``wrt`` path.
    fisher : Tensor
        Each image's Fisher information, ``[B, P, P]`` in float64, over the parameters in
        ``labels`` order: every image's own parameters, then the shared ones.
    covariance : Tensor
        Each image's marginal of the bound's covariance, ``[B, P, P]`` in float64: the image's
        own parameters together with the shared ones.
    labels : tuple of str
        The P parameter names within an image, such as ``"beads.position[0, 3].z"``.
    shared : int
        How many of the P parameters (the last ones) are shared by all images.
    units : Mapping[str, str]
        The unit of every bound, by path.
    method : {"forward", "central"}
        How the Jacobian was computed.
    notes : tuple of str
        Findings: the noise model's accuracy and any warnings raised on the way.
    reconstruction : str, optional
        The reconstruction the information was taken through; None for the raw frames.
    measurements : int, default 1
        How many independent measurements the information sums.
    layout : tuple, default ()
        Where each ``wrt`` entry's parameters sit, and their shapes (used to combine bounds).
    """

    std: Mapping[str, Tensor]
    fisher: Tensor
    covariance: Tensor
    labels: tuple[str, ...]
    shared: int
    units: Mapping[str, str]
    method: str
    notes: tuple[str, ...] = ()
    reconstruction: str | None = None
    measurements: int = 1
    layout: tuple[_Slot, ...] = ()

    def __getitem__(self, path: str) -> Tensor:
        try:
            return self.std[path]
        except KeyError:
            raise KeyError(f"no bound for {path!r}; bounds: {sorted(self.std)}") from None

    def __iter__(self) -> Iterator[str]:
        return iter(self.std)

    def __len__(self) -> int:
        return len(self.std)

    def __add__(self, other: object) -> CRLB:
        """Return the bound from both measurements: their Fisher information adds.

        Parameters
        ----------
        other : CRLB
            A bound on the same parameters from an independent measurement.

        Returns
        -------
        CRLB
            The combined bound, differentiable in both measurements' tensors.

        Raises
        ------
        StructureError
            If the two bound different parameters.
        """
        if not isinstance(other, CRLB):
            return NotImplemented
        if self.labels != other.labels or self.layout != other.layout:
            raise StructureError(
                "only bounds on the same parameters combine",
                fix="bound the same wrt paths in both measurements",
            )
        info = self.fisher + other.fisher
        own = len(self.labels) - self.shared
        covariance, singular = _covariance(info, own)
        notes = [*self.notes, *(n for n in other.notes if n not in self.notes)]
        if bool(singular.any()):
            bad = torch.nonzero(singular).flatten().tolist()
            notes.append(
                f"the Fisher information of images {bad} is singular: their bounds are NaN"
            )
        through = [r for r in (self.reconstruction, other.reconstruction) if r]
        return CRLB(
            std=_bounds(covariance, self.layout, own, order=tuple(self.std)),
            fisher=info,
            covariance=covariance,
            labels=self.labels,
            shared=self.shared,
            units=self.units,
            method=self.method if self.method == other.method else "mixed",
            notes=tuple(notes),
            reconstruction=" + ".join(through) if through else None,
            measurements=self.measurements + other.measurements,
            layout=self.layout,
        )

    def __radd__(self, other: object) -> CRLB:
        """Let ``sum()`` start from 0.

        Parameters
        ----------
        other : object
            0, or a bound.

        Returns
        -------
        CRLB
            This bound, or the sum.
        """
        if isinstance(other, int) and other == 0:
            return self
        return self.__add__(other)

    def explain(self) -> str:
        """Return a report: the parameters, the method, the bounds and the findings.

        Returns
        -------
        str
            The report.
        """
        own = len(self.labels) - self.shared
        through = f" · through {self.reconstruction}" if self.reconstruction else ""
        combined = f" · {self.measurements} measurements" if self.measurements > 1 else ""
        how = {"forward": "forward mode", "central": "central differences"}.get(
            self.method, "several methods"
        )
        lines = [
            f"CRLB · {own} parameter(s) per image, {self.shared} shared · Jacobian by "
            f"{how}{through}{combined}"
        ]
        for path, std in self.std.items():
            finite = std[torch.isfinite(std)]
            unit = self.units.get(path, "")
            spread = "no finite bound"
            if finite.numel():
                low, mid, high = (float(finite.min()), float(finite.median()), float(finite.max()))
                spread = f"median {mid:.4g} {unit} (from {low:.4g} to {high:.4g})"
            missing = int((~torch.isfinite(std)).sum())
            gone = f", {missing} without information" if missing else ""
            lines.append(f"  {path:24s} {tuple(std.shape)!s:14s} {spread}{gone}")
        lines.extend(f"note: {note}" for note in self.notes)
        return "\n".join(lines)


@dataclasses.dataclass(frozen=True)
class _Field:
    """One ``wrt`` entry: its place in the Chain, its value and which entries are parameters."""

    path: str
    name: str  # the logical field path, without a component
    tree_path: str
    spec: FieldSpec
    value: Tensor
    per_image: bool
    entries: tuple[int, ...]  # one image's slice (per image) or the whole value (shared)
    index: tuple[int, ...]  # flat indices of the parameters within ``entries``
    component: int | None

    @property
    def count(self) -> int:
        return len(self.index)

    @property
    def shape(self) -> tuple[int, ...]:
        """Shape of the bound: the value's, without the component axis."""
        entries = self.entries if self.component is None else self.entries[:-1]
        return (self.value.shape[0], *entries) if self.per_image else entries

    def labels(self) -> list[str]:
        components = self.spec.components or ()
        named = bool(self.entries) and len(components) == self.entries[-1]
        out: list[str] = []
        for flat in self.index:
            idx = _unravel(flat, self.entries)
            if named:
                head = f"[{', '.join(map(str, idx[:-1]))}]" if len(idx) > 1 else ""
                out.append(f"{self.name}{head}.{components[idx[-1]]}")
            else:
                out.append(f"{self.name}[{', '.join(map(str, idx))}]" if idx else self.name)
        return out

    def direction(self, k: int) -> Tensor:
        """Return the tangent that moves parameter ``k`` by one unit (in every image)."""
        flat = torch.zeros(self.value.numel(), dtype=self.value.dtype, device=self.value.device)
        if self.per_image:
            flat.view(self.value.shape[0], -1)[:, self.index[k]] = 1.0
        else:
            flat[self.index[k]] = 1.0
        return flat.view(self.value.shape)


def _unravel(flat: int, shape: tuple[int, ...]) -> list[int]:
    out: list[int] = []
    for size in reversed(shape):
        out.append(flat % size)
        flat //= size
    return out[::-1]


def crlb(
    pipe: Pipeline | Plan | Chain,
    chain: Chain | Mapping[str, object] | None = None,
    wrt: str | Sequence[str] = (),
    *,
    method: Literal["auto", "forward", "central"] = "auto",
    reconstruction: Reconstruction | Callable[[Tensor], Tensor] | None = None,
    tolerance: float = 1e-8,
    iterations: int = 1000,
) -> CRLB:
    """Return the Cramér–Rao lower bounds of fields of a Chain under its camera's noise.

    The bound is evaluated at the Chain's values (the truth, or a fit). Fields named in ``wrt``
    are the unknowns; every other field is treated as known, so name nuisance parameters too
    (photons, background) to account for their uncertainty. A field whose value has one entry
    per image gives one set of parameters per image; a value shared by all images (a Python
    number, or a tensor without a batch axis) is one parameter that every image informs, as in
    a calibration over a bead stack. Give per-image values for independent per-image estimates.

    Parameters
    ----------
    pipe : Pipeline, Plan or Chain
        The render whose grids and statics are used. A Chain is rendered on its own (a
        Pipeline is built for it, with outputs ``("expected",)``), and ``chain`` must be None.
    chain : Chain or Mapping[str, object], optional
        The values: a structure-compatible Chain, or values bound by field path onto the
        template (``pipe.bind``); the template when None.
    wrt : str or sequence of str
        Logical field paths, such as ``"beads.position"``, ``"beads.photons"``,
        ``"background"`` or ``"objective.pupil.0.coeffs"``; a component path such as
        ``"beads.position.z"`` takes one component, the others being known.
    method : {"auto", "forward", "central"}, default "auto"
        How the Jacobian is computed: forward-mode AD (``torch.func.jvp``), or central
        differences (two renders per parameter, for elements without forward mode). ``"auto"``
        uses forward mode and falls back to central differences with a warning.
    reconstruction : Reconstruction or callable, optional
        Bound what a reconstruction keeps: a :class:`gradix.recon.Reconstruction` (its
        ``linear`` part), or a callable affine in frames ``[B, C, H, W]``. None bounds the raw
        frames. The reconstruction is held fixed: gradients flow through the frames' mean and
        variance, not through its own parameters.
    tolerance : float, default 1e-8
        With a reconstruction: the conjugate gradients stop once ten iterations raise the
        information by less than this fraction. An early stop underestimates the information
        (the stationary form is a lower bound for any iterate), so the bound errs on the
        conservative side.
    iterations : int, default 1000
        Most conjugate-gradient iterations (with a reconstruction).

    Returns
    -------
    CRLB
        The bounds by path, with the Fisher information, the covariance and a report
        (:meth:`CRLB.explain`). They stay differentiable with respect to the Chain's other
        tensors, so a design can minimise them (reverse mode over the forward-mode Jacobian).

    Raises
    ------
    StructureError
        If a path names no real tensor field, two paths overlap, the camera has no noise
        model, or the reconstruction is not affine in the frames.
    GradientPathError
        If a field has no gradient path to the expected frames, so no image informs it.

    Warns
    -----
    GradixWarning
        When the imaging element renders scalar PSFs above NA/n = 0.45, where quantitative
        bounds need the vectorial model (§5.3); when images leave the Pipeline's envelope;
        when some images' Fisher information is singular; when ``"auto"`` falls back to
        central differences; and when the conjugate gradients reach ``iterations`` while the
        information still grows.

    Notes
    -----
    The Fisher information of image b is ``F_b = J_bᵀ W_b J_b``, with ``J_b = ∂μ_b/∂θ`` the
    Jacobian of its expected frames and ``W_b`` the inverse pixel variances of the camera
    (:func:`gradix.detect.fisher`: exact for shot noise, a Gaussian approximation for read
    noise and EMCCD gain, and only the mean's dependence on θ is counted). With per-image
    parameters θ_b and shared parameters φ, the batch's information is block-structured; the
    bound on (θ_b, φ) is the matching block of its inverse, computed through the Schur
    complement ``S = Σ_b (D_b − C_bᵀ A_b⁻¹ C_b)`` of the per-image blocks
    ``F_b = [[A_b, C_b], [C_bᵀ, D_b]]``. The bound is local: it cannot see ambiguities such as
    the symmetry of an aberration-free PSF about focus.

    With a reconstruction R, image b's information is that of z = R·y, whose mean is R·μ_b and
    whose covariance is C_b = R·Σ_b·Rᵀ (Σ_b the pixel variances; real and imaginary parts of
    complex outputs stacked): ``F_R = (RJ)ᵀ C⁺ (RJ) ≤ F``, with equality when R is invertible
    on the Jacobian's span (ADR-44). Conjugate gradients solve C·X = RJ from X = 0, which
    reaches the pseudo-inverse's solution because RJ lies in C's range; each product
    C·x = R(Σ·Rᵀx) costs one application of R and one of its adjoint (reverse mode). The
    result, ``XᵀRJ + (RJ)ᵀX − XᵀCX`` with X held fixed, is stationary in X, so its gradient
    with respect to the Chain's tensors is exact without differentiating the solver. The
    propagated noise is taken as Gaussian, and only the mean's dependence on θ is counted.

    References
    ----------
    R. J. Ober, S. Ram and E. S. Ward, "Localization accuracy in single-molecule microscopy",
    Biophys. J. 86, 1185–1200 (2004), doi:10.1016/S0006-3495(04)74193-4.

    Examples
    --------
    >>> import torch, gradix as gx
    >>> camera = gx.Camera(pixel_size=6.5, shape=(16, 16), noise=gx.noise.Ideal(), unit="e")
    >>> beads = gx.Emitters(
    ...     position=torch.tensor([[[1.3, 1.3, 0.4]], [[1.3, 1.3, -0.4]]]),
    ...     photons=1000.0,
    ...     emission=gx.Spectrum.line(0.6),
    ... )
    >>> psf = gx.imaging.PointPSF(gx.Objective(NA=0.5, magnification=40), camera, psf="scalar")
    >>> chain = gx.Chain(
    ...     emitters={"beads": beads}, imaging=psf, environment=gx.env.Homogeneous(1.33),
    ...     background=5.0,
    ... )
    >>> bound = gx.crlb(chain, wrt="beads.position")
    >>> tuple(bound["beads.position"].shape)  # per image and emitter: σ_x, σ_y, σ_z in µm
    (2, 1, 3)
    """
    if method not in METHODS:
        raise StructureError(f"unknown method {method!r}", fix=f"methods: {METHODS}")
    pipeline, base = _resolve(pipe, chain)
    paths = (wrt,) if isinstance(wrt, str) else tuple(wrt)
    if not paths:
        raise StructureError("wrt names no field", fix="pass wrt=('beads.position', ...)")
    batch = tree_axis_sizes(base).get("B", 1)
    fields = [_field(base, path, batch, pipeline) for path in paths]
    _check_overlap(fields)
    _check_quality(base, fields)
    camera = base.camera
    if not isinstance(camera, Camera) or camera.noise is None:
        raise StructureError(
            "the Fisher information needs the camera's noise model, and this camera has none",
            fix="give the camera noise=gx.noise.Ideal() (shot noise) or a sensor model",
        )
    notes: list[str] = []
    noise = camera.noise
    if isinstance(noise, Ideal):
        notes.append("shot noise: the Fisher information is exact")
    else:
        notes.append(
            f"{type(noise).__name__} noise: the Fisher information is the Gaussian "
            "approximation, which underestimates it at a few photoelectrons per pixel"
        )
    psf = _scalar_psf(base)
    if psf is not None:
        _warn(notes, psf)
    base = tree_replace(base, {f.tree_path: f.value for f in fields})  # numbers as tensors
    flags = in_envelope(pipeline.envelope, base, batch)
    if not bool(flags.all()):
        bad = torch.nonzero(~flags).flatten().tolist()
        _warn(notes, f"images {bad} leave the envelope; their grids were planned for others")
    ordered = [f for f in fields if f.per_image] + [f for f in fields if not f.per_image]
    linear = None if reconstruction is None else _linear_part(reconstruction)
    mu, columns, used = _jacobian(pipeline, base, ordered, method, notes)
    if linear is None:
        info = fisher(torch.stack(columns, -1).double(), mu.double(), camera)  # [B, P, P]
    else:
        info = _through(linear, mu, columns, camera, tolerance, iterations, notes)
    own = sum(f.count for f in ordered if f.per_image)
    covariance, singular = _covariance(info, own)
    if bool(singular.any()):
        bad = torch.nonzero(singular).flatten().tolist()
        _warn(notes, f"the Fisher information of images {bad} is singular: their bounds are NaN")
    layout = tuple(_Slot(f.path, f.per_image, f.count, f.shape, f.value.dtype) for f in ordered)
    return CRLB(
        std=_bounds(covariance, layout, own, order=tuple(f.path for f in fields)),
        fisher=info,
        covariance=covariance,
        labels=tuple(label for f in ordered for label in f.labels()),
        shared=sum(f.count for f in ordered if not f.per_image),
        units={f.path: _unit(f.spec) for f in fields},
        method=used,
        notes=tuple(notes),
        reconstruction=None if reconstruction is None else _name(reconstruction),
        layout=layout,
    )


def _bounds(
    covariance: Tensor, layout: tuple[_Slot, ...], own: int, order: tuple[str, ...]
) -> dict[str, Tensor]:
    """Return each ``wrt`` entry's bounds, shaped like its value, in ``order``."""
    std = covariance.diagonal(dim1=-2, dim2=-1).sqrt()  # [B, P]
    bounds: dict[str, Tensor] = {}
    start = {"own": 0, "shared": own}
    for slot in layout:
        scope = "own" if slot.per_image else "shared"
        block = std[:, start[scope] : start[scope] + slot.count]
        start[scope] += slot.count
        block = block if slot.per_image else block[0]
        bounds[slot.path] = block.reshape(slot.shape).to(slot.dtype)
    return {path: bounds[path] for path in order}


def _name(reconstruction: object) -> str:
    kind = type(reconstruction)
    return getattr(reconstruction, "__name__", None) or kind.__name__


def _linear_part(reconstruction: object) -> Callable[[Tensor], Tensor]:
    """Return the affine map a bound is taken through."""
    if isinstance(reconstruction, Reconstruction):
        if not reconstruction.is_linear:
            raise StructureError(
                f"{type(reconstruction).__name__} is not affine in the frames: it calibrates on "
                "each frame's median, so the bound cannot be taken through it",
                fix="give it a fixed reference or background frame (from_chain renders one)",
            )
        return reconstruction.linear
    if callable(reconstruction):
        return cast("Callable[[Tensor], Tensor]", reconstruction)
    raise StructureError(
        f"reconstruction must be a gx.recon.Reconstruction or a callable, not "
        f"{type(reconstruction).__name__}"
    )


def _dot(a: Tensor, b: Tensor) -> Tensor:
    """Per-image real inner products ``[B]`` (complex values as stacked real and imaginary)."""
    product = (a.conj() * b).real if a.is_complex() or b.is_complex() else a * b
    return product.reshape(product.shape[0], -1).sum(1)


def _through(
    linear: Callable[[Tensor], Tensor],
    mu: Tensor,
    columns: list[Tensor],
    camera: Camera,
    tolerance: float,
    iterations: int,
    notes: list[str],
) -> Tensor:
    """Return each image's information through an affine map: ``F_R = (RJ)ᵀ(RΣRᵀ)⁺(RJ)``.

    ``[B, P, P]`` in float64, differentiable through the frames' mean and variance. With
    A = R·Σ^(1/2) and c = Σ^(−1/2)·J, F_R = ‖P c‖², P the projection onto A's row space:
    CGLS on min_y ‖Aᵀy − c‖ finds it, and v = Rᵀy gives the stationary form in pixel space.
    """
    mu = mu.double()
    variance = torch.clamp(camera.variance(mu), min=1e-12).expand_as(mu)
    fixed = variance.detach()
    offset = linear(torch.zeros_like(mu).detach()).detach()  # the affine map's constant

    def apply(x: Tensor) -> Tensor:
        return linear(x) - offset

    with torch.enable_grad():
        probe = torch.zeros_like(mu, requires_grad=True)
        image = apply(probe)  # its graph gives the adjoint: Rᵀw = ∂⟨w, R·x⟩/∂x

    def adjoint(w: Tensor) -> Tensor:
        with torch.enable_grad():
            product = (w.conj() * image).real if image.is_complex() else w * image
            (out,) = torch.autograd.grad(product.sum(), probe, retain_graph=True)
        return out

    # Work in the basis that whitens the raw information: radius and index of a small sphere
    # are nearly degenerate (1 − ρ² ~ 1e-12), and a tolerance relative to each column's own
    # information resolves the degenerate direction only when that direction is a column.
    # The basis is held fixed (a constant reparameterisation), so gradients stay exact.
    whiten, unwhiten = _whitening(fisher(torch.stack(columns, -1).double(), mu, camera).detach())
    shape = (-1,) + (1,) * (mu.dim() - 1)
    stacked = torch.stack([column.double() for column in columns])  # [P, B, ...]
    mixed = [
        (stacked * whiten[:, :, k].transpose(0, 1).reshape(len(columns), *shape)).sum(0)
        for k in range(len(columns))
    ]
    solutions: list[Tensor] = []
    worst, steps = 0.0, 0
    for column in mixed:
        with torch.no_grad():
            v, used, growth = _least_squares(
                apply, adjoint, fixed, column.detach(), tolerance, iterations
            )
        solutions.append(v)
        worst, steps = max(worst, growth), max(steps, used)
    if steps >= iterations and worst > tolerance:
        _warn(
            notes,
            f"the conjugate gradients stopped after {steps} iterations while the information "
            f"still grew by {worst:.2g} per ten (tolerance {tolerance:g}): it is underestimated, "
            "so the bound is conservative",
            4,
        )
    else:
        notes.append(f"through the reconstruction: conjugate gradients in {steps} iterations")
    notes.append("the propagated noise is taken as Gaussian (sums over many pixels)")
    # the stationary form in pixel space, F_ij = ⟨v_i, J_j⟩ + ⟨J_i, v_j⟩ − ⟨v_i, Σ·v_j⟩: exact
    # gradients through J and Σ with v held fixed, and a lower bound for any v
    count = len(mixed)
    rows = []
    for i in range(count):
        row = [
            _dot(solutions[i], mixed[j])
            + _dot(mixed[i], solutions[j])
            - _dot(solutions[i], variance * solutions[j])
            for j in range(count)
        ]
        rows.append(torch.stack(row, -1))
    info = torch.stack(rows, -2)  # [B, P, P] in the whitened basis
    info = unwhiten @ ((info + info.transpose(-1, -2)) / 2.0) @ unwhiten
    return (info + info.transpose(-1, -2)) / 2.0


def _whitening(raw: Tensor) -> tuple[Tensor, Tensor]:
    """Return ``F^(−1/2)`` and ``F^(1/2)`` of symmetric information matrices ``[B, P, P]``.

    Directions without information (eigenvalues below 1e-14 of the largest) keep unit scale.
    """
    values, vectors = torch.linalg.eigh(raw)
    top = values.amax(-1, keepdim=True).clamp_min(torch.finfo(raw.dtype).tiny)
    safe = torch.where(values > 1e-14 * top, values, top)
    root = safe.sqrt()
    whiten = vectors @ torch.diag_embed(1.0 / root) @ vectors.transpose(-1, -2)
    unwhiten = vectors @ torch.diag_embed(root) @ vectors.transpose(-1, -2)
    return whiten, unwhiten


def _least_squares(
    apply: Callable[[Tensor], Tensor],
    adjoint: Callable[[Tensor], Tensor],
    variance: Tensor,
    jacobian: Tensor,
    tolerance: float,
    iterations: int,
) -> tuple[Tensor, int, float]:
    """Project a Jacobian column onto what R keeps, by CGLS in whitened pixel space.

    Solves min_y ‖Aᵀy − c‖ per image, with Aᵀ = Σ^(1/2)·Rᵀ and c = Σ^(−1/2)·J, from y = 0
    (conjugate gradients on the normal equations R·Σ·Rᵀ·y = R·J, carried on the pixel-space
    residual s = c − Aᵀy). The information ‖c‖² − ‖s‖² rises at every step; an image stops once
    ten steps raise it by less than ``tolerance`` (relative), once a step would not shrink its
    residual (rounding, on an extreme range of variances), or at ``iterations``. Returns
    v = Rᵀy in pixel space, the iterations used and the worst relative growth over the last
    ten steps.
    """
    sqrt = variance.sqrt()
    shape = (jacobian.shape[0],) + (1,) * (jacobian.dim() - 1)
    c = jacobian / sqrt
    s = c.clone()
    v = torch.zeros_like(c)
    g = apply(sqrt * s)  # A·s, in R's output space
    p = g.clone()
    gamma = _dot(g, g)
    ss = _dot(s, s)
    total = ss.clone()  # ‖c‖², the raw information of the column
    active = gamma > 0
    recent: list[Tensor] = []
    used = 0
    for used in range(1, iterations + 1):  # noqa: B007 - the count is reported
        back = adjoint(p)  # Rᵀp
        q = sqrt * back  # Aᵀp
        qq = _dot(q, q)
        step = active & (qq > 0)
        alpha = torch.where(step, gamma / torch.where(step, qq, 1.0), 0.0)
        s_new = s - alpha.reshape(shape) * q
        ss_new = _dot(s_new, s_new)
        step = step & (ss_new <= ss)  # the residual must shrink
        alpha = torch.where(step, alpha, 0.0)
        s = torch.where(step.reshape(shape), s_new, s)
        v = v + alpha.reshape(shape) * back
        gain = torch.where(step, ss - ss_new, 0.0)
        ss = torch.where(step, ss_new, ss)
        recent = [*recent[-9:], gain]
        information = total - ss
        growth = torch.stack(recent).sum(0) / torch.where(information > 0, information, 1.0)
        settled = (len(recent) == 10) & (growth <= tolerance)
        g = apply(sqrt * s)
        gamma_new = _dot(g, g)
        active = step & ~settled & (gamma_new > 0)
        if not bool(active.any()):
            break
        beta = torch.where(active, gamma_new / torch.where(gamma > 0, gamma, 1.0), 0.0)
        p = torch.where(active.reshape(shape), g + beta.reshape(shape) * p, torch.zeros_like(p))
        gamma = gamma_new
    information = total - ss
    growth = torch.stack(recent).sum(0) / torch.where(information > 0, information, 1.0)
    return v, used, float(growth.max()) if growth.numel() else 0.0


def _unit(spec: FieldSpec) -> str:
    quantity = QUANTITIES.get(spec.quantity or "")
    unit = quantity.unit if quantity is not None else ""
    return "" if unit == "1" else unit  # dimensionless


def _resolve(
    pipe: Pipeline | Plan | Chain, chain: Chain | Mapping[str, object] | None
) -> tuple[Pipeline, Chain]:
    """Return the Pipeline to render with and the Chain holding the values."""
    if isinstance(pipe, Chain):
        if chain is not None:
            raise StructureError(
                "gx.crlb takes a Chain alone, or a Pipeline (or Plan) and its values",
                fix="gx.crlb(chain, wrt=...) or gx.crlb(pipe, chain, wrt=...)",
            )
        return Pipeline(pipe, outputs=("expected",)), pipe
    if isinstance(pipe, Plan):
        pipe = pipe.pipeline
    if not isinstance(pipe, Pipeline):
        msg = f"gx.crlb takes a Pipeline, a Plan or a Chain, got {type(pipe).__name__}"
        raise StructureError(msg)
    if chain is None:
        return pipe, pipe.template
    if isinstance(chain, Mapping):
        return pipe, pipe.bind(chain)
    if not isinstance(chain, Chain):
        raise StructureError(f"the values must be a Chain or a mapping, got {type(chain).__name__}")
    return pipe, chain


def _field(chain: Chain, path: str, batch: int, pipeline: Pipeline) -> _Field:
    """Locate a ``wrt`` path: its tree path, schema, value and parameter entries."""
    (tree_path,) = chain.resolve(path)
    component_name: str | None = None
    try:
        value = tree_get(chain, tree_path)
    except StructureError as err:  # "beads.position.z": a component of a field, or a typo
        field_path, _, component_name = tree_path.rpartition(".")
        try:
            value = tree_get(chain, field_path)
        except StructureError:
            raise err from None
        if not isinstance(value, (Tensor, int, float)):
            raise err from None
        tree_path = field_path
    owner_path, _, name = tree_path.rpartition(".")
    owner = tree_get(chain, owner_path) if owner_path else chain
    spec = type(owner).schema().get(name) if isinstance(owner, Node) else None
    if spec is None or spec.kind != "tensor":
        raise StructureError(f"{path!r} is not a tensor field, so it has no bound")
    if value is None:
        raise StructureError(f"{path!r} is absent in the Chain", fix="give it a value")
    real = spec.dtype in ("real", "number")
    if isinstance(value, Tensor) and value.is_floating_point():
        # an expanded view (stride 0) cannot carry a forward-mode tangent: materialise it
        tensor = value.contiguous() if 0 in value.stride() and value.numel() > 1 else value
    elif isinstance(value, (int, float)) and not isinstance(value, bool) and real:
        # a Python number: one parameter shared by every image
        tensor = torch.tensor(float(value), dtype=pipeline.dtype, device=pipeline.device)
    else:
        kind = value.dtype if isinstance(value, Tensor) else type(value).__name__
        msg = f"{path!r} holds {kind} values; bounds need real-valued parameters"
        raise StructureError(msg)
    axes = leading_axes(spec, tuple(tensor.shape)) if spec.role in ROLE_RULES else ()
    per_image = bool(axes) and axes[0] == "B" and tensor.shape[0] == batch
    entries = tuple(tensor.shape[1:] if per_image else tensor.shape)
    component: int | None = None
    if component_name is not None:
        names = spec.components or ()
        if component_name not in names or not entries or entries[-1] != len(names):
            known = f"components: {list(names)}" if names else "the field has no components"
            raise StructureError(f"{path!r} names no component of its field", fix=known)
        component = names.index(component_name)
    count = math.prod(entries)
    last = entries[-1] if entries else 1
    index = tuple(i for i in range(count) if component is None or i % last == component)
    logical = path if component_name is None else path.rpartition(".")[0]
    return _Field(
        path=path,
        name=logical,
        tree_path=tree_path,
        spec=spec,
        value=tensor,
        per_image=per_image,
        entries=entries,
        index=index,
        component=component,
    )


def _check_overlap(fields: list[_Field]) -> None:
    """Refuse a field named twice, or whole and by component; distinct components may mix."""
    for i, f in enumerate(fields):
        for g in fields[:i]:
            if f.tree_path == g.tree_path and (
                f.component is None or g.component is None or f.component == g.component
            ):
                raise StructureError(f"{g.path!r} and {f.path!r} overlap in wrt")


def _check_quality(chain: Chain, fields: list[_Field]) -> None:
    table = gradient_table(chain, {"expected": Expected()})
    for f in fields:
        if table.rows.get(f.tree_path, {}).get("expected", "zero") == "zero":
            raise GradientPathError(
                f"{f.path!r} does not reach the expected frames, so no image informs it",
                fix="bound fields that change the image (positions, photons, background, ...)",
            )


def _scalar_psf(chain: Chain) -> str | None:
    """Return a note when scalar PSFs make quantitative bounds optimistic (§5.3)."""
    imaging = chain.imaging
    objective: object = None
    if isinstance(imaging, PointPSF) and imaging.psf != "vectorial":
        objective = imaging.objective
    elif isinstance(imaging, Strata):
        objective = imaging.objective
    if not isinstance(objective, Objective):
        return None
    ratio = ratio_per_image(objective, chain.environment)
    if ratio is None or ratio <= QUANTITATIVE_NA_N:
        return None
    return (
        f"scalar PSFs at NA/n = {ratio:.3g} > {QUANTITATIVE_NA_N}: the bounds are optimistic "
        "until the vectorial model lands (M4b, §5.3)"
    )


def _warn(notes: list[str], message: str, stacklevel: int = 3) -> None:
    notes.append(message)
    warnings.warn(message, GradixWarning, stacklevel=stacklevel)  # the caller of gx.crlb


def _renderer(pipeline: Pipeline, chain: Chain, f: _Field) -> Callable[[Tensor], Tensor]:
    def render(value: Tensor) -> Tensor:
        return pipeline._expected(tree_replace(chain, {f.tree_path: value}))[0]

    return render


def _jacobian(
    pipeline: Pipeline, chain: Chain, fields: list[_Field], method: str, notes: list[str]
) -> tuple[Tensor, list[Tensor], str]:
    """Return the expected frames, one Jacobian column per parameter and the method used."""
    mu, _flags = pipeline._expected(chain)
    if method in ("auto", "forward"):
        try:
            return mu, _columns(pipeline, chain, fields, _forward), "forward"
        except (NotImplementedError, RuntimeError) as err:
            if method == "forward" or not _no_forward_mode(err):
                raise
            _warn(notes, f"forward mode is unavailable ({err}); used central differences", 4)
    return mu, _columns(pipeline, chain, fields, _central), "central"


def _no_forward_mode(err: Exception) -> bool:
    if isinstance(err, NotImplementedError):
        return True
    text = str(err).lower()
    return any(s in text for s in ("forward ad", "forward-mode", "forward mode", "jvp"))


def _columns(
    pipeline: Pipeline,
    chain: Chain,
    fields: list[_Field],
    column: Callable[[Callable[[Tensor], Tensor], _Field, int], Tensor],
) -> list[Tensor]:
    return [column(_renderer(pipeline, chain, f), f, k) for f in fields for k in range(f.count)]


def _forward(render: Callable[[Tensor], Tensor], f: _Field, k: int) -> Tensor:
    """Return ``∂μ/∂θ_k`` by one forward-mode pass (every image's column at once)."""
    out = torch.func.jvp(render, (f.value,), (f.direction(k),))
    return cast("Tensor", out[1])


def _central(render: Callable[[Tensor], Tensor], f: _Field, k: int) -> Tensor:
    """Return ``∂μ/∂θ_k`` by central differences, with a step of ε^⅓ of the value's scale."""
    direction = f.direction(k)
    eps = torch.finfo(f.value.dtype).eps ** (1.0 / 3.0)
    if f.per_image:
        theta = f.value.detach().reshape(f.value.shape[0], -1)[:, f.index[k]]
        step = eps * theta.abs().clamp_min(1.0)  # [B]
        moved = step.reshape(-1, *([1] * (f.value.ndim - 1))) * direction
        plus, minus = render(f.value + moved), render(f.value - moved)
        return (plus - minus) / (2.0 * step.reshape(-1, *([1] * (plus.ndim - 1))))
    theta = float(f.value.detach().reshape(-1)[f.index[k]])
    step = eps * max(abs(theta), 1.0)
    return (render(f.value + step * direction) - render(f.value - step * direction)) / (2 * step)


def _inverse(matrix: Tensor, dead: Tensor) -> tuple[Tensor, Tensor]:
    """Invert symmetric positive matrices ``[..., n, n]`` whose ``dead`` entries carry nothing.

    Dead rows and columns (no information) are replaced by the identity, and the rest is
    scaled by its diagonal before inversion (Jacobi), which keeps mixed units well conditioned.
    Returns the inverses and a flag per matrix that is True where it is singular all the same.
    """
    n = matrix.shape[-1]
    live = ~dead
    keep = live[..., :, None] & live[..., None, :]
    eye = torch.eye(n, dtype=matrix.dtype, device=matrix.device)
    matrix = torch.where(keep, matrix, eye)
    scale = matrix.diagonal(dim1=-2, dim2=-1).rsqrt()
    inverse, info = torch.linalg.inv_ex(matrix * scale[..., :, None] * scale[..., None, :])
    return inverse * scale[..., :, None] * scale[..., None, :], info != 0


def _covariance(info: Tensor, own: int) -> tuple[Tensor, Tensor]:
    """Return each image's marginal covariance ``[B, P, P]`` and which images are singular.

    ``info`` holds each image's Fisher information with its own ``own`` parameters first and
    the shared ones last. Parameters without information get an infinite variance.
    """
    b, p, _ = info.shape
    diag = info.diagonal(dim1=-2, dim2=-1)
    dead_own = diag[:, :own] == 0  # [B, own]
    shared_info = info[:, own:, own:].sum(0)  # [s, s]: every image informs the shared ones
    dead_shared = shared_info.diagonal() == 0  # [s]
    if own:
        a_inv, singular = _inverse(info[:, :own, :own], dead_own)
    else:
        a_inv = info[:, :0, :0]
        singular = torch.zeros(b, dtype=torch.bool, device=info.device)
    if p == own:
        covariance = a_inv
    else:
        c = info[:, :own, own:].masked_fill(dead_own[..., None], 0.0)  # [B, own, s]
        schur = shared_info - torch.einsum("bij,bik,bkl->jl", c, a_inv, c)
        s_inv, singular_shared = _inverse(schur[None], dead_shared[None])
        s_inv = s_inv[0]
        gain = a_inv @ c @ s_inv  # [B, own, s]
        top = torch.cat([a_inv + gain @ c.transpose(-1, -2) @ a_inv, -gain], -1)
        bottom = torch.cat([-gain.transpose(-1, -2), s_inv.expand(b, -1, -1)], -1)
        covariance = torch.cat([top, bottom], -2)
        singular = singular | singular_shared
    dead = torch.cat([dead_own, dead_shared.expand(b, -1)], -1)  # [B, P]
    covariance = covariance.masked_fill(dead[..., :, None] | dead[..., None, :], 0.0)
    covariance = covariance + torch.diag_embed(torch.where(dead, math.inf, 0.0))
    return torch.where(singular[:, None, None], math.nan, covariance), singular
