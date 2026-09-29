"""Media (``gx.env``): the single source of truth for the refractive index around the sample."""

from __future__ import annotations

import dataclasses
import math
from typing import ClassVar

import torch
from torch import Tensor

from gradix._core.errors import StructureError
from gradix.objects.materials import D_LINE, Constant, Material
from gradix.schema.base import DataObject
from gradix.schema.fields import child, field
from gradix.schema.layout import canonical

__all__ = ["Homogeneous", "LayeredMedium", "Medium", "water_on_coverslip"]


@dataclasses.dataclass(frozen=True, kw_only=True, eq=False)
class Medium(DataObject):
    """Base of media: homogeneous (:class:`Homogeneous`) or layered (:class:`LayeredMedium`)."""

    layered: ClassVar[bool] = False
    """Whether imaging elements use the layered collection rule (``pupil_amplitude``)."""

    def index(
        self,
        *,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str | None = None,
        wavelength: Tensor | float | None = None,
    ) -> Tensor:
        """Return the refractive index of the sample medium at vacuum wavelengths.

        Parameters
        ----------
        dtype : torch.dtype, default torch.float32
            Dtype of the result (complex indices keep their complex dtype).
        device : torch.device or str, optional
            Device of the result.
        wavelength : Tensor or float, optional
            Vacuum wavelengths, µm: a number, ``[L]`` or ``[B|1, …]`` (such as a
            ``[B|1, L]`` spectrum). None means the d line (587.6 nm) for dispersive media.

        Returns
        -------
        Tensor
            The index, broadcastable against ``[B|1, *wavelength.shape[1:]]``: ``[B|1, 1]``
            for a number (or a non-dispersive medium).
        """
        raise NotImplementedError(type(self).__name__)

    def immersion_index(
        self,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str | None = None,
        wavelength: Tensor | float | None = None,
    ) -> Tensor:
        """Return the index of the medium the objective's pupil is defined in.

        Parameters
        ----------
        dtype : torch.dtype, default torch.float32
            Dtype of the result.
        device : torch.device or str, optional
            Device of the result.
        wavelength : Tensor or float, optional
            Vacuum wavelengths, µm, as for :meth:`index`.

        Returns
        -------
        Tensor
            ``[B|1]`` (or broadcastable against the wavelengths): the medium itself unless a
            layered medium says otherwise (its immersion).
        """
        n = self.index(dtype=dtype, device=device, wavelength=wavelength)
        return n.reshape(-1) if math.prod(n.shape[1:]) == 1 else n

    def pupil_amplitude(self, u2: Tensor, wavelength: Tensor, z: Tensor) -> tuple[Tensor, Tensor]:
        """Return the scalar pupil amplitude of isotropic emitters (layered media only).

        Parameters
        ----------
        u2 : Tensor
            ``|u|²`` of the pupil samples, NA units squared.
        wavelength : Tensor
            Vacuum wavelengths, µm.
        z : Tensor
            Emitter heights, µm.

        Returns
        -------
        tuple of Tensor
            The amplitude, normalised per emitted photon, and the index the pupil is defined in.

        Raises
        ------
        NotImplementedError
            For media that are not layered (:attr:`layered` is False).
        """
        raise NotImplementedError(f"{type(self).__name__} is not a layered medium")


@dataclasses.dataclass(frozen=True, eq=False)
class Homogeneous(Medium):
    """A homogeneous, index-matched medium.

    Parameters
    ----------
    n : Tensor or float
        Refractive index (possibly complex), or a per-image ``[B]`` tensor.

    Examples
    --------
    >>> Homogeneous(1.33).n
    1.33
    """

    registry_name: ClassVar[str | None] = "env.homogeneous"
    """Stable name for signatures and saved inputs."""

    n: Tensor | float = field(
        quantity="index",
        role="image",
        dtype="number",
        shape_affecting=True,
        constraint="positive",
        doc="refractive index",
    )

    def index(
        self,
        *,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str | None = None,
        wavelength: Tensor | float | None = None,
    ) -> Tensor:
        """Return the refractive index, ``[B|1, 1]`` (one value for every wavelength bin).

        Parameters
        ----------
        dtype : torch.dtype, default torch.float32
            Dtype of a real result; complex indices keep a complex dtype.
        device : torch.device or str, optional
            Device of the result.
        wavelength : Tensor or float, optional
            Unused: the medium is non-dispersive.

        Returns
        -------
        Tensor
            The index.
        """
        n = self.n
        is_complex = isinstance(n, complex) or (isinstance(n, Tensor) and n.is_complex())
        if is_complex:
            dtype = torch.complex128 if dtype == torch.float64 else torch.complex64
        return canonical(n, self.schema()["n"], dtype=dtype, device=device)[:, None]


_THICKNESS = 170.0
"""Default coverslip thickness, µm (#1.5)."""


def _wavelengths(wavelength: Tensor | float | None) -> Tensor | float:
    """Return wavelengths for a material: the d line by default, ``[L]`` as ``[1, L]``."""
    if wavelength is None:
        return D_LINE
    if isinstance(wavelength, Tensor) and wavelength.ndim == 1:
        return wavelength[None]  # per wavelength, not per image
    return wavelength


def _kz(n: Tensor, u2: Tensor, wavelength: Tensor) -> Tensor:
    """Return the complex ``k_z = 2π·√(n² − |u|²)/λ`` on the branch ``Im k_z ≥ 0``, rad/µm."""
    arg = n * n - u2
    cdtype = torch.complex128 if arg.dtype == torch.float64 else torch.complex64
    arg = arg.to(cdtype)
    zero = arg == 0
    root = torch.where(zero, torch.zeros_like(arg), torch.sqrt(torch.where(zero, 1.0, arg)))
    root = torch.where(root.imag < 0, -root, root)
    return (2.0 * math.pi) * root / wavelength


@dataclasses.dataclass(frozen=True, kw_only=True, eq=False)
class LayeredMedium(Medium):
    """Sample, coverslip and immersion: the layered collection rule of §4.1.

    z = 0 is the sample-side surface of the coverslip and the sample lies at z < 0; +z points
    toward the objective. An emitter at depth ``h = −z`` radiates into the sample; its angular
    spectrum is propagated *only toward the interface* (evanescent components decay),
    transmitted by the stack (the s amplitude on the scalar path, for every ``|u| ≤ NA``,
    including ``n_sample < |u|``: supercritical collection), and apodised by the power flux in
    the immersion medium. The scalar pupil power density per d²f (f = u/λ) is
    ``(2π)²·|t|²·exp(−2·Im k_z,s·h)·Re k_z,i / (|k_z,s|²·4π·k_s)`` with
    ``t = 2k_z,s/(k_z,s + k_z,g)`` (times the glass-immersion transmission when they differ):
    0.709 of the emitted photons at NA 1.45, 680 nm, water on glass, h = 0, and 0.575 at
    h = 0.1 µm. Multiple reflections in the coverslip are neglected. ``design_*`` values describe
    what the objective is corrected for; a mismatch adds the Gibson–Lanni phase.

    Parameters
    ----------
    sample : Material, Tensor or float
        The sample medium: a material (``gx.materials.Water()``) or an index, per image
        ``[B]``.
    coverslip : Material, Tensor, float or tuple, default 1.518
        The coverslip: a material or an index, or ``(material, thickness[, design])`` where
        ``design`` is the design coverslip's material (or ``(material, thickness)``).
    immersion : Material, Tensor or float, default 1.518
        The immersion medium: a material or an index.
    thickness : Tensor or float, default 170.0
        Coverslip thickness, µm.
    design_coverslip : Material, Tensor or float, optional
        Design coverslip material or index; the actual one by default.
    design_thickness : Tensor or float, optional
        Design coverslip thickness, µm; the actual one by default.

    Examples
    --------
    >>> import gradix as gx
    >>> medium = LayeredMedium(sample=gx.materials.Water(), coverslip=(gx.materials.Glass(), 170.0))
    >>> round(float(medium.index().reshape(())), 3)
    1.333
    """

    registry_name: ClassVar[str | None] = "env.layered"
    """Stable name for signatures and saved inputs."""
    schema_version: ClassVar[int] = 2

    layered: ClassVar[bool] = True
    """Imaging elements use :meth:`pupil_amplitude` (the layered collection rule)."""

    sample: Material | Tensor | float = child(doc="the sample medium")
    coverslip: Material | Tensor | float | tuple = child(
        default_factory=lambda: Constant(1.518), doc="the coverslip"
    )
    immersion: Material | Tensor | float = child(
        default_factory=lambda: Constant(1.518), doc="the immersion medium"
    )
    thickness: Tensor | float = field(
        quantity="length",
        role="image",
        constraint="positive",
        default=170.0,
        doc="coverslip thickness",
    )
    design_coverslip: Material | Tensor | float | None = child(
        default=None, doc="design coverslip material"
    )
    design_thickness: Tensor | float | None = field(
        quantity="length", role="image", default=None, doc="design coverslip thickness"
    )

    def __post_init__(self) -> None:
        coverslip = self.coverslip
        if isinstance(coverslip, tuple):
            # the plan's (material, thickness[, design]) form
            if not 1 <= len(coverslip) <= 3:
                raise StructureError("coverslip=(material, thickness[, design]) takes 1–3 items")
            object.__setattr__(self, "coverslip", coverslip[0])
            if len(coverslip) > 1:  # the tuple's thickness wins over thickness=
                object.__setattr__(self, "thickness", coverslip[1])
            if len(coverslip) > 2:
                design = coverslip[2]
                if isinstance(design, tuple):
                    object.__setattr__(self, "design_coverslip", design[0])
                    if len(design) > 1:
                        object.__setattr__(self, "design_thickness", design[1])
                else:
                    object.__setattr__(self, "design_coverslip", design)
        for name in ("sample", "coverslip", "immersion", "design_coverslip"):
            value = getattr(self, name)
            if value is not None and not isinstance(value, Material):
                object.__setattr__(self, name, Constant(value))
        super().__post_init__()

    def _material(self, name: str) -> Material:
        value = getattr(self, name)
        if not isinstance(value, Material):  # pragma: no cover - normalised in __post_init__
            raise StructureError(f"LayeredMedium.{name} is not a material")
        return value

    def _get(self, name: str, dtype: torch.dtype, device: torch.device | str | None) -> Tensor:
        value = getattr(self, name)
        return canonical(value, self.schema()[name], dtype=dtype, device=device)

    def index(
        self,
        *,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str | None = None,
        wavelength: Tensor | float | None = None,
    ) -> Tensor:
        """Return the index of the sample medium at vacuum wavelengths.

        Parameters
        ----------
        dtype : torch.dtype, default torch.float32
            Dtype of the result.
        device : torch.device or str, optional
            Device of the result.
        wavelength : Tensor or float, optional
            Vacuum wavelengths, µm: a number, ``[L]`` or ``[B|1, …]``; the d line by default.

        Returns
        -------
        Tensor
            The sample index: ``[B|1, 1]`` for a number, else broadcastable against
            ``[B|1, *wavelength.shape[1:]]``.
        """
        n = self._material("sample").index(_wavelengths(wavelength), dtype=dtype, device=device)
        return n.reshape(-1, 1) if n.ndim <= 1 else n

    def immersion_index(
        self,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str | None = None,
        wavelength: Tensor | float | None = None,
    ) -> Tensor:
        """Return the immersion index, ``[B|1]`` (or broadcastable against ``wavelength``).

        Parameters
        ----------
        dtype : torch.dtype, default torch.float32
            Dtype of the result.
        device : torch.device or str, optional
            Device of the result.
        wavelength : Tensor or float, optional
            Vacuum wavelengths, µm: a number, ``[L]`` or ``[B|1, …]``; the d line by default.

        Returns
        -------
        Tensor
            The index of the medium between the coverslip and the objective.
        """
        n = self._material("immersion").index(_wavelengths(wavelength), dtype=dtype, device=device)
        return n.reshape(-1) if n.ndim <= 1 else n

    def pupil_amplitude(self, u2: Tensor, wavelength: Tensor, z: Tensor) -> tuple[Tensor, Tensor]:
        """Return the scalar pupil amplitude of isotropic emitters at heights ``z``.

        Normalised so that ``Σ|P|²·(du/λ)²`` over the aperture is the collected fraction of one
        emitted photon: multiply by √photons.

        Parameters
        ----------
        u2 : Tensor
            ``|u|²`` of the pupil samples, NA units squared, ``[Np, Np]``.
        wavelength : Tensor
            Vacuum wavelengths, µm, broadcastable to ``[B, A, N, L, 1, 1]``.
        z : Tensor
            Emitter heights, µm (the sample is z < 0), broadcastable likewise.

        Returns
        -------
        tuple of Tensor
            The complex amplitude ``[B, A, N, L, Np, Np]`` (propagation to the interface,
            transmission, flux apodisation and the Gibson–Lanni mismatch phase), and the
            immersion index, broadcastable to ``[B|1, 1, N|1, L, 1, 1]``: the medium the pupil is
            defined in.
        """
        real, device = u2.dtype, u2.device
        shape6 = (-1, 1, 1, 1, 1, 1)
        wl = wavelength if wavelength.ndim == 6 else wavelength.reshape(shape6)

        def material(name: str) -> Tensor:
            n = self._material(name).index(wl, dtype=real, device=device)
            n = n.real if n.is_complex() else n
            return n if n.ndim == 6 else n.reshape(shape6)

        n_s, n_g, n_i = material("sample"), material("coverslip"), material("immersion")
        kzs = _kz(n_s, u2, wavelength)
        kzg = _kz(n_g, u2, wavelength)
        kzi = _kz(n_i, u2, wavelength)
        k_s = 2.0 * math.pi * n_s / wavelength
        depth = torch.clamp(-z, min=0.0)
        # emitter-independent factors first; only the depth propagation is per emitter.
        # t/k_z,s = 2/(k_z,s + k_z,g) is finite at |u| = n_s
        amplitude = 2.0 / (kzs + kzg)
        amplitude = amplitude * (2.0 * kzg / (kzg + kzi))  # 1 when the immersion matches
        # safe square root: finite forward-mode tangents where the immersion wave is evanescent
        re = kzi.real
        positive = re > 0
        flux = torch.where(positive, torch.sqrt(torch.where(positive, re, 1.0)), 0.0)
        norm = torch.sqrt((2.0 * math.pi) ** 2 / (4.0 * math.pi * k_s))
        amplitude = amplitude * flux * norm
        if self.design_coverslip is not None or self.design_thickness is not None:
            t = self._get("thickness", real, device).reshape(shape6)
            n_d = n_g if self.design_coverslip is None else material("design_coverslip")
            t_d = t
            if self.design_thickness is not None:
                t_d = self._get("design_thickness", real, device).reshape(shape6)
            kzd = _kz(n_d, u2, wavelength)
            # the immersion layer shrinks by the coverslip's excess, keeping the working
            # distance; only the phase applies (evanescent cells carry no power through the
            # stack, and e^{-Im} terms of fictitious design layers would overflow)
            mismatch = (kzg * t).real - (kzd * t_d).real - (kzi * (t - t_d)).real
            amplitude = amplitude * torch.polar(torch.ones_like(mismatch), mismatch)
        return amplitude * torch.exp(1j * kzs * depth), n_i


def water_on_coverslip(
    *,
    sample: Material | Tensor | float = 1.333,
    immersion: Material | Tensor | float = 1.518,
) -> LayeredMedium:
    """Return the common case: an aqueous sample on a #1.5 coverslip, oil immersion.

    Non-dispersive by default (indices at the d line); pass materials such as
    ``gx.materials.Water()`` and ``gx.materials.Oil()`` for dispersion.

    Parameters
    ----------
    sample : Material, Tensor or float, default 1.333
        The sample medium.
    immersion : Material, Tensor or float, default 1.518
        The immersion oil.

    Returns
    -------
    LayeredMedium
        The medium.
    """
    return LayeredMedium(sample=sample, coverslip=1.518, immersion=immersion, thickness=170.0)
