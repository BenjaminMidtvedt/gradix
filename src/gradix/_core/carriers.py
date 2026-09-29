"""Carriers: how light and emitters are represented between elements (§4.4).

The incoherent carriers (:class:`EmitterSet`, :class:`EmitterDensity`, :class:`Irradiance`) are
frozen at M0. The coherent ones (:class:`PlaneWaves`, :class:`Field`, :class:`ObjectSpectra`)
are drafts until the M3 exit, after the optical-theorem, direct-summation and Fresnel tests.

Every carrier field declares a full dimension pattern; patterns ``"B|1"`` and ``"A|1"`` let a
carrier broadcast over images and acquisition frames (§4.2).
"""

from __future__ import annotations

import dataclasses
import math
from typing import ClassVar, Protocol, runtime_checkable

import torch
from torch import Tensor

from gradix._core.axes import AcqIndex
from gradix._core.errors import StructureError
from gradix._core.grid import FreqGrid, Grid2D, VolumeGrid
from gradix._core.precision import wrap_phase
from gradix.conventions import safe_sqrt
from gradix.schema.base import Node
from gradix.schema.fields import child, field, knob

__all__ = [
    "Background",
    "Carrier",
    "Contribution",
    "EmitterDensity",
    "EmitterSet",
    "Field",
    "GaussianSheet",
    "Irradiance",
    "ObjectSpectra",
    "PlaneWaves",
]


@dataclasses.dataclass(frozen=True, kw_only=True, eq=False)
class Carrier(Node):
    """Base of carriers: frozen pytrees with fixed dimension patterns."""


@dataclasses.dataclass(frozen=True, kw_only=True, eq=False)
class EmitterSet(Carrier):
    """Incoherent point sources, never voxelised by default (frozen at M0).

    Parameters
    ----------
    position : Tensor
        ``[B|1, A|1, N, 3]`` µm, world frame (§4.1).
    photons : Tensor
        ``[B|1, A|1, N]`` expected emitted photons per exposure.
    presence : Tensor, optional
        ``[B|1, A|1, N]`` in [0, 1]; weights each emitter linearly. None means all present.
    species : Tensor, optional
        ``[B|1, 1, N]`` int64 row of each emitter in the emission table. None means row 0.
    wavelengths : Tensor
        ``[B|1, S, L]`` µm: the emission table, one row per species, L wavelength bins.
    weights : Tensor
        ``[B|1, S, L]`` power weights of the bins (each row sums to 1).
    dipole : Tensor, optional
        ``[B|1, A|1, N, 6]`` dipole second moments ⟨μᵢμⱼ⟩; None means isotropic.
    id : Tensor, optional
        ``[B|1, 1, N]`` int64 persistent identity, for tracking labels.
    acq : AcqIndex, default AcqIndex()
        What the acquisition axis A holds.
    """

    registry_name: ClassVar[str | None] = "carrier.emitter_set"
    """Stable name for signatures and saved inputs."""

    position: Tensor = field(
        quantity="length",
        role="carrier",
        dims=("B|1", "A|1", "N", 3),
        event=(3,),
        components=("x", "y", "z"),
        shape_affecting=True,
    )
    photons: Tensor = field(quantity="photons", role="carrier", dims=("B|1", "A|1", "N"))
    presence: Tensor | None = field(
        quantity="dimensionless", role="carrier", dims=("B|1", "A|1", "N"), default=None
    )
    species: Tensor | None = field(
        quantity="index_map", role="carrier", dims=("B|1", 1, "N"), dtype="integer", default=None
    )
    wavelengths: Tensor = field(
        quantity="wavelength", role="carrier", dims=("B|1", "S", "L"), shape_affecting=True
    )
    weights: Tensor = field(quantity="dimensionless", role="carrier", dims=("B|1", "S", "L"))
    dipole: Tensor | None = field(
        quantity="dimensionless", role="carrier", dims=("B|1", "A|1", "N", 6), default=None
    )
    id: Tensor | None = field(
        quantity="index_map", role="carrier", dims=("B|1", 1, "N"), dtype="integer", default=None
    )
    acq: AcqIndex = knob(default=AcqIndex())

    @property
    def slots(self) -> int:
        """The slot count N.

        Returns
        -------
        int
            N.
        """
        return int(self.position.shape[2])


@dataclasses.dataclass(frozen=True, kw_only=True, eq=False)
class EmitterDensity(Carrier):
    """Dense emitter densities: photons per voxel per species on a volume grid (frozen at M0).

    Parameters
    ----------
    data : Tensor
        ``[B|1, A|1, S, Z, Y, X]`` photons per voxel per exposure.
    wavelengths : Tensor
        ``[B|1, S, L]`` µm: the emission table.
    weights : Tensor
        ``[B|1, S, L]`` power weights of the bins.
    grid : VolumeGrid
        The volume grid.
    acq : AcqIndex, default AcqIndex()
        What the acquisition axis A holds.
    """

    registry_name: ClassVar[str | None] = "carrier.emitter_density"
    """Stable name for signatures and saved inputs."""

    data: Tensor = field(
        quantity="photons", role="carrier", dims=("B|1", "A|1", "S", "Z", "Y", "X")
    )
    wavelengths: Tensor = field(
        quantity="wavelength", role="carrier", dims=("B|1", "S", "L"), shape_affecting=True
    )
    weights: Tensor = field(quantity="dimensionless", role="carrier", dims=("B|1", "S", "L"))
    grid: VolumeGrid = knob()
    acq: AcqIndex = knob(default=AcqIndex())

    def __post_init__(self) -> None:
        super().__post_init__()
        want = (self.grid.nz, *self.grid.xy.shape)
        if tuple(self.data.shape[-3:]) != want:
            got = list(self.data.shape[-3:])
            msg = f"EmitterDensity.data ends in {got} but the grid is {list(want)}"
            raise StructureError(msg)


@dataclasses.dataclass(frozen=True, kw_only=True, eq=False)
class Irradiance(Carrier):
    """The incoherent currency on the detection grid (frozen at M0).

    Parameters
    ----------
    data : Tensor
        ``[B|1, A|1, L|1, Y, X]`` photons per exposure per grid cell, before the camera.
    grid : Grid2D
        The detection grid (the camera grid, or finer by an integer factor).
    acq : AcqIndex, default AcqIndex()
        What the acquisition axis A holds.
    """

    registry_name: ClassVar[str | None] = "carrier.irradiance"
    """Stable name for signatures and saved inputs."""

    data: Tensor = field(quantity="photons", role="carrier", dims=("B|1", "A|1", "L|1", "Y", "X"))
    grid: Grid2D = knob()
    acq: AcqIndex = knob(default=AcqIndex())

    def __post_init__(self) -> None:
        super().__post_init__()
        if tuple(self.data.shape[-2:]) != self.grid.shape:
            got, want = list(self.data.shape[-2:]), list(self.grid.shape)
            msg = f"Irradiance.data ends in {got} but the grid is {want}"
            raise StructureError(msg)

    def __add__(self, other: Irradiance) -> Irradiance:
        if not isinstance(other, Irradiance):
            return NotImplemented
        if other.grid != self.grid or other.acq != self.acq:
            msg = "only irradiances on the same grid and acquisition can be added"
            raise StructureError(msg)
        return self.replace(data=self.data + other.data)


@dataclasses.dataclass(frozen=True, kw_only=True, eq=False)
class PlaneWaves(Carrier):
    """Analytic plane waves: illumination modes, sample-side references, reflections (draft).

    Parameters
    ----------
    amplitude : Tensor
        Complex ``[B|1, A|1, M, J, L, P]`` in √(photons/µm²): M mutually incoherent modes, J
        mutually coherent waves per mode, L wavelength bins, P polarisation components.
    u : Tensor
        ``[B|1, A|1, M, J, 2]`` directions ``n·sinθ·(cosφ, sinφ)``; ``|u| > n`` is evanescent.
    wavelengths : Tensor
        ``[B|1, L]`` vacuum wavelengths in µm.
    z0 : Tensor or float, default 0.0
        Plane at which ``amplitude`` is referenced, µm.
    travel : {1, -1}, default 1
        +1 toward the detection objective, −1 away.
    """

    registry_name: ClassVar[str | None] = "carrier.plane_waves"
    """Stable name for signatures and saved inputs."""

    amplitude: Tensor = field(
        quantity="dimensionless",
        role="carrier",
        dims=("B|1", "A|1", "M", "J", "L", "P"),
        dtype="complex",
    )
    u: Tensor = field(quantity="dimensionless", role="carrier", dims=("B|1", "A|1", "M", "J", 2))
    wavelengths: Tensor = field(
        quantity="wavelength", role="carrier", dims=("B|1", "L"), shape_affecting=True
    )
    z0: Tensor | float = field(quantity="length", role="shared", default=0.0)
    travel: int = knob(default=1, choices=(1, -1))

    def at(self, points: Tensor, n: Tensor | float) -> Tensor:
        """Evaluate the waves exactly at points, including evanescent decay.

        Parameters
        ----------
        points : Tensor
            ``[B|1, A|1, N, 3]`` positions in µm.
        n : Tensor or float
            Refractive index of the medium, a scalar or ``[B|1, L]``.

        Returns
        -------
        Tensor
            Complex ``[B, A, M, L, P, N]``: the coherent sum over J at each point, in the
            amplitude's complex dtype. Waves travelling toward −z (``travel = −1``) propagate
            and decay along −z; the gradient is finite at ``|u| = n``.

        Raises
        ------
        StructureError
            If ``n`` is 1-d (ambiguous: per image or per wavelength).
        """
        amp, phasor = self._amplitudes_and_phasors(points, n)
        out = torch.einsum("bamjlp,bamjln->bamlpn", amp, phasor)
        return out.to(self.amplitude.dtype)

    def at_waves(self, points: Tensor, n: Tensor | float) -> Tensor:
        """Evaluate each coherent wave separately at points (no sum over J).

        Direction-dependent scatterers (Mie) need every incident wave on its own, because the
        scattering angle differs from wave to wave.

        Parameters
        ----------
        points : Tensor
            ``[B|1, A|1, N, 3]`` positions in µm.
        n : Tensor or float
            Refractive index of the medium, a scalar or ``[B|1, L]``.

        Returns
        -------
        Tensor
            Complex ``[B, A, M, J, L, P, N]`` in the amplitude's complex dtype; summing over J
            gives :meth:`at`.
        """
        amp, phasor = self._amplitudes_and_phasors(points, n)
        out = amp[..., None] * phasor[..., None, :]  # [B, A, M, J, L, P, N]
        return out.to(self.amplitude.dtype)

    def _amplitudes_and_phasors(self, points: Tensor, n: Tensor | float) -> tuple[Tensor, Tensor]:
        """Return fp64 amplitudes ``[B|1, A|1, M, J, L, P]`` and phasors ``[B, A, M, J, L, N]``."""
        device = points.device  # waves built from scalar parameters live on the CPU
        wl = self.wavelengths.to(device=device, dtype=torch.float64)
        k0 = (2.0 * math.pi / wl)[:, None, None, None, :, None]  # [B,1,1,1,L,1]
        u = self.u.to(device=device, dtype=torch.float64)
        ux = u[..., 0][..., None, None]  # [B,A,M,J,1,1]
        uy = u[..., 1][..., None, None]
        pts = points.to(torch.float64)
        x = pts[..., 0][:, :, None, None, None, :]  # [B,A,1,1,1,N]
        y = pts[..., 1][:, :, None, None, None, :]
        z = pts[..., 2][:, :, None, None, None, :]
        n_t = torch.as_tensor(n, dtype=torch.complex128, device=wl.device)
        if n_t.ndim == 1:
            msg = f"n must be a scalar or [B|1, L], got shape {list(n_t.shape)}"
            raise StructureError(msg, fix="use n[:, None] per image or n[None, :] per wavelength")
        if n_t.ndim == 2:
            n_t = n_t[:, None, None, None, :, None]
        uz = safe_sqrt(n_t * n_t - (ux * ux + uy * uy))  # principal branch: Im ≥ 0
        z0 = torch.as_tensor(self.z0, dtype=torch.float64, device=wl.device)
        lateral = wrap_phase(k0 * (ux * x + uy * y))
        axial = self.travel * k0 * uz * (z - z0)
        phasor = torch.exp(1j * (lateral + axial))  # [B,A,M,J,L,N]
        amp = self.amplitude.to(device=device, dtype=torch.complex128)
        return amp, phasor


@runtime_checkable
class Background(Protocol):
    """An analytic field that can be evaluated exactly at points (§4.4): the Background protocol.

    :class:`PlaneWaves` implements it, as do :class:`GaussianSheet` and the Gaussian beams of
    M3. Consumers that only need the field at emitters or scatterers (excitation, the incident
    field of sparse producers) accept any Background.

    Parameters
    ----------
    *args : object
        Not used: a protocol is a type to check against, never instantiated.
    **kwargs : object
        Not used.
    """

    wavelengths: Tensor

    def at(self, points: Tensor, n: Tensor | float) -> Tensor:
        """Evaluate the field at points.

        Parameters
        ----------
        points : Tensor
            ``[B|1, A|1, N, 3]`` positions, µm.
        n : Tensor or float
            Refractive index of the medium, a scalar or ``[B|1, L]``.

        Returns
        -------
        Tensor
            Complex ``[B, A, M, L, P, N]`` in √(photons/µm²).
        """
        ...


@dataclasses.dataclass(frozen=True, kw_only=True, eq=False)
class GaussianSheet(Carrier):
    """A Gaussian light sheet: a beam focused in z by a cylindrical lens, uniform across (draft).

    The paraxial field of a 1-D Gaussian beam travelling along ``axis``, with its waist ``waist``
    (1/e² half-width in z) at ``focus`` along the axis, centred on the plane ``z = center``:
    ``|E|² = irradiance·(w₀/w)·exp(−2(z − center)²/w²)`` with ``w = w₀·√(1 + (s/s_R)²)``,
    ``s`` the distance past the waist and ``s_R = π·w₀²·n/λ`` its Rayleigh range.

    Parameters
    ----------
    irradiance : Tensor
        Peak irradiance at the waist, photons/µm² per exposure, ``[B|1]``.
    waist : Tensor
        Waist 1/e² half-width, µm, ``[B|1]``.
    center : Tensor
        Height of the sheet's plane, µm, ``[B|1]``.
    focus : Tensor
        Position of the waist along the axis, µm, ``[B|1]``.
    wavelengths : Tensor
        ``[B|1, L]`` vacuum wavelengths, µm.
    axis : {"x", "y"}, default "x"
        Direction of travel in the sample plane.
    """

    registry_name: ClassVar[str | None] = "carrier.gaussian_sheet"
    """Stable name for signatures and saved inputs."""

    irradiance: Tensor = field(quantity="irradiance", role="carrier", dims=("B|1",))
    waist: Tensor = field(quantity="length", role="carrier", dims=("B|1",))
    center: Tensor = field(quantity="length", role="carrier", dims=("B|1",))
    focus: Tensor = field(quantity="length", role="carrier", dims=("B|1",))
    wavelengths: Tensor = field(
        quantity="wavelength", role="carrier", dims=("B|1", "L"), shape_affecting=True
    )
    axis: str = knob(default="x", choices=("x", "y"))

    def at(self, points: Tensor, n: Tensor | float) -> Tensor:
        """Evaluate the sheet's field at points (paraxial, with its Gouy phase).

        Parameters
        ----------
        points : Tensor
            ``[B|1, A|1, N, 3]`` positions, µm.
        n : Tensor or float
            Refractive index of the medium, a scalar or ``[B|1, L]``.

        Returns
        -------
        Tensor
            Complex ``[B, A, 1, L, 1, N]`` (one mode, scalar) in √(photons/µm²).
        """
        device = points.device
        real = torch.float64
        wl = self.wavelengths.to(device=device, dtype=real)  # [B|1, L]
        n_t = torch.as_tensor(n, device=device)
        n_t = (n_t.real if n_t.is_complex() else n_t).to(real)
        if n_t.ndim == 1:
            msg = f"n must be a scalar or [B|1, L], got shape {list(n_t.shape)}"
            raise StructureError(msg, fix="use n[:, None] per image or n[None, :] per wavelength")
        k = 2.0 * math.pi * n_t / wl  # [B|1, L]
        k = k.reshape(k.shape[0], 1, 1, k.shape[1], 1, 1)  # [B|1, 1, 1, L, 1, 1]

        def image(value: Tensor) -> Tensor:
            return value.to(device=device, dtype=real).reshape(-1, 1, 1, 1, 1, 1)

        pts = points.to(real)
        along = pts[..., 0 if self.axis == "x" else 1][:, :, None, None, None, :]
        z = pts[..., 2][:, :, None, None, None, :]  # [B|1, A|1, 1, 1, 1, N]
        w0 = image(self.waist)
        s = along - image(self.focus)
        rayleigh = 0.5 * k * w0 * w0  # π·w₀²·n/λ
        width = w0 * torch.sqrt(1.0 + (s / rayleigh) ** 2)
        dz = z - image(self.center)
        amplitude = torch.sqrt(image(self.irradiance) * w0 / width) * torch.exp(
            -((dz / width) ** 2)
        )
        curvature = s / (s * s + rayleigh * rayleigh)  # 1/R
        phase = k * s + 0.5 * k * dz * dz * curvature - 0.5 * torch.atan2(s, rayleigh)
        field = amplitude * torch.polar(torch.ones_like(phase), phase)
        return field.to(torch.complex64 if points.dtype == torch.float32 else torch.complex128)


@dataclasses.dataclass(frozen=True, kw_only=True, eq=False)
class Field(Carrier):
    """Coherent carrier: an analytic background plus a sampled scattered part (draft until M3).

    Parameters
    ----------
    scattered : Tensor, optional
        Complex ``[B|1, A|1, M, L, P, Y, X]`` (space) or ``[..., Ky, Kx]`` (angular) in
        √(photons/µm²) per mode; None means zero.
    background : PlaneWaves, optional
        The analytic background (zero order and references).
    grid : Grid2D or FreqGrid
        Grid of ``scattered``.
    domain : {"space", "angular"}, default "space"
        Domain of ``scattered``.
    z_ref : Tensor or float, default 0.0
        Reference plane of ``scattered``, µm.
    travel : {1, -1}, default 1
        +1 toward the detection objective, −1 away.
    residual_tilt : Tensor, optional
        ``[B|1, A|1, M, 2]`` δk of a dense march's demodulated frame, cycles/µm.
    n_medium : Tensor or float, default 1.0
        Index of the homogeneous medium at ``z_ref``: a scalar or ``[B|1, L]``.
    wavelengths : Tensor
        ``[B|1, L]`` vacuum wavelengths, µm.
    weights : Tensor
        ``[B|1, L]`` power weights of the wavelength bins.
    mode_weights : Tensor, optional
        ``[B|1, A|1, M]`` power weights of the modes; None means 1.
    basis : {"scalar", "jones", "cartesian"}, default "scalar"
        Polarisation basis: P = 1, 2 or 3.
    """

    registry_name: ClassVar[str | None] = "carrier.field"
    """Stable name for signatures and saved inputs."""

    scattered: Tensor | None = field(
        quantity="dimensionless",
        role="carrier",
        dims=("B|1", "A|1", "M", "L", "P", "*", "*"),
        dtype="complex",
        default=None,
    )
    background: PlaneWaves | None = child(default=None)
    grid: Grid2D | FreqGrid = knob()
    domain: str = knob(default="space", choices=("space", "angular"))
    z_ref: Tensor | float = field(quantity="length", role="shared", default=0.0)
    travel: int = knob(default=1, choices=(1, -1))
    residual_tilt: Tensor | None = field(
        quantity="frequency", role="carrier", dims=("B|1", "A|1", "M", 2), default=None
    )
    n_medium: Tensor | float = field(quantity="index", role="any", dtype="number", default=1.0)
    wavelengths: Tensor = field(quantity="wavelength", role="carrier", dims=("B|1", "L"))
    weights: Tensor = field(quantity="dimensionless", role="carrier", dims=("B|1", "L"))
    mode_weights: Tensor | None = field(
        quantity="dimensionless", role="carrier", dims=("B|1", "A|1", "M"), default=None
    )
    basis: str = knob(default="scalar", choices=("scalar", "jones", "cartesian"))


@dataclasses.dataclass(frozen=True, kw_only=True, eq=False)
class ObjectSpectra(Carrier):
    """Sparse coherent producers, evaluated lazily on pupil support by the consumer (draft).

    ``evaluate`` arrives with the first sparse producer (M1 smoke thread, M3 Mie).

    Parameters
    ----------
    evaluator : str
        Registry name of the evaluator, such as ``"mie"`` or ``"dipole"``.
    params : Node
        The lowered view the evaluator reads (radii, indices, polarisabilities), SoA
        ``[B|1, 1, N, ...]``.
    position : Tensor
        ``[B|1, A|1, N, 3]`` µm.
    presence : Tensor, optional
        ``[B|1, A|1, N]`` in [0, 1]; None means all present.
    travel : {1, -1}, default 1
        +1 toward the detection objective, −1 away.
    """

    registry_name: ClassVar[str | None] = "carrier.object_spectra"
    """Stable name for signatures and saved inputs."""

    evaluator: str = knob()
    params: Node = child()
    position: Tensor = field(
        quantity="length",
        role="carrier",
        dims=("B|1", "A|1", "N", 3),
        event=(3,),
        components=("x", "y", "z"),
        shape_affecting=True,
    )
    presence: Tensor | None = field(
        quantity="dimensionless", role="carrier", dims=("B|1", "A|1", "N"), default=None
    )
    travel: int = knob(default=1, choices=(1, -1))


Contribution = Field | ObjectSpectra | PlaneWaves
"""What an interaction element returns a tuple of (§4.4)."""
