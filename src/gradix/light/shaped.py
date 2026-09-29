"""Coherent shaped illumination from a pupil (``source.shaped``; §4.5 construction 3, §5.11)."""

from __future__ import annotations

import dataclasses
from typing import ClassVar

import torch
from torch import Tensor

from gradix import register
from gradix._core.carriers import PlaneWaves
from gradix._core.contract import Capabilities, Description, Element, Slot, Static, Violation
from gradix._core.envelope import Envelope
from gradix._core.errors import StructureError
from gradix._core.precision import result_dtype
from gradix.devices.slm import PhaseSLM
from gradix.objects.environment import LayeredMedium, Medium
from gradix.ops.pupil import soft_aperture
from gradix.schema.base import iter_leaves
from gradix.schema.fields import child, field, knob
from gradix.schema.layout import canonical, common_device

__all__ = ["Shaped"]


@register.element("source.shaped")
@dataclasses.dataclass(frozen=True, eq=False)
class Shaped(Element[PlaneWaves]):
    """Coherent illumination shaped in a pupil: one plane wave per pupil sample (J of them).

    The illumination pupil (the condenser's back focal plane, or the objective's for epi light)
    holds a complex field A(u) on a Q×Q grid of cells spanning ``center + [−NA, NA]²`` in NA
    units: a map (``pupil_field``), a device's transmission (a :class:`~gradix.devices.PhaseSLM`)
    or, with neither, a uniform pupil of ``samples`` cells. Each cell becomes a plane wave of
    direction u_j and amplitude ``√(I/J_eff)·w_j·A(u_j)/√cosθ_j``, all mutually coherent (one
    mode, J waves): w_j is a soft circular aperture of radius NA, J_eff = Σ w_j², and the
    aplanatic factor 1/√cosθ, in the medium the light is defined in, gives every cell of a
    uniform pupil the same flux through the focal plane. A phase-only pupil therefore keeps
    the irradiance I, and a passive amplitude mask (|A| ≤ 1) lowers it. Scalar light at
    moderate NA; vectorial focusing is v1.x (cap-25).

    Consumers see J plane waves: the coherent imaging element sums every wave's background and
    Mie scattering, the latter over J in chunks.

    Parameters
    ----------
    wavelength : Tensor or float
        Vacuum wavelength in µm, or a per-image ``[B]`` tensor.
    na : Tensor or float
        The pupil's NA (the condenser's, for transmitted light), per image ``[B]``.
    pupil_field : Tensor, optional
        Complex pupil field ``[Q, Q]`` or per image ``[B, Q, Q]`` (rows along u_y).
    device : PhaseSLM, optional
        A device whose transmission is the pupil field.
    samples : int, default 16
        Cells per axis of a uniform pupil (neither ``pupil_field`` nor ``device``).
    irradiance : Tensor or float, default 1.0
        Irradiance of the unshaped (uniform-pupil) illumination through the focal plane, in
        photons/µm² per exposure, per image ``[B]``.
    center : Tensor, optional
        Direction of the pupil's centre ``[2]`` or per image ``[B, 2]``, NA units; None is the
        optical axis.
    travel : {1, -1}, default 1
        +1 toward the objective (transmitted light, defined in the sample medium); −1 away
        from it (epi light, defined in the coverslip of a layered medium).

    Examples
    --------
    >>> import torch, gradix as gx
    >>> slm = gx.devices.PhaseSLM(phase=torch.zeros(8, 8))
    >>> waves = Shaped(0.532, na=0.4, device=slm, irradiance=1e3)(gx.env.Homogeneous(1.33))
    >>> tuple(waves.amplitude.shape)  # [B, A, M, J, L, P]: one mode of 64 waves
    (1, 1, 1, 64, 1, 1)
    """

    slot: ClassVar[Slot] = Slot.SOURCE
    caps: ClassVar[Capabilities] = Capabilities(accepts=frozenset(), produces=PlaneWaves)
    reads_medium: ClassVar[bool] = True
    """The executor passes the Chain's medium: the aplanatic factor needs its index."""

    wavelength: Tensor | float = field(
        quantity="wavelength",
        role="image",
        shape_affecting=True,
        constraint="positive",
        doc="vacuum wavelength",
    )
    _: dataclasses.KW_ONLY
    na: Tensor | float = field(
        quantity="dimensionless", role="image", constraint="positive", doc="pupil NA"
    )
    pupil_field: Tensor | None = field(
        quantity="dimensionless",
        role="image",
        event=(-1, -1),
        dtype="number",
        default=None,
        doc="complex pupil field",
    )
    device: PhaseSLM | None = child(default=None, doc="device whose transmission is the pupil")
    samples: int = knob(default=16, doc="cells per axis of a uniform pupil")
    irradiance: Tensor | float = field(
        quantity="irradiance",
        role="image",
        constraint="nonnegative",
        default=1.0,
        doc="photons per µm² per exposure of the unshaped illumination",
    )
    center: Tensor | None = field(
        quantity="dimensionless",
        role="image",
        event=(2,),
        components=("x", "y"),
        default=None,
        doc="direction of the pupil's centre",
    )
    travel: int = knob(default=1, choices=(1, -1), doc="+1 toward the objective, -1 away")

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.pupil_field is not None and self.device is not None:
            raise StructureError("give pupil_field= or device=, not both")
        rows, cols = self.grid_shape
        if rows != cols:
            raise StructureError(f"the pupil grid must be square, got {rows}×{cols}")

    @property
    def grid_shape(self) -> tuple[int, int]:
        """The pupil grid ``(Q, Q)``: the device's, the map's, or ``samples``.

        Returns
        -------
        tuple of int
            Rows and columns.
        """
        if self.device is not None:
            return self.device.grid
        if self.pupil_field is not None:
            rows, cols = torch.as_tensor(self.pupil_field).shape[-2:]
            return int(rows), int(cols)
        return self.samples, self.samples

    @property
    def waves(self) -> int:
        """J, the number of plane waves: one per pupil cell.

        Returns
        -------
        int
            Q².
        """
        rows, cols = self.grid_shape
        return rows * cols

    def validity(self, desc: Description, envelope: Envelope) -> list[Violation]:
        """Require the pupil inside the medium's index: its waves must propagate.

        Parameters
        ----------
        desc : Description
            Static description.
        envelope : Envelope
            The envelope.

        Returns
        -------
        list of Violation
            An error when NA plus the centre's offset reaches the index of the medium the
            light is defined in.
        """
        medium = desc.nodes.get(desc.part("environment")) or desc.nodes.get("environment")
        if not isinstance(medium, Medium):
            return []
        n = float(self._index(medium, torch.float64, None, None).detach().min())
        reach = float(torch.as_tensor(self.na).detach().max())
        if self.center is not None:
            reach += float(torch.as_tensor(self.center).detach().reshape(-1, 2).norm(dim=-1).max())
        if reach < n:
            return []
        return [
            Violation(
                "error",
                f"the shaped pupil reaches |u| = {reach:.3g}, beyond the medium's index "
                f"{n:.3g}: its outer waves would not propagate",
                entry="light.na",
                value=reach,
                limit=n,
                element=desc.path or "light",
                fix="lower the pupil's NA (or its centre's offset) below the medium's index",
            )
        ]

    def _index(
        self,
        medium: Medium,
        dtype: torch.dtype,
        device: torch.device | None,
        wavelength: Tensor | None,
    ) -> Tensor:
        """Return the index the light is defined in: the sample's, or the coverslip's for epi."""
        if isinstance(medium, LayeredMedium) and self.travel == -1:
            n = medium.coverslip_index(dtype=dtype, device=device, wavelength=wavelength)
        else:
            n = medium.index(dtype=dtype, device=device, wavelength=wavelength)
        n = n.real if n.is_complex() else n
        return n.reshape(-1)

    def _pupil(self, wl: Tensor, dtype: torch.dtype, device: torch.device) -> Tensor:
        """Return the pupil field, complex ``[B|1, L, Q, Q]``."""
        cdtype = torch.complex128 if dtype == torch.float64 else torch.complex64
        if self.device is not None:
            return self.device.transmission(wl).to(cdtype)
        if self.pupil_field is not None:
            spec = self.schema()["pupil_field"]
            value = canonical(self.pupil_field, spec, device=device)  # keeps complex values
            return value.to(cdtype)[:, None]
        q = self.samples
        return torch.ones(1, 1, q, q, dtype=cdtype, device=device)

    def forward(self, *inputs: object, static: Static) -> PlaneWaves:
        """Build the plane waves of the pupil's cells.

        Parameters
        ----------
        *inputs : object
            The medium the light is defined in (the executor passes the Chain's).
        static : Static
            The (empty) configuration.

        Returns
        -------
        PlaneWaves
            ``amplitude [B|1, 1, 1, J, 1, 1]``, ``u [B|1, 1, 1, J, 2]`` and ``wavelengths
            [B|1, 1]``, J = Q².

        Raises
        ------
        StructureError
            If no medium is given.
        """
        medium = inputs[0] if inputs else None
        if not isinstance(medium, Medium):
            raise StructureError(
                "Shaped needs the medium it focuses into (the aplanatic factor)",
                fix="call it with the medium: Shaped(...)(gx.env.Homogeneous(n))",
            )
        leaves = [v for _, _, v in iter_leaves(self)]
        dtype = result_dtype(*leaves)
        device = common_device(*leaves)
        spec = self.schema()
        wl = canonical(self.wavelength, spec["wavelength"], dtype=dtype, device=device)[:, None]
        irr = canonical(self.irradiance, spec["irradiance"], dtype=dtype, device=device)
        na = canonical(self.na, spec["na"], dtype=dtype, device=device).reshape(-1, 1, 1)
        pupil = self._pupil(wl, dtype, device)  # [B|1, L, Q, Q]
        q = pupil.shape[-1]
        cells = (torch.arange(q, dtype=dtype, device=device) + 0.5) * (2.0 / q) - 1.0
        ty, tx = torch.meshgrid(cells, cells, indexing="ij")  # [Q, Q] in units of NA
        radius = torch.sqrt(tx * tx + ty * ty) * na  # [B|1, Q, Q]
        du = 2.0 * float(na.detach().max()) / q
        weight = soft_aperture(radius, na, du)  # amplitude, [B|1, Q, Q]
        ux, uy = tx * na, ty * na
        if self.center is not None:
            c = canonical(self.center, spec["center"], dtype=dtype, device=device)  # [B|1, 2]
            ux, uy = ux + c[:, 0, None, None], uy + c[:, 1, None, None]
        n = self._index(medium, dtype, device, wl).reshape(-1, 1, 1)
        cos = torch.sqrt(torch.clamp(1.0 - (ux * ux + uy * uy) / (n * n), min=1e-6))
        flux = (weight * weight).sum((-2, -1))  # J_eff: [B|1]
        scale = torch.sqrt(irr / flux.clamp_min(torch.finfo(dtype).tiny))  # [B|1]
        amp = scale.reshape(-1, 1, 1, 1) * (weight / torch.sqrt(cos))[:, None] * pupil
        b, bins = amp.shape[0], amp.shape[1]
        amplitude = amp.reshape(b, bins, q * q).transpose(1, 2)  # [B|1, J, L]
        amplitude = amplitude[:, None, None, :, :, None]  # [B|1, 1, 1, J, L, 1]
        u = torch.stack([ux, uy], -1).expand(max(ux.shape[0], 1), q, q, 2)
        u = u.reshape(u.shape[0], 1, 1, q * q, 2)
        return PlaneWaves(
            amplitude=amplitude,
            u=u,
            wavelengths=wl.to(torch.promote_types(wl.dtype, dtype)),
            z0=0.0,
            travel=self.travel,
        )

    def __call__(
        self, *inputs: object, grid: object = None, static: Static | None = None
    ) -> PlaneWaves:
        """Build the plane waves eagerly, in a medium.

        Parameters
        ----------
        *inputs : object
            The medium.
        grid : object, optional
            Unused; the waves are analytic.
        static : Static, optional
            Unused.

        Returns
        -------
        PlaneWaves
            The waves.
        """
        return self.forward(*inputs, static=static or Static())
