"""A minimal point scatterer (``gx.interact.Dipole``; M1 coherent smoke thread, §4.4).

Each small sphere scatters as a scalar point dipole: ``E_s = f·e^{ikR}/R·E_inc(r_p)`` with the
Rayleigh amplitude ``f = k²α/(4π)``, ``α = 4πa³·(m² − 1)/(m² + 2)`` (m the relative index), and
the radiation-reaction correction ``f → f/(1 − i·k·f)``, which makes the optical theorem exact:
``Im f = k·|f|²`` for a lossless scatterer, so extinction equals scattering (§4.1). The output is
an :class:`~gradix.ObjectSpectra` evaluated lazily on the consumer's pupil support: in the
continuous convention its angular spectrum is ``A(k⊥) = i·f·E_inc(r_p)/(2π·k_z)·e^{−i k⊥·r_p⊥}``
toward +z (Weyl's expansion of the spherical wave).
"""

from __future__ import annotations

import dataclasses
import math
from typing import ClassVar

import torch
from torch import Tensor

from gradix import register
from gradix._core.carriers import ObjectSpectra, PlaneWaves
from gradix._core.contract import Capabilities, Element, Slot, Static
from gradix._core.errors import StructureError
from gradix.objects.environment import Medium
from gradix.objects.objectset import Spheres
from gradix.schema.base import DataObject
from gradix.schema.fields import child, field
from gradix.schema.layout import canonical

__all__ = ["Dipole", "DipoleParams", "scattering_amplitude"]


@dataclasses.dataclass(frozen=True, kw_only=True, eq=False)
class DipoleParams(DataObject):
    """The lowered view a dipole evaluator reads: one scalar scattering amplitude per object.

    Parameters
    ----------
    amplitude : Tensor
        Complex scattering amplitude f (µm), ``[B|1, A|1, N, L]``.
    """

    registry_name: ClassVar[str | None] = "params.dipole"
    """Stable name for signatures and saved inputs."""

    amplitude: Tensor = field(
        quantity="length",
        role="carrier",
        dims=("B|1", "A|1", "N", "L"),
        dtype="complex",
        doc="scattering amplitude",
    )


def scattering_amplitude(radius: Tensor, m: Tensor, k: Tensor) -> Tensor:
    """Return the radiation-corrected scalar Rayleigh amplitude ``f/(1 − i·k·f)``, µm.

    Parameters
    ----------
    radius : Tensor
        Sphere radii, µm.
    m : Tensor
        Relative (possibly complex) refractive indices.
    k : Tensor
        Wavenumbers in the medium, rad/µm, broadcastable to the others.

    Returns
    -------
    Tensor
        Complex amplitudes.
    """
    m = m.to(torch.complex128 if radius.dtype == torch.float64 else torch.complex64)
    alpha = 4.0 * math.pi * radius**3 * (m * m - 1.0) / (m * m + 2.0)
    f = k * k * alpha / (4.0 * math.pi)
    return f / (1.0 - 1j * k * f)


@register.element("interact.dipole")
@dataclasses.dataclass(frozen=True, eq=False)
class Dipole(Element[tuple]):
    """Scalar point-dipole scattering of small spheres (the minimal coherent producer).

    Valid for size parameters ``k·a ≪ 1``; the Mie element (M3) covers larger spheres.

    Parameters
    ----------
    objects : Spheres
        The scatterers (positions, radii, refractive indices in ``material``).

    Examples
    --------
    >>> import torch, gradix as gx
    >>> beads = gx.Spheres(position=torch.zeros(1, 1, 3), radius=0.02, material=1.59)
    >>> Dipole(beads).objects.material
    1.59
    """

    slot: ClassVar[Slot] = Slot.INTERACT
    caps: ClassVar[Capabilities] = Capabilities(
        accepts=frozenset({PlaneWaves}), produces=ObjectSpectra, linear_in_field=True
    )
    population_field: ClassVar[str] = "objects"

    objects: Spheres = child(doc="the scatterers")

    def forward(self, *inputs: object, static: Static) -> tuple[ObjectSpectra]:
        """Produce the scatterers' spectra for the incident modes.

        Parameters
        ----------
        *inputs : object
            The incident :class:`~gradix.PlaneWaves`, then the medium.
        static : Static
            The (empty) configuration.

        Returns
        -------
        tuple of ObjectSpectra
            One contribution with evaluator ``"dipole"``.
        """
        if len(inputs) != 2 or not isinstance(inputs[0], PlaneWaves):
            raise StructureError("Dipole takes (PlaneWaves, medium)")
        waves, medium = inputs[0], inputs[1]
        if not isinstance(medium, Medium):
            raise StructureError("Dipole needs a medium", fix="pass gx.env.Homogeneous(n)")
        if self.objects.material is None:
            raise StructureError("Dipole needs the spheres' material (refractive index)")
        spec = self.objects.schema()
        pos = canonical(self.objects.position, spec["position"])
        dtype, device = pos.dtype, pos.device
        radius = canonical(self.objects.radius, spec["radius"], dtype=dtype, device=device)
        material = canonical(self.objects.material, spec["material"], device=device)
        n = medium.index(dtype=dtype, device=device)  # [B|1, 1]
        n_real = (n.real if n.is_complex() else n).reshape(-1, 1)
        wl = waves.wavelengths.to(device=device, dtype=dtype)  # [B|1, L]
        k = (2.0 * math.pi * n_real / wl)[:, None, None, :]  # [B|1, 1, 1, L]
        relative = material / n.reshape(-1, 1, 1)  # [B|1, T|1, N]: image-aligned
        m = relative[..., None]
        amplitude = scattering_amplitude(radius[..., None], m, k)  # [B|1, T|1, N|1, L]
        presence = None
        if self.objects.presence is not None:
            presence = canonical(
                self.objects.presence, spec["presence"], dtype=dtype, device=device
            )
        # per-object fields broadcast over N (a padded population keeps one position for all
        # its slots); the carrier holds every slot explicitly
        count = max(t.shape[2] for t in (pos, amplitude, presence) if t is not None)
        pos = pos.expand(*pos.shape[:2], count, 3)
        amplitude = amplitude.expand(*amplitude.shape[:2], count, amplitude.shape[3])
        if presence is not None:
            presence = presence.expand(*presence.shape[:2], count)
        spectra = ObjectSpectra(
            evaluator="dipole",
            params=DipoleParams(amplitude=amplitude),
            position=pos,
            presence=presence,
        )
        return (spectra,)

    def __call__(
        self, *inputs: object, grid: object = None, static: Static | None = None
    ) -> tuple[ObjectSpectra]:
        """Produce the spectra eagerly (no configuration is needed).

        Parameters
        ----------
        *inputs : object
            The plane waves and the medium.
        grid : object, optional
            Unused: the spectra are evaluated on the consumer's pupil support.
        static : Static, optional
            Unused.

        Returns
        -------
        tuple of ObjectSpectra
            The contributions.
        """
        return self.forward(*inputs, static=static or Static())
