"""Lorenz–Mie spheres and Mie-dressed dipoles (``gx.interact.Mie``, ``gx.interact.Dipole``; §5.12).

Both elements lower spheres to :class:`~gradix.ObjectSpectra` with the ``"mie"`` evaluator: the
coefficients a_n and b_n per sphere and wavelength, from which a consumer evaluates the far-field
amplitudes S₁ and S₂ on its own pupil samples (§4.1: A(k⊥) = −S/(2πk_m k_z)·e^{−i(k − k_in)·r_p}).
The Mie-dressed dipole is the same spectrum truncated at n = 1: its electric and magnetic
polarisabilities α_e = i6πa₁/k³ and α_m = i6πb₁/k³ already contain radiation reaction, so the
optical theorem holds exactly at every order.
"""

from __future__ import annotations

import dataclasses
import math
from typing import ClassVar

import torch
from torch import Tensor

from gradix import register
from gradix._core.carriers import ObjectSpectra, PlaneWaves
from gradix._core.contract import Capabilities, Description, Element, Slot, Static
from gradix._core.envelope import Envelope
from gradix._core.errors import StructureError
from gradix._core.rules import Decision
from gradix.objects.environment import Medium
from gradix.objects.objectset import Spheres
from gradix.schema.base import DataObject
from gradix.schema.fields import child, field, knob
from gradix.schema.layout import canonical
from gradix.special import mie

__all__ = ["Dipole", "Mie", "MieParams", "MieStatic"]

_M_DEFAULT = 2.5
"""Largest relative index assumed when the envelope has none (sets where D_n(mx) starts)."""


@dataclasses.dataclass(frozen=True, kw_only=True, eq=False)
class MieParams(DataObject):
    """The lowered view the ``"mie"`` evaluator reads: Mie coefficients per sphere and bin.

    Parameters
    ----------
    a : Tensor
        Electric coefficients a_n, complex ``[B|1, A|1, N, L, n]``.
    b : Tensor
        Magnetic coefficients b_n, complex ``[B|1, A|1, N, L, n]``.
    """

    registry_name: ClassVar[str | None] = "params.mie"
    """Stable name for signatures and saved inputs."""

    a: Tensor = field(
        quantity="dimensionless",
        role="carrier",
        dims=("B|1", "A|1", "N", "L", "*"),
        dtype="complex",
        doc="electric Mie coefficients",
    )
    b: Tensor = field(
        quantity="dimensionless",
        role="carrier",
        dims=("B|1", "A|1", "N", "L", "*"),
        dtype="complex",
        doc="magnetic Mie coefficients",
    )


@dataclasses.dataclass(frozen=True)
class MieStatic(Static):
    """Static configuration of :class:`Mie`.

    Parameters
    ----------
    decisions : tuple of Decision, default ()
        Sizing decisions.
    terms : int, default 1
        Number of Mie orders.
    start : int, default 17
        Order at which the downward recurrence of D_n(mx) starts.
    """

    terms: int = 1
    start: int = 17


def _range(envelope: Envelope, *paths: str) -> tuple[float, float] | None:
    for path in paths:
        interval = envelope.range(path)
        if interval is not None:
            return interval
    return None


def _extreme(value: object, pick: str) -> float:
    """Return the largest or smallest real magnitude of a number or tensor (a host read)."""
    t = torch.as_tensor(value).detach()
    t = t.abs() if t.is_complex() else t
    t = t.reshape(-1).to(torch.float64)
    if not t.numel():  # an empty population needs no orders
        return 0.0
    return float(t.max() if pick == "hi" else t.min())


class _MieSpheres(Element[tuple]):
    """Shared implementation of :class:`Mie` and :class:`Dipole`."""

    slot: ClassVar[Slot] = Slot.INTERACT
    caps: ClassVar[Capabilities] = Capabilities(
        accepts=frozenset({PlaneWaves}),
        produces=ObjectSpectra,
        ports=frozenset({"+z", "-z"}),
        linear_in_field=True,
        grad_quality={"objects.rotation": "zero", "objects.id": "zero"},
    )
    population_field: ClassVar[str] = "objects"
    objects: Spheres

    def _terms(self, x_max: float) -> int:
        raise NotImplementedError

    def configure(self, desc: Description, envelope: Envelope) -> MieStatic:
        """Choose the number of orders from the envelope of radius, wavelength and index.

        Parameters
        ----------
        desc : Description
            Static description; its population names the envelope entries.
        envelope : Envelope
            The envelope.

        Returns
        -------
        MieStatic
            The configuration.
        """
        pop = desc.populations[0] if desc.populations else "objects"
        radius = _range(envelope, f"{pop}.radius")
        wl = _range(envelope, "light.wavelength", "light.wavelengths")
        n = _range(envelope, "environment.n")
        material = _range(envelope, f"{pop}.material")
        r_max = radius[1] if radius is not None else _extreme(self.objects.radius, "hi")
        wl_min = wl[0] if wl is not None else 0.4
        n_max = n[1] if n is not None else self._medium_index(desc)
        m_max = (material[1] / (n[0] if n is not None else 1.0)) if material else _M_DEFAULT
        x_max = 2.0 * math.pi * n_max * r_max / wl_min
        return self._static(x_max, m_max, "envelope")

    def _medium_index(self, desc: Description) -> float:
        medium = desc.nodes.get("environment")
        if isinstance(medium, Medium):
            return _extreme(medium.index(), "hi")
        return 1.5

    def _static(self, x_max: float, m_max: float, source: str) -> MieStatic:
        terms = self._terms(x_max)
        start = terms + math.ceil(max(m_max, 1.0) * x_max) + 16
        decision = Decision(
            "mie.terms",
            terms,
            "x_max + 4.05·x_max^⅓ + 2 (Wiscombe)" if terms > 1 else "dipole: n = 1",
            {"x_max": x_max, "source": source},
        )
        return MieStatic(decisions=(decision,), terms=terms, start=start)

    def eager_static(self, *inputs: object, envelope: Envelope | None = None) -> Static:
        """Configure for an eager call from the spheres' own values and the waves' wavelengths.

        Parameters
        ----------
        *inputs : object
            The incident :class:`~gradix.PlaneWaves`, then the medium.
        envelope : Envelope, optional
            Unused: the configuration reads the values directly.

        Returns
        -------
        Static
            A :class:`MieStatic`.
        """
        waves = inputs[0] if inputs and isinstance(inputs[0], PlaneWaves) else None
        medium = inputs[1] if len(inputs) > 1 and isinstance(inputs[1], Medium) else None
        wl_min = _extreme(waves.wavelengths, "lo") if waves is not None else 0.4
        n_max = _extreme(medium.index(), "hi") if medium is not None else 1.5
        n_min = _extreme(medium.index(), "lo") if medium is not None else 1.0
        r_max = _extreme(self.objects.radius, "hi")
        material = self.objects.material
        m_max = _extreme(material, "hi") / n_min if material is not None else _M_DEFAULT
        return self._static(2.0 * math.pi * n_max * r_max / wl_min, m_max, "values")

    def forward(self, *inputs: object, static: Static) -> tuple[ObjectSpectra]:
        """Lower the spheres to their Mie spectra for the incident light.

        Parameters
        ----------
        *inputs : object
            The incident :class:`~gradix.PlaneWaves`, then the medium.
        static : Static
            A :class:`MieStatic`.

        Returns
        -------
        tuple of ObjectSpectra
            One contribution with evaluator ``"mie"``: coefficients ``[B|1, A|1, N, L, n]``.
        """
        if len(inputs) != 2 or not isinstance(inputs[0], PlaneWaves):
            raise StructureError(f"{type(self).__name__} takes (PlaneWaves, medium)")
        waves, medium = inputs[0], inputs[1]
        if not isinstance(medium, Medium):
            raise StructureError("Mie spheres need a medium", fix="pass gx.env.Homogeneous(n)")
        if self.objects.material is None:
            raise StructureError("Mie spheres need their material (refractive index)")
        if not isinstance(static, MieStatic):
            static = self.eager_static(waves, medium)
            assert isinstance(static, MieStatic)
        spec = self.objects.schema()
        pos = canonical(self.objects.position, spec["position"])
        dtype, device = pos.dtype, pos.device
        radius = canonical(self.objects.radius, spec["radius"], dtype=dtype, device=device)
        material = canonical(self.objects.material, spec["material"], device=device)
        wl = waves.wavelengths.to(device=device, dtype=torch.float64)  # [B|1, L]
        index = medium.index(dtype=torch.float64, device=device, wavelength=wl)  # [B|1, L|1]
        n_real = index.real if index.is_complex() else index
        k = (2.0 * math.pi * n_real / wl)[:, None, None, :]  # [B|1, 1, 1, L]
        x = k * radius.to(torch.float64)[..., None]  # [B|1, T|1, N, L]
        relative = (
            material.to(torch.complex128)[..., None] / index.to(torch.complex128)[:, None, None, :]
        )  # [B|1, T|1, N, L]
        a, b = mie.coefficients(x, relative, static.terms, start=static.start)
        presence = None
        if self.objects.presence is not None:
            presence = canonical(
                self.objects.presence, spec["presence"], dtype=dtype, device=device
            )
        # every slot explicit: per-object fields broadcast over N in a padded population
        count = max(t.shape[2] for t in (pos, a, presence) if t is not None)
        pos = pos.expand(*pos.shape[:2], count, 3)
        a = a.expand(*a.shape[:2], count, *a.shape[3:])
        b = b.expand(*b.shape[:2], count, *b.shape[3:])
        if presence is not None:
            presence = presence.expand(*presence.shape[:2], count)
        spectra = ObjectSpectra(
            evaluator="mie", params=MieParams(a=a, b=b), position=pos, presence=presence
        )
        return (spectra,)

    def __call__(
        self, *inputs: object, grid: object = None, static: Static | None = None
    ) -> tuple[ObjectSpectra]:
        """Lower the spheres eagerly: the orders come from their own values.

        Parameters
        ----------
        *inputs : object
            The plane waves and the medium.
        grid : object, optional
            Unused: the spectra are evaluated on the consumer's pupil samples.
        static : Static, optional
            A configuration; derived from the inputs when None.

        Returns
        -------
        tuple of ObjectSpectra
            The contributions.
        """
        return self.forward(*inputs, static=static or self.eager_static(*inputs))


@register.element("interact.mie")
@dataclasses.dataclass(frozen=True, eq=False)
class Mie(_MieSpheres):
    """Lorenz–Mie scattering of homogeneous spheres (fp64 coefficients, any size and index).

    Parameters
    ----------
    objects : Spheres
        The scatterers: positions, radii and refractive indices (``material``, complex for
        absorbing spheres) in the host medium.
    terms : int or "auto", default "auto"
        Number of Mie orders; ``"auto"`` applies Wiscombe's rule to the largest size parameter
        of the envelope (or of the values, in an eager call).

    Examples
    --------
    >>> import torch, gradix as gx
    >>> beads = gx.Spheres(position=torch.zeros(1, 1, 3), radius=0.1, material=1.59)
    >>> waves = gx.light.PlaneWave(0.532)()
    >>> spectra = Mie(beads)(waves, gx.env.Homogeneous(1.33))[0]
    >>> spectra.evaluator, tuple(spectra.params.a.shape)  # x = 1.57: 9 orders
    ('mie', (1, 1, 1, 1, 9))
    """

    objects: Spheres = child(doc="the scatterers")
    _: dataclasses.KW_ONLY
    terms: int | str = knob(default="auto", doc="Mie orders")

    def _terms(self, x_max: float) -> int:
        return mie.terms(x_max) if self.terms == "auto" else int(self.terms)


@register.element("interact.dipole")
@dataclasses.dataclass(frozen=True, eq=False)
class Dipole(_MieSpheres):
    """Mie-dressed electric and magnetic dipoles: Mie truncated at n = 1 (same cost as Rayleigh).

    Within 5 % of Mie over all angles for dielectrics up to x ≈ 0.75 (§5.2); a₁ and b₁ carry the
    radiation reaction, so extinction equals scattering plus absorption exactly.

    Parameters
    ----------
    objects : Spheres
        The scatterers (positions, radii, refractive indices in ``material``).

    Examples
    --------
    >>> import torch, gradix as gx
    >>> beads = gx.Spheres(position=torch.zeros(1, 1, 3), radius=0.02, material=1.59)
    >>> waves = gx.light.PlaneWave(0.532)()
    >>> tuple(Dipole(beads)(waves, gx.env.Homogeneous(1.33))[0].params.a.shape)
    (1, 1, 1, 1, 1)
    """

    schema_version: ClassVar[int] = 2
    """2: Mie-dressed (a₁, b₁) instead of the scalar Rayleigh amplitude (M3a)."""

    objects: Spheres = child(doc="the scatterers")

    def _terms(self, x_max: float) -> int:
        return 1
