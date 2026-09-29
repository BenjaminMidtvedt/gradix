"""Quadriwave lateral shearing interferometry: harmonics, phase gradients, integration (§5.12).

The camera behind a detection grating (``gx.optics.Grating``) records the interference of the
grating's orders, each a copy of the image field sheared by λd·G_o. Two orders o and p
interfere at the harmonic G_o − G_p, and their product E(r − a_o)·E*(r − a_p) carries the
phase difference over the shear a_o − a_p. :class:`QWLSI` demodulates the harmonics and the
constant term (the linear part), takes the harmonics' phases and the constant's logarithm
(pointwise), and inverts the shears by least squares on the frame's FFT grid (a Poisson solver
that knows the finite shears).
"""

from __future__ import annotations

import dataclasses
import math
import warnings

import torch
from torch import Tensor

from gradix._core.errors import GradixWarning, StructureError
from gradix.compose.chain import Chain
from gradix.optics.grating import Grating, order_indices
from gradix.recon.base import Reconstruction, optics_of, without_scatterers

__all__ = ["QWLSI"]


def _scalar(value: object, name: str) -> float:
    t = torch.as_tensor(value).detach().reshape(-1)
    if t.numel() != 1 and not bool((t == t[0]).all()):
        raise StructureError(f"the reconstruction needs one grating {name} for all images")
    return float(t[0])


@dataclasses.dataclass(frozen=True)
class QWLSI(Reconstruction):
    """Recover the complex image field from quadriwave lateral shearing interferograms.

    The linear part demodulates the constant term and the harmonics named by ``harmonics``
    (order-index differences, (2, 0) and (0, 2) by default: the fringes along the grating's
    axes) inside soft circular windows, dividing out the pixel MTF at their original
    frequencies. The pointwise step takes each harmonic's phase relative to the empty field's
    (``background`` frames, or the grating's own coefficients) and half the logarithm of the
    constant term's ratio. Each harmonic's phase is, to first order in the object, the phase
    difference over its pairs' shears, a filter T_j(k) known from the orders; the phase is
    Σ_j T_j*·ψ̂_j / (Σ_j |T_j|² + ε), and the log-amplitude is the constant term deconvolved
    by the shears' average the same way. The field returned is exp(α + iφ): the image field
    over the empty one, with the frame-mean phase set to zero.

    Parameters
    ----------
    pitch : float
        Object-space pixel pitch, µm.
    wavelength : float
        Vacuum wavelength, µm.
    na : float
        Numerical aperture of the objective.
    magnification : float
        Lateral magnification.
    period : float
        The grating's fringe period Λ on the camera, µm.
    distance : float
        Grating-to-camera distance d, µm.
    rotation : float, default 0.0
        Rotation of the grating's axes, rad.
    kind : {"hartmann", "checkerboard", "custom"}, default "hartmann"
        The grating's mask (:class:`~gradix.optics.Grating`).
    max_order : int, default 3
        The checkerboard's largest odd index.
    orders : tuple of (float, float), optional
        A custom mask's order indices.
    coefficients : tuple of complex, optional
        A custom mask's coefficients.
    focused : {"camera", "grating"}, default "camera"
        The plane in focus; the field is recovered at the camera either way.
    harmonics : tuple of (float, float), default ((2, 0), (0, 2))
        Order-index differences of the harmonics to use; adding the diagonals (2, 2) and
        (2, −2) uses more of the light.
    background : Tensor, optional
        Frames of the empty field ``[B|1, C|1, H, W]``, which calibrate the harmonics' phases
        and the constant term; None uses the grating's coefficients and the frame median.
    radius : float, optional
        The harmonics' window radius in NA units; None is NA plus two frequency cells, at
        most half the fringe frequency (so the windows never overlap). The constant term's
        window reaches 2NA (its |E|² band) or the harmonics' windows, whichever is nearer.
    mtf : bool, default True
        Undo the pixel MTF.
    regularization : float, default 1e-6
        The Tikhonov term ε of the integration, relative to the largest Σ|T_j|².

    Examples
    --------
    >>> import torch
    >>> recon = QWLSI(pitch=0.065, wavelength=0.532, na=0.8, magnification=100.0,
    ...               period=26.0, distance=500.0)
    >>> frames = torch.ones(1, 1, 32, 32)  # no fringes: the grating's own calibration
    >>> recon.linear(frames).shape
    torch.Size([1, 1, 3, 32, 32])
    """

    pitch: float
    wavelength: float
    na: float
    magnification: float
    period: float
    distance: float
    rotation: float = 0.0
    kind: str = "hartmann"
    max_order: int = 3
    orders: tuple[tuple[float, float], ...] | None = None
    coefficients: tuple[complex, ...] | None = None
    focused: str = "camera"
    harmonics: tuple[tuple[float, float], ...] = ((2.0, 0.0), (0.0, 2.0))
    background: Tensor | None = None
    radius: float | None = None
    mtf: bool = True
    regularization: float = 1e-6

    @classmethod
    def from_chain(
        cls,
        chain: Chain,
        grating: str = "grating",
        *,
        normalize: bool = True,
        harmonics: tuple[tuple[float, float], ...] = ((2.0, 0.0), (0.0, 2.0)),
        radius: float | None = None,
        mtf: bool = True,
    ) -> QWLSI:
        """Read the optics and the grating from a coherent Chain.

        Parameters
        ----------
        chain : Chain
            A coherent Chain with a :class:`~gradix.optics.Grating` in ``detection_optics``.
        grating : str, default "grating"
            The grating's name in ``chain.detection_optics``.
        normalize : bool, default True
            Calibrate on the Chain rendered without its scatterers (``background``).
        harmonics : tuple of (float, float), default ((2, 0), (0, 2))
            The harmonics to use.
        radius : float, optional
            Window radius in NA units.
        mtf : bool, default True
            Undo the pixel MTF.

        Returns
        -------
        QWLSI
            The reconstruction.

        Raises
        ------
        StructureError
            If the Chain has no such grating, or its values differ between images.
        """
        o = optics_of(chain)
        stage = chain.detection_optics.get(grating)
        if not isinstance(stage, Grating):
            known = sorted(chain.detection_optics)
            raise StructureError(f"no grating {grating!r}", fix=f"detection_optics: {known}")
        coefficients = None
        if stage.coefficients is not None:
            values = torch.as_tensor(stage.coefficients).detach().reshape(-1)
            coefficients = tuple(complex(v) for v in values.tolist())
        background = None
        if normalize:
            empty = without_scatterers(chain)
            background = empty(outputs=("expected",))["expected"].detach()
        return cls(
            pitch=o.pitch,
            wavelength=o.wavelength,
            na=o.na,
            magnification=o.magnification,
            period=_scalar(stage.period, "period"),
            distance=_scalar(stage.distance, "distance"),
            rotation=_scalar(stage.rotation, "rotation"),
            kind=stage.kind,
            max_order=stage.max_order,
            orders=stage.orders,
            coefficients=coefficients,
            focused=stage.focused,
            harmonics=harmonics,
            background=background,
            radius=radius,
            mtf=mtf,
        )

    def __post_init__(self) -> None:
        clashes = self.aliased()
        if clashes:
            listed = ", ".join(f"{h} onto {t}" for h, t in clashes)
            warnings.warn(
                f"the camera aliases the grating's harmonics into the windows this "
                f"reconstruction reads ({listed}): choose another period or rotation, or the "
                "Hartmann mask",
                GradixWarning,
                stacklevel=3,
            )

    def aliased(self) -> list[tuple[tuple[float, float], tuple[float, float]]]:
        """Return the harmonics the camera's sampling aliases into the windows read.

        Every pair of the mask's orders interferes at a harmonic; a multi-order mask (the
        checkerboard) has harmonics beyond the fringe frequency, which the pixel grid folds
        back. One whose alias, with its sideband, reaches the window of a harmonic in use (or
        of the constant term) biases the reconstruction.

        Returns
        -------
        list of ((float, float), (float, float))
            (harmonic, window) index differences; the constant term is (0, 0). Harmonics
            weaker than 1 % of the weakest one in use are ignored.
        """
        mn, weighted, _shear, _g = self._orders()
        weights: dict[tuple[float, float], complex] = {}
        for i in range(mn.shape[0]):
            for j in range(mn.shape[0]):
                if i != j:
                    key = (float(mn[i, 0] - mn[j, 0]), float(mn[i, 1] - mn[j, 1]))
                    weights[key] = weights.get(key, 0j) + complex(weighted[i] * weighted[j].conj())
        used = [(float(h[0]), float(h[1])) for h in self.harmonics]
        if any(h not in weights for h in used):
            return []  # reported when the filters are built
        weakest = min(abs(weights[h]) for h in used)
        p, mag = self.pitch, self.magnification

        def centre(h: tuple[float, float]) -> Tensor:
            return mag * self._rotate(torch.tensor(h, dtype=torch.float64))

        fringe = min(float(centre(h).norm()) for h in used)
        na_band = self.na / self.wavelength
        band = min(na_band, 0.5 * fringe) if self.radius is None else self.radius / self.wavelength
        band0 = max(band, min(2.0 * na_band, fringe - band))
        targets = [((0.0, 0.0), torch.zeros(2, dtype=torch.float64), band0)]
        targets += [(h, centre(h), band) for h in used]
        out = []
        for key, weight in sorted(weights.items()):
            if key in used or abs(weight) < 0.01 * weakest:
                continue
            f = centre(key)
            alias = f - torch.round(f * p) / p  # folded into the camera's band
            for target, where, radius in targets:
                gap = float((alias - where).norm())
                if gap < (radius + band) * (1.0 - 1e-6) and not (
                    target == (0.0, 0.0) and bool(torch.allclose(alias, f))
                ):
                    out.append((key, target))
        return out

    # ---- the orders ---------------------------------------------------------------------------

    def _rotate(self, mn: Tensor) -> Tensor:
        """Frequencies of index pairs ``[..., 2]`` on the camera side, cycles/µm."""
        c, s = math.cos(self.rotation), math.sin(self.rotation)
        m, n = mn[..., 0], mn[..., 1]
        return torch.stack([c * m - s * n, s * m + c * n], dim=-1) / (2.0 * self.period)

    def _orders(self) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Return the orders' indices, weights, shears and frequencies.

        The weights are the coefficients with each order's constant propagation phase
        ``[O]``, the shears are in object space ``[O, 2]`` (µm) and the frequencies on the
        camera side ``[O, 2]`` (cycles/µm).
        """
        indices, values = order_indices(self.kind, self.max_order, self.orders)
        if values is None:
            if self.coefficients is None or len(self.coefficients) != len(indices):
                raise StructureError("a custom grating needs one coefficient per order")
            values = self.coefficients
        mn = torch.tensor(indices, dtype=torch.float64)
        g = self._rotate(mn)  # [O, 2] cycles/µm, camera side
        c = torch.tensor(values, dtype=torch.complex128)
        a = 1.0 / self.wavelength**2
        g2 = (g * g).sum(-1)
        root = torch.sqrt(torch.clamp(a - g2, min=0.0))
        constant = 2.0 * math.pi * self.distance * (-g2 / (root + math.sqrt(a)))  # θ_o(0)
        weighted = c * torch.polar(torch.ones_like(constant), constant)
        shear = self.distance * g / root[:, None] / self.magnification  # object space, µm
        return mn, weighted, shear, g

    def _pairs(self, difference: tuple[float, float]) -> list[tuple[int, int]]:
        mn = self._orders()[0]
        target = torch.tensor(difference, dtype=torch.float64)
        return [
            (i, j)
            for i in range(mn.shape[0])
            for j in range(mn.shape[0])
            if bool(torch.allclose(mn[i] - mn[j], target))
        ]

    # ---- the linear part ------------------------------------------------------------------

    def linear(self, frames: Tensor) -> Tensor:
        """Return the constant term and the harmonics, demodulated (linear in the frames).

        Parameters
        ----------
        frames : Tensor
            Interferograms ``[B, C, H, W]`` in photons per pixel.

        Returns
        -------
        Tensor
            Complex ``[B, C, 1 + J, H, W]`` in photons/µm²: the constant term, then the J
            harmonics at baseband, at the pixel centres.
        """
        if frames.ndim != 4:
            raise StructureError(f"QWLSI takes frames [B, C, H, W], got {list(frames.shape)}")
        real = torch.float64 if frames.dtype == torch.float64 else torch.float32
        cdtype = torch.complex128 if real == torch.float64 else torch.complex64
        frames = frames.to(real)
        height, width = frames.shape[-2:]
        device, p = frames.device, self.pitch
        fy = torch.fft.fftfreq(height, d=p, dtype=torch.float64, device=device)[:, None]
        fx = torch.fft.fftfreq(width, d=p, dtype=torch.float64, device=device)[None, :]
        ys = (torch.arange(height, dtype=torch.float64, device=device) + 0.5) * p
        xs = (torch.arange(width, dtype=torch.float64, device=device) + 0.5) * p
        fringe = self.magnification * float(
            self._rotate(torch.tensor(self.harmonics, dtype=torch.float64)).norm(dim=-1).min()
        )  # the closest harmonic, object space
        cell = 1.0 / (p * max(height, width))
        if self.radius is None:
            band = min(self.na / self.wavelength + 2.0 * cell, 0.5 * fringe)
        else:
            band = self.radius / self.wavelength
        # the constant term reaches 2NA/λ (|E|²): its window extends to the harmonics' windows
        band0 = max(band, min(2.0 * self.na / self.wavelength + 2.0 * cell, fringe - band))
        radius = torch.sqrt(fx * fx + fy * fy)

        def soft(limit: float) -> Tensor:
            edge = torch.clamp((limit - radius) / cell + 0.5, 0.0, 1.0)
            return edge * edge * (3.0 - 2.0 * edge)

        def demodulate(data: Tensor, fc: tuple[float, float], window: Tensor) -> Tensor:
            w = window
            if self.mtf:  # the term passed the pixel MTF at its original frequencies f + F
                mtf = torch.sinc((fx + fc[0]) * p) * torch.sinc((fy + fc[1]) * p)
                w = torch.where(window > 0, window / mtf, torch.zeros_like(window))
            return torch.fft.ifft2(torch.fft.fft2(data) * w.to(cdtype)) / (p * p)

        window = soft(band)
        out = [demodulate(frames.to(cdtype), (0.0, 0.0), soft(band0))]
        zero_mean = frames - frames.mean((-2, -1), keepdim=True)  # the constant cannot leak
        centres = self.magnification * self._rotate(
            torch.tensor(self.harmonics, dtype=torch.float64)
        )
        for fc in centres.tolist():
            phase = torch.remainder(
                -2.0 * math.pi * (fc[0] * xs[None, :] + fc[1] * ys[:, None]), 2.0 * math.pi
            )
            carrier = torch.polar(torch.ones_like(phase), phase).to(cdtype)
            out.append(demodulate(zero_mean * carrier, (fc[0], fc[1]), window))
        return torch.stack(out, dim=2)

    # ---- the pointwise step and the integration --------------------------------------------

    def _filters(self, height: int, width: int, device: torch.device) -> tuple[Tensor, Tensor]:
        """Return the phase filters T_j ``[J, H, W]`` and the amplitude filter T_0 ``[H, W]``."""
        mn, weighted, shear, _g = self._orders()
        p = self.pitch
        ky = 2.0 * math.pi * torch.fft.fftfreq(height, d=p, dtype=torch.float64)[:, None]
        kx = 2.0 * math.pi * torch.fft.fftfreq(width, d=p, dtype=torch.float64)[None, :]

        def delay(i: int, sign: float = 1.0) -> Tensor:  # e^{−ik·a_o}
            phase = -sign * (kx * float(shear[i, 0]) + ky * float(shear[i, 1]))
            return torch.polar(torch.ones_like(phase), phase)

        filters = []
        for difference in self.harmonics:
            pairs = self._pairs(difference)
            if not pairs:
                raise StructureError(f"the grating has no orders {difference} apart")
            w = torch.stack([weighted[i] * weighted[j].conj() for i, j in pairs])
            w = w / w.sum()
            b = torch.stack([delay(i) - delay(j) for i, j in pairs])  # [pairs, H, W]
            b_neg = torch.stack([delay(i, -1.0) - delay(j, -1.0) for i, j in pairs])
            forward = (w[:, None, None] * b).sum(0)
            backward = (w[:, None, None] * b_neg).sum(0)
            filters.append((forward + backward.conj()) / 2.0)  # the phase part: (B(k) + B*(−k))/2
        power = weighted.abs() ** 2  # the constant term: Σ_o |c_o|²·|E(r − a_o)|²
        delays = torch.stack([delay(i) for i in range(mn.shape[0])])
        t0 = (power[:, None, None] * delays).sum(0) / power.sum()
        return torch.stack(filters).to(device), t0.to(device)

    def _calibration(self, z: Tensor) -> Tensor:
        """Return the empty field's constant term and harmonics, like ``z``."""
        if self.background is not None:
            return self.linear(torch.as_tensor(self.background, device=z.device))
        _mn, weighted, _shear, _g = self._orders()
        level = z[:, :, :1].real.flatten(-2).median(-1).values[..., None, None]  # [B, C, 1, 1, 1]
        power = float((weighted.abs() ** 2).sum())
        values = [torch.ones((), dtype=torch.complex128)]
        for difference in self.harmonics:
            pairs = self._pairs(difference)
            w = torch.stack([weighted[i] * weighted[j].conj() for i, j in pairs])
            values.append(w.sum() / power)
        ratios = torch.stack(values).to(device=z.device, dtype=z.dtype)
        return level * ratios[:, None, None]

    def __call__(self, frames: Tensor) -> Tensor:
        """Reconstruct the complex field over the empty field.

        Parameters
        ----------
        frames : Tensor
            Interferograms ``[B, C, H, W]`` in photons per pixel.

        Returns
        -------
        Tensor
            Complex ``[B, C, H, W]``: exp(α + iφ) at the pixel centres, with the frame-mean
            phase zero.
        """
        z = self.linear(frames)
        z0 = self._calibration(z)
        height, width = z.shape[-2:]
        filters, t0 = self._filters(height, width, z.device)
        cdtype = z.dtype
        psi = torch.angle(z[:, :, 1:] * z0[:, :, 1:].conj())  # [B, C, J, H, W]
        spectra = torch.fft.fft2(psi.to(cdtype))
        f = filters.to(cdtype)
        power = (f.abs() ** 2).sum(0)
        eps = self.regularization * float(power.max())
        phase_hat = (f.conj() * spectra).sum(2) / (power + eps)
        phase_hat[..., 0, 0] = 0.0  # the piston is unobservable
        phase = torch.fft.ifft2(phase_hat).real
        ratio = (z[:, :, 0].real / z0[:, :, 0].real).clamp(min=1e-12)
        log_amp_hat = torch.fft.fft2((0.5 * torch.log(ratio)).to(cdtype))
        t0c = t0.to(cdtype)
        log_amp_hat = log_amp_hat * t0c.conj() / (t0c.abs() ** 2 + self.regularization)
        log_amp = torch.fft.ifft2(log_amp_hat).real
        return torch.polar(torch.exp(log_amp), phase)
