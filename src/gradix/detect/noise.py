"""Camera noise models (``gx.noise``): sampling with explicit keys and likelihoods (§5.5, §6.2).

Noise models work in electrons; the :class:`~gradix.Camera` converts to and from its output
unit. Each model samples with a gradient estimator and evaluates a log-likelihood.
"""

from __future__ import annotations

import dataclasses
from typing import ClassVar

import torch
from torch import Tensor

from gradix import register
from gradix._core.errors import StructureError
from gradix.ops import detect as ops
from gradix.schema.base import DataObject
from gradix.schema.fields import field, knob

__all__ = ["EMCCD", "SCMOS", "Ideal", "NoiseModel", "PoissonGaussian"]

ESTIMATORS = ("scaled_st", "straight_through")
"""Poisson gradient estimators: the default scaled straight-through, and plain ST (flagged)."""


def _poisson(lam: Tensor, key: int | Tensor, stream: str, estimator: str) -> Tensor:
    counts = ops.poisson(lam, key, f"{stream}/poisson")
    if estimator == "scaled_st":
        return ops.scaled_st(lam, counts)
    return ops.straight_through(lam, counts)


@dataclasses.dataclass(frozen=True, kw_only=True, eq=False)
class NoiseModel(DataObject):
    """Base of camera noise models; they work in electrons."""

    image_keyed: ClassVar[bool] = True
    """Whether the model supports image keys (``Tensor[B]``); EMCCD is batch-keyed until 1.x."""

    def sample(self, lam: Tensor, key: int | Tensor, stream: str) -> Tensor:
        """Draw noisy electron counts with gradient estimators attached.

        Parameters
        ----------
        lam : Tensor
            Expected electrons ``[B, ...]``.
        key : int or Tensor
            A batch key or image keys.
        stream : str
            Stream name prefix.

        Returns
        -------
        Tensor
            Noisy electrons, same shape as ``lam``.
        """
        raise NotImplementedError(type(self).__name__)

    def log_prob(self, k: Tensor, lam: Tensor) -> Tensor:
        """Return the log-likelihood of observed electrons.

        Parameters
        ----------
        k : Tensor
            Observed electrons.
        lam : Tensor
            Expected electrons, broadcastable to ``k``.

        Returns
        -------
        Tensor
            Log-likelihood per element (an electron density; the camera adds the Jacobian of
            its unit).
        """
        raise NotImplementedError(type(self).__name__)

    def mean(self, electrons: Tensor) -> Tensor:
        """Return the expected output electrons of expected photoelectrons.

        The camera's expected frames go through this hook, so stages that change the mean (EM
        gain, dark current, clock-induced charge) keep ``expected`` and ``sample`` consistent.

        Parameters
        ----------
        electrons : Tensor
            Expected photoelectrons ``[B, ..., H, W]``.

        Returns
        -------
        Tensor
            Expected output electrons; the identity by default.
        """
        return electrons

    def inverse_mean(self, electrons: Tensor) -> Tensor:
        """Return the expected photoelectrons of expected output electrons (:meth:`mean` inverse).

        Parameters
        ----------
        electrons : Tensor
            Expected output electrons.

        Returns
        -------
        Tensor
            Expected photoelectrons, the rate :meth:`sample` draws from; the identity by default.
        """
        return electrons

    def variance(self, lam: Tensor) -> Tensor:
        """Return the variance of the output electrons, electrons² (for Fisher information).

        Parameters
        ----------
        lam : Tensor
            Expected photoelectrons.

        Returns
        -------
        Tensor
            The variance; the Poisson variance ``λ`` by default.
        """
        return torch.clamp(lam, min=0.0)

    def grad_quality(self) -> str:
        """Return the gradient quality of samples: ``"biased"`` for straight-through estimators.

        Returns
        -------
        str
            A gradient quality (§6.2).
        """
        return "biased"

    def field_quality(self, name: str) -> str:
        """Return the gradient quality of noisy images with respect to one of the model's fields.

        Parameters
        ----------
        name : str
            Field path within the noise model, such as ``"read"``.

        Returns
        -------
        str
            ``"exact"`` by default: parameters of reparameterised noise stages (read noise)
            have exact pathwise gradients.
        """
        return "exact"

    def mean_quality(self, name: str) -> str:
        """Return the gradient quality of the expected frames with respect to a model field.

        Parameters
        ----------
        name : str
            Field path within the noise model.

        Returns
        -------
        str
            ``"zero"`` by default; models whose :meth:`mean` reads a field (EM gain, clock-induced
            charge) return ``"exact"`` for it.
        """
        return "zero"


@register.noise("sensor.ideal")
@dataclasses.dataclass(frozen=True, kw_only=True, eq=False)
class Ideal(NoiseModel):
    """A photon-counting detector: Poisson shot noise only.

    Parameters
    ----------
    estimator : {"scaled_st", "straight_through"}, default "scaled_st"
        Gradient estimator of the Poisson draw (§6.2).
    """

    estimator: str = knob(default="scaled_st", choices=ESTIMATORS, doc="Poisson estimator")

    def sample(self, lam: Tensor, key: int | Tensor, stream: str) -> Tensor:
        """Draw Poisson electrons.

        Parameters
        ----------
        lam : Tensor
            Expected electrons ``[B, ...]``.
        key : int or Tensor
            A batch key or image keys.
        stream : str
            Stream name prefix.

        Returns
        -------
        Tensor
            Integer-valued electrons with estimator gradients.
        """
        return _poisson(lam, key, stream, self.estimator)

    def log_prob(self, k: Tensor, lam: Tensor) -> Tensor:
        """Return the exact Poisson log-likelihood.

        Parameters
        ----------
        k : Tensor
            Observed electrons.
        lam : Tensor
            Expected electrons.

        Returns
        -------
        Tensor
            Log-likelihood per element.
        """
        return ops.poisson_log_prob(k, lam)


@register.noise("sensor.poisson_gaussian")
@dataclasses.dataclass(frozen=True, kw_only=True, eq=False)
class PoissonGaussian(NoiseModel):
    """Poisson shot noise plus Gaussian read noise.

    Parameters
    ----------
    read : Tensor or float, default 0.0
        Read-noise standard deviation in electrons: a scalar, per image ``[B]``, or a map.
    estimator : {"scaled_st", "straight_through"}, default "scaled_st"
        Gradient estimator of the Poisson draw; read noise is reparameterised.
    likelihood : {"gaussian", "shifted_poisson", "convolution"}, default "gaussian"
        The Poisson ⊛ Gaussian likelihood: a Gaussian with variance λ + σ², the shifted
        Poisson, or the exact truncated convolution
        (:func:`gradix.ops.detect.poisson_gaussian_log_prob`, 64 terms per pixel). The Q6
        benchmark (``benchmarks/q6_scmos.py``, 0.1–100 photons, σ 0.8–3 e⁻) finds the Gaussian
        unbiased for λ, gain, offset and read noise (efficiency ≥ 0.93 for λ, down to 0.48 for
        read noise at σ = 0.8 e⁻), the shifted Poisson biased at low light (σ = 0.8 e⁻,
        λ = 0.1–1: read noise −0.7 to −1.0 and offset +0.7 to +3 one-pixel standard errors, the
        offset estimate divergent at λ = 0.1), and the convolution exact at about 50× the cost:
        use it for calibration and exact bounds.
    """

    schema_version: ClassVar[int] = 2

    read: Tensor | float = field(
        quantity="electrons",
        role="pixels",
        constraint="nonnegative",
        default=0.0,
        doc="read-noise standard deviation",
    )
    estimator: str = knob(default="scaled_st", choices=ESTIMATORS, doc="Poisson estimator")
    likelihood: str = knob(
        default="gaussian",
        choices=("gaussian", "shifted_poisson", "convolution"),
        doc="Poisson ⊛ Gaussian likelihood",
    )

    def read_std(self, like: Tensor) -> Tensor:
        """Return the read noise broadcastable to a frame tensor ``[B, ..., H, W]``.

        Parameters
        ----------
        like : Tensor
            Frames the result must broadcast against.

        Returns
        -------
        Tensor
            Read-noise standard deviation in electrons.
        """
        from gradix.detect.camera import pixel_map

        return pixel_map(self.read, self.schema()["read"], like)

    def sample(self, lam: Tensor, key: int | Tensor, stream: str) -> Tensor:
        """Draw Poisson electrons plus reparameterised Gaussian read noise.

        Parameters
        ----------
        lam : Tensor
            Expected electrons ``[B, ...]``.
        key : int or Tensor
            A batch key or image keys.
        stream : str
            Stream name prefix.

        Returns
        -------
        Tensor
            Noisy electrons with estimator gradients (and exact gradients to ``read``).
        """
        n = _poisson(lam, key, stream, self.estimator)
        z = ops.normal(n, key, f"{stream}/read")
        return n + self.read_std(n) * z

    def log_prob(self, k: Tensor, lam: Tensor) -> Tensor:
        """Return the approximate Poisson ⊛ Gaussian log-likelihood.

        Parameters
        ----------
        k : Tensor
            Observed electrons.
        lam : Tensor
            Expected electrons.

        Returns
        -------
        Tensor
            Log-likelihood per element.
        """
        var = self.read_std(k) ** 2
        if self.likelihood == "shifted_poisson":
            return ops.shifted_poisson_log_prob(k, lam, var)
        if self.likelihood == "convolution":
            return ops.poisson_gaussian_log_prob(k, lam, var)
        total = torch.clamp(lam, min=0.0) + var
        total = torch.clamp(total, min=1e-12)
        return -0.5 * ((k - lam) ** 2 / total + torch.log(2.0 * torch.pi * total))

    def variance(self, lam: Tensor) -> Tensor:
        """Return the output variance ``λ + σ²``, electrons².

        Parameters
        ----------
        lam : Tensor
            Expected photoelectrons.

        Returns
        -------
        Tensor
            The variance.
        """
        return torch.clamp(lam, min=0.0) + self.read_std(lam) ** 2


@register.noise("sensor.emccd")
@dataclasses.dataclass(frozen=True, kw_only=True, eq=False)
class EMCCD(NoiseModel):
    """An electron-multiplying CCD: Poisson → clock-induced charge → EM gain → read noise (§5.5).

    The EM register multiplies each photoelectron count n by a Gamma(n, G) variate, so the mean
    output is ``G·(λ + cic)`` (the camera's expected frames include it through :meth:`mean`) and
    the excess noise factor is √2. The forward pass is the exact Gamma–Poisson chain; gradients
    are implicit-reparameterisation gradients for n > 0 and ``G`` straight through at n = 0,
    which keeps dE[y]/dλ unbiased at every light level. Losses that read the variance are
    biased low at low light (about −24 % at λ = 0.05, −9 % at λ = 1, < 1 % above λ = 3): the
    Gamma stage's variance gradient is not captured at n = 0. Batch-keyed only until 1.x.

    Parameters
    ----------
    gain : Tensor or float, default 100.0
        Mean EM gain G: a constant, per image ``[B]``, or a map.
    cic : Tensor or float, default 0.0
        Clock-induced charge, electrons per pixel per frame (before the EM register).
    read : Tensor or float, default 0.0
        Read-noise standard deviation after the EM register, electrons.
    estimator : {"scaled_st", "straight_through"}, default "scaled_st"
        Gradient estimator of the Poisson draw.
    """

    image_keyed: ClassVar[bool] = False

    gain: Tensor | float = field(
        quantity="dimensionless", role="pixels", constraint="positive", default=100.0, doc="EM gain"
    )
    cic: Tensor | float = field(
        quantity="electrons",
        role="pixels",
        constraint="nonnegative",
        default=0.0,
        doc="clock-induced charge",
    )
    read: Tensor | float = field(
        quantity="electrons",
        role="pixels",
        constraint="nonnegative",
        default=0.0,
        doc="read-noise standard deviation",
    )
    estimator: str = knob(default="scaled_st", choices=ESTIMATORS, doc="Poisson estimator")

    def _map(self, name: str, like: Tensor) -> Tensor:
        from gradix.detect.camera import pixel_map

        return pixel_map(getattr(self, name), self.schema()[name], like)

    def mean(self, electrons: Tensor) -> Tensor:
        """Return the mean output electrons, ``G·(λ + cic)``.

        Parameters
        ----------
        electrons : Tensor
            Expected photoelectrons ``[B, ..., H, W]``.

        Returns
        -------
        Tensor
            Expected electrons after the EM register.
        """
        return self._map("gain", electrons) * (electrons + self._map("cic", electrons))

    def inverse_mean(self, electrons: Tensor) -> Tensor:
        """Return the expected photoelectrons of expected output electrons.

        Parameters
        ----------
        electrons : Tensor
            Expected electrons after the EM register.

        Returns
        -------
        Tensor
            ``electrons/G − cic``.
        """
        return electrons / self._map("gain", electrons) - self._map("cic", electrons)

    def sample(self, lam: Tensor, key: int | Tensor, stream: str) -> Tensor:
        """Draw EM-amplified electrons plus read noise.

        Parameters
        ----------
        lam : Tensor
            Expected photoelectrons ``[B, ...]``.
        key : int
            A batch key.
        stream : str
            Stream name prefix.

        Returns
        -------
        Tensor
            Output electrons (mean ``G·(λ + cic)``).

        Raises
        ------
        StructureError
            For image keys (EMCCD is batch-keyed until 1.x).
        """
        if isinstance(key, Tensor):
            raise StructureError("EMCCD is batch-keyed until 1.x", fix="pass an int key")
        n = _poisson(lam + self._map("cic", lam), key, stream, self.estimator)
        gain = self._map("gain", n)
        positive = n.detach() > 0
        n_safe = torch.where(positive, n, torch.ones_like(n))
        amplified = gain * ops.gamma(n_safe, key, f"{stream}/em")
        # n = 0: no Gamma draw; the straight-through term keeps dy/dn = G there
        y = torch.where(positive, amplified, gain * (n - n.detach()))
        return y + self._map("read", y) * ops.normal(y, key, f"{stream}/read")

    def log_prob(self, k: Tensor, lam: Tensor) -> Tensor:
        """Return the high-gain Gaussian approximation of the EMCCD log-likelihood.

        The output is treated as Gaussian with mean ``G·(λ + cic)`` and variance
        ``2·G²·(λ + cic) + read²`` (excess noise factor √2); accurate when λ + cic ≫ 1 (the
        output distribution is far from Gaussian at a few photoelectrons, whatever the gain).

        Parameters
        ----------
        k : Tensor
            Observed output electrons.
        lam : Tensor
            Expected photoelectrons.

        Returns
        -------
        Tensor
            Log-likelihood per element.
        """
        mean = self.mean(lam)
        var = torch.clamp(self.variance(lam), min=1e-12)
        return -0.5 * ((k - mean) ** 2 / var + torch.log(2.0 * torch.pi * var))

    def variance(self, lam: Tensor) -> Tensor:
        """Return the output variance, ``2·G²·(λ + cic) + read²``, electrons².

        Parameters
        ----------
        lam : Tensor
            Expected photoelectrons.

        Returns
        -------
        Tensor
            The variance.
        """
        gain = self._map("gain", lam)
        signal = torch.clamp(lam + self._map("cic", lam), min=0.0)
        return 2.0 * gain * gain * signal + self._map("read", lam) ** 2

    def field_quality(self, name: str) -> str:
        """Return the gradient quality of images with respect to one of the model's fields.

        Parameters
        ----------
        name : str
            ``"gain"``, ``"cic"`` or ``"read"``.

        Returns
        -------
        str
            ``"biased"`` for ``cic`` (it passes through the Poisson estimator), ``"exact"``
            otherwise.
        """
        return "biased" if name == "cic" else "exact"

    def mean_quality(self, name: str) -> str:
        """Return the gradient quality of expected frames: gain and cic set the mean.

        Parameters
        ----------
        name : str
            ``"gain"``, ``"cic"`` or ``"read"``.

        Returns
        -------
        str
            ``"exact"`` for ``gain`` and ``cic``, ``"zero"`` for ``read``.
        """
        return "exact" if name in ("gain", "cic") else "zero"


@register.noise("sensor.scmos")
@dataclasses.dataclass(frozen=True, kw_only=True, eq=False)
class SCMOS(PoissonGaussian):
    """An sCMOS sensor: Poisson shot noise plus per-pixel Gaussian read noise (maps).

    The same model as :class:`PoissonGaussian`, named for its usual calibration: ``read`` is a
    per-pixel map ``[H, W]`` (and the camera's ``gain`` and ``offset`` are maps too; see
    ``gx.Camera.scmos``).

    Parameters
    ----------
    read : Tensor or float, default 0.0
        Read-noise standard deviation in electrons: a map ``[H, W]``, per image, or a scalar.
    estimator : {"scaled_st", "straight_through"}, default "scaled_st"
        Gradient estimator of the Poisson draw.
    likelihood : {"gaussian", "shifted_poisson", "convolution"}, default "gaussian"
        The Poisson ⊛ Gaussian likelihood (see :class:`PoissonGaussian`).
    """
