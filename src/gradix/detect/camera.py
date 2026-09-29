"""The camera element: expected frames, keyed samples and likelihoods in one output unit (§5.5).

The sensor chain follows QE → Poisson → (dark, full well, EM gain: M1) → read noise →
gain/offset → ADC (straight-through round) → saturation. Background photons are added before
the Poisson stage.

Output frames are ``[B, C, H, W]``, or ``[B, T, C, H, W]`` when the acquisition has a time
axis; other acquisition axes are flattened into C (§4.2).
"""

from __future__ import annotations

import dataclasses
import os
from collections.abc import Mapping
from typing import ClassVar

import torch
from torch import Tensor

from gradix import register
from gradix._core.axes import AcqIndex
from gradix._core.carriers import Irradiance
from gradix._core.contract import (
    Capabilities,
    Description,
    Element,
    Slot,
    Static,
    Violation,
    worst_quality,
)
from gradix._core.envelope import Envelope
from gradix._core.errors import StructureError
from gradix._core.keys import check_key
from gradix.detect.noise import NoiseModel
from gradix.ops import detect as ops
from gradix.schema.base import iter_leaves
from gradix.schema.fields import FieldSpec, child, field, knob
from gradix.schema.layout import axis_sizes, canonical

__all__ = ["UNITS", "Camera", "check_pixel_map", "check_pixel_maps", "format_frames", "pixel_map"]

UNITS = ("adu", "e", "photons")
"""Camera output units: analog-digital units, electrons, or photon equivalents (e/QE)."""


def pixel_map(value: Tensor | float, spec: FieldSpec, like: Tensor) -> Tensor:
    """Return a ``pixels``-role value broadcastable to frames ``[B, ..., H, W]``.

    Parameters
    ----------
    value : Tensor or float
        A constant, per-image ``[B]``, a map ``[H, W]`` or per-image maps ``[B, H, W]``.
    spec : FieldSpec
        The field's declaration (role ``"pixels"``).
    like : Tensor
        Frames ``[B, ..., H, W]`` the result must broadcast against.

    Returns
    -------
    Tensor
        ``[B|1, 1, …, 1, H|1, W|1]`` with as many dims as ``like``.
    """
    t = canonical(
        value, spec, dtype=like.dtype if not like.is_complex() else None, device=like.device
    )
    b, h, w = t.shape
    return t.reshape(b, *([1] * (like.ndim - 3)), h, w)


def check_pixel_maps(tree: object, shape: tuple[int, int], *, where: str = "") -> None:
    """Check that every pixel map of a tree matches a camera's ``(H, W)``.

    A 2-d ``pixels`` value is always a map ``[H, W]``: per-image values are ``[B]`` or
    ``[B, 1, 1]``, so a ``[B, 1]`` column is read as a map with one column and fails here unless
    the camera has one column.

    Parameters
    ----------
    tree : object
        A node (a camera with its noise model, a Chain, a Microscope) or a tensor.
    shape : tuple of int
        The camera's ``(H, W)``.
    where : str, optional
        Path prefix for error messages.

    Raises
    ------
    StructureError
        If a map's H or W is neither 1 nor the camera's.
    """
    for path, spec, value in iter_leaves(tree, where):
        if spec is not None and spec.role == "pixels" and isinstance(value, Tensor):
            check_pixel_map(value, spec, shape, where=path)


def check_pixel_map(
    value: Tensor, spec: FieldSpec, shape: tuple[int, int], *, where: str = ""
) -> None:
    """Check one ``pixels``-role value against a camera's ``(H, W)``.

    Parameters
    ----------
    value : Tensor
        The value.
    spec : FieldSpec
        Its declaration (role ``"pixels"``).
    shape : tuple of int
        The camera's ``(H, W)``.
    where : str, optional
        Field path for error messages.

    Raises
    ------
    StructureError
        If a map's H or W is neither 1 nor the camera's.
    """
    sizes = axis_sizes(spec, tuple(value.shape), where=where)
    for axis, size in zip(("H", "W"), shape, strict=True):
        got = sizes.get(axis, 1)
        if got not in (1, size):
            msg = (
                f"{where} is a map with {axis} = {got} but the camera has {axis} = {size} "
                f"(shape {list(value.shape)})"
            )
            fix = "give per-image values as [B] or [B, 1, 1]; a 2-d value is an [H, W] map"
            raise StructureError(msg, fix=fix)


def format_frames(data: Tensor, acq: AcqIndex) -> Tensor:
    """Unflatten the acquisition axis of ``[B, A, C, H, W]`` into output frames.

    Parameters
    ----------
    data : Tensor
        ``[B, A, C, H, W]``.
    acq : AcqIndex
        What A holds.

    Returns
    -------
    Tensor
        ``[B, T, C', H, W]`` with a time axis, else ``[B, C', H, W]``, where C' flattens the
        non-time acquisition axes into the channels (a view).
    """
    b, a, c, h, w = data.shape
    frames, channels = acq.output_layout(c)
    if a != acq.size and a != 1:
        msg = f"the acquisition axis has {a} frames but the acquisition declares {acq.size}"
        raise StructureError(msg)
    if a == 1 and acq.size != 1:
        data = data.expand(b, acq.size, c, h, w)
    if frames is None:
        return data.reshape(b, channels, h, w)
    return data.reshape(b, frames, channels, h, w)


@dataclasses.dataclass(frozen=True)
class CameraStatic(Static):
    """Static configuration of a camera: its output layout.

    Parameters
    ----------
    decisions : tuple of Decision, default ()
        Sizing decisions (none for cameras).
    """


@register.element("detect.camera")
@dataclasses.dataclass(frozen=True, kw_only=True, eq=False)
class Camera(Element[Tensor]):
    """A camera: pixel grid, sensor chain and output unit.

    Every calibration field is a tensor field, so it can be learned or sampled per image.

    Parameters
    ----------
    pixel_size : Tensor or float
        Physical pixel pitch in µm (image side); the object-space pitch divides it by the
        objective's magnification.
    shape : tuple of int
        ``(H, W)`` pixels.
    qe : Tensor or float, default 1.0
        Quantum efficiency: a constant, per image ``[B]``, or a map.
    gain : Tensor or float, default 1.0
        Conversion gain in ADU/e⁻: a constant, per image ``[B]``, or a map.
    offset : Tensor or float, default 0.0
        Offset in ADU: a constant, per image ``[B]``, or a map.
    bit_depth : int, optional
        ADC bit depth: output in ``"adu"`` is rounded (straight-through) and clipped to
        ``[0, 2**bit_depth − 1]``. None models no ADC.
    noise : NoiseModel, optional
        Noise model; None is a noise-free camera whose image equals its expected frames.
    unit : {"adu", "e", "photons"}, default "adu"
        Output unit of every frame the camera returns.

    Examples
    --------
    >>> from gradix.units import um
    >>> Camera(pixel_size=6.5 * um, shape=(128, 128)).unit
    'adu'
    """

    slot: ClassVar[Slot] = Slot.DETECT
    caps: ClassVar[Capabilities] = Capabilities(
        accepts=frozenset({Irradiance}),
        produces=Tensor,
        reads={"irradiance": "exact", "background": "exact"},
    )

    pixel_size: Tensor | float = field(
        quantity="length",
        role="image",
        shape_affecting=True,
        constraint="positive",
        doc="physical pixel pitch",
    )
    shape: tuple[int, int] = knob(doc="(H, W) pixels")
    qe: Tensor | float = field(
        quantity="dimensionless",
        role="pixels",
        constraint="unit_interval",
        default=1.0,
        doc="quantum efficiency",
    )
    gain: Tensor | float = field(
        quantity="gain", role="pixels", constraint="positive", default=1.0, doc="ADU per electron"
    )
    offset: Tensor | float = field(quantity="adu", role="pixels", default=0.0, doc="offset in ADU")
    bit_depth: int | None = knob(default=None, doc="ADC bit depth")
    noise: NoiseModel | None = child(default=None, doc="noise model")
    unit: str = knob(default="adu", choices=UNITS, doc="output unit")

    def __post_init__(self) -> None:
        super().__post_init__()
        if len(self.shape) != 2 or not all(isinstance(s, int) and s > 0 for s in self.shape):
            msg = f"Camera.shape must be two positive ints (H, W), got {self.shape!r}"
            raise StructureError(msg)
        check_pixel_maps(self, self.shape, where="camera")

    def validity(self, desc: Description, envelope: Envelope) -> list[Violation]:
        """Report findings: an ADC without the ADU unit.

        Parameters
        ----------
        desc : Description
            Static description of the camera's context.
        envelope : Envelope
            The envelope.

        Returns
        -------
        list of Violation
            Findings.
        """
        out: list[Violation] = []
        if self.bit_depth is not None and self.unit != "adu":
            out.append(
                Violation(
                    "info",
                    f"bit_depth is ignored in unit {self.unit!r}; the ADC acts on ADU",
                    element=desc.path or "camera",
                )
            )
        return out

    def _noisy(self, quality: str, output: str) -> str:
        """Quality after the sensor for a quantity upstream of the shot noise."""
        if output == "image" and self.noise is not None and quality != "zero":
            quality = worst_quality(quality, self.noise.grad_quality())
        return self._adc(quality, output)

    def _adc(self, quality: str, output: str) -> str:
        if output == "image" and quality != "zero" and self.bit_depth is not None:
            if self.unit == "adu":
                return worst_quality(quality, "exact-a.e.")
        return quality

    def field_quality(self, path: str, output: str = "expected") -> str:
        """Return the gradient quality of an output with respect to a camera field.

        Parameters
        ----------
        path : str
            Field path within the camera, such as ``"gain"`` or ``"noise.read"``.
        output : {"expected", "image"}, default "expected"
            The output kind.

        Returns
        -------
        str
            ``qe`` is upstream of the shot noise (the noise model's estimator quality in
            images); ``gain`` and ``offset`` act after it (exact); noise fields reach images
            only; ``pixel_size`` reaches outputs through the imaging element, not the sensor.
        """
        head, _, rest = path.partition(".")
        if head == "noise":
            if self.noise is None:
                return "zero"
            if output != "image":
                return self.noise.mean_quality(rest)
            return self._adc(self.noise.field_quality(rest), output)
        if head == "qe":
            return self._noisy("exact", output)
        if head in ("gain", "offset"):
            return self._adc("exact", output)
        if head == "pixel_size":
            return "zero"
        return self._adc(self.caps.quality(path), output)

    def input_quality(self, name: str, output: str = "expected") -> str:
        """Return the gradient quality of an output with respect to the camera's inputs.

        Parameters
        ----------
        name : str
            ``"irradiance"`` or ``"background"``; both enter before the shot noise.
        output : {"expected", "image"}, default "expected"
            The output kind.

        Returns
        -------
        str
            Exact for expected frames; the noise model's estimator quality (and the ADC's
            almost-everywhere derivative) for images.
        """
        return self._noisy("exact", output)

    def _map(self, name: str, like: Tensor) -> Tensor:
        return pixel_map(getattr(self, name), self.schema()[name], like)

    def electrons(self, frames: Tensor) -> Tensor:
        """Convert frames in the camera's unit into electrons.

        Parameters
        ----------
        frames : Tensor
            ``[B, …, H, W]`` in :attr:`unit`.

        Returns
        -------
        Tensor
            Electrons, same shape.
        """
        if self.unit == "adu":
            return (frames - self._map("offset", frames)) / self._map("gain", frames)
        if self.unit == "photons":
            return frames * self._map("qe", frames)
        return frames

    def from_electrons(self, electrons: Tensor) -> Tensor:
        """Convert electrons into the camera's unit (without the ADC).

        Parameters
        ----------
        electrons : Tensor
            ``[B, …, H, W]`` electrons.

        Returns
        -------
        Tensor
            Frames in :attr:`unit`.
        """
        if self.unit == "adu":
            return electrons * self._map("gain", electrons) + self._map("offset", electrons)
        if self.unit == "photons":
            return electrons / self._map("qe", electrons)
        return electrons

    def log_jacobian(self, frames: Tensor) -> Tensor:
        """Return ``log |d(unit)/d(electrons)|``, per pixel.

        Parameters
        ----------
        frames : Tensor
            Frames the result must broadcast against.

        Returns
        -------
        Tensor
            ``log(gain)`` for ADU, ``−log(qe)`` for photons, 0 for electrons.
        """
        if self.unit == "adu":
            return torch.log(self._map("gain", frames))
        if self.unit == "photons":
            return -torch.log(self._map("qe", frames))
        return torch.zeros((), dtype=frames.dtype, device=frames.device)

    def expected(self, irradiance: Irradiance, background: Tensor | float | None = None) -> Tensor:
        """Return the noise-free expected frames in the camera's unit.

        Parameters
        ----------
        irradiance : Irradiance
            Photons per pixel on the camera grid, ``[B|1, A|1, L|1, H, W]``.
        background : Tensor or float, optional
            Background photons per pixel per exposure, added before the sensor: a constant,
            per image ``[B]``, a map ``[H, W]`` or maps ``[B, H, W]``.

        Returns
        -------
        Tensor
            ``[B, C, H, W]`` (or ``[B, T, C, H, W]`` with a time axis) in :attr:`unit`.
        """
        return self.forward(irradiance, background, static=CameraStatic())

    def forward(self, *inputs: object, static: Static) -> Tensor:
        """Run the camera's expected-value chain (the element contract).

        Parameters
        ----------
        *inputs : object
            The :class:`~gradix.Irradiance`, then optionally the background.
        static : Static
            The configuration.

        Returns
        -------
        Tensor
            Expected frames in :attr:`unit`.
        """
        irradiance = inputs[0]
        background = inputs[1] if len(inputs) > 1 else None
        if not isinstance(irradiance, Irradiance):
            msg = f"Camera takes an Irradiance, got {type(irradiance).__name__}"
            raise StructureError(msg, fix="render emitters with an imaging element first")
        if irradiance.grid.shape != tuple(self.shape):
            msg = f"the irradiance grid {irradiance.grid.shape} is not the camera grid {self.shape}"
            raise StructureError(msg, fix="pixel integration from finer grids arrives with M1")
        photons = irradiance.data.sum(dim=2, keepdim=True)  # [B, A, 1, H, W]
        frames = format_frames(photons, irradiance.acq)
        if isinstance(background, (Tensor, float, int)):
            frames = frames + pixel_map(background, _BACKGROUND, frames)
        elif background is not None:
            msg = f"background must be a tensor or a number, got {type(background).__name__}"
            raise StructureError(msg)
        electrons = frames * self._map("qe", frames)
        if self.noise is not None:
            electrons = self.noise.mean(electrons)
        return self.from_electrons(electrons)

    def sample(
        self, mu: Tensor, key: int | Tensor | None = None, *, stream: str = "camera"
    ) -> Tensor:
        """Draw noisy frames from expected frames with an explicit key.

        Parameters
        ----------
        mu : Tensor
            Expected frames in :attr:`unit`, ``[B, …, H, W]``.
        key : int or Tensor, optional
            A batch key (``int``) or image keys (int64 ``Tensor[B]``); required with noise.
        stream : str, default "camera"
            Stream name; a Pipeline uses the camera's path, so direct calls with the default
            reproduce a Pipeline's noise.

        Returns
        -------
        Tensor
            Noisy frames in :attr:`unit`, same shape as ``mu``. Gradients follow the noise
            model's estimator (scaled straight-through by default).

        Raises
        ------
        StructureError
            If the camera has noise and ``key`` is None, or the key does not fit the batch.
        """
        y = mu
        if self.noise is not None:
            if key is None:
                msg = "this camera has noise, so its image needs a key"
                raise StructureError(msg, fix="pass key= (an int, or an int64 Tensor[B])")
            k = check_key(key, None)
            if isinstance(k, Tensor):
                if not self.noise.image_keyed:
                    msg = f"{type(self.noise).__name__} is batch-keyed until 1.x"
                    raise StructureError(msg, fix="pass an int key")
                if mu.shape[0] == 1 and k.shape[0] > 1:
                    mu = mu.expand(k.shape[0], *mu.shape[1:])
                check_key(k, mu.shape[0])
            # No clamp: the samplers clamp the detached rate, and the estimator must see λ ≤ 0
            # (scaled straight-through gives g = 1 there, so a dark pixel can still switch on).
            lam = self.noise.inverse_mean(self.electrons(mu))
            if k is None:  # pragma: no cover - guarded above
                raise StructureError("missing key")
            y = self.from_electrons(self.noise.sample(lam, k, stream))
        if self.bit_depth is not None and self.unit == "adu":
            # Clip first, then round: pixels that merely round onto 0 or full scale keep their
            # (almost-everywhere exact) gradient; only truly clipped pixels lose it.
            y = ops.st_round(torch.clamp(y, 0.0, float(2**self.bit_depth - 1)))
        return y

    def variance(self, mu: Tensor) -> Tensor:
        """Return the per-pixel variance of frames in the camera's unit, given expected frames.

        Parameters
        ----------
        mu : Tensor
            Expected frames in :attr:`unit`.

        Returns
        -------
        Tensor
            The noise model's variance of the output electrons, converted to the unit (× gain²
            for ADU, ÷ QE² for photons); the ADC's quantisation adds 1/12 ADU².

        Raises
        ------
        StructureError
            If the camera has no noise model.
        """
        if self.noise is None:
            raise StructureError("a noise-free camera has no variance", fix="add a noise model")
        lam = self.noise.inverse_mean(self.electrons(mu))
        var = self.noise.variance(lam)
        if self.unit == "adu":
            var = var * self._map("gain", mu) ** 2
            if self.bit_depth is not None:
                # quantisation noise; accurate when the pre-ADC noise exceeds ~0.5 ADU
                var = var + 1.0 / 12.0
        elif self.unit == "photons":
            var = var / self._map("qe", mu) ** 2
        return var

    @classmethod
    def scmos(
        cls,
        *,
        pixel_size: Tensor | float,
        maps: str | os.PathLike[str] | Mapping[str, object] | None = None,
        roi: tuple[int, int, int, int] | Tensor | None = None,
        shape: tuple[int, int] | None = None,
        gain: Tensor | float | None = None,
        offset: Tensor | float | None = None,
        read: Tensor | float | None = None,
        qe: Tensor | float | None = None,
        bit_depth: int | None = 16,
        unit: str = "adu",
    ) -> Camera:
        """Return an sCMOS camera from its calibration maps.

        Parameters
        ----------
        pixel_size : Tensor or float
            Physical pixel pitch, µm.
        maps : str, path or Mapping, optional
            Calibration maps: an ``.npz`` file or a mapping with ``offset`` (ADU), ``gain``
            (ADU/e⁻) and ``variance`` (read-noise variance, ADU²) or ``read`` (e⁻), and
            optionally ``qe``. Explicit ``gain``/``offset``/``read`` arguments override them.
        roi : tuple of int or Tensor, optional
            Crop of the maps: ``(y0, x0, height, width)``, or per-image top-left corners
            ``[B, 2]`` (then ``shape`` gives the crop size), giving per-image maps.
        shape : tuple of int, optional
            ``(H, W)``; taken from the (cropped) maps by default.
        gain : Tensor or float, optional
            Gain map ``[H, W]`` in ADU/e⁻, or a scalar.
        offset : Tensor or float, optional
            Offset map ``[H, W]`` in ADU, or a scalar.
        read : Tensor or float, optional
            Read-noise map ``[H, W]`` in electrons, or a scalar.
        qe : Tensor or float, optional
            Quantum efficiency; the maps' ``qe`` by default, else 1.
        bit_depth : int, optional, default 16
            ADC bit depth.
        unit : {"adu", "e", "photons"}, default "adu"
            Output unit.

        Returns
        -------
        Camera
            The camera, with an :class:`~gradix.noise.SCMOS` noise model.

        Raises
        ------
        StructureError
            If a map is missing, a crop leaves the maps, or the shape cannot be determined.
        """
        from gradix.detect.noise import SCMOS

        table = _load_maps(maps)
        g = gain if gain is not None else table.get("gain", 1.0)
        o = offset if offset is not None else table.get("offset", 0.0)
        if read is not None:
            r: Tensor | float = read
        elif "read" in table:
            r = table["read"]
        elif "variance" in table:
            g_t = torch.as_tensor(g, dtype=torch.float64)
            variance = torch.as_tensor(table["variance"], dtype=torch.float64)
            r = (torch.sqrt(variance) / g_t).to(table["variance"].dtype)
        else:
            r = 0.0
        q = qe if qe is not None else table.get("qe", 1.0)
        if roi is not None:
            g, o, r, q = (_crop(v, roi, shape) for v in (g, o, r, q))
        if shape is None:
            dims = [v.shape[-2:] for v in (g, o, r, q) if isinstance(v, Tensor) and v.ndim >= 2]
            if not dims:
                raise StructureError("Camera.scmos needs shape= or maps with a shape")
            shape = (int(dims[0][0]), int(dims[0][1]))
        return cls(
            pixel_size=pixel_size,
            shape=shape,
            qe=q,
            gain=g,
            offset=o,
            bit_depth=bit_depth,
            noise=SCMOS(read=r),
            unit=unit,
        )

    def log_prob(self, observed: Tensor, mu: Tensor) -> Tensor:
        """Return the per-pixel log-likelihood of observed frames given expected frames.

        The density is in the camera's unit: the Jacobian of the unit conversion is included,
        so gain and QE can be fitted.

        Parameters
        ----------
        observed : Tensor
            Observed frames in :attr:`unit`.
        mu : Tensor
            Expected frames in :attr:`unit`, broadcastable to ``observed``.

        Returns
        -------
        Tensor
            Log-likelihood per pixel.

        Raises
        ------
        StructureError
            If the camera has no noise model.
        """
        if self.noise is None:
            msg = "a noise-free camera has no likelihood"
            raise StructureError(msg, fix="give the camera a noise model, e.g. gx.noise.Ideal()")
        k = self.electrons(observed)
        lam = self.noise.inverse_mean(self.electrons(mu))
        return self.noise.log_prob(k, lam) - self.log_jacobian(observed)


_BACKGROUND = FieldSpec(kind="tensor", quantity="photons", role="pixels")


def _load_maps(maps: str | os.PathLike[str] | Mapping[str, object] | None) -> dict[str, Tensor]:
    """Read calibration maps into float32 tensors (from an ``.npz`` file or a mapping)."""
    if maps is None:
        return {}
    if isinstance(maps, Mapping):
        items = dict(maps)
    else:
        import numpy as np

        with np.load(os.fspath(maps)) as data:
            items = {k: data[k] for k in data.files}
    return {
        str(k): torch.as_tensor(v, dtype=torch.float32)
        for k, v in items.items()
        if k in ("offset", "gain", "variance", "read", "qe")
    }


def _crop(
    value: Tensor | float, roi: tuple[int, int, int, int] | Tensor, shape: tuple[int, int] | None
) -> Tensor | float:
    """Crop a map to a window, or gather per-image windows ``[B, h, w]``."""
    if not isinstance(value, Tensor) or value.ndim < 2:
        return value
    height, width = value.shape[-2:]
    if isinstance(roi, Tensor):
        if shape is None:
            raise StructureError("per-image ROIs need shape=(height, width)")
        h, w = shape
        corners = roi.to(device="cpu", dtype=torch.int64)
        if (
            bool((corners < 0).any())
            or bool((corners[:, 0] + h > height).any())
            or bool((corners[:, 1] + w > width).any())
        ):
            raise StructureError(f"a per-image ROI of {shape} leaves the {height}×{width} maps")
        rows = corners[:, 0, None] + torch.arange(h)[None, :]  # [B, h]
        cols = corners[:, 1, None] + torch.arange(w)[None, :]  # [B, w]
        return value[rows[:, :, None], cols[:, None, :]]
    y0, x0, h, w = roi
    if y0 < 0 or x0 < 0 or y0 + h > height or x0 + w > width:
        raise StructureError(f"the ROI {tuple(roi)} leaves the {height}×{width} maps")
    return value[..., y0 : y0 + h, x0 : x0 + w]
