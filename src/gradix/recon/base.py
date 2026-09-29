"""The reconstruction protocol and the optics a reconstruction reads from a Chain (§5.12)."""

from __future__ import annotations

import dataclasses

import torch
from torch import Tensor

from gradix._core.errors import StructureError
from gradix.compose.chain import Chain
from gradix.detect.camera import Camera
from gradix.objects.environment import Medium
from gradix.optics.objective import Objective

__all__ = ["ChainOptics", "Reconstruction", "optics_of", "without_scatterers"]


class Reconstruction:
    """A differentiable reconstruction: frames in, reconstructed data out (ADR-44).

    ``linear(frames)`` is the part that is linear (affine) in the frames, the part that can
    lose information; ``__call__`` adds an optional invertible pointwise step (phase
    extraction, amplitude), which by default is the identity. :func:`gradix.crlb` measures the
    information that ``linear`` keeps. Subclasses are frozen dataclasses of physical
    parameters, with a ``from_chain`` constructor where the Chain holds them.
    """

    @property
    def is_linear(self) -> bool:
        """Whether :meth:`linear` is affine in the frames, as :func:`gradix.crlb` requires.

        Returns
        -------
        bool
            True unless a calibration is estimated from the frames themselves (a median).
        """
        return True

    def linear(self, frames: Tensor) -> Tensor:
        """Return the linear (affine) part of the reconstruction.

        Parameters
        ----------
        frames : Tensor
            Frames ``[B, C, H, W]`` in the camera's unit.

        Returns
        -------
        Tensor
            The reconstructed data.
        """
        raise NotImplementedError(type(self).__name__)

    def __call__(self, frames: Tensor) -> Tensor:
        """Reconstruct: the linear part, then the pointwise step (the identity by default).

        Parameters
        ----------
        frames : Tensor
            Frames ``[B, C, H, W]`` in the camera's unit.

        Returns
        -------
        Tensor
            The reconstructed data.
        """
        return self.linear(frames)


@dataclasses.dataclass(frozen=True)
class ChainOptics:
    """The optics a reconstruction needs, read once from a coherent Chain (host numbers).

    Parameters
    ----------
    pitch : float
        Object-space pixel pitch, µm.
    wavelength : float
        Vacuum wavelength of the light, µm.
    n : float
        Refractive index of the medium.
    na : float
        Numerical aperture of the objective.
    magnification : float
        Lateral magnification.
    shape : tuple of int
        Camera shape ``(H, W)``.
    """

    pitch: float
    wavelength: float
    n: float
    na: float
    magnification: float
    shape: tuple[int, int]


def _number(value: object, name: str) -> float:
    if isinstance(value, (int, float)):
        return float(value)  # Python numbers stay exact
    t = torch.as_tensor(value).detach().reshape(-1)
    if t.numel() != 1 and not bool((t == t[0]).all()):
        msg = f"the reconstruction needs one {name} for all images"
        raise StructureError(msg, fix="build one reconstruction per value")
    t = t[0]
    return float(t.real if t.is_complex() else t)


def optics_of(chain: Chain) -> ChainOptics:
    """Return the pitch, wavelength, index, NA and magnification of a coherent Chain.

    Parameters
    ----------
    chain : Chain
        A Chain with plane-wave light, an objective, a camera and a medium.

    Returns
    -------
    ChainOptics
        The numbers, read on the host.

    Raises
    ------
    StructureError
        If a part is missing, or a value differs between images.
    """
    objective, camera, medium = chain.objective, chain.camera, chain.environment
    light = chain.light
    if not isinstance(objective, Objective) or not isinstance(camera, Camera):
        raise StructureError("the Chain needs an objective and a camera")
    if not isinstance(medium, Medium) or light is None:
        raise StructureError("the Chain needs light and a medium")
    wavelength = getattr(light, "wavelength", None)
    if wavelength is None:
        raise StructureError(f"{type(light).__name__} has no wavelength")
    magnification = _number(objective.magnification, "magnification")
    return ChainOptics(
        pitch=_number(camera.pixel_size, "pixel size") / magnification,
        wavelength=_number(wavelength, "wavelength"),
        n=_number(medium.index(dtype=torch.float64), "medium index"),
        na=_number(objective.NA, "NA"),
        magnification=magnification,
        shape=(int(camera.shape[0]), int(camera.shape[1])),
    )


def without_scatterers(chain: Chain) -> Chain:
    """Return the Chain with every scatterer switched off: the empty field, rendered alike.

    Each population keeps its objects with presence 0, so the empty render keeps the Chain's
    precision and static configuration (a Chain emptied of its populations could compute in
    float32 where the original computes in float64, and a calibration that differs from the
    frames by rounding biases every reconstruction that divides by it).

    Parameters
    ----------
    chain : Chain
        A coherent Chain.

    Returns
    -------
    Chain
        The Chain whose scattered field is exactly zero.
    """
    changes: dict[str, object] = {}
    for name, element in chain.scatterers.items():
        field = getattr(element, "population_field", "objects")
        objects = getattr(element, field, None)
        if objects is None or not hasattr(objects, "presence"):
            raise StructureError(f"scatterers[{name!r}] has no population to switch off")
        changes[name] = element.replace(**{field: objects.replace(presence=0.0)})
    return chain.replace(scatterers=changes)
