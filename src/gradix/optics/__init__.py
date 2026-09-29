"""L2 optics: the objective, pupil modifiers, Fresnel coefficients and detection gratings.

``gx.Objective``, ``gx.pupil``, ``fresnel`` and ``Grating`` (ADR-45); the sampled stages
arrive in M3. The pupil filter is ``gx.pupil.Filter``: ``gx.optics.Filter`` is reserved for
the 4f stage of §5.11.
"""

from gradix.optics import fresnel
from gradix.optics.grating import Grating
from gradix.optics.objective import Objective
from gradix.optics.pupil import PixelPupil, PupilContext, PupilModifier, Zernike

__all__ = [
    "Grating",
    "Objective",
    "PixelPupil",
    "PupilContext",
    "PupilModifier",
    "Zernike",
    "fresnel",
]
