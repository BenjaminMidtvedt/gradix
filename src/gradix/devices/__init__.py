"""Optical device models (``gx.devices``; §5.11): parameters in, transmission maps out.

``PhaseSLM`` arrives with M3a's shaped illumination; ``HeightMap`` and the v1.x library
(``DMD``, ``DeformableMirror``, ``SegmentedMirror``, ``Metasurface``) follow in M3.
"""

from gradix.devices.slm import PhaseSLM

__all__ = ["PhaseSLM"]
