"""L2 interaction elements (``gx.interact``; §4.4): coherent scatterers.

M1 ships the minimal point dipole of the coherent smoke thread; Mie, Born and multislice arrive
with M3.
"""

from gradix.interact.dipole import Dipole, DipoleParams, scattering_amplitude

__all__ = ["Dipole", "DipoleParams", "scattering_amplitude"]
