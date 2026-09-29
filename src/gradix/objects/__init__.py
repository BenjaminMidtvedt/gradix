"""L2 data objects: object sets, volumes, spectra, media and dipole models (§4.6)."""

from gradix.objects.environment import Homogeneous, Medium
from gradix.objects.labeling import Isotropic
from gradix.objects.objectset import Emitters, ObjectSet, Spheres
from gradix.objects.spectrum import Spectrum
from gradix.objects.volumes import Voxels

__all__ = [
    "Emitters",
    "Homogeneous",
    "Isotropic",
    "Medium",
    "ObjectSet",
    "Spectrum",
    "Spheres",
    "Voxels",
]
