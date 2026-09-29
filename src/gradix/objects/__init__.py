"""L2 data objects: object sets, volumes, spectra, media and dipole models (§4.6)."""

from gradix.objects.environment import Homogeneous, Medium
from gradix.objects.labeling import Isotropic, Labeling
from gradix.objects.objectset import Emitters, ObjectSet, Solid, Spheres
from gradix.objects.shapes import Boxes, Capsules, Cylinders, Ellipsoids, Gaussians
from gradix.objects.spectrum import Spectrum
from gradix.objects.volumes import Voxels

__all__ = [
    "Boxes",
    "Capsules",
    "Cylinders",
    "Ellipsoids",
    "Emitters",
    "Gaussians",
    "Homogeneous",
    "Isotropic",
    "Labeling",
    "Medium",
    "ObjectSet",
    "Solid",
    "Spectrum",
    "Spheres",
    "Voxels",
]
