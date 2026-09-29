"""Media (``gx.env``): the refractive index around the sample, and turbid slabs."""

from gradix.objects.environment import Homogeneous, LayeredMedium, Medium
from gradix.objects.environment import water_on_coverslip as WaterOnCoverslip
from gradix.objects.turbid import TurbidSlab

__all__ = ["Homogeneous", "LayeredMedium", "Medium", "TurbidSlab", "WaterOnCoverslip"]
