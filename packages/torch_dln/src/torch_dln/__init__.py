"""torch_dln: Deep Linear Network construction utilities for PyTorch."""

from .models import zoo
from .models.constructors import DeepLinearNetwork, InitMethod, bndln, dln, pdln, wdln

__version__ = "0.1.0"

__all__ = ["DeepLinearNetwork", "bndln", "dln", "pdln", "wdln", "zoo"]
