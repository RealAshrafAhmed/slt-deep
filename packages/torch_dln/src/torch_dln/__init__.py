"""torch_dln: Deep Linear Network construction utilities for PyTorch."""

from .models import zoo
from .models.constructors import DLN, BottleneckDLN, PyramidDLN, WideDLN

__version__ = "0.1.0"

__all__ = ["DLN", "WideDLN", "BottleneckDLN", "PyramidDLN", "zoo"]
