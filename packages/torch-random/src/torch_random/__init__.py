"""
torch-random: Data sampling from conditional Gibbs measures

Generates synthetic (x, y) datasets from PyTorch models and loss functions.
"""

from .gibbs import gibbs

__version__ = "0.1.0"

__all__ = ["gibbs"]
