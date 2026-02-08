"""
torch_bdn: Bayesian Deep Networks for PyTorch

Provides PyMC-style Bayesian inference for PyTorch models with pluggable
MCMC sampling backends.
"""

from .bn.bayesian_net import BayesianNet
from .sampling import list_backends, register_backend

__version__ = "0.1.0"

__all__ = ["BayesianNet", "register_backend", "list_backends"]
