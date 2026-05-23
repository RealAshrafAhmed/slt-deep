"""
torch_bdn: Bayesian Deep Networks for PyTorch

Provides typed MCMC sampling for PyTorch models.
"""

from .bn.bayesian_net import BayesianNet
from .sampling import (
    HMC,
    NUTS,
    SGHMC,
    SGLD,
    ChainResult,
    InitStrategy,
    MultiChainResult,
    Perturb,
    Prior,
    Sampler,
    SamplerConfig,
)

__version__ = "0.1.0"

__all__ = [
    "HMC",
    "NUTS",
    "SGHMC",
    "SGLD",
    "BayesianNet",
    "ChainResult",
    "InitStrategy",
    "MultiChainResult",
    "Perturb",
    "Prior",
    "Sampler",
    "SamplerConfig",
]
