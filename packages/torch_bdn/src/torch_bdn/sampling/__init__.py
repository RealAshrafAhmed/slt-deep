"""
Sampling module for torch_bdn.

Provides typed sampler configuration and MCMC backends.

Usage::

    from torch_bdn.sampling import Sampler, NUTS, SGLD

    sampler = Sampler(bn, x_data, y_data)
    result = sampler.sample(NUTS(n_warmup=500), n_samples=2000)
"""

from .configs import (
    HMC,
    NUTS,
    RMHMC,
    SGHMC,
    SGLD,
    InitStrategy,
    Perturb,
    Prior,
    SamplerConfig,
)
from .sampler import ChainResult, MultiChainResult, Sampler

__all__ = [
    "HMC",
    "NUTS",
    "SGHMC",
    "SGLD",
    "ChainResult",
    "InitStrategy",
    "MultiChainResult",
    "Perturb",
    "Prior",
    "Sampler",
    "SamplerConfig",
]
