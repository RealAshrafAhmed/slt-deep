"""
Samplers: MCMC algorithms for Deep Linear Network posterior sampling.

This package provides implementations of key sampling algorithms for Bayesian
analysis of Deep Linear Networks (DLNs) in Singular Learning Theory research.

Main Components:
- SGLDSampler: Stochastic Gradient Langevin Dynamics
- HMCSampler: Hamiltonian Monte Carlo (via blackjax)
- HybridSampler: Two-phase SGD→Langevin sampling
- RegularDLN: Regular (non-singular) DLN models for testing
"""

from .hmc import HMCSampler
from .hybrid import HybridSampler
from .sgld import SGLDSampler
from .toy_dln import DLNState, RegularDLN, create_minimal_dln, create_simple_dln

__all__ = [
    "SGLDSampler",
    "HMCSampler",
    "HybridSampler",
    "RegularDLN",
    "DLNState",
    "create_simple_dln",
    "create_minimal_dln",
]

__version__ = "0.1.0"
