"""
Sampling backends registry for torch_bdn.

Provides a plugin system for MCMC sampling methods, similar to PyMC's backend architecture.
Allows users to register custom sampling backends or use built-in ones.
"""

import importlib
from typing import Any, Protocol

from .sampler import Sampler


class SamplingBackend(Protocol):
    """Protocol that all sampling backends must implement."""

    def sample(self, logp_fn, init_params: Any, n_samples: int, **kwargs) -> Any:
        """
        Sample from posterior using this backend.

        Args:
            logp_fn: Log probability function
            init_params: Initial parameter values
            n_samples: Number of samples to generate
            **kwargs: Backend-specific parameters

        Returns:
            Samples from the posterior
        """
        ...


# Global registry of available backends
_BACKENDS: dict[str, str] = {
    "sgld": "torch_bdn.sampling.backends.sgld.sgld",
    "hmc": "torch_bdn.sampling.backends.hmc.hmc",
}

# Create a registry that directly maps to functions (like the BayesianNet expects)
DEFAULT_BACKEND_REGISTRY = {}


def _initialize_default_registry():
    """Initialize the default registry with function references."""
    from .backends.hmc import hmc
    from .backends.sgld import sgld

    DEFAULT_BACKEND_REGISTRY.update(
        {
            "sgld": sgld,
            "hmc": hmc,
        }
    )


# Initialize on module import
_initialize_default_registry()


def register_backend(name: str, backend_class_path: str) -> None:
    """
    Register a custom sampling backend.

    Args:
        name: Name to identify the backend (e.g., "jax-hmc")
        backend_class_path: Full import path to backend class

    Example:
        register_backend("custom-sampler", "my_package.samplers.CustomSampler")
    """
    _BACKENDS[name] = backend_class_path


def get_backend(name: str) -> type[SamplingBackend]:
    """
    Get a sampling backend by name.

    Args:
        name: Backend name (e.g., "torch-hmc")

    Returns:
        Backend class

    Raises:
        KeyError: If backend not found
        ImportError: If backend class cannot be imported
    """
    if name not in _BACKENDS:
        available = list(_BACKENDS.keys())
        raise KeyError(f"Backend '{name}' not found. Available backends: {available}")

    class_path = _BACKENDS[name]
    module_path, class_name = class_path.rsplit(".", 1)

    try:
        module = importlib.import_module(module_path)
        backend_class = getattr(module, class_name)
        return backend_class
    except (ImportError, AttributeError) as e:
        raise ImportError(
            f"Could not import backend '{name}' from '{class_path}'"
        ) from e


def list_backends() -> dict[str, str]:
    """List all registered backends."""
    return _BACKENDS.copy()


__all__ = [
    "DEFAULT_BACKEND_REGISTRY",
    "Sampler",
    "SamplingBackend",
    "get_backend",
    "list_backends",
    "register_backend",
]
