"""
MCMC Sampler implementation for BayesianNet models.
"""

from typing import Any

import torch

from ..bn.bayesian_net import BayesianNet
from . import DEFAULT_BACKEND_REGISTRY


class Sampler:
    """MCMC sampler for BayesianNet models."""

    def __init__(
        self,
        bayes_net: BayesianNet,
        x: torch.Tensor,
        y: torch.Tensor,
    ):
        """Initialize sampler with a BayesianNet model.

        Args:
            bayes_net: BayesianNet instance defining the posterior
            x: Input data tensor
            y: Target data tensor
        """
        self.bayes_net = bayes_net
        self.x = x
        self.y = y

    def sample(
        self,
        num_samples: int = 1000,
        backend: str = "sgld",
        warmup: int | None = None,
        **backend_kwargs,
    ) -> dict[str, Any]:
        """
        Sample from posterior distribution.

        Args:
            num_samples: Number of posterior samples to collect
            backend: Sampling backend ('sgld', 'hmc', etc.)
            warmup: Number of warmup/burnin samples (backend-dependent)
            **backend_kwargs: Additional arguments passed to backend

        Returns:
            Dictionary containing posterior samples and diagnostics
        """
        # Get backend from registry
        if backend not in DEFAULT_BACKEND_REGISTRY:
            available = list(DEFAULT_BACKEND_REGISTRY.keys())
            raise ValueError(
                f"Backend '{backend}' not available. Available: {available}"
            )

        backend_fn = DEFAULT_BACKEND_REGISTRY[backend]

        # Get initial parameters from BayesianNet
        init_params: torch.Tensor = self.bayes_net.get_parameters(flat=True)

        print(f"Working with parameters of shape {init_params.size()}")
        print(f"Data shapes - x: {self.x.shape}, y: {self.y.shape}")

        # Call backend with direct method references
        kwargs = backend_kwargs.copy()
        if warmup is not None:
            kwargs["n_burnin"] = warmup

        samples = backend_fn(
            set_params_fn=self.bayes_net.set_parameters,
            log_likelihood_fn=self.bayes_net.log_likelihood,
            log_prior_fn=self.bayes_net.log_prior_from_params,
            x_data=self.x,
            y_data=self.y,
            init_params=init_params,
            n_samples=num_samples,
            **kwargs,
        )

        return samples
