"""
Bayesian Neural Network implementation for PyTorch.

Provides PyMC-style interface for Bayesian inference on PyTorch models
using pluggable MCMC sampling backends.
"""

from typing import Any, Dict

import torch
import torch.nn as nn

from ..sampling import DEFAULT_BACKEND_REGISTRY


class BayesianNet:
    """
    Bayesian wrapper for PyTorch models.

    Enables posterior sampling from PyTorch models using various MCMC backends.
    Automatically maps PyTorch loss functions to appropriate likelihood distributions:

    - MSELoss → Gaussian likelihood
    - L1Loss → Laplace likelihood
    - CrossEntropyLoss → Categorical likelihood
    - BCELoss → Bernoulli likelihood

    Example:
        >>> model = nn.Sequential(nn.Linear(2, 1))
        >>> bayes_net = BayesianNet(model)
        >>> samples = bayes_net.sample(
        ...     x_data, y_data,
        ...     num_samples=1000,
        ...     backend='sgld'
        )
    """

    def __init__(
        self,
        model: nn.Module,
        loss_fn: nn.Module = None,
        prior_std: float = 1.0,
        temperature: float = 1.0,
    ):
        """
        Initialize Bayesian network.

        Args:
            model: PyTorch model (e.g., nn.Sequential, custom nn.Module)
            loss_fn: Loss function (determines likelihood type). If None, defaults to MSELoss.
            prior_std: Standard deviation for Gaussian priors on all parameters
            temperature: Temperature for tempered posteriors (1.0 = standard)
        """
        self.model = model
        self.loss_fn = loss_fn or nn.MSELoss()
        self.prior_std = prior_std
        self.temperature = temperature

        # Map loss functions to likelihood distributions
        self._loss_to_likelihood = {
            nn.MSELoss: "gaussian",
            nn.L1Loss: "laplace",
            nn.CrossEntropyLoss: "categorical",
            nn.BCELoss: "bernoulli",
        }

    def logp(self, x_data: torch.Tensor, y_data: torch.Tensor) -> float:
        """
        Compute log probability: log p(θ, y|x) = log p(y|x, θ) + log p(θ)

        Args:
            x_data: Input data
            y_data: Target data

        Returns:
            Log probability (scalar)
        """
        loglik = self._log_likelihood(x_data, y_data)
        logprior = self._log_prior()
        return (loglik + logprior) / self.temperature

    def _log_likelihood(self, x_data: torch.Tensor, y_data: torch.Tensor) -> float:
        """Compute log-likelihood from PyTorch loss function."""
        y_pred = self.model(x_data)
        loss_value = self.loss_fn(y_pred, y_data)

        # Convert loss to log-likelihood (negative log-likelihood)
        return -loss_value.item()

    def _log_prior(self) -> float:
        """Compute log-prior: independent Gaussian priors on all parameters."""
        logprior = 0.0
        for param in self.model.parameters():
            # Gaussian prior: -0.5 * (θ/σ)²  (ignoring normalization constant)
            logprior += -0.5 * torch.sum((param / self.prior_std) ** 2).item()
        return logprior

    def sample(
        self,
        x_data: torch.Tensor,
        y_data: torch.Tensor,
        num_samples: int = 1000,
        backend: str = "sgld",
        warmup: int = None,
        **backend_kwargs,
    ) -> Dict[str, Any]:
        """
        Sample from posterior distribution.

        Args:
            x_data: Input data tensor
            y_data: Target data tensor
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

        # Create log probability function
        def logp_fn():
            return self.logp(x_data, y_data)

        # Get initial parameters
        init_params = torch.cat([p.flatten() for p in self.model.parameters()])

        # Call backend with appropriate arguments
        kwargs = backend_kwargs.copy()
        if warmup is not None:
            kwargs["n_burnin"] = warmup

        samples = backend_fn(
            logp_fn=logp_fn, init_params=init_params, n_samples=num_samples, **kwargs
        )

        return samples

    def get_params_dict(self) -> Dict[str, torch.Tensor]:
        """Extract model parameters as dictionary."""
        return {name: param.clone() for name, param in self.model.named_parameters()}

    def set_params_dict(self, params_dict: Dict[str, torch.Tensor]):
        """Load parameters from dictionary into model."""
        for name, param in self.model.named_parameters():
            param.data = params_dict[name]

    def get_likelihood_type(self) -> str:
        """Get likelihood distribution type based on loss function."""
        loss_type = type(self.loss_fn)
        return self._loss_to_likelihood.get(loss_type, "unknown")
