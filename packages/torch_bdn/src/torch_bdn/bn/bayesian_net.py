"""
Bayesian Neural Network implementation for PyTorch.

Provides PyMC-style interface for Bayesian inference on PyTorch models
using pluggable MCMC sampling backends.
"""

from collections.abc import Callable
from enum import Enum
from typing import overload

import torch
import torch.nn as nn


class LikelihoodType(Enum):
    """Enumeration of supported likelihood distribution types."""

    GAUSSIAN = (nn.MSELoss,)
    LAPLACE = (nn.L1Loss,)
    CATEGORICAL = (nn.CrossEntropyLoss,)
    BERNOULLI = (nn.BCELoss,)

    def __init__(self, loss_class):
        self.loss_class = loss_class

    @classmethod
    def from_loss(cls, loss_fn: nn.Module) -> "LikelihoodType":
        """Map PyTorch loss function to corresponding likelihood type.

        Args:
            loss_fn: PyTorch loss function instance

        Returns:
            LikelihoodType enum value for the loss function

        Raises:
            ValueError: If loss function is not supported
        """
        loss_type = type(loss_fn)
        for likelihood in cls:
            if likelihood.loss_class == loss_type:
                return likelihood

        supported_losses = [lik.loss_class.__name__ for lik in cls]
        raise ValueError(
            f"Unsupported loss function: {loss_type.__name__}. "
            f"Supported loss functions: {supported_losses}"
        )


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
        >>> import torch
        >>> model = nn.Sequential(nn.Linear(2, 1))
        >>> loss_fn = nn.MSELoss()
        >>> def prior_fn(params): return torch.distributions.Normal(0, 1).log_prob(params).sum()
        >>> bayes_net = BayesianNet(model, loss_fn, prior_fn)
        >>> from ..sampling.sampler import Sampler
        >>> sampler = Sampler(bayes_net)
        >>> samples = sampler.sample(
        ...     x_data, y_data,
        ...     num_samples=1000,
        ...     backend='sgld'
        )
    """

    def __init__(
        self,
        model: nn.Module,
        loss_fn: nn.Module,
        prior_logp: Callable[[torch.Tensor], torch.Tensor],
        temperature: float = 1.0,
    ):
        """
        Initialize Bayesian network.

        Args:
            model: PyTorch model (e.g., nn.Sequential, custom nn.Module)
            loss_fn: Loss function (determines likelihood type)
            prior_logp: Callable that takes flattened parameter tensor and returns log probability
            temperature: Temperature for tempered posteriors (1.0 = standard)
        """
        self.model = model
        self.loss_fn = loss_fn
        self.prior_logp = prior_logp
        self.temperature = temperature

    def logp(self, x_data: torch.Tensor, y_data: torch.Tensor) -> torch.Tensor:
        """
        Compute log probability: log p(θ, y|x) = log p(y|x, θ) + log p(θ)

        Args:
            x_data: Input data
            y_data: Target data

        Returns:
            Log probability (tensor, supports batching)
        """
        loglik = self.log_likelihood(x_data, y_data)
        logprior = self.log_prior_from_params(self.get_parameters(flat=True))
        return (loglik + logprior) / self.temperature

    def log_likelihood(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute log-likelihood: log p(y|x, θ)

        Similar to PyMC's likelihood evaluation, this computes the probability
        of observing the data given the current model parameters.

        Args:
            x: Input data tensor
            y: Target/observed data tensor

        Returns:
            Log-likelihood (tensor, supports batching)
        """
        y_pred = self.model(x)
        loss_value = self.loss_fn(y_pred, y)

        # Convert loss to log-likelihood (negative log-likelihood)
        return -loss_value

    def log_prior_from_params(self, flat_params: torch.Tensor) -> torch.Tensor:
        """
        Compute log-prior: log p(θ) from given parameter tensor.

        This version allows computing the prior directly from a parameter tensor
        without breaking gradient connections, useful for MCMC sampling.

        Args:
            flat_params: Flattened parameter tensor

        Returns:
            Log-prior probability (tensor scalar)
        """
        return self.prior_logp(flat_params)

    def get_params_dict(self) -> dict[str, torch.Tensor]:
        """Extract model parameters as dictionary."""
        return {name: param.clone() for name, param in self.model.named_parameters()}

    def set_params_dict(self, params_dict: dict[str, torch.Tensor]):
        """Load parameters from dictionary into model."""
        for name, param in self.model.named_parameters():
            param.data = params_dict[name]

    def set_parameters(self, flattened_params: torch.Tensor) -> None:
        """
        Set model parameters from flattened parameter vector.

        This is similar to how optimizers update parameters in PyTorch,
        or how PyMC updates model parameters during sampling.

        Args:
            flattened_params: Flattened tensor containing all model parameters
        """
        param_idx = 0
        for param in self.model.parameters():
            param_size = param.numel()
            param.data = flattened_params[param_idx : param_idx + param_size].view(
                param.shape
            )
            param_idx += param_size

    @overload
    def get_parameters(self, flat: bool = True) -> torch.Tensor: ...

    @overload
    def get_parameters(self, flat: bool = False) -> dict[str, torch.Tensor]: ...

    def get_parameters(
        self, flat: bool = True
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        """
        Get current model parameters.

        Args:
            flat: If True, return flattened tensor; if False, return parameter dict

        Returns:
            Flattened tensor containing all model parameters (if flat=True)
            or dictionary mapping parameter names to tensors (if flat=False)
        """
        if flat:
            return torch.cat([p.flatten() for p in self.model.parameters()])
        else:
            return {
                name: param.clone() for name, param in self.model.named_parameters()
            }
