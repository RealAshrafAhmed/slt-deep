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
from torch.func import functional_call, hessian
from torch.nn.utils import parameters_to_vector, vector_to_parameters


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


def unflatten(named_params, params_vec):
    param_dict = dict(named_params)
    param_names = list(param_dict.keys())
    param_shapes = [p.shape for p in param_dict.values()]
    param_numels = [p.numel() for p in param_dict.values()]

    out, offset = {}, 0
    for name, shape, n in zip(param_names, param_shapes, param_numels):
        out[name] = params_vec[offset : offset + n].view(shape)
        offset += n
    return out


def make_fast_unflatten(model: nn.Module):
    """Cache parameter metadata for fast repeated unflattening."""
    names = []
    shapes = []
    numels = []
    offsets = [0]
    for name, p in model.named_parameters():
        names.append(name)
        shapes.append(p.shape)
        numels.append(p.numel())
        offsets.append(offsets[-1] + p.numel())

    def fast_unflatten(params_vec):
        return {
            n: params_vec[offsets[i] : offsets[i + 1]].view(s)
            for i, (n, s) in enumerate(zip(names, shapes))
        }

    return fast_unflatten


def mean_loss_factory(model, loss_fn):
    def fun(params_vec, x, y):
        params = unflatten(named_params=model.named_parameters(), params_vec=params_vec)
        pred = functional_call(model, params, x)
        return loss_fn(pred, y)  # make sure this uses mean reduction, not sum

    return fun


def approx_hessian(
    model, loss_fn, x: torch.Tensor, y: torch.Tensor, chunk_size=32
) -> torch.Tensor:
    """
    Compute Hessian of negative log-posterior with respect to model parameters.

    This is useful for second-order MCMC methods like Riemannian HMC or Laplace approximation.

    Args:
        model: PyTorch model
        loss_fn: Loss function
        x: Input data tensor
        y: Target data tensor
        chunk_size: Size of data chunks for memory-efficient computation

    Returns:
        Hessian matrix (tensor of shape [num_params, num_params])
    """

    params_vec = parameters_to_vector(model.parameters()).detach()

    if chunk_size is not None and x.shape[0] > chunk_size:
        # Compute Hessian in chunks to save memory
        hessians = []
        for i in range(0, x.shape[0], chunk_size):
            x_chunk = x[i : i + chunk_size]
            y_chunk = y[i : i + chunk_size]

            hessian_fn = hessian(mean_loss_factory(model=model, loss_fn=loss_fn))
            h_chunk = hessian_fn(params_vec, x_chunk, y_chunk)
            hessians.append(h_chunk)

        result = torch.stack(hessians).mean(dim=0)
    else:
        # Compute on full data
        hessian_fn = hessian(mean_loss_factory(model=model, loss_fn=loss_fn))
        result = hessian_fn(params_vec, x, y)

    assert isinstance(result, torch.Tensor)
    return result


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
        compile: bool = False,
    ):
        """
        Initialize Bayesian network.

        Args:
            model: PyTorch model (e.g., nn.Sequential, custom nn.Module)
            loss_fn: Loss function (determines likelihood type)
            prior_logp: Callable that takes flattened parameter tensor and returns log probability
            temperature: Temperature for tempered posteriors (1.0 = standard)
            compile: If True, use torch.compile on the forward pass for faster
                     repeated evaluations (e.g. during MCMC sampling).
        """
        self.model = model
        self.loss_fn = loss_fn
        self.prior_logp = prior_logp
        self.temperature = temperature
        self._fast_unflatten = make_fast_unflatten(model)

        # Compile the forward+loss kernel for repeated calls (MCMC hot path)
        def _forward_and_loss(params_dict, x, y):
            y_pred = functional_call(model, params_dict, x)
            return -loss_fn(y_pred, y)

        if compile:
            self._forward_and_loss = torch.compile(_forward_and_loss)
        else:
            self._forward_and_loss = _forward_and_loss

    # ── pickle / deepcopy support (torch.compile objects aren't picklable) ─

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop("_forward_and_loss", None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        model = self.model
        loss_fn = self.loss_fn

        def _forward_and_loss(params_dict, x, y):
            y_pred = functional_call(model, params_dict, x)
            return -loss_fn(y_pred, y)

        self._forward_and_loss = _forward_and_loss

    # ── device helpers ────────────────────────────────────────────────────

    @property
    def device(self) -> torch.device:
        """Device of the first model parameter (all should be on same device)."""
        return next(self.model.parameters()).device

    def to(self, device: torch.device | str) -> "BayesianNet":
        """Move model to *device* and return self for chaining."""
        self.model = self.model.to(device)
        return self

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

    def log_likelihood_from_flat(
        self, flat_params: torch.Tensor, x: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        """
        Differentiable log-likelihood that preserves the autograd graph
        from *flat_params* through to the returned scalar.

        Uses ``torch.func.functional_call`` so that gradients flow
        back through ``flat_params`` — unlike ``set_parameters`` followed
        by ``log_likelihood``, which severs the graph.
        """
        params_dict = self._fast_unflatten(flat_params)
        return self._forward_and_loss(params_dict, x, y)

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

        Uses ``torch.nn.utils.vector_to_parameters`` for a single
        vectorised C++ call instead of a Python-level loop.

        Args:
            flattened_params: Flattened tensor containing all model parameters
        """
        vector_to_parameters(flattened_params, self.model.parameters())

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
