"""
Gibbs sampling for PyTorch models.

Generates synthetic (x, y) datasets from conditional Gibbs measures
defined by PyTorch models and loss functions.
"""

from typing import Literal, Tuple

import torch
import torch.nn as nn


def gibbs(
    model: nn.Module,
    loss_fn: nn.Module,
    n_samples: int,
    input_dim: int = None,
    input_dist: Literal["gaussian", "uniform"] = "gaussian",
    noise_std: float = 0.1,
    device: str = "cpu",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Sample (x, y) data from conditional Gibbs measure defined by model and loss.

    The loss function determines the conditional distribution p(y|x):
    - MSELoss → Gaussian: y = model(x) + N(0, σ²)
    - L1Loss → Laplace: y = model(x) + Laplace(0, β)
    - CrossEntropyLoss → Categorical: y ~ Categorical(softmax(model(x)))

    Args:
        model: PyTorch model to generate data from
        loss_fn: Loss function (determines likelihood type)
        n_samples: Number of (x, y) pairs to generate
        input_dim: Input dimension (inferred from model if None)
        input_dist: Distribution for sampling x ("gaussian" or "uniform")
        noise_std: Noise parameter (σ for Gaussian, β for Laplace)
        device: Device to run on

    Returns:
        Tuple of (x_data, y_data) tensors

    Example:
        model = nn.Sequential(nn.Linear(2, 3), nn.Linear(3, 1))
        x, y = gibbs(model, nn.MSELoss(), n_samples=1000)
    """
    model = model.to(device)
    model.eval()

    # Infer input dimension if not provided
    if input_dim is None:
        input_dim = _infer_input_dim(model)

    # Sample inputs x
    x_data = _sample_inputs(n_samples, input_dim, input_dist, device)

    # Generate outputs y based on loss function type
    with torch.no_grad():
        predictions = model(x_data)
        y_data = _add_noise_by_loss_type(predictions, loss_fn, noise_std)

    return x_data, y_data


def _infer_input_dim(model: nn.Module) -> int:
    """Try to infer input dimension from first layer."""
    for module in model.modules():
        if isinstance(module, nn.Linear):
            return module.in_features
        elif isinstance(module, nn.Conv2d):
            # For conv networks, might need different logic
            raise NotImplementedError("Conv2d input inference not implemented")

    raise ValueError("Could not infer input dimension from model")


def _sample_inputs(
    n_samples: int, input_dim: int, input_dist: str, device: str
) -> torch.Tensor:
    """Sample input data x."""
    if input_dist == "gaussian":
        return torch.randn(n_samples, input_dim, device=device)
    elif input_dist == "uniform":
        return torch.rand(n_samples, input_dim, device=device) * 2 - 1  # Uniform[-1, 1]
    else:
        raise ValueError(f"Unknown input distribution: {input_dist}")


def _add_noise_by_loss_type(
    predictions: torch.Tensor, loss_fn: nn.Module, noise_std: float
) -> torch.Tensor:
    """Add noise to predictions based on loss function type."""
    loss_type = type(loss_fn)

    if loss_type == nn.MSELoss:
        # Gaussian noise: y = μ + N(0, σ²)
        noise = torch.randn_like(predictions) * noise_std
        return predictions + noise

    elif loss_type == nn.L1Loss:
        # Laplace noise: y = μ + Laplace(0, β)
        # Laplace distribution using inverse CDF
        uniform = torch.rand_like(predictions)
        # Laplace inverse CDF: F^(-1)(u) = -β*sign(u-0.5)*log(1-2|u-0.5|)
        sign = torch.sign(uniform - 0.5)
        noise = -noise_std * sign * torch.log(1 - 2 * torch.abs(uniform - 0.5))
        return predictions + noise

    elif loss_type == nn.CrossEntropyLoss:
        # Categorical: sample from softmax(predictions)
        probs = torch.softmax(predictions, dim=-1)
        return torch.multinomial(probs, num_samples=1).squeeze(-1)

    elif loss_type == nn.BCELoss:
        # Bernoulli: sample from sigmoid(predictions)
        probs = torch.sigmoid(predictions)
        return torch.bernoulli(probs)

    else:
        # Default to Gaussian if unknown loss type
        noise = torch.randn_like(predictions) * noise_std
        return predictions + noise
