"""
Gibbs sampling for PyTorch models.

Generates synthetic (x, y) datasets from conditional Gibbs measures
defined by PyTorch models and loss functions.
"""

import torch
import torch.nn as nn


def gibbs(
    model: nn.Module,
    loss_fn: nn.Module,
    n_samples: int,
    input_dist: torch.distributions.Distribution,
    device: str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Sample (x, y) data from conditional Gibbs measure defined by model and loss.

    The loss function determines the conditional distribution p(y|x):
    - MSELoss → Gaussian: y = model(x) + N(0, σ²)
    - L1Loss → Laplace: y = model(x) + Laplace(0, β)
    - CrossEntropyLoss → Categorical: y ~ Categorical(softmax(model(x)))

    Args:
        model: PyTorch model to generate data from (must start with nn.Linear layer)
        loss_fn: Loss function (determines likelihood type and noise scale)
        n_samples: Number of (x, y) pairs to generate
        input_dist: Distribution for sampling inputs
        device: Device to run on

    Returns:
        Tuple of (x_data, y_data) tensors

    Note:
        IMPORTANT LIMITATIONS:

        INPUT: Only supports models that start with nn.Linear layers for vector inputs.
        Conv2D/Conv1D/other input layers are not supported - no image/sequence inputs.

        OUTPUT: Only supports scalar and vector outputs like [batch, 1] or [batch, n_features].
        Multi-dimensional tensor outputs (images, sequences) are not yet implemented.
        Shapes like [batch, channels, height, width] will raise NotImplementedError.

    Examples:
        # Standard Normal inputs
        model = nn.Sequential(nn.Linear(2, 3), nn.Linear(3, 1))
        normal_dist = torch.distributions.Normal(0, 1)
        x, y = gibbs(model, nn.MSELoss(), 1000, normal_dist)

        # Uniform[-2, 2] inputs
        uniform_dist = torch.distributions.Uniform(-2, 2)
        x, y = gibbs(model, nn.MSELoss(), 1000, uniform_dist)
    """
    model = model.to(device)
    # Set to eval mode to disable dropout and fix batch norm statistics
    # for consistent synthetic data generation
    model.eval()

    # Infer input shape from model (excluding batch dimension)
    input_shape = _infer_input_shape(model)

    # Sample inputs x
    x_data = _sample_inputs(n_samples, input_shape, input_dist, device)

    # Generate outputs y based on loss function type
    with torch.no_grad():
        mus = model(x_data)
        y_data = _sample_outputs(mus, loss_fn)

    return x_data, y_data


def _infer_input_shape(model: nn.Module) -> tuple:
    """Infer input shape (excluding batch dimension) from model's first layer."""
    for module in model.modules():
        if isinstance(module, nn.Linear):
            return (module.in_features,)

    raise ValueError(
        "Could not infer input shape from model. Only Linear input layers are currently supported."
    )


def _sample_inputs(
    n_samples: int,
    input_shape: tuple,
    distribution: torch.distributions.Distribution,
    device: str,
) -> torch.Tensor:
    """Sample input data x using the provided distribution."""
    full_shape = (n_samples, *input_shape)

    # Note: Sampling happens on whatever device the distribution parameters are on,
    # then we move to target device. For efficiency, users should create distributions
    # with parameters already on the target device.
    samples = distribution.sample(full_shape)
    return samples.to(device)


def _sample_outputs(mus: torch.Tensor, loss_fn: nn.Module) -> torch.Tensor:
    """Sample outputs using proper multivariate distributions based on loss function."""
    loss_type = type(loss_fn)
    # Extract shape: first dim is number of vectorized samples, rest is output shape
    n_samples_dim, *output_shape = mus.shape

    if loss_type == nn.MSELoss:
        # Always use MultivariateNormal for proper covariance handling
        if len(output_shape) == 1:
            # Vector output: y ~ N(μ, \sigma**2 *I) with \sigma**2 = 1/2 to match MSE
            output_dim = output_shape[0]
            if output_dim == 1:
                # Single scalar output: use univariate Normal for efficiency
                return (
                    torch.distributions.Normal(mus.squeeze(-1), (0.5) ** 0.5)
                    .sample()
                    .unsqueeze(-1)
                )
            else:
                # Multi-dimensional output: use batched MultivariateNormal
                covariance = torch.eye(output_dim, device=mus.device) * 0.5
                # Expand covariance to match batch dimension [n_samples, output_dim, output_dim]
                covariance_batch = covariance.expand(n_samples_dim, -1, -1)
                dist = torch.distributions.MultivariateNormal(mus, covariance_batch)
                return dist.sample()
        else:
            raise NotImplementedError(
                f"MSE with output shape {output_shape} not implemented"
            )

    elif loss_type == nn.L1Loss:
        # Independent Laplace (L1 doesn't have natural multivariate extension)
        return torch.distributions.Laplace(mus, 1.0).sample()

    elif loss_type == nn.CrossEntropyLoss:
        # Categorical: sample from softmax(logits)
        if len(output_shape) == 1:
            # Standard case: [n_samples, num_classes] -> [n_samples]
            probs = torch.softmax(mus, dim=-1)
            return torch.multinomial(probs, num_samples=1).squeeze(-1)
        else:
            # Multi-dimensional case: [n_samples, ...dims..., num_classes]
            # Need to apply multinomial over each ...dims... independently
            original_shape = mus.shape
            # Flatten all but last dimension: [n_samples * prod(dims), num_classes]
            mus_flat = mus.reshape(-1, original_shape[-1])
            probs_flat = torch.softmax(mus_flat, dim=-1)
            samples_flat = torch.multinomial(probs_flat, num_samples=1).squeeze(-1)
            # Reshape back: [n_samples, ...dims...]
            return samples_flat.reshape(original_shape[:-1])

    elif loss_type == nn.BCELoss:
        # Bernoulli: sample from sigmoid(logits)
        probs = torch.sigmoid(mus)
        return torch.bernoulli(probs)

    else:
        # Default to multivariate Gaussian (same as MSE)
        if len(output_shape) == 1 and output_shape[0] > 1:
            output_dim = output_shape[0]
            covariance = torch.eye(output_dim, device=mus.device) * 0.5
            dist = torch.distributions.MultivariateNormal(mus, covariance)
            return dist.sample()
        else:
            return torch.distributions.Normal(mus, (0.5) ** 0.5).sample()
