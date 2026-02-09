"""
SGLD (Stochastic Gradient Langevin Dynamics) sampler implementation.

Implements SGLD using pure PyTorch operations for Bayesian inference.
"""

from collections.abc import Callable
from typing import Any

import torch


def _suggest_batch_size(n_data: int, lr: float) -> int:
    """
    Suggest adaptive batch size based on dataset size and learning rate.

    Args:
        n_data: Number of data points
        lr: Learning rate

    Returns:
        Suggested batch size (power of 2, between 32 and 512)
    """
    # Base batch size from dataset size
    if n_data < 1000:
        base_size = 32
    elif n_data < 5000:
        base_size = 64
    elif n_data < 20000:
        base_size = 128
    elif n_data < 100000:
        base_size = 256
    else:
        base_size = 512

    # Adjust based on learning rate
    # Higher lr → larger batches for stability
    # Lower lr → smaller batches for more exploration
    if lr > 0.1:
        size_multiplier = 2  # Use larger batches for high lr
    elif lr < 0.001:
        size_multiplier = 0.5  # Use smaller batches for low lr
    else:
        size_multiplier = 1

    # Apply multiplier and constrain to valid range
    suggested_size = int(base_size * size_multiplier)

    # Clamp to [32, 512] and round to nearest power of 2
    suggested_size = max(32, min(512, suggested_size))

    # Round to nearest power of 2
    import math

    power = round(math.log2(suggested_size))
    return 2**power


def sgld(
    set_params_fn: Callable[[torch.Tensor], None],
    log_likelihood_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    log_prior_fn: Callable[[torch.Tensor], torch.Tensor],
    x_data: torch.Tensor,
    y_data: torch.Tensor,
    init_params: torch.Tensor,
    n_samples: int = 1000,
    batch_size: int | None = None,  # Now optional with adaptive selection
    lr: float = 0.01,
    noise_scale: float | None = None,
    n_burnin: int = 1000,
    thin: int = 1,
    **kwargs,
) -> dict[str, Any]:
    """
    Generate samples using Stochastic Gradient Langevin Dynamics (SGLD).

    Implements proper SGLD with mini-batch gradient computation for both likelihood and prior.
    SGLD update: θ_new = θ + (lr/2) * (∇log p_likelihood(θ|batch) + ∇log p_prior(θ)) + η
    where η ~ N(0, lr * I) and gradients are computed on mini-batches

    Args:
        set_params_fn: Function that sets model parameters from flattened tensor
        log_likelihood_fn: Function that takes (x_batch, y_batch) and returns log likelihood
        x_data: Input data tensor
        y_data: Target data tensor
        init_params: Initial parameter values
        n_samples: Number of samples to collect
        batch_size: Size of mini-batches for stochastic gradients (if None, auto-selected)
        lr: Learning rate / step size
        noise_scale: Noise scaling factor (if None, computed from lr)
        n_burnin: Number of burnin samples to discard
        thin: Thinning factor (keep every thin-th sample)
        **kwargs: Ignored for compatibility

    Returns:
        Dictionary with samples and diagnostics
    """
    # Auto-select batch size if not provided
    batch_size_was_auto = batch_size is None
    if batch_size is None:
        batch_size = _suggest_batch_size(x_data.shape[0], lr)

    # Set noise scale based on learning rate if not provided
    if noise_scale is None:
        noise_scale = (2 * lr) ** 0.5

    # Type checker assistance: noise_scale is guaranteed to be float here
    assert noise_scale is not None  # For type checker

    # Initialize parameter tensor (requires gradient)
    params = init_params.clone().detach().requires_grad_(True)

    samples = []
    logp_vals = []

    total_steps = n_burnin + n_samples * thin
    n_data = x_data.shape[0]
    n_batches_per_epoch = (n_data + batch_size - 1) // batch_size  # Ceiling division

    for step in range(total_steps):
        # Determine which batch within epoch we're in
        batch_idx = step % n_batches_per_epoch

        # Shuffle data indices for each new epoch (when batch_idx == 0)
        if batch_idx == 0:
            indices = torch.randperm(n_data)

        # Get current mini-batch
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, n_data)
        batch_indices = indices[start_idx:end_idx]
        x_batch = x_data[batch_indices]
        y_batch = y_data[batch_indices]

        # Zero gradients
        if params.grad is not None:
            params.grad.zero_()

        # Set current parameters in the model
        set_params_fn(params)

        # Compute log likelihood using model at current parameters
        log_likelihood = log_likelihood_fn(x_batch, y_batch)

        # Compute log prior directly from params tensor (preserves gradient connection)
        log_prior = log_prior_fn(params)

        # Compute gradients with respect to the params tensor directly
        likelihood_grad = torch.autograd.grad(
            outputs=log_likelihood,
            inputs=params,
            retain_graph=True,
            create_graph=False,
            allow_unused=True,
        )[0]

        prior_grad = torch.autograd.grad(
            outputs=log_prior,
            inputs=params,
            retain_graph=False,
            create_graph=False,
            allow_unused=True,
        )[0]

        # Handle case where some parameters might not be used (gradients could be None)
        if likelihood_grad is None:
            likelihood_grad = torch.zeros_like(params)
        if prior_grad is None:
            prior_grad = torch.zeros_like(params)

        # Scale likelihood gradient by dataset size / batch size for unbiased estimate
        likelihood_grad = likelihood_grad * (n_data / len(x_batch))

        # Clip gradients to prevent explosion
        total_grad = likelihood_grad + prior_grad
        grad_norm = torch.norm(total_grad)
        max_grad_norm = 10.0  # Gradient clipping threshold

        if grad_norm > max_grad_norm:
            total_grad = total_grad * (max_grad_norm / grad_norm)
            likelihood_grad = likelihood_grad * (max_grad_norm / grad_norm)
            prior_grad = prior_grad * (max_grad_norm / grad_norm)

        # Total log probability for recording (approximate with current batch)
        total_logp = log_likelihood + log_prior

        # SGLD update: θ_new = θ + (lr/2) * ∇log p(θ) + η
        with torch.no_grad():
            # Gradient step (stochastic gradient from mini-batch)
            gradient_step = (lr / 2) * (likelihood_grad + prior_grad)

            # Noise step
            noise_step = noise_scale * torch.randn_like(params)

            # Update parameters
            params += gradient_step + noise_step

            # Clip parameters to prevent extreme values
            params.clamp_(-50.0, 50.0)  # Prevent parameters from exploding

        # Collect samples after burnin
        if step >= n_burnin and (step - n_burnin) % thin == 0:
            samples.append(params.clone().detach())
            logp_vals.append(total_logp.item())

    return {
        "parameters": samples,
        "log_probabilities": logp_vals,
        "backend": "sgld",
        "diagnostics": {
            "lr": lr,
            "noise_scale": noise_scale,
            "batch_size": batch_size,
            "batch_size_auto_selected": batch_size_was_auto,
            "n_data": n_data,
            "n_batches_per_epoch": n_batches_per_epoch,
            "n_burnin": n_burnin,
            "thin": thin,
            "total_steps": total_steps,
        },
    }
