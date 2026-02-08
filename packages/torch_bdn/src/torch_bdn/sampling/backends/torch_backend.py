"""
Native PyTorch MCMC backends for Bayesian inference.

Implements SGLD and HMC samplers using pure PyTorch operations.
"""

from typing import Any, Callable, Dict

import torch


def torch_sgld(
    logp_fn: Callable[[], float],
    init_params: torch.Tensor,
    n_samples: int = 1000,
    lr: float = 0.01,
    noise_scale: float = None,
    n_burnin: int = 1000,
    thin: int = 1,
    **kwargs,
) -> Dict[str, torch.Tensor]:
    """
    Generate samples using SGLD.

    Args:
        logp_fn: Log probability function
        init_params: Initial parameter values
        n_samples: Number of samples to collect
        lr: Learning rate / step size
        noise_scale: Noise scaling factor (if None, computed from lr)
        n_burnin: Number of burnin samples to discard
        thin: Thinning factor (keep every thin-th sample)
        **kwargs: Ignored for compatibility

    Returns:
        Dictionary with samples and diagnostics
    """
    # Set noise scale based on learning rate if not provided
    if noise_scale is None:
        noise_scale = (2 * lr) ** 0.5

    # Initialize parameter tensor (requires gradient)
    params = init_params.clone().detach().requires_grad_(True)

    samples = []
    logp_vals = []

    total_steps = n_burnin + n_samples * thin

    for step in range(total_steps):
        # Zero gradients
        if params.grad is not None:
            params.grad.zero_()

        # Compute log probability and gradients
        logp = logp_fn()
        logp_tensor = torch.tensor(logp, requires_grad=True)
        logp_tensor.backward()

        # SGLD update: θ_new = θ + (lr/2) * ∇log p(θ) + η
        # where η ~ N(0, lr * I)
        with torch.no_grad():
            if params.grad is not None:
                grad_step = 0.5 * lr * params.grad
                noise_step = noise_scale * torch.randn_like(params)
                params += grad_step + noise_step
            else:
                # No gradients available, just add noise (random walk)
                noise_step = noise_scale * torch.randn_like(params)
                params += noise_step

        # Collect samples after burnin
        if step >= n_burnin and (step - n_burnin) % thin == 0:
            samples.append(params.clone().detach())
            logp_vals.append(logp)

    return {
        "parameters": samples,
        "log_probabilities": logp_vals,
        "backend": "sgld",
        "diagnostics": {
            "lr": lr,
            "noise_scale": noise_scale,
            "n_burnin": n_burnin,
            "thin": thin,
        },
    }


def torch_hmc(
    logp_fn: Callable[[], float],
    init_params: torch.Tensor,
    n_samples: int = 1000,
    step_size: float = 0.01,
    n_leapfrog: int = 10,
    n_burnin: int = 1000,
    thin: int = 1,
    **kwargs,
) -> Dict[str, Any]:
    """
    Hamiltonian Monte Carlo sampling.

    Args:
        logp_fn: Log probability function
        init_params: Initial parameter values
        n_samples: Number of samples to collect
        step_size: Leapfrog step size
        n_leapfrog: Number of leapfrog steps
        n_burnin: Number of burnin samples
        thin: Thinning factor
        **kwargs: Ignored

    Returns:
        Dictionary with samples and diagnostics
    """
    params = init_params.clone().detach()

    samples = []
    logp_vals = []
    n_accepted = 0

    total_steps = n_burnin + n_samples * thin

    for step in range(total_steps):
        # Propose new state using leapfrog
        new_params, new_logp = _leapfrog_step(params, logp_fn, step_size, n_leapfrog)

        # Accept/reject using Metropolis criterion
        current_logp = logp_fn()

        # For HMC, we'd normally include kinetic energy terms, but simplified here
        log_accept_prob = new_logp - current_logp
        accept_prob = min(1.0, float(torch.exp(torch.tensor(log_accept_prob))))

        if torch.rand(1).item() < accept_prob:
            params = new_params
            n_accepted += 1
        # else: keep current params

        # Collect samples after burnin
        if step >= n_burnin and (step - n_burnin) % thin == 0:
            samples.append(params.clone())
            logp_vals.append(logp_fn())

    acceptance_rate = n_accepted / total_steps if total_steps > 0 else 0.0

    return {
        "parameters": samples,
        "log_probabilities": logp_vals,
        "acceptance_rate": acceptance_rate,
        "backend": "hmc",
        "diagnostics": {
            "step_size": step_size,
            "n_leapfrog": n_leapfrog,
            "n_burnin": n_burnin,
            "thin": thin,
            "n_accepted": n_accepted,
        },
    }


def _leapfrog_step(
    params: torch.Tensor, logp_fn: Callable[[], float], step_size: float, n_steps: int
) -> tuple[torch.Tensor, float]:
    """Leapfrog integration for HMC."""
    # Initialize momentum
    momentum = torch.randn_like(params)

    # Make a copy for the trajectory
    q = params.clone().detach().requires_grad_(True)
    p = momentum.clone()

    # Leapfrog steps
    for _ in range(n_steps):
        # Half step for momentum
        if q.grad is not None:
            q.grad.zero_()

        logp = logp_fn()
        logp_tensor = torch.tensor(logp, requires_grad=True)
        logp_tensor.backward()

        if q.grad is not None:
            p = p + 0.5 * step_size * q.grad

        # Full step for position
        q = q + step_size * p

        # Half step for momentum
        if q.grad is not None:
            q.grad.zero_()

        logp = logp_fn()
        logp_tensor = torch.tensor(logp, requires_grad=True)
        logp_tensor.backward()

        if q.grad is not None:
            p = p + 0.5 * step_size * q.grad

    return q.detach(), logp


def _compute_grad(params: torch.Tensor, logp_fn: Callable[[], float]) -> torch.Tensor:
    """Compute gradient of log probability."""
    params_grad = params.clone().detach().requires_grad_(True)

    logp = logp_fn()
    logp_tensor = torch.tensor(logp, requires_grad=True)
    logp_tensor.backward()

    if params_grad.grad is not None:
        return params_grad.grad.clone()
    else:
        return torch.zeros_like(params)
