"""
HMC (Hamiltonian Monte Carlo) sampler implementation.

Implements HMC using pure PyTorch operations for Bayesian inference.
"""

from collections.abc import Callable
from typing import Any

import torch


def hmc(
    set_params_fn: Callable[[torch.Tensor], None],
    log_likelihood_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    log_prior_fn: Callable[[], torch.Tensor],
    x_data: torch.Tensor,
    y_data: torch.Tensor,
    init_params: torch.Tensor,
    n_samples: int = 1000,
    batch_size: int = 32,  # For API consistency, though HMC typically uses full data
    step_size: float = 0.01,
    n_leapfrog: int = 10,
    n_burnin: int = 1000,
    thin: int = 1,
    **kwargs,
) -> dict[str, Any]:
    """
    Hamiltonian Monte Carlo sampling (simplified implementation).

    Note: This is a simplified version. Full HMC would implement proper
    leapfrog integration with momentum variables. Unlike SGLD, HMC typically
    uses the full dataset for each gradient computation.

    Args:
        set_params_fn: Function that sets model parameters from flattened tensor
        log_likelihood_fn: Function that takes (x_data, y_data) and returns log likelihood
        log_prior_fn: Function that returns log prior at current params
        x_data: Input data tensor
        y_data: Target data tensor
        init_params: Initial parameter values
        n_samples: Number of samples to collect
        batch_size: Batch size (for API consistency, HMC uses full data)
        step_size: Leapfrog step size
        n_leapfrog: Number of leapfrog steps (unused in this simplified version)
        n_burnin: Number of burnin samples
        thin: Thinning factor
        **kwargs: Ignored

    Returns:
        Dictionary with samples and diagnostics
    """
    params = init_params.clone().detach().requires_grad_(True)

    samples = []
    logp_vals = []
    n_accepted = 0

    total_steps = n_burnin + n_samples * thin

    for step in range(total_steps):
        # Simplified HMC - just random walk for now
        # Real HMC would need proper leapfrog integration with momentum
        proposed_params = params + step_size * torch.randn_like(params)

        # Compute current log probability (HMC uses full dataset)
        set_params_fn(params)
        current_log_likelihood = log_likelihood_fn(x_data, y_data)
        current_log_prior = log_prior_fn()
        current_logp = current_log_likelihood + current_log_prior

        # Compute proposed log probability
        # Temporarily switch to proposed params for evaluation
        old_params = params.clone()
        params = proposed_params.clone().detach().requires_grad_(True)

        set_params_fn(params)
        new_log_likelihood = log_likelihood_fn(x_data, y_data)
        new_log_prior = log_prior_fn()
        new_logp = new_log_likelihood + new_log_prior

        # Accept/reject using Metropolis criterion
        log_accept_prob = new_logp - current_logp
        accept_prob = min(1.0, float(torch.exp(log_accept_prob)))

        if torch.rand(1).item() < accept_prob:
            # Keep proposed params
            n_accepted += 1
        else:
            # Revert to old params
            params = old_params

        # Collect samples after burnin
        if step >= n_burnin and (step - n_burnin) % thin == 0:
            with torch.no_grad():
                samples.append(params.clone().detach())
            # Recompute logp for recording
            set_params_fn(params)
            final_log_likelihood = log_likelihood_fn(x_data, y_data)
            final_log_prior = log_prior_fn()
            logp_vals.append((final_log_likelihood + final_log_prior).item())

    acceptance_rate = n_accepted / total_steps if total_steps > 0 else 0.0

    return {
        "parameters": samples,
        "log_probabilities": logp_vals,
        "acceptance_rate": acceptance_rate,
        "backend": "hmc",
        "diagnostics": {
            "step_size": step_size,
            "n_leapfrog": n_leapfrog,
            "batch_size": batch_size,
            "n_data": x_data.shape[0],
            "n_burnin": n_burnin,
            "thin": thin,
            "n_accepted": n_accepted,
            "total_steps": total_steps,
        },
    }
