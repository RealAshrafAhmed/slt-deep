"""
Hamiltonian Monte Carlo (HMC) sampler using blackjax.

Provides HMC sampling for DLN posterior distributions with automatic
step size tuning and mass matrix adaptation.
"""

from typing import List, Tuple

import blackjax
import jax
import jax.numpy as jnp
import numpy as np
from jax import random
from tqdm import tqdm

from .toy_dln import DLNState, RegularDLN


class HMCSampler:
    """
    Hamiltonian Monte Carlo sampler using blackjax.

    Provides efficient sampling from DLN posteriors using Hamiltonian dynamics
    with automatic tuning of step size and mass matrix.
    """

    def __init__(
        self,
        n_leapfrog_steps: int = 10,
        target_accept_rate: float = 0.8,
        adaptation_window: int = 1000,
        initial_step_size: float = 0.01,
    ):
        """
        Args:
            n_leapfrog_steps: Number of leapfrog steps per HMC proposal
            target_accept_rate: Target acceptance rate for step size adaptation
            adaptation_window: Number of steps for adaptation phase
            initial_step_size: Initial step size for HMC
        """
        self.n_leapfrog_steps = n_leapfrog_steps
        self.target_accept_rate = target_accept_rate
        self.adaptation_window = adaptation_window
        self.initial_step_size = initial_step_size

    def _flatten_params(self, params: DLNState) -> jnp.ndarray:
        """Flatten parameter structure to vector."""
        return jnp.concatenate([params.W1.flatten(), params.W2.flatten()])

    def _unflatten_params(
        self, flat_params: jnp.ndarray, model: RegularDLN
    ) -> DLNState:
        """Reconstruct parameter structure from vector."""
        W1_size = model.hidden_dim * model.input_dim
        W2_size = model.output_dim * model.hidden_dim

        W1 = flat_params[:W1_size].reshape(model.hidden_dim, model.input_dim)
        W2 = flat_params[W1_size : W1_size + W2_size].reshape(
            model.output_dim, model.hidden_dim
        )

        return DLNState(W1=W1, W2=W2)

    def _create_log_posterior_fn(
        self, model: RegularDLN, x_data: jnp.ndarray, y_data: jnp.ndarray
    ):
        """Create log posterior function for blackjax."""

        def log_posterior_fn(flat_params):
            params = self._unflatten_params(flat_params, model)
            return model.log_posterior(params, x_data, y_data)

        return log_posterior_fn

    def sample(
        self,
        model: RegularDLN,
        x_data: jnp.ndarray,
        y_data: jnp.ndarray,
        init_params: DLNState,
        n_samples: int,
        n_burnin: int = 1000,
        key: jnp.ndarray = None,
        verbose: bool = True,
    ) -> Tuple[List[DLNState], jnp.ndarray, dict]:
        """
        Sample from posterior using HMC.

        Args:
            model: DLN model to sample for
            x_data: Input data (input_dim, n_data)
            y_data: Output data (output_dim, n_data)
            init_params: Initial parameter values
            n_samples: Number of samples to collect
            n_burnin: Number of burn-in steps
            key: Random key
            verbose: Show progress bar

        Returns:
            samples: List of parameter samples
            log_posts: Log-posterior values
            info: Sampling diagnostics (acceptance rates, etc.)
        """
        if key is None:
            key = random.PRNGKey(42)

        # Setup
        log_posterior_fn = self._create_log_posterior_fn(model, x_data, y_data)
        init_position = self._flatten_params(init_params)

        # Initialize HMC sampler
        hmc = blackjax.hmc(
            log_posterior_fn,
            step_size=self.initial_step_size,
            num_integration_steps=self.n_leapfrog_steps,
        )

        # Initial state
        hmc_state = hmc.init(init_position)

        # Adaptation phase
        if verbose:
            print("HMC adaptation phase...")

        adaptation = blackjax.window_adaptation(
            hmc, log_posterior_fn, target_acceptance_rate=self.target_accept_rate
        )

        key, subkey = random.split(key)

        # Run adaptation
        (hmc_state, adaptation_state), adaptation_info = adaptation.run(
            subkey, hmc_state, num_steps=self.adaptation_window
        )

        # Update HMC with adapted parameters
        adapted_hmc = blackjax.hmc(
            log_posterior_fn,
            step_size=adaptation_info["step_size"].mean(),
            inverse_mass_matrix=adaptation_info["inverse_mass_matrix"],
            num_integration_steps=self.n_leapfrog_steps,
        )

        # Sampling phase
        if verbose:
            print(f"HMC sampling phase: {n_samples} samples...")

        samples = []
        log_posts = []
        acceptance_rates = []

        total_steps = n_burnin + n_samples
        iterator = tqdm(range(total_steps)) if verbose else range(total_steps)

        for step in iterator:
            key, subkey = random.split(key)

            # HMC step
            hmc_state, hmc_info = adapted_hmc.step(subkey, hmc_state)

            # Collect samples after burn-in
            if step >= n_burnin:
                # Convert back to structured parameters
                params = self._unflatten_params(hmc_state.position, model)
                samples.append(params)

                log_post = log_posterior_fn(hmc_state.position)
                log_posts.append(log_post)
                acceptance_rates.append(hmc_info.is_accepted)

                if verbose and len(samples) % 100 == 0:
                    recent_accept = jnp.mean(jnp.array(acceptance_rates[-100:]))
                    iterator.set_description(
                        f"Samples: {len(samples)}, LogPost: {log_post:.3f}, "
                        f"Accept: {recent_accept:.3f}"
                    )

        # Diagnostics
        info = {
            "acceptance_rate": jnp.mean(jnp.array(acceptance_rates)),
            "adapted_step_size": adaptation_info["step_size"].mean(),
            "adapted_mass_matrix": adaptation_info["inverse_mass_matrix"],
            "adaptation_info": adaptation_info,
        }

        return samples, jnp.array(log_posts), info

    def find_map(
        self,
        model: RegularDLN,
        x_data: jnp.ndarray,
        y_data: jnp.ndarray,
        init_params: DLNState,
        key: jnp.ndarray = None,
        verbose: bool = True,
    ) -> Tuple[DLNState, float]:
        """
        Find MAP estimate using scipy optimization.

        Uses JAX gradients with scipy L-BFGS for efficient MAP finding.
        """
        if key is None:
            key = random.PRNGKey(123)

        from scipy.optimize import minimize

        log_posterior_fn = self._create_log_posterior_fn(model, x_data, y_data)

        # Objective function for scipy (negative log posterior)
        def objective(flat_params):
            return -float(log_posterior_fn(jnp.array(flat_params)))

        # Gradient function
        grad_fn = jax.grad(log_posterior_fn)

        def objective_grad(flat_params):
            return -np.array(grad_fn(jnp.array(flat_params)))

        # Initial point
        init_position = self._flatten_params(init_params)

        if verbose:
            print("Finding MAP with L-BFGS...")

        # Optimize
        result = minimize(
            objective,
            x0=np.array(init_position),
            jac=objective_grad,
            method="L-BFGS-B",
            options={"disp": verbose, "maxiter": 1000},
        )

        if verbose:
            print(f"Optimization completed. Success: {result.success}")
            print(f"Final log-posterior: {-result.fun:.6f}")

        # Convert back to structured parameters
        map_params = self._unflatten_params(jnp.array(result.x), model)
        map_log_post = -result.fun

        return map_params, map_log_post
