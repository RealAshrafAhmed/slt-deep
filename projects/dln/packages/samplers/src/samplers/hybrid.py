"""
Hybrid 2-phase sampler: SGD → Langevin.

Combines deterministic optimization (SGD) to find the mode with stochastic
sampling (SGLD) for local posterior exploration around the MAP estimate.
"""

import jax
import jax.numpy as jnp
from jax import random
from typing import List, Tuple, Optional
import numpy as np
from tqdm import tqdm

from .toy_dln import DLNState, RegularDLN
from .sgld import SGLDSampler


class HybridSampler:
    """
    Hybrid two-phase sampler.

    Phase 1: Deterministic SGD to find MAP estimate
    Phase 2: SGLD sampling around the MAP for posterior exploration

    This approach is efficient when we want to understand the local posterior
    geometry around the maximum a posteriori estimate.
    """

    def __init__(
        self,
        sgd_learning_rate: float = 0.1,
        sgld_learning_rate: float = 0.01,
        sgld_temperature: float = 1.0,
        gradient_clipping: Optional[float] = None,
        batch_size: Optional[int] = None
    ):
        """
        Args:
            sgd_learning_rate: Learning rate for SGD phase (finding MAP)
            sgld_learning_rate: Learning rate for SGLD phase (sampling)
            sgld_temperature: Temperature for SGLD phase
            gradient_clipping: Max gradient norm (applied in both phases)
            batch_size: Mini-batch size (None = full batch)
        """
        self.sgd_learning_rate = sgd_learning_rate
        self.sgld_learning_rate = sgld_learning_rate
        self.sgld_temperature = sgld_temperature
        self.gradient_clipping = gradient_clipping
        self.batch_size = batch_size

        # SGLD sampler for phase 2
        self.sgld_sampler = SGLDSampler(
            learning_rate=sgld_learning_rate,
            temperature=sgld_temperature,
            batch_size=batch_size,
            gradient_clipping=gradient_clipping
        )

    def _flatten_params(self, params: DLNState) -> jnp.ndarray:
        """Flatten parameter structure to vector."""
        return jnp.concatenate([params.W1.flatten(), params.W2.flatten()])

    def _unflatten_params(self, flat_params: jnp.ndarray, model: RegularDLN) -> DLNState:
        """Reconstruct parameter structure from vector."""
        W1_size = model.hidden_dim * model.input_dim
        W2_size = model.output_dim * model.hidden_dim

        W1 = flat_params[:W1_size].reshape(model.hidden_dim, model.input_dim)
        W2 = flat_params[W1_size:W1_size + W2_size].reshape(model.output_dim, model.hidden_dim)

        return DLNState(W1=W1, W2=W2)

    def _compute_gradient(
        self,
        params: DLNState,
        model: RegularDLN,
        x_batch: jnp.ndarray,
        y_batch: jnp.ndarray
    ) -> DLNState:
        """Compute gradient of log-posterior."""
        from jax import grad

        def log_post_flat(flat_params):
            params_struct = self._unflatten_params(flat_params, model)
            return model.log_posterior(params_struct, x_batch, y_batch)

        flat_params = self._flatten_params(params)
        flat_grad = grad(log_post_flat)(flat_params)

        # Convert back to structured gradient
        grad_params = self._unflatten_params(flat_grad, model)

        # Apply gradient clipping if specified
        if self.gradient_clipping is not None:
            grad_norm = jnp.sqrt(jnp.sum(flat_grad ** 2))
            if grad_norm > self.gradient_clipping:
                clip_factor = self.gradient_clipping / grad_norm
                grad_params = DLNState(
                    W1=grad_params.W1 * clip_factor,
                    W2=grad_params.W2 * clip_factor
                )

        return grad_params

    def _get_batch(
        self,
        x: jnp.ndarray,
        y: jnp.ndarray,
        key: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Get mini-batch or full batch."""
        if self.batch_size is None or self.batch_size >= x.shape[1]:
            return x, y

        # Random mini-batch
        n_data = x.shape[1]
        batch_indices = random.choice(
            key, n_data, shape=(self.batch_size,), replace=False
        )

        return x[:, batch_indices], y[:, batch_indices]

    def _sgd_step(
        self,
        params: DLNState,
        model: RegularDLN,
        x_batch: jnp.ndarray,
        y_batch: jnp.ndarray
    ) -> DLNState:
        """Single SGD update step (no noise)."""

        # Compute gradient
        grad_params = self._compute_gradient(params, model, x_batch, y_batch)

        # Pure gradient ascent
        new_W1 = params.W1 + self.sgd_learning_rate * grad_params.W1
        new_W2 = params.W2 + self.sgd_learning_rate * grad_params.W2

        return DLNState(W1=new_W1, W2=new_W2)

    def find_map(
        self,
        model: RegularDLN,
        x_data: jnp.ndarray,
        y_data: jnp.ndarray,
        init_params: DLNState,
        n_steps: int = 5000,
        key: jnp.ndarray = None,
        verbose: bool = True,
        convergence_threshold: float = 1e-6,
        patience: int = 100
    ) -> Tuple[DLNState, float, dict]:
        \"\"\"
        Phase 1: Find MAP estimate using SGD.

        Args:
            model: DLN model
            x_data: Input data
            y_data: Output data
            init_params: Initial parameters
            n_steps: Maximum number of SGD steps
            key: Random key
            verbose: Show progress
            convergence_threshold: Stop when log-post improvement < threshold
            patience: Number of steps to wait for improvement

        Returns:
            map_params: MAP estimate
            map_log_post: Log-posterior at MAP
            info: Optimization diagnostics
        \"\"\"
        if key is None:
            key = random.PRNGKey(123)

        params = init_params
        log_posts = []
        best_log_post = -jnp.inf
        best_params = params
        steps_without_improvement = 0

        iterator = tqdm(range(n_steps), desc=\"SGD → MAP\") if verbose else range(n_steps)

        for step in iterator:
            key, batch_key = random.split(key)

            # Get batch
            x_batch, y_batch = self._get_batch(x_data, y_data, batch_key)

            # SGD step
            params = self._sgd_step(params, model, x_batch, y_batch)

            # Evaluate progress
            if step % 10 == 0:  # Check every 10 steps
                log_post = model.log_posterior(params, x_data, y_data)
                log_posts.append(log_post)

                if log_post > best_log_post:
                    improvement = log_post - best_log_post
                    best_log_post = log_post
                    best_params = params
                    steps_without_improvement = 0

                    if verbose and step % 100 == 0:
                        iterator.set_description(
                            f\"SGD → MAP | LogPost: {log_post:.6f} | Δ: {improvement:.2e}\"
                        )

                    # Check convergence
                    if improvement < convergence_threshold and step > 100:
                        steps_without_improvement += 1
                        if steps_without_improvement >= patience:
                            if verbose:
                                print(f\"\\nConverged at step {step} (improvement < {convergence_threshold:.2e})\")\n                            break
                else:
                    steps_without_improvement += 1
                    if steps_without_improvement >= patience:
                        if verbose:
                            print(f\"\\nStopped at step {step} (no improvement for {patience} evaluations)\")\n                        break

        info = {
            'n_steps_taken': step + 1,
            'log_post_history': jnp.array(log_posts),
            'converged': steps_without_improvement >= patience,
            'final_log_post': best_log_post
        }

        return best_params, best_log_post, info

    def sample_around_map(
        self,
        model: RegularDLN,
        x_data: jnp.ndarray,
        y_data: jnp.ndarray,
        map_params: DLNState,
        n_samples: int,
        n_burnin: int = 1000,
        key: jnp.ndarray = None,
        verbose: bool = True,
        adaptive_lr: bool = True
    ) -> Tuple[List[DLNState], jnp.ndarray]:
        \"\"\"
        Phase 2: SGLD sampling around MAP estimate.

        Args:
            model: DLN model
            x_data: Input data
            y_data: Output data
            map_params: MAP estimate from phase 1
            n_samples: Number of samples to collect
            n_burnin: Burn-in steps for SGLD
            key: Random key
            verbose: Show progress
            adaptive_lr: Adapt learning rate based on distance from MAP

        Returns:
            samples: Parameter samples around MAP
            log_posts: Log-posterior values
        \"\"\"
        if key is None:
            key = random.PRNGKey(456)

        if verbose:
            print(f\"\\nPhase 2: SGLD sampling around MAP (T={self.sgld_temperature})\")\n            \n        # Optionally add small perturbation to avoid starting exactly at mode\n        key, noise_key = random.split(key)\n        noise_scale = 0.01  # Small perturbation\n        \n        noise_W1 = random.normal(noise_key, map_params.W1.shape) * noise_scale\n        key, noise_key2 = random.split(key) \n        noise_W2 = random.normal(noise_key2, map_params.W2.shape) * noise_scale\n        \n        perturbed_params = DLNState(\n            W1=map_params.W1 + noise_W1,\n            W2=map_params.W2 + noise_W2\n        )\n        \n        # Sample using SGLD\n        samples, log_posts = self.sgld_sampler.sample(\n            model=model,\n            x_data=x_data,\n            y_data=y_data,\n            init_params=perturbed_params,\n            n_samples=n_samples,\n            n_burnin=n_burnin,\n            key=key,\n            verbose=verbose\n        )\n        \n        return samples, log_posts\n    \n    def sample(\n        self,\n        model: RegularDLN,\n        x_data: jnp.ndarray,\n        y_data: jnp.ndarray, \n        init_params: DLNState,\n        n_samples: int,\n        map_steps: int = 5000,\n        n_burnin: int = 1000,\n        key: jnp.ndarray = None,\n        verbose: bool = True\n    ) -> Tuple[DLNState, List[DLNState], jnp.ndarray, dict]:\n        \"\"\"  \n        Full hybrid sampling: SGD → MAP, then SGLD around MAP.\n        \n        Args:\n            model: DLN model\n            x_data: Input data\n            y_data: Output data\n            init_params: Initial parameters\n            n_samples: Number of SGLD samples around MAP\n            map_steps: Number of SGD steps for MAP finding\n            n_burnin: SGLD burn-in steps\n            key: Random key\n            verbose: Show progress\n            \n        Returns:\n            map_params: MAP estimate from phase 1\n            samples: SGLD samples from phase 2\n            log_posts: Log-posterior values for samples\n            info: Combined diagnostics from both phases\n        \"\"\"  \n        if key is None:\n            key = random.PRNGKey(42)\n            \n        key1, key2 = random.split(key)\n        \n        # Phase 1: Find MAP\n        if verbose:\n            print(\"=== Hybrid Sampler: Phase 1 (SGD → MAP) ===\")\n        map_params, map_log_post, map_info = self.find_map(\n            model, x_data, y_data, init_params, \n            n_steps=map_steps, key=key1, verbose=verbose\n        )\n        \n        # Phase 2: Sample around MAP  \n        if verbose:\n            print(\"\\n=== Phase 2 (SGLD around MAP) ===\")\n        samples, log_posts = self.sample_around_map(\n            model, x_data, y_data, map_params,\n            n_samples=n_samples, n_burnin=n_burnin, \n            key=key2, verbose=verbose\n        )\n        \n        # Combined info\n        info = {\n            'map_log_post': map_log_post,\n            'map_info': map_info,\n            'sgld_mean_log_post': jnp.mean(log_posts),\n            'sgld_std_log_post': jnp.std(log_posts)\n        }\n        \n        if verbose:\n            print(f\"\\n=== Hybrid Sampling Complete ===\")\n            print(f\"MAP log-posterior: {map_log_post:.6f}\")\n            print(f\"SGLD mean log-posterior: {info['sgld_mean_log_post']:.6f} ± {info['sgld_std_log_post']:.6f}\")\n            print(f\"Collected {len(samples)} samples around MAP\")\n            \n        return map_params, samples, log_posts, info"
