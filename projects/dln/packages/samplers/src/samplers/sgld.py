"""
Stochastic Gradient Langevin Dynamics (SGLD) sampler.

Implements SGLD for sampling from the posterior distribution of DLN parameters.
Proper gradient noise scaling and temperature control for effective sampling.
"""

from typing import List, Optional, Tuple

import jax.numpy as jnp
from jax import grad, random
from tqdm import tqdm

from .toy_dln import DLNState, RegularDLN


class SGLDSampler:
    """
    Stochastic Gradient Langevin Dynamics sampler.

    Implements the SGLD update rule:
    θ_{t+1} = θ_t + (ε/2) ∇_θ log p(θ|D) + √ε η_t

    where η_t ~ N(0, I) is Gaussian noise and ε is the learning rate.
    """

    def __init__(
        self,
        learning_rate: float = 0.01,
        temperature: float = 1.0,
        batch_size: Optional[int] = None,
        gradient_clipping: Optional[float] = None,
    ):
        """
        Args:
            learning_rate: Step size ε in SGLD update
            temperature: Temperature parameter (1.0 = standard posterior)
            batch_size: Mini-batch size for stochastic gradients (None = full batch)
            gradient_clipping: Max gradient norm (None = no clipping)
        """
        self.learning_rate = learning_rate
        self.temperature = temperature
        self.batch_size = batch_size
        self.gradient_clipping = gradient_clipping

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

    def _compute_gradient(
        self,
        params: DLNState,
        model: RegularDLN,
        x_batch: jnp.ndarray,
        y_batch: jnp.ndarray,
    ) -> DLNState:
        """Compute gradient of log-posterior with respect to parameters."""

        def log_post_flat(flat_params):
            params_struct = self._unflatten_params(flat_params, model)
            return (
                model.log_posterior(params_struct, x_batch, y_batch) / self.temperature
            )

        flat_params = self._flatten_params(params)
        flat_grad = grad(log_post_flat)(flat_params)

        # Convert back to structured gradient
        grad_params = self._unflatten_params(flat_grad, model)

        # Apply gradient clipping if specified
        if self.gradient_clipping is not None:
            grad_norm = jnp.sqrt(jnp.sum(flat_grad**2))
            if grad_norm > self.gradient_clipping:
                clip_factor = self.gradient_clipping / grad_norm
                grad_params = DLNState(
                    W1=grad_params.W1 * clip_factor, W2=grad_params.W2 * clip_factor
                )

        return grad_params

    def _sgld_step(
        self,
        params: DLNState,
        model: RegularDLN,
        x_batch: jnp.ndarray,
        y_batch: jnp.ndarray,
        key: jnp.ndarray,
    ) -> DLNState:
        """Single SGLD update step."""

        # Compute gradient
        grad_params = self._compute_gradient(params, model, x_batch, y_batch)

        # Generate noise
        key1, key2 = random.split(key)
        noise_W1 = random.normal(key1, params.W1.shape)
        noise_W2 = random.normal(key2, params.W2.shape)

        # SGLD update: θ_{t+1} = θ_t + (ε/2) ∇log p(θ|D) + √ε η_t
        noise_scale = jnp.sqrt(self.learning_rate * self.temperature)

        new_W1 = (
            params.W1
            + 0.5 * self.learning_rate * grad_params.W1
            + noise_scale * noise_W1
        )

        new_W2 = (
            params.W2
            + 0.5 * self.learning_rate * grad_params.W2
            + noise_scale * noise_W2
        )

        return DLNState(W1=new_W1, W2=new_W2)

    def _get_batch(
        self, x: jnp.ndarray, y: jnp.ndarray, key: jnp.ndarray
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

    def sample(
        self,
        model: RegularDLN,
        x_data: jnp.ndarray,
        y_data: jnp.ndarray,
        init_params: DLNState,
        n_samples: int,
        n_burnin: int = 1000,
        thin: int = 1,
        key: jnp.ndarray = None,
        verbose: bool = True,
    ) -> Tuple[List[DLNState], jnp.ndarray]:
        """
        Sample from posterior using SGLD.

        Args:
            model: DLN model to sample for
            x_data: Input data (input_dim, n_data)
            y_data: Output data (output_dim, n_data)
            init_params: Initial parameter values
            n_samples: Number of samples to collect
            n_burnin: Number of burn-in steps
            thin: Thinning interval
            key: Random key
            verbose: Show progress bar

        Returns:
            samples: List of parameter samples
            log_posts: Log-posterior values for each sample
        """
        if key is None:
            key = random.PRNGKey(42)

        samples = []
        log_posts = []
        params = init_params

        total_steps = n_burnin + n_samples * thin
        iterator = tqdm(range(total_steps)) if verbose else range(total_steps)

        for step in iterator:
            key, subkey, batch_key = random.split(key, 3)

            # Get batch
            x_batch, y_batch = self._get_batch(x_data, y_data, batch_key)

            # SGLD step
            params = self._sgld_step(params, model, x_batch, y_batch, subkey)

            # Collect samples after burn-in
            if step >= n_burnin and (step - n_burnin) % thin == 0:
                samples.append(params)
                log_post = model.log_posterior(params, x_data, y_data)
                log_posts.append(log_post)

                if verbose and len(samples) % 100 == 0:
                    iterator.set_description(
                        f"Samples: {len(samples)}, LogPost: {log_post:.3f}"
                    )

        return samples, jnp.array(log_posts)

    def find_map(
        self,
        model: RegularDLN,
        x_data: jnp.ndarray,
        y_data: jnp.ndarray,
        init_params: DLNState,
        n_steps: int = 5000,
        key: jnp.ndarray = None,
        verbose: bool = True,
    ) -> Tuple[DLNState, float]:
        """
        Find Maximum A Posteriori (MAP) estimate using gradient ascent.

        Uses SGLD without noise (pure gradient ascent) to find mode.
        """
        if key is None:
            key = random.PRNGKey(123)

        # Temporarily disable noise and increase learning rate
        original_temp = self.temperature
        self.temperature = 0.0  # No noise

        params = init_params
        best_log_post = -jnp.inf
        best_params = params

        iterator = tqdm(range(n_steps)) if verbose else range(n_steps)

        for step in iterator:
            key, subkey, batch_key = random.split(key, 3)

            # Get batch
            x_batch, y_batch = self._get_batch(x_data, y_data, batch_key)

            # Pure gradient step (no noise)
            params = self._sgld_step(params, model, x_batch, y_batch, subkey)

            # Track best
            if step % 10 == 0:
                log_post = model.log_posterior(params, x_data, y_data)
                if log_post > best_log_post:
                    best_log_post = log_post
                    best_params = params

                if verbose and step % 500 == 0:
                    iterator.set_description(f"MAP LogPost: {log_post:.3f}")

        # Restore original temperature
        self.temperature = original_temp

        return best_params, best_log_post
