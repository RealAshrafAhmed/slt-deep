"""
Toy Deep Linear Network models for testing MCMC samplers.

Regular (non-singular) DLN implementations for controlled experiments where
we can compute analytical or semi-analytical posteriors for validation.
"""

from typing import NamedTuple, Tuple

import jax.numpy as jnp
from jax import random


class DLNState(NamedTuple):
    """Parameters for a Deep Linear Network."""

    W1: jnp.ndarray  # First layer weights
    W2: jnp.ndarray  # Second layer weights


class RegularDLN:
    """
    Regular (non-singular) Deep Linear Network: y = W2 @ W1 @ x + noise

    This model is designed to be regular (non-singular) for testing samplers
    before moving to truly singular models. Uses proper initialization and
    noise injection to avoid degenerate cases.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        noise_std: float = 0.1,
        prior_std: float = 1.0,
    ):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.noise_std = noise_std
        self.prior_std = prior_std

    def init_params(self, key: jnp.ndarray) -> DLNState:
        """Initialize parameters from prior."""
        key1, key2 = random.split(key)

        W1 = random.normal(key1, (self.hidden_dim, self.input_dim)) * self.prior_std
        W2 = random.normal(key2, (self.output_dim, self.hidden_dim)) * self.prior_std

        return DLNState(W1=W1, W2=W2)

    def forward(self, params: DLNState, x: jnp.ndarray) -> jnp.ndarray:
        """Forward pass: y = W2 @ W1 @ x"""
        hidden = (
            params.W1 @ x
        )  # (hidden_dim, batch_size) if x is (input_dim, batch_size)
        output = params.W2 @ hidden  # (output_dim, batch_size)
        return output

    def log_likelihood(self, params: DLNState, x: jnp.ndarray, y: jnp.ndarray) -> float:
        """Log-likelihood: p(y|x, θ) under Gaussian noise."""
        y_pred = self.forward(params, x)

        # Sum over output dimensions and batch
        mse = jnp.sum((y - y_pred) ** 2)
        log_lik = -0.5 * mse / (self.noise_std**2)
        log_lik -= 0.5 * y.size * jnp.log(2 * jnp.pi * self.noise_std**2)

        return log_lik

    def log_prior(self, params: DLNState) -> float:
        """Log-prior: Independent Gaussian priors on all weights."""
        log_p_W1 = -0.5 * jnp.sum(params.W1**2) / (self.prior_std**2)
        log_p_W1 -= 0.5 * params.W1.size * jnp.log(2 * jnp.pi * self.prior_std**2)

        log_p_W2 = -0.5 * jnp.sum(params.W2**2) / (self.prior_std**2)
        log_p_W2 -= 0.5 * params.W2.size * jnp.log(2 * jnp.pi * self.prior_std**2)

        return log_p_W1 + log_p_W2

    def log_posterior(self, params: DLNState, x: jnp.ndarray, y: jnp.ndarray) -> float:
        """Log-posterior: log p(θ|x,y) ∝ log p(y|x,θ) + log p(θ)"""
        return self.log_likelihood(params, x, y) + self.log_prior(params)

    def sample_data(
        self,
        key: jnp.ndarray,
        true_params: DLNState,
        n_samples: int,
        x_distribution: str = "gaussian",
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Generate synthetic data from the model."""
        key_x, key_noise = random.split(key)

        # Generate inputs
        if x_distribution == "gaussian":
            x = random.normal(key_x, (self.input_dim, n_samples))
        elif x_distribution == "uniform":
            x = random.uniform(key_x, (self.input_dim, n_samples), minval=-1, maxval=1)
        else:
            raise ValueError(f"Unknown x_distribution: {x_distribution}")

        # Forward pass + noise
        y_clean = self.forward(true_params, x)
        noise = random.normal(key_noise, y_clean.shape) * self.noise_std
        y = y_clean + noise

        return x, y


def create_simple_dln() -> RegularDLN:
    """Create a simple 2D → 3D → 1D DLN for quick testing."""
    return RegularDLN(
        input_dim=2, hidden_dim=3, output_dim=1, noise_std=0.1, prior_std=0.5
    )


def create_minimal_dln() -> RegularDLN:
    """Create minimal 1D → 2D → 1D DLN for visualization."""
    return RegularDLN(
        input_dim=1, hidden_dim=2, output_dim=1, noise_std=0.05, prior_std=0.3
    )
