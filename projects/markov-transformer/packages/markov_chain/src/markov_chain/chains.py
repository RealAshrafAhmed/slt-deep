"""Transition matrix sampling and stationary distribution utilities.

T[i, j] = P(X_{t+1} = j | X_t = i).  Rows sum to 1.
"""

from __future__ import annotations

import numpy as np


def sample_transition_matrix(
    n_states: int,
    alpha: float,
    n: int = 1,
    rng: np.random.Generator | int | None = None,
) -> np.ndarray:
    """Sample a random N×N row-stochastic transition matrix.

    Each row is drawn i.i.d. from the symmetric Dirichlet distribution
    ``Dirichlet(alpha, alpha, ..., alpha)`` with ``n_states`` components.

    The concentration parameter ``alpha`` controls the shape of each row:

    * **alpha → 0** : rows become one-hot (chain stays in one state or teleports)
    * **alpha = 1**  : rows are uniform on the probability simplex (Jeffreys prior)
    * **alpha → ∞** : rows converge to the uniform distribution (1/N, ..., 1/N)

    Args:
        n: Number of transition matrices to generate
        n_states: Number of states N ≥ 2.
        alpha: Dirichlet concentration parameter.  Must be > 0.
        rng: Reproducibility seed.  Accepts ``int``, ``np.random.Generator``,
             or ``None`` (uses the global numpy RNG).

    Returns:
        T: float64 array of shape (n_states, n_states), rows sum to 1.
    """
    if n_states < 2:
        raise ValueError(f"n_states must be >= 2, got {n_states=}")
    if alpha <= 0:
        raise ValueError(f"alpha must be > 0, got {alpha=}")
    rng = np.random.default_rng(rng)
    result = rng.dirichlet(np.full(n_states, alpha), size=(n, n_states))
    return result.squeeze(0) if n == 1 else result


def gamblers_ruin(p: float, num_states: int):
    """Build the Gambler's Ruin transition matrix.

    States 0 and ``num_states - 1`` are absorbing boundaries.  Every interior
    state ``i`` transitions to ``i + 1`` with probability ``p`` and to ``i - 1``
    with probability ``1 - p``.

    Args:
        p: Step-right probability.  Must satisfy 0 < p < 1.
        num_states: Total number of states (≥ 3, including both boundaries).

    Returns:
        T: torch.FloatTensor of shape (num_states, num_states), rows sum to 1.
    """
    import torch

    if not 0.0 < p < 1.0:
        raise ValueError(f"p must be in (0, 1), got {p=}")
    if num_states < 3:
        raise ValueError(f"num_states must be >= 3, got {num_states=}")

    T = np.zeros((num_states, num_states))
    T[0, 0] = 1.0
    T[-1, -1] = 1.0
    interior = np.arange(1, num_states - 1)
    T[interior, interior + 1] = p
    T[interior, interior - 1] = 1.0 - p
    return torch.tensor(T, dtype=torch.float32)


def stationary_distribution(T: np.ndarray) -> np.ndarray:
    """Compute the unique stationary distribution π s.t. π T = π, Σπᵢ = 1.

    π is a row probability vector (measure acting from the right).  The
    stationarity equation π T = π is solved in its transposed form
    (T^T − I)π = 0, with the last equation replaced by the normalisation
    constraint Σπᵢ = 1.  Requires the chain to be ergodic (irreducible +
    aperiodic) so π is unique.

    Args:
        T: Row-stochastic transition matrix or array of matrices.
           Shape (N, N) for a single matrix or (M, N, N) for a batch.

    Returns:
        pi: float64 array of shape (N,) for a single matrix, or (M, N)
            for a batch, each row summing to 1.
    """
    if T.ndim == 3:
        return np.stack([stationary_distribution(t) for t in T])
    n = T.shape[0]
    A = T.T - np.eye(n)
    b = np.zeros(n)
    A[-1, :] = 1.0
    b[-1] = 1.0
    return np.linalg.solve(A, b)
