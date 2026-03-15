"""Transition matrix sampling and stationary distribution utilities.

T[i, j] = P(X_{t+1} = j | X_t = i).  Rows sum to 1.
"""

from __future__ import annotations

import numpy as np


def sample_transition_matrix(
    n_states: int,
    alpha: np.ndarray,
    n: int = 1,
    rng: np.random.Generator | int | None = None,
) -> np.ndarray:
    """Sample ``n`` random N×N row-stochastic transition matrices.

    Every row of every sampled matrix is drawn i.i.d. from the same Dirichlet
    distribution parameterised by ``alpha``.

    ``alpha`` must be a 1-D array of shape ``(n_states,)``.  For a symmetric
    Dirichlet pass ``np.full(n_states, c)``; for an asymmetric one targeted at
    a desired limiting distribution **π** pass ``c * π`` for any ``c > 0``.

    **Choosing alpha to target a limiting distribution π:**
    Because every row shares the same mean, the expected transition matrix is
    rank-1 with each row equal to ``alpha / alpha.sum()``.  Its unique
    stationary distribution is therefore ``alpha / alpha.sum()``.  To sample
    chains whose stationary distribution concentrates near a desired **π**,
    set ``alpha = c * π`` for any ``c > 0``.  The scalar ``c`` (total
    concentration) controls sharpness: larger ``c`` pulls each sampled row
    closer to **π**, while smaller ``c`` adds more row-to-row noise.

    Args:
        n_states: Number of states N ≥ 2.
        alpha: Dirichlet concentration.  1-D array of shape ``(n_states,)``.
            All entries must be > 0.  For a symmetric Dirichlet use
            ``np.full(n_states, c)``.
        n: Number of independent transition matrices to generate.
        rng: Reproducibility seed.  Accepts ``int``, ``np.random.Generator``,
            or ``None`` (uses the global numpy RNG).

    Returns:
        T: float64 array of shape ``(n_states, n_states)`` when ``n == 1``,
           or ``(n, n_states, n_states)`` when ``n > 1``.  Rows sum to 1.
    """
    if n_states < 2:
        raise ValueError(f"n_states must be >= 2, got {n_states=}")
    if alpha.ndim != 1 or alpha.shape != (n_states,):
        raise ValueError(
            f"alpha must be a 1-D array of shape ({n_states},), got shape {alpha.shape}"
        )
    if np.any(alpha <= 0):
        raise ValueError("alpha values must be > 0")
    rng = np.random.default_rng(rng)

    result = np.empty((n, n_states, n_states), dtype=float)
    for k in range(n):
        for i in range(n_states):
            result[k, i] = rng.dirichlet(alpha)

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
