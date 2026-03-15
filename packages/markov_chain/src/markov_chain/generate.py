"""Sequence generation from discrete Markov chains.

Uses numpy's Generator API (``np.random.default_rng``) so experiments are
reproducible without touching global RNG state.  Return dtype is ``np.intp``
(int64 on 64-bit platforms) so sequences index directly into embedding tables.
"""

from __future__ import annotations

import numpy as np


def sample_sequences(
    T: np.ndarray,
    n: int,
    seq_len: int,
    prior: np.ndarray,
    rng: np.random.Generator | int | None = None,
) -> np.ndarray:
    """Sample ``n`` independent sequences of length ``seq_len`` from a Markov chain.

    The first token of each sequence is drawn from ``prior``.  Subsequent
    tokens are sampled from the row of ``T`` corresponding to the current state.

    Args:
        T: Row-stochastic transition matrix or array of matrices.
           Shape (n_states, n_states) for a single chain, or (M, n_states,
           n_states) for a batch of M chains.
        n: Number of independent sequences per chain.
        seq_len: Length of each sequence (number of tokens including the first).
        prior: Initial-state distribution.  Shape (n_states,) — shared across
               all chains — or (M, n_states) for a per-chain prior.  Must be
               non-negative and sum to 1 (per row).  Pass the stationary
               distribution to start each chain already in equilibrium.
        rng: Reproducibility seed.  Accepts ``int``, ``np.random.Generator``,
             or ``None`` (uses the global numpy RNG).

    Returns:
        sequences: int array of shape (n, seq_len) for a single matrix, or
                   (M, n, seq_len) for a batch.
    """
    rng = np.random.default_rng(rng)

    T = np.asarray(T)
    if T.ndim == 3:
        prior = np.asarray(prior, dtype=float)
        if prior.ndim == 1:
            prior = np.broadcast_to(prior, (T.shape[0], prior.shape[0]))
        return np.stack(
            [
                sample_sequences(T[i], n, seq_len, prior=prior[i], rng=rng)
                for i in range(T.shape[0])
            ]
        )

    n_states = T.shape[0]
    states = np.arange(n_states)

    prior = np.asarray(prior, dtype=float)
    if prior.shape != (n_states,):
        raise ValueError(f"prior must have shape ({n_states},), got {prior.shape}")
    if not np.isclose(prior.sum(), 1.0):
        raise ValueError(f"prior must sum to 1, got {prior.sum()}")

    sequences = np.empty((n, seq_len), dtype=np.intp)

    # Draw initial states from the prior.
    current = rng.choice(states, size=n, p=prior).astype(np.intp)
    sequences[:, 0] = current

    for t in range(1, seq_len):
        # For each state, batch-sample next tokens for all sequences in that state.
        next_state = np.empty(n, dtype=np.intp)
        for s in range(n_states):
            mask = current == s
            count = int(mask.sum())
            if count:
                next_state[mask] = rng.choice(states, size=count, p=T[s])
        current = next_state
        sequences[:, t] = current

    return sequences
