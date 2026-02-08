"""
normal_mixture.py
A simple 1D normal mixture sampler, following the style of scipy.stats and PyMC random samplers.
"""

import numpy as np


def sample(n, weights, means, stds, random_state=None):
    """
    Sample from a 1D normal mixture model.

    Parameters
    ----------
    n : int
        Number of samples to generate.
    weights : array-like
        Mixture weights (should sum to 1).
    means : array-like
        Means of the normal components.
    stds : array-like
        Standard deviations of the normal components.
    random_state : int, np.random.Generator, or None
        Random seed or generator for reproducibility.

    Returns
    -------
    samples : np.ndarray
        Array of shape (n,) with samples from the mixture.
    components : np.ndarray
        Array of shape (n,) with the component index for each sample.
    """
    weights = np.asarray(weights)
    means = np.asarray(means)
    stds = np.asarray(stds)
    if random_state is None:
        rng = np.random.default_rng()
    elif isinstance(random_state, np.random.Generator):
        rng = random_state
    else:
        rng = np.random.default_rng(random_state)
    k = len(weights)
    components = rng.choice(k, size=n, p=weights)
    samples = rng.normal(loc=means[components], scale=stds[components])
    return samples, components
