import numpy as np
from scipy_extensions.normal_mixture import sample


def test_sample_shape():
    n = 100
    weights = [0.5, 0.5]
    means = [0.0, 2.0]
    stds = [1.0, 0.5]
    samples, components = sample(n, weights, means, stds, random_state=42)
    assert samples.shape == (n,)
    assert components.shape == (n,)


def test_sample_component_distribution():
    n = 10000
    weights = [0.7, 0.3]
    means = [0.0, 5.0]
    stds = [1.0, 1.0]
    samples, components = sample(n, weights, means, stds, random_state=123)
    unique, counts = np.unique(components, return_counts=True)
    proportions = counts / n
    assert np.allclose(proportions, weights, atol=0.02)


def test_sample_mean_std():
    n = 5000
    weights = [0.5, 0.5]
    means = [1.0, 3.0]
    stds = [1.0, 2.0]
    samples, components = sample(n, weights, means, stds, random_state=7)
    # Check sample mean is between component means
    assert samples.mean() > min(means)
    assert samples.mean() < max(means)
    # Check sample std is reasonable
    assert samples.std() > min(stds)
    assert samples.std() < max(stds) + 1
