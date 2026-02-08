"""Kullback-Leibler divergence utilities for SLT experiments."""

from typing import Callable, Optional

import numpy as np
from scipy import integrate


def kl_divergence_discrete(p: np.ndarray, q: np.ndarray) -> float:
    """
    Calculate KL divergence between two discrete probability distributions.

    Args:
        p: True distribution (reference)
        q: Approximate distribution

    Returns:
        KL(p || q) = sum(p * log(p / q))
    """
    # Add small epsilon to avoid log(0)
    epsilon = 1e-10
    q_safe = np.maximum(q, epsilon)

    # Only compute where p > 0 to avoid 0 * log(0) terms
    mask = p > epsilon
    return np.sum(p[mask] * np.log(p[mask] / q_safe[mask]))


def kl_divergence_continuous(
    p_func: Callable[[np.ndarray], np.ndarray],
    q_func: Callable[[np.ndarray], np.ndarray],
    domain: tuple[float, float],
    n_points: int = 1000,
) -> float:
    """
    Approximate KL divergence between continuous distributions using numerical integration.

    Args:
        p_func: True distribution density function
        q_func: Approximate distribution density function
        domain: Integration domain (a, b)
        n_points: Number of points for numerical integration

    Returns:
        Approximate KL(p || q)
    """

    def integrand(x):
        p_val = p_func(x)
        q_val = q_func(x)

        # Avoid log(0) and 0/0 issues
        mask = (p_val > 1e-10) & (q_val > 1e-10)
        result = np.zeros_like(x)
        result[mask] = p_val[mask] * np.log(p_val[mask] / q_val[mask])
        return result

    result, _ = integrate.quad(integrand, domain[0], domain[1])
    return result


def symmetrized_kl(p: np.ndarray, q: np.ndarray) -> float:
    """
    Calculate symmetrized KL divergence: (KL(p||q) + KL(q||p)) / 2

    Useful when you want a symmetric distance measure.
    """
    return (kl_divergence_discrete(p, q) + kl_divergence_discrete(q, p)) / 2


def kl_divergence_gaussian(
    mu1: float, sigma1: float, mu2: float, sigma2: float
) -> float:
    """
    Analytical KL divergence between two Gaussian distributions.

    KL(N(μ₁,σ₁²) || N(μ₂,σ₂²)) = log(σ₂/σ₁) + (σ₁² + (μ₁-μ₂)²)/(2σ₂²) - 1/2
    """
    return (
        np.log(sigma2 / sigma1) + (sigma1**2 + (mu1 - mu2) ** 2) / (2 * sigma2**2) - 0.5
    )
