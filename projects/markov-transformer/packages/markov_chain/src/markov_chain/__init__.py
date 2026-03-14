"""markov_chain: discrete Markov chain utilities for sequence generation."""

from .chains import gamblers_ruin, sample_transition_matrix, stationary_distribution
from .generate import generate_sequences, sample_sequences

__version__ = "0.1.0"

__all__ = [
    "gamblers_ruin",
    "generate_sequences",
    "sample_sequences",
    "sample_transition_matrix",
    "stationary_distribution",
]
