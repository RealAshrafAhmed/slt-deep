"""Unit tests for stationary distribution utilities."""

import numpy as np
import pytest
from markov_chain.chains import sample_transition_matrix, stationary_distribution


class TestStationaryDistribution:
    def test_should_match_closed_form_for_two_state_chain(self):
        """Verify the analytic solution for a simple 2-state ergodic chain."""
        T = np.array(
            [
                [0.7, 0.3],
                [0.2, 0.8],
            ],
            dtype=float,
        )

        pi = stationary_distribution(T)
        expected = np.array([0.4, 0.6], dtype=float)

        assert np.allclose(pi, expected, atol=1e-12)

    def test_should_satisfy_stationarity_fixed_point_equation(self):
        """Check that π satisfies the row-vector fixed-point equation π T = π."""
        T = np.array(
            [
                [0.1, 0.6, 0.3],
                [0.2, 0.2, 0.6],
                [0.7, 0.1, 0.2],
            ],
            dtype=float,
        )

        pi = stationary_distribution(T)

        assert np.allclose(pi @ T, pi, atol=1e-10)

    def test_should_return_valid_probability_vector(self):
        """Stationary distribution should be non-negative and normalized."""
        T = sample_transition_matrix(n_states=5, alpha=2.0, n=1, rng=0)

        pi = stationary_distribution(T)

        assert np.isclose(pi.sum(), 1.0, atol=1e-12)
        assert np.all(pi >= -1e-12)

    def test_should_raise_on_non_ergodic_chain_with_non_unique_stationary_dist(self):
        """Non-ergodic chains should fail because the linear system is singular."""
        T = np.eye(3, dtype=float)

        with pytest.raises(np.linalg.LinAlgError):
            stationary_distribution(T)
