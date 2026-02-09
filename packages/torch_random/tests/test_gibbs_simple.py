#!/usr/bin/env python
"""
Simple integration test for gibbs functionality.
Run this directly without pytest to validate the implementation.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch
import torch.nn as nn
from torch_random.gibbs import (
    _infer_input_shape,
    _sample_inputs,
    _sample_outputs,
    gibbs,
)


def test_should_infer_input_shape_correctly():
    """Should infer input shapes from different model architectures."""
    print("Testing _infer_input_shape...")

    # Test linear model
    model = nn.Linear(10, 5)
    shape = _infer_input_shape(model)
    assert shape == (10,), f"Expected (10,), got {shape}"

    # Test sequential model
    model = nn.Sequential(nn.Linear(5, 10), nn.ReLU(), nn.Linear(10, 1))
    shape = _infer_input_shape(model)
    assert shape == (5,), f"Expected (5,), got {shape}"

    print("✅ _infer_input_shape tests passed")


def test_should_sample_inputs_from_distributions():
    """Should generate input samples from various distributions."""
    print("Testing _sample_inputs...")

    # Test normal distribution
    dist = torch.distributions.Normal(0, 1)
    samples = _sample_inputs(100, (5,), dist, "cpu")
    assert samples.shape == (100, 5), f"Expected (100, 5), got {samples.shape}"
    assert samples.device.type == "cpu"

    # Test uniform distribution
    dist = torch.distributions.Uniform(-1, 1)
    samples = _sample_inputs(50, (3,), dist, "cpu")
    assert samples.shape == (50, 3), f"Expected (50, 3), got {samples.shape}"
    assert torch.all(samples >= -1) and torch.all(samples <= 1), "Values not in [-1, 1]"

    print("✅ _sample_inputs tests passed")


def test_should_sample_outputs_based_on_loss_function():
    """Should generate appropriate output samples for different loss functions."""
    print("Testing _sample_outputs...")

    # Test MSE scalar output
    mus = torch.randn(100, 1)
    loss_fn = nn.MSELoss()
    samples = _sample_outputs(mus, loss_fn)
    assert samples.shape == (100, 1), f"Expected (100, 1), got {samples.shape}"

    # Test MSE vector output
    mus = torch.randn(100, 5)
    samples = _sample_outputs(mus, loss_fn)
    assert samples.shape == (100, 5), f"Expected (100, 5), got {samples.shape}"

    # Test L1 loss
    mus = torch.randn(50, 3)
    loss_fn = nn.L1Loss()
    samples = _sample_outputs(mus, loss_fn)
    assert samples.shape == (50, 3), f"Expected (50, 3), got {samples.shape}"

    # Test CrossEntropy
    mus = torch.randn(100, 10)
    loss_fn = nn.CrossEntropyLoss()
    samples = _sample_outputs(mus, loss_fn)
    assert samples.shape == (100,), f"Expected (100,), got {samples.shape}"
    assert samples.dtype == torch.long, f"Expected torch.long, got {samples.dtype}"
    assert torch.all(samples >= 0) and torch.all(
        samples < 10
    ), "Class indices out of range"

    # Test BCE loss
    mus = torch.randn(100, 5)
    loss_fn = nn.BCELoss()
    samples = _sample_outputs(mus, loss_fn)
    assert samples.shape == (100, 5), f"Expected (100, 5), got {samples.shape}"
    assert torch.all((samples == 0) | (samples == 1)), "BCE samples not binary"

    print("✅ _sample_outputs tests passed")


def test_should_perform_end_to_end_gibbs_sampling():
    """Should perform complete gibbs sampling workflow for various model types."""
    print("Testing gibbs integration...")

    # Test scalar regression
    model = nn.Sequential(nn.Linear(5, 10), nn.ReLU(), nn.Linear(10, 1))
    loss_fn = nn.MSELoss()
    input_dist = torch.distributions.Normal(0, 1)

    x, y = gibbs(model, loss_fn, 200, input_dist)
    assert x.shape == (200, 5), f"Expected x.shape (200, 5), got {x.shape}"
    assert y.shape == (200, 1), f"Expected y.shape (200, 1), got {y.shape}"

    # Test multiclass classification
    model = nn.Sequential(nn.Linear(3, 8), nn.ReLU(), nn.Linear(8, 5))
    loss_fn = nn.CrossEntropyLoss()
    input_dist = torch.distributions.Uniform(-1, 1)

    x, y = gibbs(model, loss_fn, 150, input_dist)
    assert x.shape == (150, 3), f"Expected x.shape (150, 3), got {x.shape}"
    assert y.shape == (150,), f"Expected y.shape (150,), got {y.shape}"
    assert y.dtype == torch.long, f"Expected y.dtype torch.long, got {y.dtype}"

    # Test vector output
    model = nn.Sequential(nn.Linear(4, 6), nn.ReLU(), nn.Linear(6, 3))
    loss_fn = nn.MSELoss()
    input_dist = torch.distributions.Normal(0, 0.5)

    x, y = gibbs(model, loss_fn, 100, input_dist)
    assert x.shape == (100, 4), f"Expected x.shape (100, 4), got {x.shape}"
    assert y.shape == (100, 3), f"Expected y.shape (100, 3), got {y.shape}"

    # Test reproducibility
    model = nn.Linear(2, 1)
    loss_fn = nn.MSELoss()
    input_dist = torch.distributions.Normal(0, 1)

    torch.manual_seed(42)
    x1, y1 = gibbs(model, loss_fn, 100, input_dist)

    torch.manual_seed(42)
    x2, y2 = gibbs(model, loss_fn, 100, input_dist)

    assert torch.allclose(x1, x2), "Results not reproducible with same seed"
    assert torch.allclose(y1, y2), "Results not reproducible with same seed"

    print("✅ gibbs integration tests passed")


def test_should_handle_errors_gracefully():
    """Should raise appropriate errors for unsupported cases."""
    print("Testing error handling...")

    # Test non-Linear input layer
    try:
        model = nn.Conv2d(3, 64, 3)
        _infer_input_shape(model)
        raise AssertionError("Should have raised ValueError for Conv2d")
    except ValueError as e:
        assert "Only Linear input layers" in str(e)

    # Test unsupported multi-dimensional MSE
    try:
        mus = torch.randn(10, 3, 32, 32)
        loss_fn = nn.MSELoss()
        _sample_outputs(mus, loss_fn)
        raise AssertionError("Should have raised NotImplementedError for multi-dim MSE")
    except NotImplementedError as e:
        assert "MSE with output shape" in str(e)

    print("✅ Error handling tests passed")


def main():
    """Run all tests."""
    print("Running torch_random.gibbs tests...")
    print("=" * 50)

    test_should_infer_input_shape_correctly()
    test_should_sample_inputs_from_distributions()
    test_should_sample_outputs_based_on_loss_function()
    test_should_perform_end_to_end_gibbs_sampling()
    test_should_handle_errors_gracefully()

    print("=" * 50)
    print("🎉 ALL TESTS PASSED!")
    print("\nGibbs sampling implementation is working correctly!")


if __name__ == "__main__":
    main()
