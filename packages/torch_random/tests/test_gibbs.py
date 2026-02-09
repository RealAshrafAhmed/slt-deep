"""
Unit tests for gibbs sampling functionality.
"""

import pytest
import torch
import torch.nn as nn
from torch_random.gibbs import (
    _infer_input_shape,
    _sample_inputs,
    _sample_outputs,
    gibbs,
)


class TestInferInputShape:
    """Test input shape inference from model architecture."""

    def test_should_infer_correct_shape_from_linear_model(self):
        """Should return correct input shape when given a Linear model."""
        model = nn.Linear(10, 5)
        shape = _infer_input_shape(model)
        assert shape == (10,)

    def test_should_infer_correct_shape_from_sequential_model(self):
        """Should return input shape of first Linear layer in Sequential model."""
        model = nn.Sequential(nn.Linear(5, 10), nn.ReLU(), nn.Linear(10, 1))
        shape = _infer_input_shape(model)
        assert shape == (5,)

    def test_should_raise_error_when_model_has_no_linear_layers(self):
        """Should raise ValueError when model has no Linear input layers."""
        model = nn.Conv2d(3, 64, 3)  # No Linear layer

        with pytest.raises(ValueError, match="Only Linear input layers"):
            _infer_input_shape(model)


class TestSampleInputs:
    """Test input sampling functionality."""

    def test_should_sample_correctly_from_normal_distribution(self):
        """Should generate samples with correct shape and device from Normal distribution."""
        dist = torch.distributions.Normal(0, 1)
        samples = _sample_inputs(100, (5,), dist, "cpu")

        assert samples.shape == (100, 5)
        assert samples.device.type == "cpu"

    def test_should_sample_within_bounds_from_uniform_distribution(self):
        """Should generate samples within specified bounds from Uniform distribution."""
        dist = torch.distributions.Uniform(-1, 1)
        samples = _sample_inputs(50, (3,), dist, "cpu")

        assert samples.shape == (50, 3)
        assert torch.all(samples >= -1) and torch.all(samples <= 1)

    def test_should_transfer_samples_to_correct_device(self):
        """Should move samples to the specified device."""
        dist = torch.distributions.Normal(0, 1)
        samples = _sample_inputs(10, (2,), dist, "cpu")

        assert samples.device.type == "cpu"


class TestSampleOutputs:
    """Test output sampling for different loss functions."""

    def test_should_generate_gaussian_samples_for_mse_with_scalar_output(self):
        """Should generate Gaussian samples with correct scale for MSE loss and scalar output."""
        mus = torch.randn(100, 1)  # Scalar output
        loss_fn = nn.MSELoss()

        samples = _sample_outputs(mus, loss_fn)

        assert samples.shape == (100, 1)
        # Check if samples are reasonably distributed around means
        diff = (samples - mus).abs().mean()
        assert 0.1 < diff < 2.0  # Should have some noise but not too much

    def test_should_generate_multivariate_gaussian_samples_for_mse_with_vector_output(
        self,
    ):
        """Should generate multivariate Gaussian samples for MSE loss and vector output."""
        mus = torch.randn(100, 5)  # Vector output
        loss_fn = nn.MSELoss()

        samples = _sample_outputs(mus, loss_fn)

        assert samples.shape == (100, 5)
        diff = (samples - mus).abs().mean()
        assert 0.1 < diff < 2.0

    def test_should_generate_laplace_samples_for_l1_loss(self):
        """Should generate Laplace distribution samples for L1 loss."""
        mus = torch.randn(100, 3)
        loss_fn = nn.L1Loss()

        samples = _sample_outputs(mus, loss_fn)

        assert samples.shape == (100, 3)
        # Laplace distribution should have heavier tails than Gaussian
        diff = (samples - mus).abs().mean()
        assert 0.5 < diff < 3.0

    def test_should_generate_categorical_samples_for_cross_entropy_standard_case(self):
        """Should generate categorical samples for standard CrossEntropy classification."""
        mus = torch.randn(100, 10)  # 10 classes
        loss_fn = nn.CrossEntropyLoss()

        samples = _sample_outputs(mus, loss_fn)

        assert samples.shape == (100,)
        assert samples.dtype == torch.long
        assert torch.all(samples >= 0) and torch.all(samples < 10)

    def test_should_generate_categorical_samples_for_cross_entropy_multidimensional_case(
        self,
    ):
        """Should generate categorical samples for multi-dimensional CrossEntropy output."""
        mus = torch.randn(50, 8, 5)  # Sequence of 8 elements, 5 classes each
        loss_fn = nn.CrossEntropyLoss()

        samples = _sample_outputs(mus, loss_fn)

        assert samples.shape == (50, 8)
        assert samples.dtype == torch.long
        assert torch.all(samples >= 0) and torch.all(samples < 5)

    def test_should_generate_binary_samples_for_bce_loss(self):
        """Should generate binary (0 or 1) samples for BCE loss."""
        mus = torch.randn(100, 5)
        loss_fn = nn.BCELoss()

        samples = _sample_outputs(mus, loss_fn)

        assert samples.shape == (100, 5)
        # BCE should produce binary outputs (0 or 1)
        assert torch.all((samples == 0) | (samples == 1))

    def test_should_raise_error_for_unsupported_multidimensional_mse(self):
        """Should raise NotImplementedError for multi-dimensional MSE output shapes."""
        mus = torch.randn(10, 3, 32, 32)  # Image-like output
        loss_fn = nn.MSELoss()

        with pytest.raises(NotImplementedError, match="MSE with output shape"):
            _sample_outputs(mus, loss_fn)

    def test_should_fallback_to_gaussian_for_unknown_loss_function(self):
        """Should fall back to multivariate Gaussian for unknown loss function types."""
        mus = torch.randn(100, 3)
        loss_fn = nn.SmoothL1Loss()  # Unsupported loss type

        samples = _sample_outputs(mus, loss_fn)

        assert samples.shape == (100, 3)
        # Should fall back to multivariate normal


class TestGibbsIntegration:
    """Integration tests for the full gibbs function."""

    def test_should_generate_correlated_data_for_scalar_regression_model(self):
        """Should generate x,y pairs where y is related to x through model with noise."""
        model = nn.Sequential(nn.Linear(5, 10), nn.ReLU(), nn.Linear(10, 1))
        loss_fn = nn.MSELoss()
        input_dist = torch.distributions.Normal(0, 1)

        x, y = gibbs(model, loss_fn, 200, input_dist)

        assert x.shape == (200, 5)
        assert y.shape == (200, 1)

        # Check that y values are related to x values through the model
        with torch.no_grad():
            model.eval()
            y_pred = model(x)
            diff = (y - y_pred).abs().mean()
            assert 0.1 < diff < 2.0  # Should have noise but be related

    def test_should_generate_valid_class_labels_for_multiclass_classification(self):
        """Should generate valid class indices for multiclass classification model."""
        model = nn.Sequential(nn.Linear(3, 8), nn.ReLU(), nn.Linear(8, 5))
        loss_fn = nn.CrossEntropyLoss()
        input_dist = torch.distributions.Uniform(-1, 1)

        x, y = gibbs(model, loss_fn, 150, input_dist)

        assert x.shape == (150, 3)
        assert y.shape == (150,)
        assert y.dtype == torch.long
        assert torch.all(y >= 0) and torch.all(y < 5)

    def test_should_handle_vector_output_model_correctly(self):
        """Should generate correct shapes for models with vector outputs."""
        model = nn.Sequential(nn.Linear(4, 6), nn.ReLU(), nn.Linear(6, 3))
        loss_fn = nn.MSELoss()
        input_dist = torch.distributions.Normal(0, 0.5)

        x, y = gibbs(model, loss_fn, 100, input_dist)

        assert x.shape == (100, 4)
        assert y.shape == (100, 3)

    def test_should_respect_device_parameter_in_gibbs_sampling(self):
        """Should place all tensors on the specified device during gibbs sampling."""
        model = nn.Linear(2, 1)
        loss_fn = nn.MSELoss()
        input_dist = torch.distributions.Normal(0, 1)

        x, y = gibbs(model, loss_fn, 50, input_dist, device="cpu")

        assert x.device.type == "cpu"
        assert y.device.type == "cpu"

    def test_should_put_model_in_eval_mode_during_sampling(self):
        """Should set model to eval mode during gibbs sampling for consistent results."""
        model = nn.Sequential(nn.Linear(3, 5), nn.Dropout(0.5), nn.Linear(5, 1))
        loss_fn = nn.MSELoss()
        input_dist = torch.distributions.Normal(0, 1)

        # Set model to train mode initially
        model.train()
        assert model.training

        x, y = gibbs(model, loss_fn, 100, input_dist)

        # Model should be in eval mode after gibbs call
        assert not model.training

    def test_should_be_reproducible_when_using_same_seed(self):
        """Should generate identical results when using the same random seed."""
        model = nn.Linear(2, 1)
        loss_fn = nn.MSELoss()
        input_dist = torch.distributions.Normal(0, 1)

        # First run
        torch.manual_seed(42)
        x1, y1 = gibbs(model, loss_fn, 100, input_dist)

        # Second run with same seed
        torch.manual_seed(42)
        x2, y2 = gibbs(model, loss_fn, 100, input_dist)

        assert torch.allclose(x1, x2)
        assert torch.allclose(y1, y2)

    def test_should_handle_various_input_dimensions_correctly(self):
        """Should work correctly with models having different input dimensions."""
        # Test various input sizes
        for input_dim in [1, 5, 20, 100]:
            model = nn.Linear(input_dim, 1)
            loss_fn = nn.MSELoss()
            input_dist = torch.distributions.Normal(0, 1)

            x, y = gibbs(model, loss_fn, 50, input_dist)

            assert x.shape == (50, input_dim)
            assert y.shape == (50, 1)

    def test_should_handle_various_output_dimensions_correctly(self):
        """Should work correctly with models having different output dimensions."""
        # Test various output sizes for MSE
        for output_dim in [1, 3, 10]:
            model = nn.Linear(5, output_dim)
            loss_fn = nn.MSELoss()
            input_dist = torch.distributions.Normal(0, 1)

            x, y = gibbs(model, loss_fn, 50, input_dist)

            assert x.shape == (50, 5)
            assert y.shape == (50, output_dim)

    def test_should_run_in_no_grad_context_during_sampling(self):
        """Should disable gradients during gibbs sampling for memory efficiency."""
        model = nn.Linear(3, 1)
        model.requires_grad_(True)  # Enable gradients

        loss_fn = nn.MSELoss()
        input_dist = torch.distributions.Normal(0, 1)

        x, y = gibbs(model, loss_fn, 50, input_dist)

        # Outputs should not require gradients
        assert not x.requires_grad
        assert not y.requires_grad


if __name__ == "__main__":
    pytest.main([__file__])
