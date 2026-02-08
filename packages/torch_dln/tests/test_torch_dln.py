"""Tests for torch_dln package."""

import torch
import torch.nn as nn
from torch_dln import DLN, BottleneckDLN, PyramidDLN, WideDLN, zoo


def test_dln_basic():
    """Test basic DLN construction."""
    model = DLN([10, 5, 3, 1])

    # Check structure
    assert isinstance(model, nn.Sequential)
    assert len(model) == 3  # 3 linear layers

    # Check dimensions
    x = torch.randn(32, 10)
    y = model(x)
    assert y.shape == (32, 1)


def test_wide_dln():
    """Test WideDLN construction."""
    model = WideDLN(input_dim=5, output_dim=2, width=20, depth=4)

    x = torch.randn(16, 5)
    y = model(x)
    assert y.shape == (16, 2)


def test_bottleneck_dln():
    """Test BottleneckDLN construction."""
    model = BottleneckDLN(input_dim=10, output_dim=3, bottleneck_dim=2)

    x = torch.randn(8, 10)
    y = model(x)
    assert y.shape == (8, 3)


def test_pyramid_dln():
    """Test PyramidDLN construction."""
    model = PyramidDLN(input_dim=20, output_dim=1, n_layers=5)

    x = torch.randn(4, 20)
    y = model(x)
    assert y.shape == (4, 1)


def test_zoo_models():
    """Test zoo model construction."""
    # Test a few zoo models
    tiny = zoo.TinyDLN(input_dim=5, output_dim=2)
    x = torch.randn(10, 5)
    y = tiny(x)
    assert y.shape == (10, 2)

    # Test get_model function
    model = zoo.get_model("tiny", input_dim=3, output_dim=1)
    x = torch.randn(5, 3)
    y = model(x)
    assert y.shape == (5, 1)


def test_classic_architectures():
    """Test classic architecture approximations."""
    lenet = zoo.LeNetLinear(input_dim=784, num_classes=10)
    x = torch.randn(16, 784)
    y = lenet(x)
    assert y.shape == (16, 10)


def test_research_models():
    """Test research-oriented models."""
    autoencoder = zoo.LinearAutoencoder(input_dim=50, encoding_dim=10)
    x = torch.randn(32, 50)
    y = autoencoder(x)
    assert y.shape == (32, 50)

    low_rank = zoo.LowRankDLN(input_dim=20, output_dim=5, rank=3)
    x = torch.randn(16, 20)
    y = low_rank(x)
    assert y.shape == (16, 5)


if __name__ == "__main__":
    # Run basic tests
    test_dln_basic()
    test_wide_dln()
    test_bottleneck_dln()
    test_pyramid_dln()
    test_zoo_models()
    test_classic_architectures()
    test_research_models()
    print("All tests passed!")
