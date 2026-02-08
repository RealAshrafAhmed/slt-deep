"""
Famous DLN architectures and toy models for research.

This module provides well-known Deep Linear Network architectures from the
literature, as well as simple toy models for experimentation.
"""

import torch
import torch.nn as nn

from .constructors import DLN, BottleneckDLN


# Famous architectures from literature
def LeNetLinear(input_dim: int = 784, num_classes: int = 10) -> nn.Sequential:
    """Linear approximation of LeNet architecture."""
    return DLN([input_dim, 120, 84, num_classes])


def AlexNetLinear(
    input_dim: int = 224 * 224 * 3, num_classes: int = 1000
) -> nn.Sequential:
    """Linear approximation of AlexNet architecture."""
    return DLN([input_dim, 4096, 4096, num_classes])


def ResNetLinear18(
    input_dim: int = 224 * 224 * 3, num_classes: int = 1000
) -> nn.Sequential:
    """Linear approximation of ResNet-18 layer widths."""
    return DLN([input_dim, 64, 128, 256, 512, num_classes])


def VGGLinear11(
    input_dim: int = 224 * 224 * 3, num_classes: int = 1000
) -> nn.Sequential:
    """Linear approximation of VGG-11 architecture."""
    return DLN([input_dim, 64, 128, 256, 256, 512, 512, 512, 512, num_classes])


# Research toy models
def TinyDLN(
    input_dim: int = 2, output_dim: int = 1, num_hidden: int = 5
) -> nn.Sequential:
    """Minimal DLN for theoretical analysis."""
    return DLN([input_dim] + [num_hidden] * 2 + [output_dim])


def SymmetricDLN(
    input_dim: int = 10, bottleneck: int = 3, output_dim: int = 10
) -> nn.Sequential:
    """Symmetric encoder-decoder DLN."""
    return DLN([input_dim, 7, 5, bottleneck, 5, 7, output_dim])


def OverparameterizedDLN(
    input_dim: int = 10, output_dim: int = 1, width: int = 100, depth: int = 10
) -> nn.Sequential:
    """Highly overparameterized DLN for studying generalization."""
    return DLN([input_dim] + [width] * depth + [output_dim])


def DeepBottleneckDLN(
    input_dim: int, output_dim: int, bottleneck_dim: int, depth: int
) -> nn.Sequential:
    """
    Deep bottleneck DLN with specified total depth.

    The network tapers down to a bottleneck then expands back up.

    Args:
        input_dim: Input dimension
        output_dim: Output dimension
        bottleneck_dim: Minimum width at the bottleneck
        depth: Total number of layers

    Returns:
        nn.Sequential DLN with bottleneck structure
    """
    if depth < 3:
        raise ValueError("Deep bottleneck DLN requires at least 3 layers")

    # Calculate dimensions - taper down then back up
    layers_down = depth // 2
    layers_up = depth - layers_down - 1  # -1 for bottleneck layer

    # Dimensions tapering down to bottleneck
    down_dims = torch.linspace(input_dim, bottleneck_dim, layers_down + 1).int()

    # Dimensions expanding up from bottleneck
    up_dims = torch.linspace(bottleneck_dim, output_dim, layers_up + 1).int()

    # Combine (remove duplicate bottleneck dim)
    layer_dims = torch.cat([down_dims, up_dims[1:]]).tolist()

    return DLN(layer_dims)


# For backward compatibility, map BottleneckDLN to DeepBottleneckDLN in zoo
BottleneckDLN = DeepBottleneckDLN


def PolynomialDLN(input_dim: int = 1, degree: int = 3) -> nn.Sequential:
    """DLN that can represent polynomials up to given degree.

    Each layer doubles the dimension to create polynomial features.
    """
    layers = [input_dim]
    for _ in range(degree):
        layers.append(layers[-1] * 2)
    layers.append(1)  # Output scalar
    return DLN(layers)


# Matrix factorization models
def LowRankDLN(input_dim: int, output_dim: int, rank: int) -> nn.Sequential:
    """Two-layer DLN that learns low-rank matrix factorization."""
    return DLN([input_dim, rank, output_dim])


def MatrixCompletion(matrix_shape: tuple[int, int], rank: int) -> nn.Sequential:
    """DLN for matrix completion via factorization."""
    m, n = matrix_shape
    return DLN([m, rank, n])


# Compression/autoencoder models
def LinearAutoencoder(input_dim: int, encoding_dim: int) -> nn.Sequential:
    """Linear autoencoder with symmetric encoder-decoder."""
    return SymmetricDLN(input_dim, encoding_dim, input_dim)


def ProgressiveCompression(
    input_dim: int, compression_ratio: float = 0.5, num_layers: int = 4
) -> nn.Sequential:
    """DLN that progressively compresses input."""
    layers = [input_dim]
    current_dim = input_dim

    for _ in range(num_layers - 1):
        current_dim = max(1, int(current_dim * compression_ratio))
        layers.append(current_dim)

    return DLN(layers)


# Function approximation models
def FourierDLN(input_dim: int = 1, num_frequencies: int = 10) -> nn.Sequential:
    """DLN for learning Fourier-like representations."""
    return DLN([input_dim, num_frequencies * 2, num_frequencies, 1])


def ChebyshevDLN(input_dim: int = 1, degree: int = 10) -> nn.Sequential:
    """DLN for Chebyshev polynomial approximation."""
    return DLN([input_dim, degree, degree // 2, 1])


# Ensemble/multi-task models
def MultiTaskDLN(
    input_dim: int, num_tasks: int, shared_layers: list[int], task_layers: list[int]
) -> dict[str, nn.Sequential]:
    """Create multiple DLNs with shared initial layers."""
    shared = DLN([input_dim] + shared_layers)

    tasks = {}
    for i in range(num_tasks):
        task_specific = DLN([shared_layers[-1]] + task_layers)
        tasks[f"task_{i}"] = nn.Sequential(shared, task_specific)

    return tasks


# Testing and validation models
def IdentityDLN(dim: int) -> nn.Sequential:
    """DLN that should learn identity function."""
    return DLN([dim, dim * 2, dim])


def LinearRegressionDLN(input_dim: int) -> nn.Sequential:
    """Single layer for linear regression baseline."""
    return DLN([input_dim, 1])


def DeepLinearRegression(
    input_dim: int, depth: int = 5, width: int = None
) -> nn.Sequential:
    """Deep linear regression with specified depth."""
    if width is None:
        width = input_dim
    return DLN([input_dim] + [width] * depth + [1])


# Pathological cases for analysis
def RankDeficientDLN(input_dim: int, output_dim: int, min_rank: int) -> nn.Sequential:
    """DLN designed to have rank deficiency issues."""
    return DLN([input_dim, min_rank, min_rank, output_dim])


def VanishingGradientDLN(
    input_dim: int, output_dim: int, depth: int = 20
) -> nn.Sequential:
    """Very deep DLN to study vanishing gradients."""
    narrow_width = max(1, min(input_dim, output_dim) // 4)
    return DLN([input_dim] + [narrow_width] * depth + [output_dim])


def ExplodingGradientDLN(input_dim: int, output_dim: int) -> nn.Sequential:
    """Wide DLN that may have exploding gradient issues."""
    wide_width = input_dim * 10
    return DLN([input_dim, wide_width, wide_width, output_dim])


# Model collections
CLASSIC_MODELS = {
    "lenet": LeNetLinear,
    "alexnet": AlexNetLinear,
    "resnet18": ResNetLinear18,
    "vgg11": VGGLinear11,
}

TOY_MODELS = {
    "tiny": TinyDLN,
    "symmetric": SymmetricDLN,
    "overparameterized": OverparameterizedDLN,
    "polynomial": PolynomialDLN,
    "deep_bottleneck": DeepBottleneckDLN,
}

RESEARCH_MODELS = {
    "low_rank": LowRankDLN,
    "matrix_completion": MatrixCompletion,
    "autoencoder": LinearAutoencoder,
    "progressive_compression": ProgressiveCompression,
    "fourier": FourierDLN,
    "chebyshev": ChebyshevDLN,
}

PATHOLOGICAL_MODELS = {
    "rank_deficient": RankDeficientDLN,
    "vanishing_gradient": VanishingGradientDLN,
    "exploding_gradient": ExplodingGradientDLN,
}


# Convenience function
def get_model(name: str, **kwargs) -> nn.Sequential:
    """Get a model by name from all collections."""
    all_models = {
        **CLASSIC_MODELS,
        **TOY_MODELS,
        **RESEARCH_MODELS,
        **PATHOLOGICAL_MODELS,
    }

    if name not in all_models:
        available = list(all_models.keys())
        raise ValueError(f"Model '{name}' not found. Available: {available}")

    return all_models[name](**kwargs)
