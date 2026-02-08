"""
Deep Linear Network constructors.

Provides convenient helpers for building DLN architectures using PyTorch's
nn.Sequential and nn.Linear layers.
"""

from typing import List, Optional

import torch
import torch.nn as nn


def DLN(
    layer_dims: List[int],
    bias: bool = False,
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
) -> nn.Sequential:
    """
    Construct a Deep Linear Network.

    Creates a sequential model of linear layers without activation functions.
    This is the fundamental building block for DLN research.

    Args:
        layer_dims: List of layer dimensions [input_dim, hidden1, hidden2, ..., output_dim]
        bias: Whether to include bias terms (typically False for DLNs)
        dtype: Data type for parameters
        device: Device to place parameters on

    Returns:
        nn.Sequential model representing the DLN

    Example:
        # 2D input → 5D hidden → 3D hidden → 1D output
        model = DLN([2, 5, 3, 1])

        # Equivalent to:
        # nn.Sequential(
        #     nn.Linear(2, 5, bias=False),
        #     nn.Linear(5, 3, bias=False),
        #     nn.Linear(3, 1, bias=False)
        # )
    """
    if len(layer_dims) < 2:
        raise ValueError("Need at least input and output dimensions")

    layers = []
    for i in range(len(layer_dims) - 1):
        layer = nn.Linear(
            layer_dims[i], layer_dims[i + 1], bias=bias, dtype=dtype, device=device
        )
        layers.append(layer)

    return nn.Sequential(*layers)


def WideDLN(
    input_dim: int, output_dim: int, width: int, depth: int, **kwargs
) -> nn.Sequential:
    """
    Create a wide DLN with constant hidden layer width.

    Args:
        input_dim: Input dimension
        output_dim: Output dimension
        width: Width of all hidden layers
        depth: Number of hidden layers
        **kwargs: Passed to DLN()

    Returns:
        Wide DLN model

    Example:
        model = WideDLN(input_dim=2, output_dim=1, width=10, depth=3)
        # Creates: 2 → 10 → 10 → 10 → 1
    """
    layer_dims = [input_dim] + [width] * depth + [output_dim]
    return DLN(layer_dims, **kwargs)


def BottleneckDLN(
    input_dim: int, output_dim: int, bottleneck_dim: int, **kwargs
) -> nn.Sequential:
    """
    Create a bottleneck DLN: input → bottleneck → output

    This creates interesting singular structure when bottleneck_dim < min(input_dim, output_dim).

    Args:
        input_dim: Input dimension
        output_dim: Output dimension
        bottleneck_dim: Bottleneck (hidden layer) dimension
        **kwargs: Passed to DLN()

    Returns:
        Bottleneck DLN model

    Example:
        model = BottleneckDLN(input_dim=10, output_dim=5, bottleneck_dim=2)
        # Creates: 10 → 2 → 5 (singular due to rank constraint)
    """
    layer_dims = [input_dim, bottleneck_dim, output_dim]
    return DLN(layer_dims, **kwargs)


def PyramidDLN(
    input_dim: int, output_dim: int, n_layers: int, **kwargs
) -> nn.Sequential:
    """
    Create a pyramid-shaped DLN with linearly decreasing width.

    Args:
        input_dim: Input dimension
        output_dim: Output dimension
        n_layers: Total number of layers (including input→hidden and hidden→output)
        **kwargs: Passed to DLN()

    Returns:
        Pyramid DLN model

    Example:
        model = PyramidDLN(input_dim=10, output_dim=2, n_layers=4)
        # Creates something like: 10 → 7 → 4 → 2
    """
    if n_layers < 2:
        raise ValueError("Need at least 2 layers")

    # Create linearly spaced dimensions
    dims = torch.linspace(input_dim, output_dim, n_layers + 1)
    layer_dims = [int(round(float(d))) for d in dims]

    # Ensure we hit exact input and output dims
    layer_dims[0] = input_dim
    layer_dims[-1] = output_dim

    return DLN(layer_dims, **kwargs)
