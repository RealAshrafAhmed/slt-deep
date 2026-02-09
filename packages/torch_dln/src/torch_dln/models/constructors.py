"""
Deep Linear Network constructors.

Provides convenient helpers for building DLN architectures using PyTorch's
nn.Sequential and nn.Linear layers.
"""

import time
from enum import Enum

import torch
import torch.nn as nn


class InitMethod(Enum):
    """Initialization methods for DeepLinearNetwork weights."""

    XAVIER_NORMAL = "xavier_normal"
    XAVIER_UNIFORM = "xavier_uniform"
    KAIMING = "kaiming"
    NORMAL = "normal"
    ZEROS = "zeros"


class DeepLinearNetwork(nn.Module):
    """
    Deep Linear Network: A multi-layer linear transformation without activations.

    This is the fundamental building block for studying singular learning theory,
    providing a clean interface for research-focused functionality like analyzing
    weight matrices, tracking gradients, and studying generalization properties.
    """

    def __init__(
        self,
        layer_dims: list[int],
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
        init_method: InitMethod = InitMethod.XAVIER_NORMAL,
        name: str | None = None,
        store_activations: bool = False,
    ):
        """
        Initialize Deep Linear Network.

        Args:
            layer_dims: List of layer dimensions [input_dim, hidden1, hidden2, ..., output_dim]
            dtype: Data type for parameters
            device: Device to place parameters on
            init_method: Initialization method (InitMethod enum)
            name: Optional name for the model (useful for experiments)
            store_activations: Whether to store intermediate activations during forward pass
        """
        super().__init__()

        if len(layer_dims) < 2:
            raise ValueError("Need at least input and output dimensions")

        # Store architecture metadata
        self.layer_dims = layer_dims.copy()
        self.n_layers = len(layer_dims) - 1
        self.input_dim = layer_dims[0]
        self.output_dim = layer_dims[-1]
        self.hidden_dims = layer_dims[1:-1]
        self.init_method = init_method
        self.name = name
        self.store_activations = store_activations

        # Store creation metadata (useful for research)
        self._creation_time = time.time()
        self._last_activations: list[torch.Tensor] = []

        # Build layers
        self.layers: nn.ModuleList[nn.Linear] = nn.ModuleList()
        for i in range(len(layer_dims) - 1):
            layer = nn.Linear(
                layer_dims[i], layer_dims[i + 1], bias=False, dtype=dtype, device=device
            )
            self.layers.append(layer)

        # Initialize parameters
        self.reset_parameters(init_method)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the deep linear network.

        Args:
            x: Input tensor of shape (batch_size, input_dim)

        Returns:
            Output tensor of shape (batch_size, output_dim)
        """
        if self.store_activations:
            self._last_activations = []

        for _i, layer in enumerate(self.layers):
            x = layer(x)
            if self.store_activations:
                self._last_activations.append(x.detach().clone())

        return x

    def reset_parameters(self, init_method: InitMethod | None = None) -> None:
        """Reset all parameters using specified initialization method."""
        method = init_method or self.init_method

        layer: nn.Linear
        for layer in self.layers:
            if method == InitMethod.XAVIER_NORMAL:
                nn.init.xavier_normal_(layer.weight)
            elif method == InitMethod.XAVIER_UNIFORM:
                nn.init.xavier_uniform_(layer.weight)
            elif method == InitMethod.KAIMING:
                nn.init.kaiming_normal_(layer.weight)
            elif method == InitMethod.NORMAL:
                nn.init.normal_(layer.weight, std=0.01)
            elif method == InitMethod.ZEROS:
                nn.init.zeros_(layer.weight)
            else:
                raise ValueError(f"Unknown initialization method: {method}")

    @property
    def depth(self) -> int:
        """Number of linear transformations."""
        return self.n_layers

    @property
    def width(self) -> int:
        """Maximum hidden layer width (useful for wide networks)."""
        return max(self.hidden_dims) if self.hidden_dims else 0

    def effective_rank(self, layer_idx: int | None = None) -> dict[str, torch.Tensor]:
        """
        Compute effective rank of weight matrices.

        Args:
            layer_idx: If specified, only compute for that layer. Otherwise compute for all.

        Returns:
            Dictionary mapping layer names to effective rank values
        """
        ranks = {}
        layers_to_analyze = (
            [layer_idx] if layer_idx is not None else range(len(self.layers))
        )

        for i in layers_to_analyze:
            layer: nn.Linear = self.layers[i]
            # Effective rank using normalized singular values
            _, s, _ = torch.svd(layer.weight)
            # Effective rank as sum of normalized singular values
            s_normalized = s / s.max()
            effective_rank = torch.sum(s_normalized)
            ranks[f"layer_{i}"] = effective_rank

        return ranks

    def get_weight_norms(self) -> dict[str, float]:
        """Get L2 norms of all weight matrices."""
        norms = {}
        layer: nn.Linear
        for i, layer in enumerate(self.layers):
            norms[f"layer_{i}"] = float(torch.norm(layer.weight.detach(), p=2))
        return norms

    def get_activations(self) -> list[torch.Tensor]:
        """Get stored activations from last forward pass."""
        if not self.store_activations:
            raise RuntimeError(
                "store_activations=False. Set to True to capture activations."
            )
        return self._last_activations.copy()

    def copy(self) -> "DeepLinearNetwork":
        """Create a deep copy for experiments."""
        new_model = DeepLinearNetwork(
            layer_dims=self.layer_dims,
            dtype=next(self.parameters()).dtype,
            device=next(self.parameters()).device,
            init_method=self.init_method,
            name=f"{self.name}_copy" if self.name else None,
            store_activations=self.store_activations,
        )
        new_model.load_state_dict(self.state_dict())
        return new_model

    def __repr__(self) -> str:
        dims_str = " → ".join(map(str, self.layer_dims))
        layers_info = f"layers={self.n_layers}"
        params_info = f"params={sum(p.numel() for p in self.parameters())}"

        extra = []
        if self.name:
            extra.append(f"name='{self.name}'")
        if self.store_activations:
            extra.append("store_activations=True")

        extra_str = f", {', '.join(extra)}" if extra else ""

        return f"DeepLinearNetwork({dims_str}, {layers_info}, {params_info}{extra_str})"


def dln(
    layer_dims: list[int],
    dtype: torch.dtype | None = None,
    device: torch.device | None = None,
    **kwargs,
) -> DeepLinearNetwork:
    """
    Construct a Deep Linear Network.

    Creates a DeepLinearNetwork model with linear layers without activation functions.
    This is the fundamental building block for DLN research.

    Args:
        layer_dims: List of layer dimensions [input_dim, hidden1, hidden2, ..., output_dim]
        dtype: Data type for parameters
        device: Device to place parameters on
        **kwargs: Additional arguments passed to DeepLinearNetwork

    Returns:
        DeepLinearNetwork model representing the deep linear network

    Example:
        # 2D input → 5D hidden → 3D hidden → 1D output
        model = dln([2, 5, 3, 1])

        # With specific device
        model = dln([2, 5, 3, 1], device='cuda')
    """
    return DeepLinearNetwork(
        layer_dims=layer_dims, dtype=dtype, device=device, **kwargs
    )


def wdln(
    input_dim: int, output_dim: int, width: int, depth: int, **kwargs
) -> DeepLinearNetwork:
    """
    Create a wide DLN with constant hidden layer width.

    Args:
        input_dim: Input dimension
        output_dim: Output dimension
        width: Width of all hidden layers
        depth: Number of hidden layers
        **kwargs: Passed to dln()

    Returns:
        Wide DLN model

    Example:
        model = wdln(input_dim=2, output_dim=1, width=10, depth=3)
        # Creates: 2 → 10 → 10 → 10 → 1
    """
    layer_dims = [input_dim] + [width] * depth + [output_dim]
    return dln(layer_dims, **kwargs)


def bndln(
    input_dim: int, output_dim: int, bottleneck_dim: int, **kwargs
) -> DeepLinearNetwork:
    """
    Create a bottleneck DLN: input → bottleneck → output

    This creates interesting singular structure when bottleneck_dim < min(input_dim, output_dim).

    Args:
        input_dim: Input dimension
        output_dim: Output dimension
        bottleneck_dim: Bottleneck (hidden layer) dimension
        **kwargs: Passed to dln()

    Returns:
        Bottleneck DLN model

    Example:
        model = bndln(input_dim=10, output_dim=5, bottleneck_dim=2)
        # Creates: 10 → 2 → 5 (singular due to rank constraint)
    """
    layer_dims = [input_dim, bottleneck_dim, output_dim]
    return dln(layer_dims, **kwargs)


def pdln(input_dim: int, output_dim: int, n_layers: int, **kwargs) -> DeepLinearNetwork:
    """
    Create a pyramid-shaped DLN with linearly decreasing width.

    Args:
        input_dim: Input dimension
        output_dim: Output dimension
        n_layers: Total number of layers (including input→hidden and hidden→output)
        **kwargs: Passed to dln()

    Returns:
        Pyramid DLN model

    Example:
        model = pdln(input_dim=10, output_dim=2, n_layers=4)
        # Creates something like: 10 → 7 → 4 → 2
    """
    if n_layers < 2:
        raise ValueError("Need at least 2 layers")

    # Create linearly spaced dimensions
    dims = torch.linspace(input_dim, output_dim, n_layers + 1)
    layer_dims = [round(d.item()) for d in dims]

    # Ensure we hit exact input and output dims
    layer_dims[0] = input_dim
    layer_dims[-1] = output_dim

    return dln(layer_dims, **kwargs)
