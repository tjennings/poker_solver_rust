"""BoundaryNet model matching the Rust architecture."""

import torch
import torch.nn as nn

from cfvnet.constants import INPUT_SIZE, OUTPUT_SIZE


class HiddenBlock(nn.Module):
    """Linear -> BatchNorm1d -> PReLU block."""

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.norm = nn.BatchNorm1d(out_features)
        self.activation = nn.PReLU(num_parameters=out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through linear, batchnorm, and PReLU."""
        return self.activation(self.norm(self.linear(x)))


class BoundaryNet(nn.Module):
    """Boundary value network for depth-bounded range solving.

    Architecture: Input(2720) -> [Linear -> BN -> PReLU] x N -> Linear(1326)
    """

    def __init__(self, num_layers: int, hidden_size: int) -> None:
        """Create a BoundaryNet.

        Args:
            num_layers: Number of hidden blocks.
            hidden_size: Width of each hidden layer.
        """
        super().__init__()
        assert num_layers > 0, "need at least one hidden layer"

        layers: list[HiddenBlock] = []
        layers.append(HiddenBlock(INPUT_SIZE, hidden_size))
        for _ in range(1, num_layers):
            layers.append(HiddenBlock(hidden_size, hidden_size))
        self.hidden = nn.ModuleList(layers)
        self.output = nn.Linear(hidden_size, OUTPUT_SIZE)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: returns [batch, OUTPUT_SIZE] normalized EVs."""
        for block in self.hidden:
            x = block(x)
        return self.output(x)
