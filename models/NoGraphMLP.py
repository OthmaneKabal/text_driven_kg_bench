"""A text-feature classifier that deliberately ignores graph structure."""

from __future__ import annotations

import torch
from torch import nn


class NoGraphMLP(nn.Module):
    """One-hidden-layer MLP for the no-graph baseline."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_classes: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if min(in_channels, hidden_channels, num_classes) < 1:
            raise ValueError("in_channels, hidden_channels and num_classes must all be >= 1")
        self.input_layer = nn.Linear(in_channels, hidden_channels)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_channels, num_classes)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.input_layer.reset_parameters()
        self.classifier.reset_parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.dropout(self.activation(self.input_layer(x))))
