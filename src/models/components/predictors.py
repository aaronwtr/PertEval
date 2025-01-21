from typing import Any, Dict, Optional, Tuple

import os
import numpy as np
import torch
import torch.nn as nn


class LinearRegressionModel(torch.nn.Module):
    def __init__(self,
                 in_dim: int):
        super().__init__()

        self.linear = torch.nn.Linear(in_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class MLP(torch.nn.Module):
    def __init__(self, in_dim: int, hidden_dim: float, out_dim: int, num_layers: int,
                 layer_activation: nn.Module = nn.ReLU()):
        super().__init__()
        hidden_dim = int(hidden_dim)
        self.layer_activation = layer_activation

        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(in_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.layers.append(nn.Linear(hidden_dim, out_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers[:-1]:
            x = self.layer_activation(layer(x))
        x = self.layers[-1](x)
        return x


class MeanExpression(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean_expr = torch.mean(x[:, :x.shape[1] // 2], dim=0)
        return mean_expr - x[:, :x.shape[1] // 2]
