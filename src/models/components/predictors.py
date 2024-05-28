from typing import Any, Dict, Optional, Tuple

import os
import numpy as np
import torch
import torch.nn as nn

class LinearRegressionModel(torch.nn.Module):
    def __init__(self, 
                 input_dim: int):
        super().__init__()
        
        self.linear = torch.nn.Linear(input_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)
    
class MLP(torch.nn.Module):
    def __init__(self, input_dim: int, 
                 hidden_dim: int,
                 output_dim: int,
                 layer_activation: nn.Module = nn.ReLU(),):
        
        super().__init__()
        self.layer_activation = layer_activation
        self.fc1 = torch.nn.Linear(input_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.layer_activation(x)
        x = self.fc2(x)
        return x