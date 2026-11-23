"""
Feed-Forward Network (FFN) Module
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .config import SentriFormerConfig


class FeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network:
    Linear(d_model -> d_ff) -> GELU -> Dropout -> Linear(d_ff -> d_model)
    """
    def __init__(self, config: SentriFormerConfig):
        super().__init__()
        self.fc1 = nn.Linear(config.d_model, config.d_ff, bias=config.use_bias)
        self.fc2 = nn.Linear(config.d_ff, config.d_model, bias=config.use_bias)
        self.dropout = nn.Dropout(config.dropout)
        
        if config.activation_function == "gelu":
            self.act_fn = nn.GELU()
        elif config.activation_function == "relu":
            self.act_fn = nn.ReLU()
        else:
            self.act_fn = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x
