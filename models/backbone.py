"""
backbone.py
===========
The shared temporal feature extractor. It reads a window of measurements
(shape: window_len x num_features) and produces one feature vector that
summarises the recent radio history of the UE.

We use a GRU (a standard, light recurrent network). The SAME backbone is used
by the proposed method and by the uncertainty baselines, so comparisons are fair.
"""

import torch
import torch.nn as nn


class GRUBackbone(nn.Module):
    """GRU encoder: (batch, window_len, num_features) -> (batch, hidden_size)."""

    def __init__(self, num_features: int, hidden_size: int = 64,
                 num_layers: int = 1, dropout: float = 0.2):
        super().__init__()
        self.gru = nn.GRU(
            input_size=num_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        # a dropout layer AFTER the GRU lets us do MC-Dropout at inference time
        self.dropout = nn.Dropout(dropout)
        self.hidden_size = hidden_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, window_len, num_features)
        out, _ = self.gru(x)          # out: (batch, window_len, hidden)
        last = out[:, -1, :]          # take the last time step
        return self.dropout(last)     # (batch, hidden)
