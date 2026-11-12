"""
b1_deep_hybrid.py  —  Baseline B1: deterministic deep hybrid predictor
======================================================================
Represents the "deterministic deep learning" class of predictive-handover
methods (LSTM + GRU + CNN + dense layers), producing a single point prediction
with a plain softmax (no uncertainty quantification).

Reference class: Wireless Networks (Springer) 2025 deep-hybrid handover predictor.
Trained and evaluated with the SAME loop as every other model.
"""

import torch
import torch.nn as nn


class DeepHybrid(nn.Module):
    """CNN over features + LSTM + GRU over time + dense classifier (softmax)."""

    def __init__(self, num_features, num_classes, hidden=64):
        super().__init__()
        # 1-D CNN across the feature dimension at each time step
        self.conv = nn.Sequential(
            nn.Conv1d(num_features, hidden, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.lstm = nn.LSTM(hidden, hidden, batch_first=True)
        self.gru = nn.GRU(hidden, hidden, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x):
        # x: (batch, window_len, num_features) -> conv wants (batch, features, time)
        h = self.conv(x.transpose(1, 2)).transpose(1, 2)   # (batch, time, hidden)
        h, _ = self.lstm(h)
        h, _ = self.gru(h)
        return self.head(h[:, -1, :])                       # logits


def build(num_features, num_classes, hidden=64):
    """Return (model, loss_fn, edl_flag) for training with the shared loop."""
    model = DeepHybrid(num_features, num_classes, hidden)
    loss_fn = nn.CrossEntropyLoss()
    return model, loss_fn, False
