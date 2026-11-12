"""
uq_baselines.py  —  Layer-B uncertainty comparators (same GRU backbone)
=======================================================================
These share the SAME GRU backbone as the proposed method, so any difference in
calibration/coverage comes from the UQ technique, not the network. They let us
prove ACC-HD's calibration + adaptive-conformal choices beat the standard ones.

Provided:
  build_softmax(...)          -> a GRU + softmax classifier (cross-entropy)
The single trained softmax model is reused for several comparators:
  - "softmax"        : raw softmax confidence (uncalibrated)
  - "temp_scaling"   : + temperature scaling (models/calibration.py)
  - "mc_dropout"     : average several dropout-on forward passes (models/train.predict_probs)
  - "static_conformal": softmax probs + fixed split conformal (models/conformal.py)
A deep ENSEMBLE trains several such models with different seeds and averages them.
"""

import numpy as np
import torch.nn as nn

from models.backbone import GRUBackbone
from models.heads import SoftmaxHead


class SoftmaxGRU(nn.Module):
    """GRU backbone + softmax head. Forward returns logits."""

    def __init__(self, num_features, num_classes, hidden_size=64, dropout=0.2):
        super().__init__()
        self.backbone = GRUBackbone(num_features, hidden_size, dropout=dropout)
        self.head = SoftmaxHead(hidden_size, num_classes)

    def forward(self, x):
        return self.head(self.backbone(x))            # logits


def build_softmax(num_features, num_classes, hidden_size=64, dropout=0.2):
    """Return (model, loss_fn, edl_flag=False) for the shared training loop."""
    model = SoftmaxGRU(num_features, num_classes, hidden_size, dropout)
    return model, nn.CrossEntropyLoss(), False


def ensemble_average(prob_list):
    """Average a list of (N, K) probability arrays -> one (N, K) array."""
    return np.mean(np.stack(prob_list, axis=0), axis=0)
