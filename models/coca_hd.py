"""
coca_hd.py
==========
Assembles the PROPOSED method: ACC-HD
(Conformal-Calibrated Handover Decisions).

It wires together the pieces from the other files:
  backbone (GRU) + evidential head  ->  calibrated probabilities
  -> per-service-class conformal prediction sets (with risk control)
  -> confidence-aware policy (HO / HOLD / CHO)

This file only BUILDS the model and its loss; training and evaluation are run by
experiments/run_all.py so that every method goes through the same procedure.
"""

import torch.nn as nn

from models.backbone import GRUBackbone
from models.heads import EvidentialHead, edl_loss


class CoCAModel(nn.Module):
    """GRU backbone + evidential head. Forward returns Dirichlet alpha."""

    def __init__(self, num_features, num_classes, hidden_size=64, dropout=0.2):
        super().__init__()
        self.backbone = GRUBackbone(num_features, hidden_size, dropout=dropout)
        self.head = EvidentialHead(hidden_size, num_classes)

    def forward(self, x):
        return self.head(self.backbone(x))        # alpha (batch, num_classes)


def build_coca(num_features, num_classes, hidden_size=64, dropout=0.2):
    """Return (model, loss_fn, edl_flag) for the proposed method.

    loss_fn(alpha, target, epoch) -> scalar (edl_loss handles KL annealing).
    """
    model = CoCAModel(num_features, num_classes, hidden_size, dropout)

    def loss_fn(alpha, target, epoch):
        return edl_loss(alpha, target, epoch)

    return model, loss_fn, True                    # edl_flag = True
