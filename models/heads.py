"""
heads.py
========
The output layers ("heads") placed on top of the GRU backbone, plus their loss
functions.

Two heads:
  1. SoftmaxHead      - standard classifier (used by the uncertainty baselines).
  2. EvidentialHead   - the PRIMARY head of ACC-HD. Instead of a plain softmax
                        it predicts Dirichlet "evidence", which gives BOTH the
                        class probabilities AND an epistemic-uncertainty signal
                        in a single forward pass (cheap -> good for real time).

Evidential deep learning (Sensoy et al., 2018; modern variants 2023-2024):
  evidence e_k >= 0,  alpha_k = e_k + 1,  S = sum_k alpha_k
  prob_k = alpha_k / S            (expected class probability)
  vacuity (epistemic uncertainty) = num_classes / S   (high when little evidence)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftmaxHead(nn.Module):
    """Plain linear classifier -> logits. Train with cross-entropy."""

    def __init__(self, hidden_size: int, num_classes: int):
        super().__init__()
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.fc(h)                     # logits (batch, num_classes)


class EvidentialHead(nn.Module):
    """Evidential (Dirichlet) head. Returns alpha parameters (all > 1)."""

    def __init__(self, hidden_size: int, num_classes: int):
        super().__init__()
        self.fc = nn.Linear(hidden_size, num_classes)
        self.num_classes = num_classes

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        evidence = F.softplus(self.fc(h))     # e_k >= 0
        alpha = evidence + 1.0                # alpha_k >= 1
        return alpha


def focal_loss(logits: torch.Tensor, target: torch.Tensor, gamma: float = 2.0):
    """Focal cross-entropy. Down-weights easy, already-correct samples, which
    improves CALIBRATION during training (the proposed method's calibration).
    """
    logp = F.log_softmax(logits, dim=1)
    p = logp.exp()
    logp_t = logp.gather(1, target.unsqueeze(1)).squeeze(1)
    p_t = p.gather(1, target.unsqueeze(1)).squeeze(1)
    return torch.mean(-((1 - p_t) ** gamma) * logp_t)


def evidential_probs(alpha: torch.Tensor) -> torch.Tensor:
    """Expected class probabilities from Dirichlet alpha."""
    return alpha / alpha.sum(dim=1, keepdim=True)


def evidential_vacuity(alpha: torch.Tensor) -> torch.Tensor:
    """Epistemic uncertainty (vacuity) = K / S, in (0, 1]."""
    K = alpha.shape[1]
    S = alpha.sum(dim=1, keepdim=True)
    return (K / S).squeeze(1)


def edl_loss(alpha: torch.Tensor, target: torch.Tensor,
             epoch: int, anneal_epochs: int = 10) -> torch.Tensor:
    """Evidential loss (Type-II MSE / Bayes-risk form, Sensoy et al. 2018).

    Uses ONLY basic tensor ops (no digamma/lgamma), so it runs on every GPU.
    (Some CUDA runtimes fail to JIT-compile the digamma kernel, which is
    why we avoid it.) It still yields calibrated Dirichlet uncertainty.

    NOTE (fix vs code-3): the previous version added a wrong-class evidence
    penalty with weight 0.1, which drove ALL evidence toward zero -> the
    Dirichlet collapsed to uniform and accuracy fell to ~random (0.118). The
    Bayes-risk term (err + var) is itself a valid evidential loss and trains
    stably; we keep only a VERY small (1e-3), annealed, (K-1)-normalized penalty
    on wrong-class evidence so it can never dominate the fit.
    """
    K = alpha.shape[1]
    S = alpha.sum(dim=1, keepdim=True)
    p = alpha / S                                   # expected probability
    y = F.one_hot(target, num_classes=K).float()

    # Bayes risk of the sum-of-squares loss = squared error + variance term
    err = torch.sum((y - p) ** 2, dim=1)
    var = torch.sum(p * (1.0 - p) / (S + 1.0), dim=1)

    # tiny, safe regulariser (normalized, weight 1e-3): discourages excess
    # wrong-class evidence WITHOUT collapsing the model.
    wrong_evidence = torch.sum((1.0 - y) * (alpha - 1.0), dim=1) / max(1, K - 1)
    anneal = min(1.0, epoch / max(1, anneal_epochs))
    lam = 1e-3 * anneal
    return torch.mean(err + var + lam * wrong_evidence)
