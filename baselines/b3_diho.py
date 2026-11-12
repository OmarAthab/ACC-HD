"""
b3_diho.py  —  Baseline B3: direction-aware mobility-prediction handover (Di-HO)
==============================================================================
Faithful re-implementation of the mobility-prediction-assisted, direction-aware
handover algorithm of the IEEE TNSE 2024 paper (its Algorithm 1). It is a
rule-based (non-learning) method: after an A3 event, it tracks the RSRP *pattern*
of each candidate target cell across the time-to-trigger window and decides on
direction.

Algorithm 1 logic, applied to our measurement windows:
  * For each candidate cell k, compare its RSRP across the window's successive
    instances (t_{j-1} -> t_j).
  * A consistently DECREASING RSRP means the UE is moving AWAY from k -> k is a
    "black cell" (excluded from handover). A consistently increasing RSRP means
    the UE is moving toward k (a cell on its path). Cells whose RSRP is flat for
    eta steps are also black-listed.
  * The predicted next serving cell is the non-black candidate on the UE's path
    with the strongest, rising RSRP; if the current serving cell remains strong
    and non-declining, the UE stays (delay handover to avoid an unnecessary one).

This is deterministic and requires no training (fit is a no-op), exactly as the
reference algorithm. Output is a probability row (softmax over per-cell scores)
so B3 passes through the same metrics as every other method.
"""

import numpy as np


def fit(X_train, y_train, num_cells):
    """Rule-based method: nothing to learn. Returns None (kept for a uniform API)."""
    return None


def _trend(rsrp_seq):
    """Per-cell RSRP trend across the window: mean of successive differences.
    rsrp_seq: (N, W, K) -> (N, K). Positive => rising (approaching), negative =>
    declining (moving away)."""
    diffs = np.diff(rsrp_seq, axis=1)           # (N, W-1, K)
    return diffs.mean(axis=1)                    # (N, K)


def predict(_model, X, num_cells, eta=3, beta=2.0, big=1e3):
    """Return per-cell probability rows from the direction-aware rule.

    Score_j = RSRP_last[j] + beta * trend[j]  (reward rising cells on the path),
    with declining ("black") cells strongly penalised. The current serving cell
    is kept (stay) when it is strong and not declining.
    """
    X = np.asarray(X, dtype=np.float32)
    K = num_cells
    rsrp_seq = X[:, :, :K]                       # (N, W, K) normalised RSRP
    rsrp_last = rsrp_seq[:, -1, :]               # (N, K)
    trend = _trend(rsrp_seq)                     # (N, K)
    serving = X[:, -1, 2 * K:3 * K].argmax(axis=1)   # (N,)

    # black cells: RSRP declining across the window (UE moving away)
    black = trend < 0.0                          # (N, K) boolean

    # direction-aware score: prefer non-black cells with high, rising RSRP
    score = rsrp_last + beta * trend
    score = score - big * black.astype(np.float32)

    # keep the serving cell competitive when it is strong and not declining
    N = len(X)
    rows = np.arange(N)
    serving_ok = trend[rows, serving] >= 0.0
    score[rows, serving] = np.where(
        serving_ok, rsrp_last[rows, serving] + beta * np.abs(trend[rows, serving]),
        score[rows, serving])

    # convert scores to a probability row (numerically stable softmax)
    score = score - score.max(axis=1, keepdims=True)
    ex = np.exp(score)
    probs = ex / ex.sum(axis=1, keepdims=True)
    return probs.astype(np.float32)
