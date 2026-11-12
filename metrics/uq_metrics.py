"""
uq_metrics.py
=============
Prediction-quality and uncertainty-quality metrics.

- accuracy, macro_f1     : how often the predicted target cell is right
- ece                    : calibration error (re-exported from models.calibration)
- confidence_reliability : fraction of test points whose confidence is
                           "reliable", i.e. the true class is inside the
                           conformal set (a practical trust measure)
- coverage, avg_set_size : from conformal prediction sets
"""

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score

from models.calibration import expected_calibration_error


def accuracy(probs, labels) -> float:
    return float((np.asarray(probs).argmax(1) == np.asarray(labels)).mean())


def macro_f1(probs, labels) -> float:
    pred = np.asarray(probs).argmax(1)
    return float(f1_score(np.asarray(labels), pred, average="macro",
                          zero_division=0))


def macro_precision_recall(probs, labels):
    """Macro precision and recall (imbalance-aware, on the FULL test set)."""
    pred = np.asarray(probs).argmax(1); labels = np.asarray(labels)
    p = precision_score(labels, pred, average="macro", zero_division=0)
    r = recall_score(labels, pred, average="macro", zero_division=0)
    return float(p), float(r)


def ece(probs, labels) -> float:
    return expected_calibration_error(probs, labels)


def adaptive_ece(probs, labels, n_bins=15) -> float:
    """Adaptive ECE (ACE): calibration error with EQUAL-MASS (quantile) bins
    instead of equal-width bins, so it is robust to the confidence histogram
    shape. Complementary to ECE (Nixon et al.; sources.md #36)."""
    probs = np.asarray(probs); labels = np.asarray(labels)
    conf = probs.max(axis=1)
    correct = (probs.argmax(axis=1) == labels).astype(float)
    order = np.argsort(conf); conf = conf[order]; correct = correct[order]
    n = len(conf)
    if n == 0:
        return float("nan")
    edges = np.linspace(0, n, n_bins + 1).astype(int)
    ace = 0.0
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        if hi <= lo:
            continue
        ace += (hi - lo) / n * abs(correct[lo:hi].mean() - conf[lo:hi].mean())
    return float(ace)


def confidence_reliability(sets, labels) -> float:
    """Fraction of samples whose true class is inside the (conformal) set.

    This equals empirical coverage; reported as a percentage-style trust score.
    """
    sets = np.asarray(sets); labels = np.asarray(labels)
    return float(sets[np.arange(len(labels)), labels].mean())


def summarize(probs, labels, sets=None) -> dict:
    """Bundle the common metrics into one dict row."""
    prec, rec = macro_precision_recall(probs, labels)
    row = {
        "accuracy": accuracy(probs, labels),
        "macro_f1": macro_f1(probs, labels),
        "macro_precision": prec,
        "macro_recall": rec,
        "ece": ece(probs, labels),
        "ace": adaptive_ece(probs, labels),
    }
    if sets is not None:
        sets = np.asarray(sets)
        row["coverage"] = confidence_reliability(sets, labels)
        row["avg_set_size"] = float(sets.sum(axis=1).mean())
    return row
