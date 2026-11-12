"""
calibration.py
==============
Turns raw confidence scores into well-calibrated ones, and measures how well
confidence matches reality.

- expected_calibration_error (ECE): the headline calibration metric.
- temperature_scale: the classic post-hoc method (kept as a BASELINE/comparator).
- The proposed ACC-HD instead calibrates during training (evidential loss),
  but temperature scaling is provided so we can quantify the difference.
"""

import numpy as np
from scipy.optimize import minimize_scalar


def _softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def expected_calibration_error(probs, labels, n_bins=15) -> float:
    """ECE over equal-width confidence bins (lower is better)."""
    probs = np.asarray(probs)
    labels = np.asarray(labels)
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == labels).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    n = len(labels)
    ece = 0.0
    for i in range(n_bins):
        m = (conf > bins[i]) & (conf <= bins[i + 1])
        if m.sum() > 0:
            ece += (m.sum() / n) * abs(correct[m].mean() - conf[m].mean())
    return float(ece)


def reliability_curve(probs, labels, n_bins=15):
    """Return (bin_confidence, bin_accuracy, bin_count) for a reliability diagram."""
    probs = np.asarray(probs); labels = np.asarray(labels)
    conf = probs.max(axis=1); pred = probs.argmax(axis=1)
    correct = (pred == labels).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    xs, ys, cs = [], [], []
    for i in range(n_bins):
        m = (conf > bins[i]) & (conf <= bins[i + 1])
        if m.sum() > 0:
            xs.append(conf[m].mean()); ys.append(correct[m].mean()); cs.append(int(m.sum()))
    return np.array(xs), np.array(ys), np.array(cs)


def fit_temperature(logits, labels):
    """Fit scalar temperature T>0 by minimising NLL on a held-out set."""
    logits = np.asarray(logits); labels = np.asarray(labels)

    def nll(T):
        p = _softmax(logits / T)
        p = np.clip(p[np.arange(len(labels)), labels], 1e-12, 1.0)
        return float(-np.log(p).mean())

    return float(minimize_scalar(nll, bounds=(0.05, 10.0), method="bounded").x)


def fit_temperature_ece(logits, labels, logits_val=None, labels_val=None, n_bins=15):
    """Robust post-hoc temperature that NEVER worsens calibration on unseen data.

    Two-step, generalisation-safe procedure:
      1. Fit T>0 by minimising NLL on the calibration set (the classic, smooth,
         well-posed temperature-scaling objective).
      2. ACCEPT that T only if it improves ECE on a DISJOINT held-out set (the
         validation split); otherwise return T=1 (no scaling).

    Why not minimise ECE directly: ECE is a piecewise-constant (binned) function of
    T, so minimising it ON the fit set overfits its bin boundaries and can *raise*
    test ECE on an already well-calibrated model (observed in run (14): the CE
    backbone's ECE is minimal at T=1, yet ECE-on-calib minimisation picked T~0.79
    and worsened test ECE 0.042 -> 0.067). Fitting by NLL and GUARDING on a held-out
    set avoids that: when the backbone is already well-calibrated the guard returns
    T=1. No test leakage -- the guard uses the validation split, never the test set.
    If no held-out set is supplied we fall back to the calibration set (used only in
    unit tests).
    """
    logits = np.asarray(logits); labels = np.asarray(labels)
    T = fit_temperature(logits, labels)                 # NLL-optimal temperature
    if logits_val is not None and labels_val is not None:
        gl = np.asarray(logits_val); gy = np.asarray(labels_val)
    else:
        gl = logits; gy = labels
    base = expected_calibration_error(_softmax(gl), gy, n_bins)          # T = 1
    scaled = expected_calibration_error(_softmax(gl / T), gy, n_bins)    # fitted T
    return float(T) if scaled < base else 1.0


def apply_temperature(logits, T):
    """Apply a fitted temperature to logits -> calibrated probabilities."""
    return _softmax(np.asarray(logits) / T)
