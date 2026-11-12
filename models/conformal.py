"""
conformal.py
============
Distribution-free guarantees for the handover decision.

Three tools (the heart of ACC-HD's novelty):
  1. split_conformal_qhat / prediction_sets
        Build a small SET of candidate target cells that contains the true cell
        with probability >= 1 - alpha (marginal coverage), calibrated on a
        held-out calibration split.
  2. adaptive_conformal_stream (ACI)
        Update the miscoverage level online so coverage STAYS valid when the
        mobility distribution drifts (non-stationary) — exchangeability no longer
        needed.
  3. conformal_risk_threshold
        Pick a set threshold so a chosen loss (e.g. wrong-target) is bounded by
        a target level alpha (conformal risk control).

Nonconformity score used throughout: s = 1 - p_true (higher = worse fit).
"""

import numpy as np


def split_conformal_qhat(probs_calib, labels_calib, alpha) -> float:
    """Return the conformal threshold q on the calibration split."""
    probs_calib = np.asarray(probs_calib); labels_calib = np.asarray(labels_calib)
    s = 1.0 - probs_calib[np.arange(len(labels_calib)), labels_calib]
    n = len(s)
    # finite-sample correction so coverage >= 1 - alpha
    level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
    return float(np.quantile(s, level, method="higher"))


def prediction_sets(probs, qhat):
    """Boolean matrix (N, K): class kept iff (1 - p) <= qhat."""
    return (1.0 - np.asarray(probs)) <= qhat


def coverage_and_size(sets, labels):
    """Empirical coverage (true label inside set) and average set size."""
    sets = np.asarray(sets); labels = np.asarray(labels)
    cov = float(sets[np.arange(len(labels)), labels].mean())
    size = float(sets.sum(axis=1).mean())
    return cov, size


def adaptive_conformal_stream(probs, labels, alpha, gamma=0.02, init_q=None):
    """Adaptive Conformal Inference (ACI) over a stream.

    Walk through the samples in order; keep a running threshold q_t and update
    the target miscoverage after each observation. Returns realised coverage,
    average set size, and the per-step q trajectory.
    """
    probs = np.asarray(probs); labels = np.asarray(labels)
    N, K = probs.shape
    s_all = 1.0 - probs                                   # nonconformity per class
    s_true = s_all[np.arange(N), labels]

    q = float(np.quantile(s_true, 1 - alpha)) if init_q is None else init_q
    alpha_t = alpha
    covered = np.zeros(N, dtype=bool)
    sizes = np.zeros(N, dtype=float)
    q_traj = np.zeros(N, dtype=float)

    for t in range(N):
        # current prediction set with threshold q
        inset = s_all[t] <= q
        covered[t] = inset[labels[t]]
        sizes[t] = inset.sum()
        q_traj[t] = q
        # ACI update of the miscoverage level, then map back to a threshold
        err = 0.0 if covered[t] else 1.0
        alpha_t = min(1.0, max(0.0, alpha_t + gamma * (alpha - err)))
        # recompute q as the (1 - alpha_t) empirical quantile of recent scores
        lo = max(0, t - 200)
        q = float(np.quantile(s_true[lo:t + 1], 1 - alpha_t)) if t > 0 else q

    return float(covered.mean()), float(sizes.mean()), q_traj


def drift_stream_coverage(probs, labels, alpha, q_static, gamma=0.02, window=200):
    """Compare STATIC vs ADAPTIVE (ACI) conformal on an ORDERED, drifting stream.

    This is ACI's intended setting (Gibbs & Candes, 2021): the test stream is
    ordered from easy to hard (e.g. pedestrian -> vehicular -> non-stationary),
    so the nonconformity-score distribution DRIFTS along the stream. A fixed
    (static) threshold `q_static` calibrated on the source data then progressively
    under-covers, while ACI adapts its threshold online and holds nominal coverage.

    Returns:
      static_run : running (cumulative) coverage using the fixed q_static
      aci_run    : running (cumulative) coverage using the ACI threshold
      q_traj     : ACI threshold trajectory
      alpha_traj : ACI miscoverage-level trajectory
    """
    probs = np.asarray(probs); labels = np.asarray(labels)
    N = probs.shape[0]
    s_all = 1.0 - probs
    s_true = s_all[np.arange(N), labels]

    # static: fixed threshold, cumulative coverage
    static_cov_step = (s_true <= q_static).astype(float)
    static_run = np.cumsum(static_cov_step) / np.arange(1, N + 1)

    # ACI: adapt the threshold after each observation
    q = float(q_static)
    alpha_t = float(alpha)
    aci_step = np.zeros(N, dtype=float)
    q_traj = np.zeros(N, dtype=float)
    alpha_traj = np.zeros(N, dtype=float)
    for t in range(N):
        covered = s_true[t] <= q
        aci_step[t] = 1.0 if covered else 0.0
        q_traj[t] = q
        alpha_traj[t] = alpha_t
        err = 0.0 if covered else 1.0
        alpha_t = min(1.0, max(0.0, alpha_t + gamma * (alpha - err)))
        lo = max(0, t - window)
        q = float(np.quantile(s_true[lo:t + 1], 1 - alpha_t))
    aci_run = np.cumsum(aci_step) / np.arange(1, N + 1)
    return static_run, aci_run, q_traj, alpha_traj


def conformal_risk_threshold(probs_calib, labels_calib, alpha):
    """Conformal risk control: smallest threshold q so empirical miss-risk <= alpha.

    Loss here = 1 if the true class is NOT in the set (a 'miss'). Scanning q from
    high to low, we stop at the smallest q that still keeps average loss <= alpha.
    (For this monotone 0/1 loss the result coincides with split conformal; the
    function is kept separate so other losses can be plugged in.)
    """
    probs_calib = np.asarray(probs_calib); labels_calib = np.asarray(labels_calib)
    s_true = 1.0 - probs_calib[np.arange(len(labels_calib)), labels_calib]
    for q in np.linspace(1.0, 0.0, 101):
        miss = float((s_true > q).mean())
        if miss <= alpha:
            return float(q)
    return 0.0
