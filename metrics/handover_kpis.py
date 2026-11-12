"""
handover_kpis.py
================
Handover KPIs derived from the predictions and the confidence-aware policy.

Because this is an offline simulation of the DECISION quality (not a full
protocol emulation), we use clear, documented proxies that are standard in
predictive-handover studies:

  Handover Failure Rate (HOF proxy)
      Among samples where a handover is actually needed (the future serving cell
      differs from the current one), the fraction where the method commits a
      handover to the WRONG cell. Deferring (HOLD/CHO) is not counted as a
      failure — the point of the policy is to avoid confident mistakes.

  Ping-Pong Rate (HOPP proxy)
      Fraction of committed handovers whose predicted target is neither the
      current cell nor the true future cell (an unnecessary/oscillating switch).

  Unnecessary-Handover Rate
      Fraction of samples where the method commits a handover although no change
      was needed (current == true future).

  Handover Success Rate
      Among committed handovers, fraction to the correct target.

These are computed the SAME way for every method, so the comparison is fair.
The exact definitions go into Section 5 of the paper.
"""

import numpy as np

from models.policy import HO


def _current_cell(X, num_cells):
    return np.asarray(X)[:, -1, :num_cells].argmax(axis=1)


def compute_kpis(X, y_true, probs, actions, num_cells) -> dict:
    """Return a dict of KPI proxies.

    X       : (N, W, 2*num_cells) test windows (to read the current cell)
    y_true  : (N,) true future serving cell
    probs   : (N, num_cells) predicted probabilities
    actions : (N,) policy actions (HO/HOLD/CHO); for methods without a policy,
              pass an all-HO array so every sample counts as a committed handover.
    """
    X = np.asarray(X); y_true = np.asarray(y_true)
    pred = np.asarray(probs).argmax(1)
    cur = _current_cell(X, num_cells)
    actions = np.asarray(actions)

    # A real handover happens only when we COMMIT (HO action) AND the predicted
    # target is a DIFFERENT cell than the current one (predicting the same cell
    # is "stay", not a handover).
    committed = (actions == HO) & (pred != cur)
    needed = y_true != cur                      # a handover was actually required

    n = len(y_true)
    n_committed = max(1, committed.sum())
    n_needed = max(1, needed.sum())

    # failures: committed to the wrong cell, among needed handovers
    wrong_commit = committed & (pred != y_true)
    hof = float((wrong_commit & needed).sum() / n_needed)

    # ping-pong: committed switch that is neither current nor correct target
    pingpong = committed & (pred != cur) & (pred != y_true)
    hopp = float(pingpong.sum() / n_committed)

    # unnecessary: committed a handover although none was needed
    unnecessary = committed & (~needed)
    unnec = float(unnecessary.sum() / max(1, (~needed).sum()))

    # success among committed handovers
    success = committed & (pred == y_true)
    ho_success = float(success.sum() / n_committed)

    return {
        "HOF_proxy": hof,
        "HOPP_proxy": hopp,
        "unnecessary_HO": unnec,
        "HO_success_rate": ho_success,
        "commit_fraction": float(committed.mean()),
    }
