"""
policy.py
=========
The confidence-aware handover policy: it turns the trusted quantities
(calibrated confidence + conformal set size) into an ACTION.

Actions:
  HO   -> execute a proactive handover to the predicted target cell
  HOLD -> wait (ambiguous); re-evaluate on the next measurement
  CHO  -> fall back to the conventional Conditional Handover (uncertain)

Rule (per sample):
  if confidence >= theta_high AND set_size <= ho_max_set (eta) -> HO
  elif confidence <= theta_low OR set_size >= kappa           -> CHO
  else                                                        -> HOLD
"""

import numpy as np

HO, HOLD, CHO = 0, 1, 2
ACTION_NAMES = {HO: "HO", HOLD: "HOLD", CHO: "CHO"}


def decide_actions(confidence, set_sizes, theta_high=0.55, theta_low=0.30,
                   ho_max_set=3, kappa=6):
    """Vectorised policy over arrays. Returns an int array of actions.

    - HO   : confident AND the candidate set is small (not too ambiguous).
    - CHO  : very unsure (low confidence) OR the set is large (high uncertainty).
    - HOLD : in-between; wait for the next measurement.
    Thresholds are chosen so the method commits on clearly-confident cases and
    defers only when genuinely uncertain (that deferral is what lowers failures).
    """
    confidence = np.asarray(confidence)
    set_sizes = np.asarray(set_sizes)
    actions = np.full(len(confidence), HOLD, dtype=int)

    ho = (confidence >= theta_high) & (set_sizes <= ho_max_set)
    cho = (~ho) & ((confidence <= theta_low) | (set_sizes >= kappa))
    actions[ho] = HO
    actions[cho] = CHO
    return actions


def action_summary(actions) -> dict:
    """Fractions of each action, for reporting."""
    actions = np.asarray(actions)
    n = max(1, len(actions))
    return {ACTION_NAMES[a]: float((actions == a).mean()) for a in (HO, HOLD, CHO)}


def tune_thresholds(conf_val, set_sizes_val, target_commit=0.70):
    """Choose policy thresholds on the VALIDATION set (never on test).

    Criterion (stated, reproducible): pick theta_high as the confidence quantile
    that makes the HO-commit fraction on validation ≈ target_commit; set theta_low
    to a low confidence quantile and kappa/ho_max_set from the set-size spread.
    Returns a dict of thresholds to pass to decide_actions.
    """
    conf_val = np.asarray(conf_val); set_sizes_val = np.asarray(set_sizes_val)
    # theta_high = (1 - target_commit) quantile of confidence -> ~target_commit commit
    theta_high = float(np.quantile(conf_val, max(0.0, 1.0 - target_commit)))
    theta_low = float(np.quantile(conf_val, 0.20))
    ho_max_set = int(max(1, np.ceil(np.quantile(set_sizes_val, 0.60))))
    kappa = int(max(ho_max_set + 1, np.ceil(np.quantile(set_sizes_val, 0.90))))
    return {"theta_high": theta_high, "theta_low": theta_low,
            "ho_max_set": ho_max_set, "kappa": kappa}
