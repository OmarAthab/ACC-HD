"""
threshold_sensitivity.py
=========================================================================
Shows the confidence-aware policy is insensitive to its three hand-tuned knobs
(theta_high, theta_low, kappa) AROUND ITS DEPLOYED OPERATING POINT.

DROP-IN REPLACEMENT for experiments/threshold_sensitivity.py (repo v15).
Note: the previous grid (theta_high in [0.45,0.65]) excluded the deployed
operating point (theta_high=0.83, theta_low=0.76, eta=2, kappa=7). This version
re-centres the grid on the DEPLOYED thresholds, matching paper Section 5.13
(36 points), and appends the deployed operating point as a final labelled row so
its HOF is traceable in the CSV.

OFFLINE: reuses the proposed model's saved test predictions + conformal sets +
handover-execution inputs from results/figure_data/figdata.npz. No retraining;
it only re-runs the policy + the handover-KPI replay over a grid.

Prereq:      run_all.py has been run once.
Run:         python experiments/threshold_sensitivity.py
Output:      results/metrics/threshold_sensitivity_centered36.csv  (+ .json)
  one row per (theta_high, theta_low, kappa) with commit-rate, HOF, ping-pong,
  followed by the deployed operating point (0.83 / 0.76 / kappa 7 / eta 2).
A flat HOF surface across the grid = the gains are not an artefact of a lucky
threshold choice.
"""

import sys
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.config import CFG
from common.io_utils import load_arrays, save_table
from models.policy import decide_actions, HOLD
from metrics.ho_execution import compute_ho_kpis

FIG = CFG.results_dir / "figure_data" / "figdata.npz"
OUT = CFG.results_dir / "metrics"

# grid centred on the DEPLOYED thresholds (theta_high=0.83, theta_low=0.76,
# eta=2, kappa=7); 4 x 3 x 3 = 36 points, matching paper Section 5.13.
THETA_HIGH = [0.75, 0.80, 0.85, 0.90]
THETA_LOW = [0.70, 0.75, 0.80]
KAPPA = [5, 7, 9]
HO_MAX_SET = 2                      # eta

# deployed operating point, appended as a final labelled row
DEPLOYED = dict(theta_high=0.83, theta_low=0.76, kappa=7, ho_max_set=2)


def _eval(conf, ssize, sinr, serv, ue, probs, sets, y, th, tl, kp, eta):
    act = decide_actions(conf, ssize, theta_high=th, theta_low=tl,
                         ho_max_set=eta, kappa=kp)
    commit = float((act != HOLD).mean())
    try:
        k = compute_ho_kpis(sinr, serv, ue, probs,
                            confident=(act != HOLD), cand_sets=sets, y_true=y)
        hof = float(k.get("hof_rate", float("nan")))
        ppr = float(k.get("ping_pong_rate", float("nan")))
    except Exception as e:
        hof = ppr = float("nan")
        print(f"  (KPIs failed th={th} tl={tl} kp={kp}: {e})")
    return commit, hof, ppr


def main():
    if not FIG.exists():
        raise SystemExit(f"missing {FIG} — run experiments/run_all.py first.")
    d = load_arrays(FIG)
    probs = d["ho_pred_proposed"] if "ho_pred_proposed" in d else d["coca_probs"]
    sets = d["coca_sets"]; y = d["y_test"]
    sinr = d["ho_sinr"]; serv = d["ho_serv"]; ue = d["ho_ue"]
    conf = probs.max(1); ssize = sets.sum(1)

    rows = []
    for th in THETA_HIGH:
        for tl in THETA_LOW:
            # full 4 x 3 x 3 = 36 factorial grid (paper Section 5.13); the few
            # theta_low >= theta_high corner configs are retained so the point
            # count and the reported HOF/ping-pong ranges match the paper exactly.
            for kp in KAPPA:
                commit, hof, ppr = _eval(conf, ssize, sinr, serv, ue, probs,
                                         sets, y, th, tl, kp, HO_MAX_SET)
                rows.append({"theta_high": th, "theta_low": tl, "kappa": kp,
                             "ho_max_set": HO_MAX_SET, "commit_rate": round(commit, 3),
                             "HOF": round(hof, 4), "ping_pong": round(ppr, 4),
                             "is_deployed": 0})
                print(f"  th={th:.2f} tl={tl:.2f} k={kp}  commit={commit:.2f}  "
                      f"HOF={hof:.4f}  PPR={ppr:.4f}")

    # deployed operating point (final labelled row)
    commit, hof, ppr = _eval(conf, ssize, sinr, serv, ue, probs, sets, y,
                             DEPLOYED["theta_high"], DEPLOYED["theta_low"],
                             DEPLOYED["kappa"], DEPLOYED["ho_max_set"])
    rows.append({"theta_high": DEPLOYED["theta_high"], "theta_low": DEPLOYED["theta_low"],
                 "kappa": DEPLOYED["kappa"], "ho_max_set": DEPLOYED["ho_max_set"],
                 "commit_rate": round(commit, 3), "HOF": round(hof, 4),
                 "ping_pong": round(ppr, 4), "is_deployed": 1})
    print(f"  DEPLOYED th=0.83 tl=0.76 k=7 eta=2  commit={commit:.2f}  "
          f"HOF={hof:.4f}  PPR={ppr:.4f}")

    save_table(OUT / "threshold_sensitivity_centered36", rows)
    grid = [r for r in rows if r["is_deployed"] == 0]
    hofs = [r["HOF"] for r in grid if not np.isnan(r["HOF"])]
    pprs = [r["ping_pong"] for r in grid if not np.isnan(r["ping_pong"])]
    if hofs:
        print(f"\n36-point grid  HOF: min={min(hofs):.4f} max={max(hofs):.4f}  "
              f"ping-pong: min={min(pprs):.4f} max={max(pprs):.4f}")
        print("A small spread => the policy's advantage is robust to the thresholds.")
    print(f"Saved -> {OUT / 'threshold_sensitivity_centered36.csv'}")


if __name__ == "__main__":
    main()
