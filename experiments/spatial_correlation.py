"""
spatial_correlation.py
==========================================
OFFLINE: reads the proposed model's saved test predictions from
results/figure_data/figdata.npz (produced by run_all.py).

Drops into:  <repo>/experiments/spatial_correlation.py
Prereq:      run_all.py (or run_seeds.py) has been run once.
Run:         python experiments/spatial_correlation.py

Non-conformity score  s = 1 - p_true  (the exact score the conformal layer uses).
Two diagnostics:
  1. within-UE lag-1 autocorrelation of s  (temporal dependence along a UE track)
  2. between-serving-cell variance fraction eta^2 = SSB/SST of s
     (how much of the score variance is explained by *which cell* the UE is in
      -> a proxy for spatial structure / shadow-fading correlation)

Verdict rule (stated, reproducible): if both |autocorr| < 0.10 and eta^2 < 0.05,
the scores are close to exchangeable and the reported marginal coverage is not
materially inflated; otherwise the limitation is real and must be stated (and ACI,
which does not rely on exchangeability, is the mitigation).
"""

import sys
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.config import CFG
from common.io_utils import load_arrays, save_table

FIG = CFG.results_dir / "figure_data" / "figdata.npz"
OUT = CFG.results_dir / "metrics"


def _scores(probs, y):
    probs = np.asarray(probs); y = np.asarray(y)
    return 1.0 - probs[np.arange(len(y)), y]


def _within_ue_autocorr(s, ue):
    """Mean lag-1 autocorrelation of s within each UE track (count-weighted)."""
    num = 0.0; den = 0.0
    for u in np.unique(ue):
        si = s[ue == u]
        if len(si) < 3 or si.std() < 1e-12:
            continue
        a, b = si[:-1], si[1:]
        if a.std() < 1e-12 or b.std() < 1e-12:
            continue
        r = float(np.corrcoef(a, b)[0, 1])
        num += r * len(si); den += len(si)
    return num / den if den > 0 else float("nan")


def _between_group_eta2(s, g):
    """Fraction of variance of s explained by group g (serving cell): SSB/SST."""
    s = np.asarray(s, float); g = np.asarray(g)
    gm = s.mean()
    sst = float(((s - gm) ** 2).sum())
    if sst <= 0:
        return float("nan")
    ssb = 0.0
    for v in np.unique(g):
        si = s[g == v]
        ssb += len(si) * (si.mean() - gm) ** 2
    return float(ssb / sst)


def main():
    if not FIG.exists():
        raise SystemExit(f"missing {FIG} — run experiments/run_all.py first.")
    d = load_arrays(FIG)
    probs = d["ho_pred_proposed"] if "ho_pred_proposed" in d else d["coca_probs"]
    y = d["y_test"]; ue = d["ho_ue"]; serv = d["ho_serv"]
    s = _scores(probs, y)

    ac = _within_ue_autocorr(s, ue)
    eta2 = _between_group_eta2(s, serv)
    exchangeable = (abs(ac) < 0.10) and (eta2 < 0.05)

    rows = [{
        "n_samples": int(len(s)),
        "n_UEs": int(len(np.unique(ue))),
        "n_serving_cells": int(len(np.unique(serv))),
        "within_UE_lag1_autocorr": round(float(ac), 4),
        "between_cell_eta2": round(float(eta2), 4),
        "near_exchangeable": int(exchangeable),
    }]
    save_table(OUT / "spatial_correlation", rows)
    print("=== spatial-correlation diagnostic (non-conformity scores) ===")
    for k, v in rows[0].items():
        print(f"  {k:28s} {v}")
    print("\nVerdict:",
          "scores are close to exchangeable; reported marginal coverage is not"
          " materially inflated." if exchangeable else
          "non-negligible dependence detected; state this limitation explicitly."
          " ACI (which does not require exchangeability) is the mitigation.")
    print(f"Saved -> {OUT / 'spatial_correlation.csv'}")


if __name__ == "__main__":
    main()
