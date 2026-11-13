"""
matched_ablation.py
=========================================================================
Isolates WHERE the handover-failure gain comes from: is it the
uncertainty-sized conformal set, or merely offering more fallback targets / the
confidence gate / the action policy?

DROP-IN REPLACEMENT for experiments/matched_ablation.py (repo v15).
Note: the previous version emitted a SINGLE matched-CHO control at
m = round(mean |C|) (top-4 only). Paper Table 13 reports a bracket of FIXED
budgets top-4 / top-5 / top-6 straddling the adaptive mean prepared (4.77). This
version sweeps m in {4, 5, 6} so every row of Table 13 is reproduced in the CSV.

OFFLINE from results/figure_data/figdata.npz (produced by run_all.py).
It re-runs ONLY the execution replay (metrics/ho_execution.compute_ho_kpis) under
matched controls that share the SAME confidence gate and the SAME execution budget:

  * proposed        : confidence gate + the adaptive conformal SET as the CHO list.
  * confidence-only : same gate, but a SINGLE target (no set).
  * matched-CHO(m)  : same gate, but prepare a FIXED m top-prob cells instead of the
                      uncertainty-sized conformal set, for m in {4, 5, 6}. This is the
                      key control: a fixed number of prepared cells, chosen by raw
                      probability rather than by calibrated uncertainty.
  * no-gate         : single target, no confidence gate (acts on every step).

If "proposed" beats "matched-CHO(top-m)" at equal-or-larger mean prepared, the gain
is the UNCERTAINTY SIZING, not just "more targets". Reported jointly: HOF, ping-pong,
HIT, handovers/UE, mean prepared.

Run:  python experiments/matched_ablation.py
Output: results/metrics/matched_ablation_top456.csv
"""

import sys, csv
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.config import CFG
from common.io_utils import load_arrays, ensure_dir
from metrics.ho_execution import compute_ho_kpis

FIG = CFG.results_dir / "figure_data" / "figdata.npz"
OUT = CFG.results_dir / "metrics"

# fixed budgets straddling the adaptive mean prepared (~4.77), matching Table 13
MATCHED_M = [4, 5, 6]


def _topm_sets(probs, m):
    """Boolean (N,K): the m highest-probability cells per row."""
    m = max(1, int(m))
    idx = np.argsort(-probs, axis=1)[:, :m]
    sets = np.zeros_like(probs, dtype=bool)
    np.put_along_axis(sets, idx, True, axis=1)
    return sets


def _kpis(sinr, serv, ue, probs, confident, cand_sets, y):
    try:
        return compute_ho_kpis(sinr, serv, ue, probs, confident=confident,
                               cand_sets=cand_sets, y_true=y)
    except Exception as e:
        print("  (kpis failed:", e, ")"); return {}


def main():
    if not FIG.exists():
        raise SystemExit(f"missing {FIG} - run experiments/run_all.py first.")
    d = load_arrays(FIG)
    pte = d["ho_pred_proposed"]; sets = d["coca_sets"]; y = d["y_test"]
    sinr = d["ho_sinr"]; serv = d["ho_serv"]; ue = d["ho_ue"]
    gate = d["ho_conf"].astype(bool)                     # the proposed confidence gate
    print(f"mean conformal set size |C| = {sets.sum(1).mean():.2f}  "
          f"(adaptive mean prepared reported over committed steps)")

    variants = [("proposed (adaptive set)",    dict(confident=gate, cand_sets=sets)),
                ("confidence-only (1 target)", dict(confident=gate, cand_sets=None))]
    for m in MATCHED_M:
        variants.append((f"matched-CHO (top-{m})",
                         dict(confident=gate, cand_sets=_topm_sets(pte, m))))
    variants.append(("no-gate (1 target)",     dict(confident=None, cand_sets=None)))

    rows = []
    for name, kw in variants:
        k = _kpis(sinr, serv, ue, pte, kw["confident"], kw["cand_sets"], y)
        rows.append({"variant": name,
                     "HOF": round(float(k.get("hof_rate", float("nan"))), 4),
                     "ping_pong": round(float(k.get("ping_pong_rate", float("nan"))), 4),
                     "HIT_ms": round(float(k.get("mean_hit_ms", float("nan"))), 1),
                     "ho_per_ue": round(float(k.get("ho_per_ue", float("nan"))), 2),
                     "mean_prepared": round(float(k.get("mean_prepared", float("nan"))), 2)})
        print(f"  {name:28s} HOF={rows[-1]['HOF']}  HIT={rows[-1]['HIT_ms']}ms  "
              f"ho/ue={rows[-1]['ho_per_ue']}  prepared={rows[-1]['mean_prepared']}")

    ensure_dir(OUT)
    with open(OUT / "matched_ablation_top456.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader()
        for r in rows: w.writerow(r)
    print(f"\nSaved -> {OUT/'matched_ablation_top456.csv'}")
    print("Interpretation: compare 'proposed' vs 'matched-CHO(top-m)' at equal-or-larger")
    print("mean_prepared -> proposed has lower HOF => the gain is UNCERTAINTY SIZING.")


if __name__ == "__main__":
    main()
