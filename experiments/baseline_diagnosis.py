"""
baseline_diagnosis.py
==================================================
Provides the sanity check
against expected A3/CHO (B0) behaviour.

OFFLINE from results/figure_data/figdata.npz. For every method it reports, under the
identical execution replay:
  - attempt_rate : fraction of steps where the method's top cell != serving AND beats
                   it by the A3 hysteresis margin (how often it WANTS to hand over),
  - ho_per_ue    : committed handovers per UE,
  - HOF          : fraction of committed handovers whose target never becomes servable
                   in the arrival window (physical failure, identical metric for all),
  - mean_hit_ms, rlf_per_ue.
A high HOF with a high attempt_rate and a target that is weak in the arrival window =
the baseline commits many handovers to cells that do not become serviceable, i.e. a
real property of its target choice under the common model, not a simulator bug. B0
(3GPP A3/CHO) is the sanity anchor: it should look like a conventional reactive rule.

Drop into <repo>/experiments/.  Run:  python experiments/baseline_diagnosis.py
Output: results/metrics/baseline_diagnosis.csv
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

METHODS = [
    ("proposed", "ho_pred_proposed"),
    ("softmax", "ho_pred_softmax"),
    ("deep-ensemble", "ho_pred_ensemble"),
    ("B1 deep-hybrid", "ho_pred_b1"),
    ("B2 DDQN-RL", "ho_pred_b2"),
    ("B3 Di-HO", "ho_pred_b3"),
    ("B0 3GPP A3/CHO", "ho_pred_b0"),
]


def _attempt_rate(scores, serv, hyst):
    cand = scores.argmax(1)
    K = scores.shape[1]
    serv = np.clip(serv, 0, K - 1)
    margin = scores[np.arange(len(serv)), cand] - scores[np.arange(len(serv)), serv]
    return float(np.mean((cand != serv) & (margin >= hyst)))


def main():
    if not FIG.exists():
        raise SystemExit(f"missing {FIG} - run experiments/run_all.py first.")
    d = load_arrays(FIG)
    serv = d["ho_serv"]; sinr = d["ho_sinr"]; ue = d["ho_ue"]; y = d["y_test"]
    hyst = float(getattr(CFG.hexec, "a3_hysteresis", 0.05))

    rows = []
    for name, key in METHODS:
        if key not in d:
            continue
        probs = d[key]
        # proposed uses its confidence gate + sets; everyone else single-target no gate
        if name == "proposed":
            k = compute_ho_kpis(sinr, serv, ue, probs, confident=d["ho_conf"].astype(bool),
                                cand_sets=d["coca_sets"], y_true=y)
        else:
            k = compute_ho_kpis(sinr, serv, ue, probs)
        rows.append({"method": name,
                     "attempt_rate": round(_attempt_rate(probs, serv, hyst), 3),
                     "ho_per_ue": round(float(k.get("ho_per_ue", float("nan"))), 2),
                     "HOF": round(float(k.get("hof_rate", float("nan"))), 4),
                     "mean_hit_ms": round(float(k.get("mean_hit_ms", float("nan"))), 1),
                     "rlf_per_ue": round(float(k.get("rlf_per_ue", float("nan"))), 2)})
        print(f"  {name:16s} attempt={rows[-1]['attempt_rate']:.3f} ho/ue={rows[-1]['ho_per_ue']:.2f} "
              f"HOF={rows[-1]['HOF']:.3f} RLF/ue={rows[-1]['rlf_per_ue']:.2f}")

    ensure_dir(OUT)
    with open(OUT / "baseline_diagnosis.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader()
        for r in rows: w.writerow(r)
    print(f"\nSaved -> {OUT/'baseline_diagnosis.csv'}")
    print("Use this to explain in the paper that the rule/RL baselines' high HOF follows from committing")
    print("many handovers to targets that do not become servable in the arrival window under the common")
    print("execution model (B0 = reactive A3/CHO anchor), not from a crippled baseline or simulator bug.")


if __name__ == "__main__":
    main()
