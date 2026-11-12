"""
recompute_ho.py  -  recompute the standard handover-KPI table OFFLINE
=====================================================================
The handover-execution KPIs depend ONLY on (i) each method's saved test
predictions and (ii) the saved replay inputs (raw SINR, serving cell, UE id) --
all of which run_all.py now stores in results/figure_data/figdata.npz. So the
whole §6.5 table can be regenerated from a results archive WITHOUT retraining or
To change the HO-execution config, edit metrics/ho_execution.py (or HandoverExecConfig), then run this.

Usage:
    python tools/recompute_ho.py  path/to/figdata.npz  [out_dir]

Writes results_ho.csv / results_ho.json next to the source (or into out_dir).
"""

import sys, json
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from metrics.ho_execution import compute_ho_kpis   # noqa: E402

METHODS = [
    ("ACC-HD (proposed)", "ho_pred_proposed", True),
    ("Softmax / ensemble predictor", "ho_pred_softmax", False),
    ("Deep ensemble", "ho_pred_ensemble", False),
    ("B1: Deep hybrid", "ho_pred_b1", False),
    ("B2: DDQN-RL", "ho_pred_b2", False),
    ("B3: Mobility prediction (Di-HO)", "ho_pred_b3", False),
    ("B0: 3GPP A3/CHO", "ho_pred_b0", False),
]


def main():
    if len(sys.argv) < 2:
        print(__doc__); sys.exit(1)
    npz = Path(sys.argv[1])
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else npz.parent
    out.mkdir(parents=True, exist_ok=True)
    d = np.load(npz)
    sinr, serv, ue = d["ho_sinr"], d["ho_serv"], d["ho_ue"]
    conf = d["ho_conf"] if "ho_conf" in d.files else None
    sets = d["coca_sets"] if "coca_sets" in d.files else None      # proposed CHO candidate list
    ytrue = d["y_test"] if "y_test" in d.files else None

    rows = []
    for name, key, use_conf in METHODS:
        if key not in d.files:
            print(f"  (skip {name}: {key} not in archive)"); continue
        # only the proposed method prepares the conformal set as its CHO candidates
        cs = sets if use_conf else None
        yt = ytrue if use_conf else None
        k = compute_ho_kpis(sinr, serv, ue, d[key],
                            confident=(conf if use_conf else None),
                            cand_sets=cs, y_true=yt)
        rows.append({"method": name, **k})

    ref = next((r.get("ho_per_ue") for r in rows if r["method"].startswith("B0")), None)
    for r in rows:
        h = r.get("ho_per_ue")
        r["ho_reduction"] = float(1 - h / ref) if ref and h is not None and ref > 0 else float("nan")

    # write CSV + JSON
    cols = ["method", "hof_rate", "ping_pong_rate", "rlf_per_ue", "reestab_per_ue",
            "mean_hit_ms", "mean_recovery_ms", "mean_tos_s", "ho_per_ue", "mean_prepared", "ho_reduction"]
    import csv
    with open(out / "results_ho.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})
    json.dump(rows, open(out / "results_ho.json", "w"), indent=2)
    print(f"wrote {out/'results_ho.csv'}")
    for r in rows:
        print(f"  {r['method']:34s} HO/ue={r.get('ho_per_ue',float('nan')):6.1f} "
              f"HOF={r.get('hof_rate',float('nan')):.3f} "
              f"PP={r.get('ping_pong_rate',float('nan')):.3f} "
              f"RLF/ue={r.get('rlf_per_ue',float('nan')):.1f} "
              f"HIT={r.get('mean_hit_ms',float('nan')):.0f}ms "
              f"red={r.get('ho_reduction',float('nan')):.2f}")


if __name__ == "__main__":
    main()
