"""
per_class_kpis.py
=============================================
Reports per-service-class handover KPIs + coverage, so a
tunable alpha is shown to meet distinct service requirements.

OFFLINE from results/figure_data/figdata.npz. Each UE carries one service class, so
filtering samples by class preserves whole-UE temporal sequences for the replay.

Drop into <repo>/experiments/.  Run:  python experiments/per_class_kpis.py
Output: results/metrics/per_class_kpis.csv
"""

import sys, csv
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.config import CFG, SERVICE_CLASSES
from common.io_utils import load_arrays, ensure_dir
from metrics.ho_execution import compute_ho_kpis

FIG = CFG.results_dir / "figure_data" / "figdata.npz"
OUT = CFG.results_dir / "metrics"


def main():
    if not FIG.exists():
        raise SystemExit(f"missing {FIG} - run experiments/run_all.py first.")
    d = load_arrays(FIG)
    pte = d["ho_pred_proposed"]; sets = d["coca_sets"]; y = d["y_test"]; cls = d["cls_test"]
    sinr = d["ho_sinr"]; serv = d["ho_serv"]; ue = d["ho_ue"]; gate = d["ho_conf"].astype(bool)

    rows = []
    for idx, sc in enumerate(SERVICE_CLASSES):
        name = getattr(sc, "name", f"class{idx}")
        m = cls == idx
        if m.sum() == 0:
            continue
        # coverage of the conformal set for this class
        cov = float(sets[m][np.arange(m.sum()), y[m]].mean())
        size = float(sets[m].sum(1).mean())
        try:
            k = compute_ho_kpis(sinr[m], serv[m], ue[m], pte[m],
                                confident=gate[m], cand_sets=sets[m], y_true=y[m])
        except Exception as e:
            print(f"  (kpis failed {name}: {e})"); k = {}
        rows.append({"class": name, "alpha": sc.alpha, "n_samples": int(m.sum()),
                     "coverage": round(cov, 4), "avg_set_size": round(size, 3),
                     "HOF": round(float(k.get("hof_rate", float("nan"))), 4),
                     "HIT_ms": round(float(k.get("mean_hit_ms", float("nan"))), 1),
                     "ho_per_ue": round(float(k.get("ho_per_ue", float("nan"))), 2),
                     "mean_prepared": round(float(k.get("mean_prepared", float("nan"))), 2),
                     "rlf_per_ue": round(float(k.get("rlf_per_ue", float("nan"))), 2)})
        print(f"  {name:6s} a={sc.alpha:.2f} cov={rows[-1]['coverage']} |C|={rows[-1]['avg_set_size']} "
              f"HOF={rows[-1]['HOF']} HIT={rows[-1]['HIT_ms']} prep={rows[-1]['mean_prepared']}")

    ensure_dir(OUT)
    with open(OUT / "per_class_kpis.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader()
        for r in rows: w.writerow(r)
    print(f"\nSaved -> {OUT/'per_class_kpis.csv'}")
    print("Report this table so differentiation is shown at the level of HOF/HIT/preparation per class,")
    print("with each class's reliability requirement stated in advance (URLLC strictest).")


if __name__ == "__main__":
    main()
