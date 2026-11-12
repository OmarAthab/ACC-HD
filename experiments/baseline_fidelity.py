"""
baseline_fidelity.py  —  Package B / B-2
========================================
It does NOT change any number; it makes the comparison TRANSPARENT so
the baselines are faithfully implemented and that the high HOF of the
rule/RL baselines is a property of driving them through the SAME execution model,
not a crippled implementation.

Drops into:  <repo>/experiments/baseline_fidelity.py
Prereq:      run_all.py has been run once (needs results/metrics/*.csv).
Run:         python experiments/baseline_fidelity.py

Output:
  results/metrics/baseline_fidelity.csv   (reproduced metrics )
  results/metrics/baseline_fidelity.md    (a short table stub for the manuscript)

Key framing the table encodes (state this in the rebuttal/§5.2):
  * B1 (deep-hybrid) is a next-cell PREDICTOR -> compare on accuracy AND HOF.
  * B2 (DDQN) and B3 (Di-HO) are handover-MANAGEMENT controllers -> next-cell
    accuracy is not their design target; their HOF is measured by driving them
    through the identical A3-style execution replay as every other method.
  * The point of the paper is reliability of the DECISION, so HOF under one common
    execution model is the fair, apples-to-apples axis.

"""

import sys
import csv
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.config import CFG

MET = CFG.results_dir / "metrics"

# map the method names as they appear in the result CSVs -> (ref, type)
BASELINES = {
    "B1: Deep hybrid":                 ("[6] deep-hybrid predictor", "predictor"),
    "B2: DDQN-RL":                     ("[8] double-DQN controller", "management"),
    "B3: Mobility prediction (Di-HO)": ("[5] Di-HO rule",            "management"),
    "B0: 3GPP A3/CHO":                 ("[34] 3GPP A3/CHO",          "reference"),
}


def _read(path):
    if not path.exists():
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def _find(rows, name, *keys):
    for r in rows:
        if r.get("method") == name:
            for k in keys:
                if k in r and r[k] not in ("", None):
                    return r[k]
    return ""


def main():
    indist = _read(MET / "results_indist.csv")
    ho = _read(MET / "results_ho.csv")
    if not ho:
        raise SystemExit(f"missing {MET/'results_ho.csv'} — run experiments/run_all.py first.")

    rows = []
    for ho_name, (ref, kind) in BASELINES.items():
        # prediction accuracy only meaningful for the predictor + proposed
        acc = _find(indist, "B1: Deep hybrid", "accuracy") if ho_name.startswith("B1") else ""
        rows.append({
            "baseline": ho_name,
            "reference": ref,
            "role": kind,
            "reproduced_accuracy": acc,
            "reproduced_HOF": _find(ho, ho_name, "hof_rate"),
            "reproduced_HIT_ms": _find(ho, ho_name, "mean_hit_ms"),
            "reproduced_RLF_per_ue": _find(ho, ho_name, "rlf_per_ue"),
            "reported_in_source": "",           # <- FILL from the cited paper
            "notes": ("predictor: compare on accuracy + HOF"
                      if kind == "predictor" else
                      "management/rule: HOF measured via the common execution replay; "
                      "next-cell accuracy is not its design target"
                      if kind == "management" else
                      "3GPP reference for handover-reduction denominator"),
        })

    cols = ["baseline", "reference", "role", "reproduced_accuracy", "reproduced_HOF",
            "reproduced_HIT_ms", "reproduced_RLF_per_ue", "reported_in_source", "notes"]
    with open(MET / "baseline_fidelity.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
        for r in rows:
            w.writerow(r)

    with open(MET / "baseline_fidelity.md", "w") as f:
        f.write("# Baseline fidelity (reproduced vs source-reported)\n\n")
        f.write("| Baseline | Role | Reproduced acc | Reproduced HOF | Reported in source | Note |\n")
        f.write("|---|---|---|---|---|---|\n")
        for r in rows:
            f.write(f"| {r['baseline']} | {r['role']} | {r['reproduced_accuracy']} | "
                    f"{r['reproduced_HOF']} | _(fill)_ | {r['notes']} |\n")
        f.write("\nFill 'Reported in source' from each cited paper.\n")

    print("=== baseline fidelity ===")
    for r in rows:
        print(f"  {r['baseline']:34s} role={r['role']:11s} "
              f"acc={r['reproduced_accuracy'] or '-':>6} HOF={r['reproduced_HOF'] or '-'}")
    print(f"\nSaved -> {MET/'baseline_fidelity.csv'} and .md")
    print("Fill 'reported_in_source' from the cited papers, then cite this table in §5.2.")


if __name__ == "__main__":
    main()
