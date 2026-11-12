"""
run_seeds.py  -  multi-seed variance protocol
===================================================================================
Runs the full pipeline for several random seeds and reports mean +/- standard
deviation for every method and metric, so the paper can report statistical
variance rather than a single point estimate.

Protocol (standard, defensible):
  * Hyper-parameters are searched ONCE (on the first seed) and REUSED for every
    seed.
  * The dataset (leakage-free UE-disjoint split); each seed varies model
    initialisation, training stochasticity and the calibration/conformal draws.

  * The first seed also renders the full figure set; later seeds skip figures.

It simply re-invokes experiments/run_all.py once per seed (each run is the exact,
already-validated pipeline, only re-seeded and pointed at its own output folder),
then aggregates the per-seed CSVs.

Run it INSTEAD of run_all.py (it calls run_all.py internally). Requires the
datasets to have been built first (run_simulator.py), same as run_all.py.

Seeds default to 0,1,2 -> override with e.g.  UA_SEEDS="0,1,2,3,4"

Outputs (under results/):
  results/metrics/...                  primary (seed 0) full run + all figures
  results/seeds/seed_<s>/metrics/...   per-seed metrics (no figures)
  results/summary/summary_indist.csv   per-method mean & std of prediction metrics
  results/summary/summary_ho.csv       per-method mean & std of handover KPIs
"""

import os
import sys
import csv
import math
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
SEEDS = [int(s) for s in os.environ.get("UA_SEEDS", "0,1,2").split(",") if s.strip() != ""]

# columns that are not numeric metrics (skip when aggregating)
NON_METRIC = {"method", "scenario", "policy_HO", "policy_HOLD", "policy_CHO"}


def _run(seed, subdir=None, hp_json=None, no_figs=False):
    env = dict(os.environ)
    env["UA_SEED"] = str(seed)
    env.pop("UA_RESULTS_SUBDIR", None)
    env.pop("UA_HP_JSON", None)
    env.pop("UA_NO_FIGS", None)
    if subdir:
        env["UA_RESULTS_SUBDIR"] = subdir
    if hp_json:
        env["UA_HP_JSON"] = str(hp_json)
    if no_figs:
        env["UA_NO_FIGS"] = "1"
    print(f"\n{'='*66}\n SEED {seed}  (output: results/{subdir or '.'})\n{'='*66}", flush=True)
    subprocess.run([sys.executable, str(ROOT / "experiments" / "run_all.py")],
                   env=env, check=True)


def _read(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def _aggregate(out_name, rel, key="method"):
    """Aggregate one CSV (by file name) across seeds into per-method mean/std."""
    tables = []
    for s in SEEDS:
        p = (RESULTS / "metrics" / rel) if s == SEEDS[0] else \
            (RESULTS / "seeds" / f"seed_{s}" / "metrics" / rel)
        if p.exists():
            tables.append(_read(p))
    if not tables:
        print(f"  (no tables found for {rel})")
        return
    if not tables[0] or key not in tables[0][0]:
        print(f"  (skip {rel}: no '{key}' column)")
        return

    def get(tb, kv):
        for r in tb:
            if r.get(key) == kv:
                return r
        return None

    keys = [r[key] for r in tables[0]]
    metrics = [k for k in tables[0][0].keys() if k not in NON_METRIC and k != key]
    out_rows = []
    for kv in keys:
        per = [get(tb, kv) for tb in tables]
        per = [r for r in per if r is not None]
        row = {key: kv, "n_seeds": len(per)}
        for met in metrics:
            vals = []
            for r in per:
                try:
                    v = float(r.get(met, ""))
                    if not math.isnan(v):
                        vals.append(v)
                except (ValueError, TypeError):
                    pass
            if vals:
                mean = sum(vals) / len(vals)
                var = sum((v - mean) ** 2 for v in vals) / len(vals)  # population std
                row[met + "_mean"] = f"{mean:.6g}"
                row[met + "_std"] = f"{math.sqrt(var):.6g}"
        out_rows.append(row)

    outdir = RESULTS / "summary"
    outdir.mkdir(parents=True, exist_ok=True)
    # fieldnames = union of all keys across rows (rows may populate different metrics)
    seen = [key, "n_seeds"]
    for r in out_rows:
        for k in r:
            if k not in seen:
                seen.append(k)
    cols = seen
    with open(outdir / out_name, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in out_rows:
            w.writerow(r)
    print(f"  wrote {outdir / out_name}")


def _metrics_dir(seed):
    return (RESULTS / "metrics") if seed == SEEDS[0] else \
        (RESULTS / "seeds" / f"seed_{seed}" / "metrics")


def _done(seed):
    """A seed is complete only once BOTH end-of-run tables exist (results_indist is
    written last), so a half-finished/interrupted seed is correctly re-run."""
    md = _metrics_dir(seed)
    return (md / "results_indist.csv").exists() and (md / "results_ho.csv").exists()


def main():
    # RESUMABLE: seeds whose results already exist are skipped, so an interrupted
    # run is not fatal -- just re-run and it continues where it stopped.
    # Set UA_FORCE=1 to recompute every seed from scratch.
    force = bool(os.environ.get("UA_FORCE"))
    print(f"Multi-seed run over seeds {SEEDS}  (resume={'OFF' if force else 'ON'})")
    # first seed: full run (HPO + figures) into results/
    if force or not _done(SEEDS[0]):
        _run(SEEDS[0])
    else:
        print(f"  seed {SEEDS[0]} already complete -> skipping (UA_FORCE=1 to rerun)")
    hp = RESULTS / "metrics" / "hpo_best.json"
    hp = hp if hp.exists() else None
    # remaining seeds: reuse HP, own subdir, skip figures
    for s in SEEDS[1:]:
        if force or not _done(s):
            _run(s, subdir=f"seeds/seed_{s}", hp_json=hp, no_figs=True)
        else:
            print(f"  seed {s} already complete -> skipping (UA_FORCE=1 to rerun)")
    print("\n=== aggregating mean +/- std across seeds ===")
    _aggregate("summary_indist.csv", "results_indist.csv")
    _aggregate("summary_ho.csv", "results_ho.csv")
    _aggregate("summary_shift.csv", "results_shift.csv", key="scenario")
    print(f"\nMulti-seed complete ({len(SEEDS)} seeds). See results/summary/.")


if __name__ == "__main__":
    main()
