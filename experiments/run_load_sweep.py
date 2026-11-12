"""
run_load_sweep.py  -  network-load sensitivity (RLF / SINR vs load_factor)
==========================================================================

For each load in UA_LOADS (default 0.2, 0.5, 1.0) this:
  1. regenerates ONLY the SOURCE (in-distribution) dataset at that load,
  2. trains the shared cross-entropy backbone once (seed UA_SWEEP_SEED, default 0),
  3. applies the SAME proposed uncertainty layer as run_all.py
     (ECE-guarded temperature + per-class conformal sets + selective/CHO policy),
  4. records the serving-SINR distribution and the handover-execution KPIs
     (RLF, HOF, ping-pong, HIT, handovers/UE) for the proposed method and for the
     single-target softmax predictor.

Output: results/summary/load_sensitivity.csv  (one row per load).


"""

import os
import sys
import json
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.config import CFG, SERVICE_CLASSES
from common.seeding import set_all_seeds
from common.io_utils import load_arrays, save_table, ensure_dir
from models.train import train_model, predict_probs
from models.calibration import fit_temperature_ece, apply_temperature, expected_calibration_error
from models import conformal as cf
from models.policy import decide_actions, tune_thresholds, HOLD
from metrics.ho_execution import compute_ho_kpis
from baselines.uq_baselines import build_softmax
from simulator import dataset as ds
from simulator.scenarios import SOURCE_SCENARIO

LOADS = [float(x) for x in os.environ.get("UA_LOADS", "0.2,0.5,1.0").split(",") if x.strip()]
SWEEP_SEED = int(os.environ.get("UA_SWEEP_SEED", "0"))
Q_OUT = CFG.hexec.q_out_sinr_db
BASE_DATA = CFG.data_dir


def _load(scn, split):
    d = load_arrays(CFG.data_dir / scn / f"{split}.npz")
    return d["X"], d["y"], d["cls"], int(d["num_cells"][0])


def _load_full(scn, split):
    d = load_arrays(CFG.data_dir / scn / f"{split}.npz")
    return (d["X"], d["y"], d["cls"], d["sinr_raw"], d["serving_now"], d["ue"],
            int(d["num_cells"][0]))


def _conformal_sets(pc, yc, cc, pt, ct):
    sets = np.zeros_like(pt, dtype=bool)
    for idx, sc in enumerate(SERVICE_CLASSES):
        mc = cc == idx
        mt = ct == idx
        if mc.sum() == 0 or mt.sum() == 0:
            continue
        q = cf.split_conformal_qhat(pc[mc], yc[mc], sc.alpha)
        sets[mt] = cf.prediction_sets(pt[mt], q)
    return sets


def _default_hp():
    t = CFG.train
    return {"hidden": t.hidden_size, "dropout": t.dropout, "lr": t.lr,
            "weight_decay": t.weight_decay, "batch": t.batch_size,
            "focal_gamma": t.focal_gamma}


def _hp():
    hp = _default_hp()
    hpj = os.environ.get("UA_HP_JSON")
    if hpj and os.path.exists(hpj):
        try:
            best = json.load(open(hpj)).get("best_params")
            if best:
                hp.update(best)
                print(f"  reusing hyper-parameters from {hpj}")
        except Exception as e:
            print(f"  (could not read {hpj}: {e}; using defaults)")
    return hp


def _train_backbone(Xtr, ytr, Xva, yva, F, K, hp, tag):
    set_all_seeds(SWEEP_SEED)
    sm, sm_loss, _ = build_softmax(F, K, hp["hidden"], hp["dropout"])
    sm, _ = train_model(
        sm, sm_loss, Xtr, ytr,
        max_epochs=CFG.train.max_epochs, batch_size=hp["batch"],
        lr=hp["lr"], weight_decay=hp["weight_decay"],
        grad_clip=CFG.train.grad_clip, patience=CFG.train.patience,
        min_delta=CFG.train.min_delta,
        sched_patience=CFG.train.lr_schedule_patience,
        sched_factor=CFG.train.lr_schedule_factor,
        name=tag, edl=False, X_val=Xva, y_val=yva)
    return sm


def main():
    hp = _hp()
    rows = []
    for load in LOADS:
        print("=" * 66)
        print(f" LOAD-SENSITIVITY  |  load_factor = {load}")
        print("=" * 66)
        CFG.channel.load_factor = load
        CFG.data_dir = BASE_DATA / f"load_{load:.2f}"
        # 1) regenerate the in-distribution dataset at this load
        set_all_seeds(SWEEP_SEED)
        ds.build_source_only()
        # 2) load splits
        Xtr, ytr, _, K = _load(SOURCE_SCENARIO, "train")
        Xva, yva, clsva, _ = _load(SOURCE_SCENARIO, "val")
        Xca, yca, clsca, _ = _load(SOURCE_SCENARIO, "calib")
        Xte, yte, clste, sinr_te, serv_te, ue_te, _ = _load_full(SOURCE_SCENARIO, "test")
        F = Xtr.shape[-1]
        # 3) shared CE backbone + proposed uncertainty layer
        sm = _train_backbone(Xtr, ytr, Xva, yva, F, K, hp, f"backbone-load{load:.2f}")
        smca = predict_probs(sm, Xca)
        T = fit_temperature_ece(np.log(np.clip(smca, 1e-12, 1)), yca,
                                logits_val=np.log(np.clip(predict_probs(sm, Xva), 1e-12, 1)),
                                labels_val=yva)

        def pp(X):
            return apply_temperature(np.log(np.clip(predict_probs(sm, X), 1e-12, 1)), T)

        pca = pp(Xca); pva = pp(Xva); pte = pp(Xte)
        sets_va = _conformal_sets(pca, yca, clsca, pva, clsva)
        thr = tune_thresholds(pva.max(1), sets_va.sum(1), target_commit=CFG.train.target_commit)
        sets_te = _conformal_sets(pca, yca, clsca, pte, clste)
        act = decide_actions(pte.max(1), sets_te.sum(1), **thr)
        ho_conf = (act != HOLD)
        smte = predict_probs(sm, Xte)

        # 4) SINR distribution (serving as chosen, and best available cell)
        serv_sinr = sinr_te[np.arange(len(serv_te)), np.clip(serv_te, 0, K - 1)]
        best_sinr = sinr_te.max(axis=1)

        kp = compute_ho_kpis(sinr_te, serv_te, ue_te, pte,
                             confident=ho_conf, cand_sets=sets_te, y_true=yte)
        ks = compute_ho_kpis(sinr_te, serv_te, ue_te, smte)

        row = {
            "load_factor": load,
            "sinr_serv_median_db": round(float(np.median(serv_sinr)), 2),
            "sinr_serv_p5_db": round(float(np.percentile(serv_sinr, 5)), 2),
            "serv_outage_pct": round(float(np.mean(serv_sinr < Q_OUT) * 100.0), 2),
            "sinr_best_median_db": round(float(np.median(best_sinr)), 2),
            "test_accuracy": round(float(np.mean(pte.argmax(1) == yte)), 4),
            "ece_proposed": round(float(expected_calibration_error(pte, yte)), 4),
            "rlf_per_ue_proposed": round(float(kp.get("rlf_per_ue", float("nan"))), 2),
            "hof_proposed": round(float(kp.get("hof_rate", float("nan"))), 4),
            "pingpong_proposed": round(float(kp.get("ping_pong_rate", float("nan"))), 4),
            "hit_ms_proposed": round(float(kp.get("mean_hit_ms", float("nan"))), 1),
            "ho_per_ue_proposed": round(float(kp.get("ho_per_ue", float("nan"))), 2),
            "rlf_per_ue_softmax": round(float(ks.get("rlf_per_ue", float("nan"))), 2),
            "hof_softmax": round(float(ks.get("hof_rate", float("nan"))), 4),
        }
        rows.append(row)
        print("  ->", {k: row[k] for k in ("sinr_serv_median_db", "serv_outage_pct",
                                           "rlf_per_ue_proposed", "hof_proposed")})

    CFG.data_dir = BASE_DATA
    out = ensure_dir(CFG.results_dir / "summary")
    save_table(out / "load_sensitivity", rows)
    print("\nwrote", out / "load_sensitivity.csv")
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
