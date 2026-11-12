"""
run_all.py  —  evaluation protocol (rigor v7)
=============================================
  * Bayesian HPO (Optuna/TPE) on the VALIDATION split.
  * Train ACC-HD on a cross-entropy backbone (+ a focal-loss ablation) and the
    comparators on SOURCE/train (early stopping, LR schedule, grad clip,
    best-val checkpoint). Temperature on SOURCE/calib,
    policy thresholds on SOURCE/val, conformal on SOURCE/calib.
  * Evaluate on SOURCE/test (in-distribution) and the genuinely out-of-source
    shifted tests; per-scenario coverage + ECE; ordered drift stream.
  * Standard handover KPIs (RLF/HOF/ping-pong/HIT/ToS) from a temporal
    handover-execution replay (metrics/ho_execution.py).
  * Selective prediction (risk-coverage), top-k accuracy, alpha-sweep trade-off,
    and a computational-cost measurement.
UE-disjoint splits throughout.
"""

import sys, json, time
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.config import CFG, SERVICE_CLASSES, ALPHA_SWEEP
from common.seeding import set_all_seeds
from common.io_utils import load_arrays, save_table, save_arrays, ensure_dir
from models.train import train_model, predict_probs
from models.heads import focal_loss
from models.hpo import search_hyperparams
from models.coca_hd import build_coca
from models.calibration import fit_temperature, fit_temperature_ece, apply_temperature
from models import conformal as cf
from models.policy import decide_actions, tune_thresholds, action_summary, HO, HOLD
from metrics import uq_metrics as um
from metrics.ho_execution import compute_ho_kpis
from baselines import b0_3gpp, b1_deep_hybrid, b3_diho, b2_dqn
from baselines.uq_baselines import SoftmaxGRU, build_softmax, ensemble_average
from simulator.scenarios import SOURCE_SCENARIO, SHIFT_SCENARIOS

# ---- multi-seed hooks (used by experiments/run_seeds.py; inert otherwise) ----
# Applied BEFORE the results dirs are derived so each seed writes to its own subdir.
import os as _os
if _os.environ.get("UA_SEED"):
    CFG.seed = int(_os.environ["UA_SEED"])
if _os.environ.get("UA_RESULTS_SUBDIR"):
    CFG.results_dir = CFG.results_dir / _os.environ["UA_RESULTS_SUBDIR"]

METRICS_DIR = CFG.results_dir / "metrics"
FIGDATA_DIR = CFG.results_dir / "figure_data"
ALL_TEST = [SOURCE_SCENARIO] + SHIFT_SCENARIOS
DRIFT_ALPHA = 0.10                                  # representative operating point


def _load(scenario, split):
    d = load_arrays(CFG.data_dir / scenario / f"{split}.npz")
    return d["X"], d["y"], d["cls"], int(d["num_cells"][0])


def _load_full(scenario, split):
    """Also return the handover-execution fields (raw SINR, serving cell, UE id)."""
    d = load_arrays(CFG.data_dir / scenario / f"{split}.npz")
    return (d["X"], d["y"], d["cls"], d["sinr_raw"], d["serving_now"], d["ue"],
            int(d["num_cells"][0]))


def _default_hp():
    t = CFG.train
    return {"hidden": t.hidden_size, "dropout": t.dropout, "lr": t.lr,
            "weight_decay": t.weight_decay, "batch": t.batch_size,
            "focal_gamma": t.focal_gamma}


def _fit(model, loss_fn, Xtr, ytr, Xva, yva, name, edl, hp):
    return train_model(
        model, loss_fn, Xtr, ytr,
        max_epochs=CFG.train.max_epochs, batch_size=hp["batch"],
        lr=hp["lr"], weight_decay=hp["weight_decay"],
        grad_clip=CFG.train.grad_clip, patience=CFG.train.patience,
        min_delta=CFG.train.min_delta,
        sched_patience=CFG.train.lr_schedule_patience,
        sched_factor=CFG.train.lr_schedule_factor,
        name=name, edl=edl, X_val=Xva, y_val=yva)


def _conformal_sets(probs_calib, y_calib, cls_calib, probs_test, cls_test):
    sets = np.zeros_like(probs_test, dtype=bool)
    for idx, sc in enumerate(SERVICE_CLASSES):
        mc = cls_calib == idx; mt = cls_test == idx
        if mc.sum() == 0 or mt.sum() == 0:
            continue
        q = cf.split_conformal_qhat(probs_calib[mc], y_calib[mc], sc.alpha)
        sets[mt] = cf.prediction_sets(probs_test[mt], q)
    return sets


def _pred_row(method, probs, y, sets=None):
    """Prediction + calibration row (no handover proxies; HO KPIs live in results_ho)."""
    r = um.summarize(probs, y, sets)
    r["method"] = method; r["scenario"] = SOURCE_SCENARIO
    return r


def _topk_acc(probs, y, k):
    idx = np.argsort(-probs, axis=1)[:, :k]
    return float(np.mean([y[i] in idx[i] for i in range(len(y))]))


def main():
    print("=" * 66)
    print(" UA-PHM (14)  |  UMi (realistic load) + shift + standard HO KPIs")
    print("=" * 66)
    set_all_seeds(CFG.seed); ensure_dir(METRICS_DIR); ensure_dir(FIGDATA_DIR)

    Xtr, ytr, _, K = _load(SOURCE_SCENARIO, "train")
    Xva, yva, clsva, _ = _load(SOURCE_SCENARIO, "val")
    Xca, yca, clsca, _ = _load(SOURCE_SCENARIO, "calib")
    F = Xtr.shape[-1]
    print(f"source train={len(ytr)} val={len(yva)} calib={len(yca)} | feat={F} cells={K}\n")

    # ---------------- HYPER-PARAMETER SEARCH ----------------
    hp = None
    # Multi-seed runs reuse the HPO result from the first seed (env UA_HP_JSON) so
    # the search is done ONCE, not re-tuned per seed (standard variance protocol).
    _hpj = _os.environ.get("UA_HP_JSON")
    if _hpj and _os.path.exists(_hpj):
        hp = json.load(open(_hpj)).get("best_params")
        print(f"--- reusing hyper-parameters from {_hpj}: {hp}")
    if hp is None and CFG.hpo.enabled:
        print("--- Bayesian hyper-parameter optimisation (Optuna/TPE) ---")
        hp = search_hyperparams(Xtr, ytr, Xva, yva, F, K, CFG.hpo, CFG.seed,
                                out_json_path=METRICS_DIR / "hpo_best.json")
    if hp is None:
        hp = _default_hp(); print(f"  using default hyper-parameters: {hp}")
    HID, DROP, GAMMA = hp["hidden"], hp["dropout"], hp["focal_gamma"]
    print(f"  -> hidden={HID} dropout={DROP:.2f} lr={hp['lr']:.2e} "
          f"wd={hp['weight_decay']:.2e} batch={hp['batch']} gamma={GAMMA:.2f}\n")

    rows = []; shift_rows = []; ho_rows = []; figdata = {}

    # ---------------- train the SHARED cross-entropy backbone ----------------
    # The proposed ACC-HD and the softmax comparator share ONE cross-entropy
    # backbone. Cross-entropy (not focal) is chosen on the EVIDENCE: a focal
    # backbone is trained below as an ablation and is LESS calibrated on this task
    # (higher ECE), so it is not used for the proposed method. ACC-HD's
    # contribution is the uncertainty layer placed ON TOP of this backbone
    # -- ECE-guarded temperature + per-class conformal prediction sets + Adaptive
    # Conformal Inference under drift + the selective / conditional-handover (CHO)
    # policy -- NOT the training loss. Sharing the backbone also makes the
    # "Softmax (uncalibrated)" row a clean ablation: exactly this backbone WITHOUT
    # the uncertainty layer.
    print("--- training shared CE backbone (softmax / proposed) ---")
    set_all_seeds(CFG.seed)
    sm, sm_loss, _ = build_softmax(F, K, HID, DROP)
    sm, hist = _fit(sm, sm_loss, Xtr, ytr, Xva, yva, "backbone", False, hp)
    smca = predict_probs(sm, Xca)
    figdata["coca_loss"] = np.array(hist["loss"])

    # ------------- PROPOSED ACC-HD = CE backbone + uncertainty layer ----------
    prop = sm                               # proposed shares the calibrated backbone
    lg_ca = np.log(np.clip(smca, 1e-12, 1))
    lg_va = np.log(np.clip(predict_probs(sm, Xva), 1e-12, 1))
    # Robust temperature: fit by NLL on calib, accept it only if it improves ECE on
    # the DISJOINT validation split, else T=1. On an already well-calibrated backbone
    # this returns T=1 (no scaling), so the proposed is NEVER worse-calibrated than
    # the uncalibrated softmax. No test leakage (guard uses val, not test).
    T = fit_temperature_ece(lg_ca, yca, logits_val=lg_va, labels_val=yva)
    def prop_probs(X):
        return apply_temperature(np.log(np.clip(predict_probs(prop, X), 1e-12, 1)), T)
    pca = prop_probs(Xca); pva = prop_probs(Xva)
    sets_va = _conformal_sets(pca, yca, clsca, pva, clsva)
    thr = tune_thresholds(pva.max(1), sets_va.sum(1), target_commit=CFG.train.target_commit)
    print("  tuned thresholds:", {k: round(v, 3) for k, v in thr.items()}, " T=%.3f" % T)

    # ------------- focal-backbone ABLATION (documented negative result) --------
    # Same architecture and the SAME ECE-guarded temperature, trained with focal
    # loss instead of cross-entropy. Reported in the in-distribution table so the
    # manuscript can state, with evidence, why the proposed backbone uses
    # cross-entropy (focal did not improve calibration here).
    print("--- training focal-backbone ablation ---")
    set_all_seeds(CFG.seed)
    prop_focal = SoftmaxGRU(F, K, HID, DROP)
    prop_focal, _ = _fit(prop_focal, lambda o, t: focal_loss(o, t, GAMMA),
                         Xtr, ytr, Xva, yva, "ACC-HD-focal", False, hp)
    Tf = fit_temperature_ece(np.log(np.clip(predict_probs(prop_focal, Xca), 1e-12, 1)), yca,
                             logits_val=np.log(np.clip(predict_probs(prop_focal, Xva), 1e-12, 1)),
                             labels_val=yva)

    # ---------------- train baselines ----------------
    print("\n--- training baselines ---")
    set_all_seeds(CFG.seed)
    b1, b1l, _ = b1_deep_hybrid.build(F, K, HID)
    b1, _ = _fit(b1, b1l, Xtr, ytr, Xva, yva, "B1-hybrid", False, hp)
    # B2 gets the SAME fair training budget/protocol as the other learned models:
    # full epoch budget + validation-based early stopping + best-checkpoint.
    qnet = b2_dqn.train_dqn(Xtr, ytr, K, epochs=CFG.train.max_epochs,
                            X_val=Xva, y_val=yva,
                            patience=CFG.train.patience, min_delta=CFG.train.min_delta)
    diho = b3_diho.fit(Xtr, ytr, K)
    ev = None
    try:
        set_all_seeds(CFG.seed)
        ev, evl, edl = build_coca(F, K, HID, DROP)
        ev, _ = _fit(ev, evl, Xtr, ytr, Xva, yva, "evidential", edl, hp)
    except Exception as e:
        print(f"  (evidential ablation skipped: {e})"); ev = None
    ens = [sm]
    for mi in range(1, CFG.train.ensemble_members):
        set_all_seeds(CFG.seed + 100 + mi)
        em, el, _ = build_softmax(F, K, HID, DROP)
        em, _ = _fit(em, el, Xtr, ytr, Xva, yva, f"ens-{mi}", False, hp)
        ens.append(em)
    set_all_seeds(CFG.seed)

    # ---------------- in-distribution test ----------------
    Xte, yte, clste, sinr_te, serv_te, ue_te, _ = _load_full(SOURCE_SCENARIO, "test")
    print(f"\n=== in-distribution test: {len(yte)} samples ===")
    pte = prop_probs(Xte)
    sets = _conformal_sets(pca, yca, clsca, pte, clste)
    act = decide_actions(pte.max(1), sets.sum(1), **thr)
    r = _pred_row("ACC-HD (proposed)", pte, yte, sets)
    r["policy"] = action_summary(act); rows.append(r)
    figdata.update(coca_probs=pte, coca_sets=sets, y_test=yte, cls_test=clste)

    # top-k accuracy (operationally relevant: true cell within the candidate set)
    topk = {f"top{k}": _topk_acc(pte, yte, k) for k in (1, 2, 3, 5)}
    save_table(METRICS_DIR / "topk_accuracy", [{"k": k, "accuracy": v}
                                               for k, v in zip((1, 2, 3, 5), topk.values())])
    print("  top-1/2/3 = %.3f / %.3f / %.3f" % (topk["top1"], topk["top2"], topk["top3"]))

    # risk-coverage (selective prediction)
    conf = pte.max(1); correct = (pte.argmax(1) == yte).astype(float)
    order = np.argsort(-conf); cum_acc = np.cumsum(correct[order]) / np.arange(1, len(yte) + 1)
    rc = [{"coverage": round(f, 2), "selective_accuracy": float(cum_acc[max(1, int(f*len(yte)))-1])}
          for f in [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4]]
    save_table(METRICS_DIR / "risk_coverage", rc)
    figdata["rc_coverage"] = np.array([x["coverage"] for x in rc])
    figdata["rc_accuracy"] = np.array([x["selective_accuracy"] for x in rc])

    # prediction/calibration rows for the predictor comparators
    smte = predict_probs(sm, Xte)
    ens_te = ensemble_average([predict_probs(m, Xte) for m in ens])
    b1_te = predict_probs(b1, Xte); b2_te = b2_dqn.predict(qnet, Xte, K)
    b3_te = b3_diho.predict(diho, Xte, K); b0_te = b0_3gpp.predict(Xte, K)
    rows.append(_pred_row("Softmax (uncalibrated)", smte, yte))
    smte_cal = apply_temperature(np.log(np.clip(smte, 1e-12, 1)),
                                 fit_temperature(np.log(np.clip(smca, 1e-12, 1)), yca))
    rows.append(_pred_row("Temperature scaling", smte_cal, yte))
    rows.append(_pred_row("MC-Dropout",
                          predict_probs(sm, Xte, mc_dropout_passes=CFG.train.mc_dropout_passes), yte))
    st_sets = _conformal_sets(smca, yca, clsca, smte, clste)
    rows.append(_pred_row("Static conformal", smte, yte, st_sets))
    rows.append(_pred_row("Deep ensemble", ens_te, yte))
    if ev is not None:
        rows.append(_pred_row("ACC-HD (evidential, ablation)", predict_probs(ev, Xte, edl=True), yte))
    # focal-backbone ablation: same architecture + same ECE-guarded temperature,
    # trained with focal loss. Documents why the proposed backbone uses CE.
    pf_te = apply_temperature(np.log(np.clip(predict_probs(prop_focal, Xte), 1e-12, 1)), Tf)
    rows.append(_pred_row("ACC-HD (focal backbone, ablation)", pf_te, yte))
    rows.append(_pred_row("B1: Deep hybrid", b1_te, yte))

    # ---------------- STANDARD HANDOVER KPIs (temporal execution replay) ------
    print("\n=== standard handover KPIs (RLF/HOF/ping-pong/HIT/ToS) ===")
    # Every method is driven through the SAME realistic execution model
    # (A3-style hysteresis + TTT trigger, physical HOF, RLF + RRC re-establishment;
    # see metrics/ho_execution.py). The proposed method ALONE adds a calibrated
    # confidence gate (defer when the policy HOLDs); baselines have none.
    ho_conf = (act != HOLD)
    def _ho(method, scores, confident=None, cand_sets=None, y_true=None):
        try:
            k = compute_ho_kpis(sinr_te, serv_te, ue_te, scores, confident=confident,
                                cand_sets=cand_sets, y_true=y_true)
        except Exception as e:
            print(f"  (HO KPIs failed for {method}: {e})"); k = {}
        k = {"method": method, **k}; return k
    # proposed: confidence gate (HOLD->defer) + CHO prepares the conformal set
    ho_rows.append(_ho("ACC-HD (proposed)", pte, confident=ho_conf, cand_sets=sets, y_true=yte))
    ho_rows.append(_ho("Softmax / ensemble predictor", smte))   # same trigger, single target, no gate
    ho_rows.append(_ho("Deep ensemble", ens_te))
    ho_rows.append(_ho("B1: Deep hybrid", b1_te))
    ho_rows.append(_ho("B2: DDQN-RL", b2_te))
    ho_rows.append(_ho("B3: Mobility prediction (Di-HO)", b3_te))
    ho_rows.append(_ho("B0: 3GPP A3/CHO", b0_te))
    # Persist every method's test predictions + the HO-replay inputs, so the HO
    # table can be recomputed OFFLINE (e.g. if the execution model is refined)
    # without retraining.
    figdata.update(ho_sinr=sinr_te, ho_serv=serv_te, ho_ue=ue_te, ho_conf=ho_conf,
                   ho_pred_proposed=pte, ho_pred_softmax=smte, ho_pred_ensemble=ens_te,
                   ho_pred_b1=b1_te, ho_pred_b2=b2_te, ho_pred_b3=b3_te, ho_pred_b0=b0_te)
    # handover-reduction referenced to the 3GPP A3/CHO baseline (B0)
    ref = next((x.get("ho_per_ue") for x in ho_rows if x["method"].startswith("B0")), None)
    for x in ho_rows:
        hp_ue = x.get("ho_per_ue")
        x["ho_reduction"] = (float(1 - hp_ue / ref) if ref and hp_ue is not None and ref > 0
                             else float("nan"))
    save_table(METRICS_DIR / "results_ho", ho_rows)
    for x in ho_rows:
        print(f"  {x['method']:34s} HOF={x.get('hof_rate',float('nan')):.3f} "
              f"PPR={x.get('ping_pong_rate',float('nan')):.3f} "
              f"RLF/ue={x.get('rlf_per_ue',float('nan')):.2f} "
              f"HIT={x.get('mean_hit_ms',float('nan')):.0f}ms red={x.get('ho_reduction',float('nan')):.2f}")

    # ---------------- SHIFT: per-scenario coverage + ECE ----------------
    print("\n=== coverage & calibration under distribution shift ===")
    cov_static = {}; cov_aci = {}
    for scn in ALL_TEST:
        Xs, ys, clss, _ = _load(scn, "test")
        ps = prop_probs(Xs)
        s_sets = _conformal_sets(pca, yca, clsca, ps, clss)
        static_cov = float(s_sets[np.arange(len(ys)), ys].mean())
        aci_c = []
        for idx, sc in enumerate(SERVICE_CLASSES):
            m = clss == idx
            if m.sum() == 0:
                continue
            c, _, _ = cf.adaptive_conformal_stream(ps[m], ys[m], sc.alpha)
            aci_c.append(c)
        cov_static[scn] = static_cov
        cov_aci[scn] = float(np.mean(aci_c)) if aci_c else float("nan")
        acc = um.accuracy(ps, ys); scn_ece = um.ece(ps, ys)
        shift_rows.append({"scenario": scn, "proposed_acc": acc, "proposed_ece": scn_ece,
                           "static_coverage": static_cov, "aci_coverage": cov_aci[scn]})
        print(f"  {scn:16s} acc={acc:.3f} ece={scn_ece:.3f} "
              f"static_cov={static_cov:.3f} aci_cov={cov_aci[scn]:.3f}")

    # ---------------- DRIFT STREAM ----------------
    print("\n=== adaptive vs static conformal on an ORDERED DRIFTING stream ===")
    q_static = cf.split_conformal_qhat(pca, yca, DRIFT_ALPHA)
    drift_probs = []; drift_y = []; seg_names = list(SHIFT_SCENARIOS)   # easy -> hard
    for scn in seg_names:
        Xs, ys, _, _ = _load(scn, "test")
        drift_probs.append(prop_probs(Xs)); drift_y.append(ys)
    seg_bounds = np.cumsum([len(a) for a in drift_y])
    drift_probs = np.concatenate(drift_probs, 0); drift_y = np.concatenate(drift_y, 0)
    st_run, aci_run, qtraj, atraj = cf.drift_stream_coverage(drift_probs, drift_y, DRIFT_ALPHA, q_static)
    print(f"  stream N={len(drift_y)} final static_cov={st_run[-1]:.3f} "
          f"aci_cov={aci_run[-1]:.3f} (target {1-DRIFT_ALPHA:.2f})")
    step = max(1, len(drift_y) // 500)
    save_table(METRICS_DIR / "coverage_drift",
               [{"index": int(i), "static_running": float(st_run[i]),
                 "aci_running": float(aci_run[i]), "aci_alpha": float(atraj[i])}
                for i in range(0, len(drift_y), step)])
    figdata["drift_static_run"] = st_run; figdata["drift_aci_run"] = aci_run
    figdata["drift_alpha_traj"] = atraj; figdata["drift_bounds"] = seg_bounds
    figdata["drift_seg_names"] = np.array(seg_names); figdata["drift_target"] = np.array([1 - DRIFT_ALPHA])

    # ---------------- alpha sweep ----------------
    print("\n=== coverage vs set-size trade-off (alpha sweep) ===")
    sweep = []
    for a in ALPHA_SWEEP:
        q = cf.split_conformal_qhat(pca, yca, a)
        s = cf.prediction_sets(pte, q)
        sweep.append({"alpha": a, "target": 1 - a,
                      "coverage": float(s[np.arange(len(yte)), yte].mean()),
                      "avg_set_size": float(s.sum(1).mean())})
        print(f"    alpha={a:.2f} coverage={sweep[-1]['coverage']:.3f} set={sweep[-1]['avg_set_size']:.2f}")
    save_table(METRICS_DIR / "coverage_vs_size", sweep)
    figdata["sweep_alpha"] = np.array([s["alpha"] for s in sweep])
    figdata["sweep_cov"] = np.array([s["coverage"] for s in sweep])
    figdata["sweep_size"] = np.array([s["avg_set_size"] for s in sweep])

    # ---------------- computational cost ----------------
    print("\n=== computational cost ===")
    cost = {}
    try:
        n_params = int(sum(p.numel() for p in prop.parameters()))
        t0 = time.time(); _ = predict_probs(prop, Xte[:2000]); dt = time.time() - t0
        cost = {"proposed_params": n_params,
                "ensemble_params": n_params * CFG.train.ensemble_members,
                "inference_ms_per_decision": float(dt / max(1, min(2000, len(Xte))) * 1000.0),
                "ensemble_relative_cost": float(CFG.train.ensemble_members)}
        with open(METRICS_DIR / "cost.json", "w") as f:
            json.dump(cost, f, indent=2)
        print(f"  params={n_params}  inference={cost['inference_ms_per_decision']:.3f} ms/decision")
    except Exception as e:
        print(f"  (cost measurement skipped: {e})")

    # ---------------- save ----------------
    for r in rows:
        pol = r.pop("policy", None)
        if pol:
            for k, v in pol.items():
                r[f"policy_{k}"] = v
    save_table(METRICS_DIR / "results_indist", rows)
    save_table(METRICS_DIR / "results_shift", shift_rows)
    figdata["shift_scenarios"] = np.array(ALL_TEST)
    figdata["shift_static"] = np.array([cov_static[s] for s in ALL_TEST])
    figdata["shift_aci"] = np.array([cov_aci[s] for s in ALL_TEST])
    save_arrays(FIGDATA_DIR / "figdata.npz", **figdata)

    print("\n===== PREDICTION SUMMARY =====")
    print(f"{'method':34s} {'acc':>6s} {'F1':>6s} {'ECE':>7s} {'ACE':>7s}")
    for r in rows:
        print(f"{r['method']:34s} {r['accuracy']*100:5.1f}% {r['macro_f1']:6.3f} "
              f"{r['ece']:7.4f} {r.get('ace', float('nan')):7.4f}")
    print(f"\nSaved -> results_indist.csv, results_ho.csv, results_shift.csv, "
          f"coverage_drift.csv, coverage_vs_size.csv, risk_coverage.csv, topk_accuracy.csv, "
          f"cost.json, hpo_best.json")

    # secondary seeds skip figure rendering (UA_NO_FIGS) to save time; the primary
    # seed produces the full figure set.
    if not _os.environ.get("UA_NO_FIGS"):
        try:
            from plotting.make_figures import make_all_figures
            make_all_figures()
        except Exception as e:
            print(f"(plotting skipped: {e})")


if __name__ == "__main__":
    main()
