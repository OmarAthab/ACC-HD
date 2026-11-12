"""
run_backbones.py
====================================================
Multi-backbone demonstration,
AND a direct test of the calibration-artifact concern.

Drops into:  <repo>/experiments/run_backbones.py   (same folder as run_all.py)
Run from repo root (after datasets are built by run_simulator.py):
    python experiments/run_backbones.py
Optional:
    UA_BACKBONES="gru,transformer,mlp"   (default)  pick which encoders to run
    UA_SEED=0                                        single seed (fast)

What it does, for EACH backbone, using the EXACT proposed-method pipeline from
run_all.py (shared-CE backbone -> ECE-guarded temperature -> per-service-class
split conformal -> confidence-aware policy -> handover KPIs):
  * trains a cross-entropy softmax model with that encoder,
  * fits the ECE-guarded temperature (NLL on calib, accept only if it lowers ECE
    on the DISJOINT val split, else T=1)  -> records whether the guard FIRED,
  * reports in-distribution accuracy + ECE,
  * reports per-class conformal coverage/size at each service-class alpha,
  * reports the headline handover KPIs (HOF / ping-pong / HIT) under the SAME
    execution replay used in the paper.

Output:  results/backbones/backbone_compare.csv   (one row per backbone)
Reading it: if coverage/ECE/HOF hold across GRU, Transformer and MLP, the
backbone-agnostic claim is demonstrated; if the guard fires (T != 1) on the
over-parameterised Transformer, that STRENGTHENS the calibration layer.


"""

import os
import sys
import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.config import CFG, SERVICE_CLASSES
from common.seeding import set_all_seeds
from common.io_utils import load_arrays, ensure_dir
from models.train import train_model, predict_probs
from models.heads import SoftmaxHead
from models.calibration import (fit_temperature_ece, apply_temperature,
                                expected_calibration_error)
from models import conformal as cf
from models.policy import decide_actions, tune_thresholds, HOLD
from metrics.ho_execution import compute_ho_kpis
from baselines.uq_baselines import build_softmax          # the GRU reference
from simulator.scenarios import SOURCE_SCENARIO

if os.environ.get("UA_SEED"):
    CFG.seed = int(os.environ["UA_SEED"])

BACKBONES = [b.strip() for b in os.environ.get("UA_BACKBONES", "gru,transformer,mlp").split(",") if b.strip()]
OUT = CFG.results_dir / "backbones"


# --------------------------------------------------------------------------- #
#  Extra encoders with the SAME interface as GRUBackbone:                      #
#      forward: (batch, window_len, num_features) -> (batch, hidden_size)      #
#  Each wrapper returns LOGITS (encoder -> SoftmaxHead), like SoftmaxGRU.      #
# --------------------------------------------------------------------------- #
class _MLPEncoder(nn.Module):
    """Lightweight MLP over the flattened window (deliberately small)."""
    def __init__(self, window_len, num_features, hidden_size=64, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(window_len * num_features, 256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, hidden_size), nn.ReLU(), nn.Dropout(dropout),
        )
        self.hidden_size = hidden_size

    def forward(self, x):
        return self.net(x)


class _TransformerEncoder(nn.Module):
    """Higher-capacity Transformer encoder (the over-parameterised stress test)."""
    def __init__(self, window_len, num_features, hidden_size=128, nhead=4,
                 nlayers=2, dropout=0.2):
        super().__init__()
        self.proj = nn.Linear(num_features, hidden_size)
        self.pos = nn.Parameter(torch.zeros(1, window_len, hidden_size))
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_size, nhead=nhead, dim_feedforward=hidden_size * 4,
            dropout=dropout, batch_first=True)
        self.enc = nn.TransformerEncoder(layer, num_layers=nlayers)
        self.drop = nn.Dropout(dropout)
        self.hidden_size = hidden_size

    def forward(self, x):
        h = self.proj(x) + self.pos[:, : x.size(1), :]
        h = self.enc(h)
        return self.drop(h[:, -1, :])


class _Net(nn.Module):
    """encoder -> softmax head (returns logits), same shape contract as SoftmaxGRU."""
    def __init__(self, encoder, num_classes):
        super().__init__()
        self.backbone = encoder
        self.head = SoftmaxHead(encoder.hidden_size, num_classes)

    def forward(self, x):
        return self.head(self.backbone(x))


def _build(kind, W, F, K, hidden, dropout):
    if kind == "gru":
        return build_softmax(F, K, hidden, dropout)                 # repo GRU
    if kind == "mlp":
        return _Net(_MLPEncoder(W, F, hidden, dropout), K), nn.CrossEntropyLoss(), False
    if kind == "transformer":
        # deliberately higher capacity than the 24.8k-param GRU
        enc = _TransformerEncoder(W, F, hidden_size=max(128, hidden), dropout=dropout)
        return _Net(enc, K), nn.CrossEntropyLoss(), False
    raise ValueError(f"unknown backbone '{kind}'")


# --------------------------------------------------------------------------- #
def _load(split):
    d = load_arrays(CFG.data_dir / SOURCE_SCENARIO / f"{split}.npz")
    return d["X"], d["y"], d["cls"], int(d["num_cells"][0])


def _load_full(split):
    d = load_arrays(CFG.data_dir / SOURCE_SCENARIO / f"{split}.npz")
    return (d["X"], d["y"], d["cls"], d["sinr_raw"], d["serving_now"], d["ue"],
            int(d["num_cells"][0]))


def _conformal_sets(p_cal, y_cal, cls_cal, p_te, cls_te):
    sets = np.zeros_like(p_te, dtype=bool)
    for idx, sc in enumerate(SERVICE_CLASSES):
        mc = cls_cal == idx; mt = cls_te == idx
        if mc.sum() == 0 or mt.sum() == 0:
            continue
        q = cf.split_conformal_qhat(p_cal[mc], y_cal[mc], sc.alpha)
        sets[mt] = cf.prediction_sets(p_te[mt], q)
    return sets


def _logit(p):
    return np.log(np.clip(p, 1e-12, 1.0))


def run_one(kind, data):
    (Xtr, ytr, Xva, yva, clsva, Xca, yca, clsca,
     Xte, yte, clste, sinr_te, serv_te, ue_te, W, F, K) = data
    hidden = CFG.train.hidden_size
    dropout = CFG.train.dropout
    print(f"\n{'='*66}\n BACKBONE: {kind}\n{'='*66}")
    set_all_seeds(CFG.seed)
    model, loss_fn, edl = _build(kind, W, F, K, hidden, dropout)
    n_params = int(sum(p.numel() for p in model.parameters()))
    model, _ = train_model(
        model, loss_fn, Xtr, ytr,
        max_epochs=CFG.train.max_epochs, batch_size=CFG.train.batch_size,
        lr=CFG.train.lr, weight_decay=CFG.train.weight_decay,
        grad_clip=CFG.train.grad_clip, patience=CFG.train.patience,
        min_delta=CFG.train.min_delta, sched_patience=CFG.train.lr_schedule_patience,
        sched_factor=CFG.train.lr_schedule_factor, name=f"{kind}-CE",
        edl=False, X_val=Xva, y_val=yva)

    p_ca = predict_probs(model, Xca); p_va = predict_probs(model, Xva)
    p_te = predict_probs(model, Xte)
    T = fit_temperature_ece(_logit(p_ca), yca, logits_val=_logit(p_va), labels_val=yva)
    guard_fired = abs(T - 1.0) > 1e-6
    p_te_cal = apply_temperature(_logit(p_te), T)
    p_va_cal = apply_temperature(_logit(p_va), T)

    acc = float((p_te_cal.argmax(1) == yte).mean())
    ece = expected_calibration_error(p_te_cal, yte)

    # per-class conformal coverage/size
    row = {"backbone": kind, "n_params": n_params, "T": round(float(T), 4),
           "guard_fired": int(guard_fired), "acc": round(acc, 4), "ece": round(ece, 4)}
    sets_te = _conformal_sets(p_ca, yca, clsca, p_te, clste)
    for idx, sc in enumerate(SERVICE_CLASSES):
        name = getattr(sc, "name", f"class{idx}")
        mt = clste == idx
        if mt.sum() == 0:
            continue
        cov = float(sets_te[mt][np.arange(mt.sum()), yte[mt]].mean())
        size = float(sets_te[mt].sum(1).mean())
        row[f"cov_{name}"] = round(cov, 4)
        row[f"size_{name}"] = round(size, 3)

    # headline handover KPIs under the same execution replay
    sets_va = _conformal_sets(p_ca, yca, clsca, p_va, clsva)
    thr = tune_thresholds(p_va_cal.max(1), sets_va.sum(1), target_commit=CFG.train.target_commit)
    act = decide_actions(p_te_cal.max(1), sets_te.sum(1), **thr)
    try:
        k = compute_ho_kpis(sinr_te, serv_te, ue_te, p_te_cal,
                            confident=(act != HOLD), cand_sets=sets_te, y_true=yte)
        row["HOF"] = round(float(k.get("hof_rate", float("nan"))), 4)
        row["ping_pong"] = round(float(k.get("ping_pong_rate", float("nan"))), 4)
        row["HIT_ms"] = round(float(k.get("mean_hit_ms", float("nan"))), 1)
    except Exception as e:
        print(f"  (HO KPIs failed for {kind}: {e})")
    print(f"  params={n_params}  T={T:.3f}  guard_fired={guard_fired}  "
          f"acc={acc:.3f}  ece={ece:.4f}  HOF={row.get('HOF','-')}")
    return row


def main():
    ensure_dir(OUT)
    Xtr, ytr, _, K = _load("train")
    Xva, yva, clsva, _ = _load("val")
    Xca, yca, clsca, _ = _load("calib")
    Xte, yte, clste, sinr_te, serv_te, ue_te, _ = _load_full("test")
    W, F = Xtr.shape[1], Xtr.shape[2]
    data = (Xtr, ytr, Xva, yva, clsva, Xca, yca, clsca,
            Xte, yte, clste, sinr_te, serv_te, ue_te, W, F, K)
    print(f"source: train={len(ytr)} val={len(yva)} calib={len(yca)} test={len(yte)} "
          f"| window={W} feat={F} cells={K} | backbones={BACKBONES}")

    rows = [run_one(kind, data) for kind in BACKBONES]

    cols = []
    for r in rows:
        for k in r:
            if k not in cols:
                cols.append(k)
    out_csv = OUT / "backbone_compare.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\nSaved -> {out_csv}")
    print("\nInterpretation:")
    print("  * coverage/ECE/HOF similar across backbones  -> backbone-agnostic claim demonstrated.")
    print("  * guard_fired=1 (T!=1) on the Transformer    -> calibration layer is a NEEDED")
    print("                                                  ")


if __name__ == "__main__":
    main()
