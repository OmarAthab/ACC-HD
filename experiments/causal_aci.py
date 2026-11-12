"""
causal_aci.py  -  Package B / B-7  [CRIT]  (+ absorbs B-10)
==========================================================
This re-runs the drift/ACI experiment with a REALISTIC feedback delay
(the outcome of a decision is only known H steps = horizon/stride samples later),
sweeps the ACI step size gamma, and compares a dynamically-tuned ACI
(DtACI) that does not need gamma hand-tuned.

It trains the proposed CE backbone with the exact pipeline (shared-CE + ECE-guarded
temperature), rebuilds the ordered drift stream, and reports, for each method x gamma
x delay: realised coverage and mean set size at the drift operating point.

Drop into <repo>/experiments/.  Run:  python experiments/causal_aci.py
Output: results/metrics/causal_aci.csv

"""

import sys, csv
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.config import CFG
from common.seeding import set_all_seeds
from common.io_utils import load_arrays, ensure_dir
from models.train import train_model, predict_probs
from models.calibration import fit_temperature_ece, apply_temperature
from models import conformal as cf
from baselines.uq_baselines import build_softmax
from simulator.scenarios import SOURCE_SCENARIO, SHIFT_SCENARIOS

OUT = CFG.results_dir / "metrics"
ALPHA = 0.10                                   # drift operating point (paper's DRIFT_ALPHA)
GAMMAS = [0.005, 0.01, 0.02, 0.05, 0.10]
WIN = 200


def _load(split):
    d = load_arrays(CFG.data_dir / SOURCE_SCENARIO / f"{split}.npz")
    return d["X"], d["y"]


def _logit(p):
    return np.log(np.clip(p, 1e-12, 1.0))


def aci_stream(s_true, s_all, labels, alpha, gamma, delay, q0):
    """ACI with an explicit feedback delay. q0 = initial threshold (from calib)."""
    N = len(labels)
    q = float(q0); a_t = float(alpha)
    errs = np.zeros(N); cov = np.zeros(N); size = np.zeros(N)
    for t in range(N):
        inset = s_all[t] <= q
        cov[t] = inset[labels[t]]; size[t] = inset.sum()
        errs[t] = 0.0 if cov[t] else 1.0
        src = t - delay                                   # outcome known `delay` steps ago
        if src >= 0:
            a_t = min(1.0, max(0.0, a_t + gamma * (alpha - errs[src])))
        lo = max(0, t - WIN)
        q = float(np.quantile(s_true[lo:t + 1], 1.0 - a_t))
    return float(cov.mean()), float(size.mean())


def dtaci_stream(s_true, s_all, labels, alpha, delay, q0, gammas=(0.005, 0.02, 0.08), eta=0.1):
    """Dynamically-tuned ACI: a small set of gamma-experts, exponentially weighted by
    their pinball loss, aggregated into one miscoverage level. No hand-tuned gamma."""
    N = len(labels); M = len(gammas)
    a_k = np.full(M, float(alpha)); w = np.ones(M) / M
    q = float(q0)
    cov = np.zeros(N); size = np.zeros(N); errs = np.zeros(N)
    for t in range(N):
        a_agg = float(np.clip(np.dot(w, a_k), 0.0, 1.0))
        inset = s_all[t] <= q
        cov[t] = inset[labels[t]]; size[t] = inset.sum()
        errs[t] = 0.0 if cov[t] else 1.0
        src = t - delay
        if src >= 0:
            e = errs[src]
            # pinball loss of each expert's level vs the realised error, then reweight
            loss = np.array([(alpha - e) * (alpha - a) for a in a_k])
            w = w * np.exp(-eta * loss); w = w / w.sum()
            a_k = np.clip(a_k + np.array(gammas) * (alpha - e), 0.0, 1.0)
        lo = max(0, t - WIN)
        q = float(np.quantile(s_true[lo:t + 1], 1.0 - a_agg))
    return float(cov.mean()), float(size.mean())


def main():
    ensure_dir(OUT)
    Xtr, ytr = _load("train"); Xva, yva = _load("val"); Xca, yca = _load("calib")
    F, K = Xtr.shape[-1], int(load_arrays(CFG.data_dir / SOURCE_SCENARIO / "train.npz")["num_cells"][0])
    set_all_seeds(CFG.seed)
    model, loss_fn, _ = build_softmax(F, K, CFG.train.hidden_size, CFG.train.dropout)
    model, _ = train_model(model, loss_fn, Xtr, ytr, max_epochs=CFG.train.max_epochs,
                           batch_size=CFG.train.batch_size, lr=CFG.train.lr,
                           weight_decay=CFG.train.weight_decay, grad_clip=CFG.train.grad_clip,
                           patience=CFG.train.patience, min_delta=CFG.train.min_delta,
                           name="CE-backbone", edl=False, X_val=Xva, y_val=yva)
    T = fit_temperature_ece(_logit(predict_probs(model, Xca)), yca,
                            logits_val=_logit(predict_probs(model, Xva)), labels_val=yva)
    prob = lambda X: apply_temperature(_logit(predict_probs(model, X)), T)

    # ordered drift stream (easy -> hard), exactly as run_all builds it
    probs, ys = [], []
    for scn in SHIFT_SCENARIOS:
        d = load_arrays(CFG.data_dir / scn / "test.npz")
        probs.append(prob(d["X"])); ys.append(d["y"])
    probs = np.concatenate(probs, 0); labels = np.concatenate(ys, 0)
    s_all = 1.0 - probs; s_true = s_all[np.arange(len(labels)), labels]
    q0 = float(cf.split_conformal_qhat(prob(Xca), yca, ALPHA))

    delay = max(1, round(CFG.task.horizon / max(1, CFG.task.window_stride)))
    print(f"stream N={len(labels)}  realistic feedback delay = {delay} samples "
          f"(horizon {CFG.task.horizon} / stride {CFG.task.window_stride})  target {1-ALPHA:.2f}")

    rows = []
    for g in GAMMAS:
        for dl in (0, delay):
            cov, sz = aci_stream(s_true, s_all, labels, ALPHA, g, dl, q0)
            rows.append({"method": "ACI", "gamma": g, "delay": dl,
                         "coverage": round(cov, 4), "avg_set_size": round(sz, 3)})
            print(f"  ACI   gamma={g:.3f} delay={dl:2d}  cov={cov:.4f}  |C|={sz:.2f}")
    for dl in (0, delay):
        cov, sz = dtaci_stream(s_true, s_all, labels, ALPHA, dl, q0)
        rows.append({"method": "DtACI", "gamma": "auto", "delay": dl,
                     "coverage": round(cov, 4), "avg_set_size": round(sz, 3)})
        print(f"  DtACI gamma=auto  delay={dl:2d}  cov={cov:.4f}  |C|={sz:.2f}")

    with open(OUT / "causal_aci.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader()
        for r in rows: w.writerow(r)
    print(f"\nSaved -> {OUT/'causal_aci.csv'}")
    print("Report coverage at the realistic delay (not delay=0). If it drops, restate the claim and show")
    print("which gamma / DtACI recovers nominal coverage under the real feedback delay.")


if __name__ == "__main__":
    main()
