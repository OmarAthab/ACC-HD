"""
latency_measure.py
===============================================
Replaces the batch-amortised 0.005 ms figure with a defensible per-decision latency
: synchronised timing, batch size = 1 (one real decision at a time) and a
batched reference, the conformal-set + policy overhead, and the latency DISTRIBUTION
(p50/p95/p99) - not just a mean. ).

Drop into <repo>/experiments/.  Run:  python experiments/latency_measure.py
Output: results/metrics/latency.csv
"""

import sys, time
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.config import CFG, SERVICE_CLASSES
from common.io_utils import load_arrays, ensure_dir, save_table
from common.seeding import set_all_seeds
from models.train import get_device
from models import conformal as cf
from baselines.uq_baselines import build_softmax
from simulator.scenarios import SOURCE_SCENARIO

OUT = CFG.results_dir / "metrics"
N = 1000          # decisions timed one-at-a-time
WARMUP = 50


def _sync(dev):
    if dev.type == "cuda":
        torch.cuda.synchronize()


def main():
    ensure_dir(OUT)
    d = load_arrays(CFG.data_dir / SOURCE_SCENARIO / "test.npz")
    X = d["X"]; K = int(d["num_cells"][0]); F = X.shape[-1]
    set_all_seeds(CFG.seed)
    model, _, _ = build_softmax(F, K, CFG.train.hidden_size, CFG.train.dropout)
    model.eval()
    dev = get_device()
    model.to(dev)
    plat = torch.cuda.get_device_name(0) if dev.type == "cuda" else "CPU"

    xb_all = torch.as_tensor(X[:N], dtype=torch.float32, device=dev)

    # warm-up
    with torch.no_grad():
        for i in range(WARMUP):
            _ = model(xb_all[i:i+1])
    _sync(dev)

    # ---- per-decision latency (batch=1), synchronised, incl. conformal+policy ----
    # build a calibration quantile once (not timed) for the conformal overhead
    import numpy as _np
    cal = d["X"][:2000]
    with torch.no_grad():
        pcal = torch.softmax(model(torch.as_tensor(cal, dtype=torch.float32, device=dev)), 1).cpu().numpy()
    qhat = cf.split_conformal_qhat(pcal, d["y"][:2000], SERVICE_CLASSES[0].alpha)

    lat = np.zeros(N)
    with torch.no_grad():
        for i in range(N):
            t0 = time.perf_counter()
            logits = model(xb_all[i:i+1])
            p = torch.softmax(logits, 1).cpu().numpy()[0]       # includes device->host
            # conformal set + policy (the real per-decision overhead)
            cset = (1.0 - p) <= qhat
            c = float(p.max()); _ssize = int(cset.sum())
            _ = (c >= 0.55 and _ssize == 1)                      # policy decision
            _sync(dev)
            lat[i] = (time.perf_counter() - t0) * 1000.0         # ms

    # ---- batched reference (amortised) ----
    _sync(dev); t0 = time.perf_counter()
    with torch.no_grad():
        _ = torch.softmax(model(xb_all), 1).cpu().numpy()
    _sync(dev)
    batched_ms = (time.perf_counter() - t0) / N * 1000.0

    row = {"platform": plat, "batch1_mean_ms": round(float(lat.mean()), 4),
           "batch1_p50_ms": round(float(np.percentile(lat, 50)), 4),
           "batch1_p95_ms": round(float(np.percentile(lat, 95)), 4),
           "batch1_p99_ms": round(float(np.percentile(lat, 99)), 4),
           "batched_amortised_ms": round(float(batched_ms), 5),
           "includes": "forward + softmax + device->host + conformal set + policy"}
    save_table(OUT / "latency", [row])
    print("=== per-decision latency (synchronised) ===")
    for k, v in row.items():
        print(f"  {k:22s} {v}")
    print(f"\nReport batch=1 p50/p95/p99 as the per-decision latency on {plat}; keep the batched number")
    print("clearly labelled as amortised.")


if __name__ == "__main__":
    main()
