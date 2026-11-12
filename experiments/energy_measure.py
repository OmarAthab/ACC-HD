"""
energy_measure.py
=====================================
Adds an energy-per-decision measurement.

Drops into:  <repo>/experiments/energy_measure.py
Run:         python experiments/energy_measure.py

What it measures, for the proposed GRU backbone at the real input shape:
  * parameter count,
  * inference latency (ms per decision), batch-amortised, averaged over repeats,
  * GPU energy per decision (millijoules) via NVML power sampling during a timed
    loop  -- ONLY if a CUDA GPU and pynvml are available; otherwise it reports
    latency and states explicitly that energy requires a hardware power meter
    (an on-device/edge measurement is the proper venue, per the paper's future
    work). No fabricated energy numbers.


"""

import sys
import time
from pathlib import Path
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.config import CFG
from common.io_utils import load_arrays, ensure_dir, save_table
from common.seeding import set_all_seeds
from models.train import predict_probs, get_device
from baselines.uq_baselines import build_softmax
from simulator.scenarios import SOURCE_SCENARIO

OUT = CFG.results_dir / "metrics"
N_EVAL = 2000
REPEATS = 20


def _gpu_name():
    try:
        return torch.cuda.get_device_name(0)
    except Exception:
        return "CPU"


def _try_nvml():
    try:
        import pynvml
        pynvml.nvmlInit()
        return pynvml, pynvml.nvmlDeviceGetHandleByIndex(0)
    except Exception:
        return None, None


def main():
    ensure_dir(OUT)
    d = load_arrays(CFG.data_dir / SOURCE_SCENARIO / "test.npz")
    X = d["X"][:N_EVAL]; K = int(d["num_cells"][0]); F = X.shape[-1]
    set_all_seeds(CFG.seed)
    model, _, _ = build_softmax(F, K, CFG.train.hidden_size, CFG.train.dropout)
    model.eval()
    dev = get_device()
    n_params = int(sum(p.numel() for p in model.parameters()))
    plat = _gpu_name()

    # warm-up
    _ = predict_probs(model, X[:256])

    # latency
    ts = []
    for _ in range(REPEATS):
        t0 = time.time(); _ = predict_probs(model, X); ts.append(time.time() - t0)
    ms_per_decision = float(np.median(ts) / len(X) * 1000.0)

    row = {"platform": plat, "n_params": n_params, "n_decisions": int(len(X)),
           "latency_ms_per_decision": round(ms_per_decision, 5)}

    # energy via NVML (GPU only)
    pynvml, handle = _try_nvml()
    if pynvml is not None and dev.type == "cuda":
        powers = []
        t0 = time.time()
        while time.time() - t0 < 2.0:                      # sample during a 2 s busy loop
            _ = predict_probs(model, X)
            powers.append(pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0)  # W
        elapsed = time.time() - t0
        decisions = len(X) * max(1, len(powers))
        avg_w = float(np.mean(powers))
        joules = avg_w * elapsed
        row["avg_power_W"] = round(avg_w, 2)
        row["energy_mJ_per_decision"] = round(joules / decisions * 1000.0, 4)
        print(f"  GPU power ~{avg_w:.1f} W -> {row['energy_mJ_per_decision']} mJ/decision")
    else:
        row["avg_power_W"] = ""
        row["energy_mJ_per_decision"] = ""
        print("  (no CUDA/pynvml -> energy not measured; report latency only and")
        print("   state that per-decision energy needs an on-device power meter.)")

    save_table(OUT / "energy_cost", [row])
    print("=== energy / latency ===")
    for k, v in row.items():
        print(f"  {k:26s} {v}")
    print(f"\nMeasured on: {plat}. This is NOT a UE/edge figure; frame UE/edge "
          f"deployment as future work (Package A / A7).")
    print(f"Saved -> {OUT / 'energy_cost.csv'}")


if __name__ == "__main__":
    main()
