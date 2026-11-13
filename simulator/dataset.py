"""
dataset.py  (scenario-aware, UE-disjoint, leakage-free)
=======================================================
Produces the machine-learning datasets following the distribution-shift protocol:
  - SOURCE scenario ("urban"): split BY UE (disjoint users) into
    train / val / calib / test  -> no temporal leakage between splits.
  - SHIFTED scenarios (vehicular_fast / high_blockage / non_stationary): all UEs
    used as held-out TEST sets (genuinely out-of-source) to measure behaviour
    under distribution shift.

Each sample = a window [RSRP, SINR, current-serving-one-hot] over `window_len`
steps; label = serving cell `horizon` steps ahead.

For the handover-execution KPIs (RLF / HOF / ping-pong / HIT / time-of-stay), each
saved split ALSO carries, per sample: the UE id (`ue`), the RAW serving+neighbour
SINR in dB at the decision step (`sinr_raw`), and the current serving cell
(`serving_now`). Within a UE the samples are in temporal order, so the handover
state machine can replay each UE's decision sequence.
"""

import numpy as np

from config.config import CFG, SERVICE_CLASSES
from common.progress import ProgressPrinter
from common.io_utils import save_arrays, ensure_dir
from simulator.network import build_cell_positions
from simulator.mobility import generate_ue_trajectory
from simulator.channel import compute_rsrp_sinr
from simulator.events import compute_serving_cells, compute_prediction_targets
from simulator.scenarios import SCENARIOS, SOURCE_SCENARIO, SHIFT_SCENARIOS


def _causal_zscore(radio, min_count: int = 20, eps: float = 1e-6):
    """PAST-ONLY (causal) per-UE z-score.

    At step t the mean/std use steps [0..t] only (an expanding estimate), never the
    future, so no future observation enters a sample. For the first `min_count`
    steps (thin statistics) we use the estimate AT min_count to avoid dividing by a
    near-zero early std (a deployable cold-start would use the training transform).

    """
    radio = np.asarray(radio, dtype=np.float64)
    T = radio.shape[0]
    csum = np.cumsum(radio, axis=0)
    csum2 = np.cumsum(radio * radio, axis=0)
    cnt = np.arange(1, T + 1, dtype=np.float64)[:, None]
    mean_c = csum / cnt
    std_c = np.sqrt(np.maximum(csum2 / cnt - mean_c * mean_c, 0.0))
    if T > min_count:
        mean_c[:min_count] = mean_c[min_count - 1]
        std_c[:min_count] = std_c[min_count - 1]
    return ((radio - mean_c) / (std_c + eps)).astype(np.float32)


def _windows_from_ue(rsrp, sinr, serving, targets, window_len, stride):
    """Cut one UE's series into samples. Returns per-sample:
      X          : (n, W, 3K) normalised features [RSRP, SINR, serving one-hot]
      y          : (n,) label = serving cell `horizon` steps ahead
      sinr_raw   : (n, K) RAW SINR (dB) at the decision step (window end)
      serving_now: (n,) current serving cell at the decision step
    """
    num_steps, num_cells = rsrp.shape
    # Causal (past-only) per-UE standardisation (avoids whole-trajectory leakage).
    radio = _causal_zscore(np.concatenate([rsrp, sinr], axis=1))
    serving_onehot = np.eye(num_cells, dtype=np.float32)[serving]
    feat = np.concatenate([radio, serving_onehot], axis=1)     # (num_steps, 3K)

    X, y, sinr_raw, serving_now = [], [], [], []
    for start in range(0, num_steps - window_len, max(1, stride)):
        end = start + window_len
        now = end - 1                       # decision step (window end)
        label = targets[now]
        if label < 0:
            continue
        X.append(feat[start:end])
        y.append(label)
        sinr_raw.append(sinr[now])          # raw dB SINR at the decision step
        serving_now.append(serving[now])
    if not X:
        return None
    return {
        "X": np.asarray(X, np.float32),
        "y": np.asarray(y, np.int64),
        "sinr_raw": np.asarray(sinr_raw, np.float32),
        "serving_now": np.asarray(serving_now, np.int64),
    }


def _one_ue(scenario, cell_positions, rng):
    positions, _reg, blk = generate_ue_trajectory(rng, scenario)
    rsrp, sinr = compute_rsrp_sinr(positions, cell_positions, rng, blockage_mult=blk)
    serving = compute_serving_cells(rsrp)
    targets = compute_prediction_targets(serving, CFG.task.horizon)
    return _windows_from_ue(rsrp, sinr, serving, targets,
                            CFG.task.window_len, CFG.task.window_stride)


def _service_class_of_ue(ue_idx, rng):
    fr = np.array([sc.fraction_of_ues for sc in SERVICE_CLASSES]); fr = fr / fr.sum()
    return int(rng.choice(len(SERVICE_CLASSES), p=fr))


def _gen_scenario_ues(scenario, num_ues, cell_positions, rng, name):
    """Return a list of per-UE dicts {X, y, cls, sinr_raw, serving_now}."""
    ues = []
    pp = ProgressPrinter(total=num_ues, name=f"Sim {name}",
                         print_every_seconds=CFG.progress.print_every_seconds)
    for u in range(num_ues):
        d = _one_ue(scenario, cell_positions, rng)
        if d is not None:
            d["cls"] = np.full(len(d["y"]), _service_class_of_ue(u, rng), dtype=np.int64)
            ues.append(d)
        pp.update(u + 1)
    pp.done()
    return ues


def _concat(ues, idxs):
    """Concatenate selected UEs, adding a per-sample UE id. Within a UE the samples
    stay in temporal order, so the handover state machine can replay each UE."""
    X, y, cls, sinr_raw, serving_now, ue = [], [], [], [], [], []
    for uid, i in enumerate(idxs):
        d = ues[i]
        m = len(d["y"])
        X.append(d["X"]); y.append(d["y"]); cls.append(d["cls"])
        sinr_raw.append(d["sinr_raw"]); serving_now.append(d["serving_now"])
        ue.append(np.full(m, uid, dtype=np.int64))
    return (np.concatenate(X, 0), np.concatenate(y, 0), np.concatenate(cls, 0),
            np.concatenate(sinr_raw, 0), np.concatenate(serving_now, 0),
            np.concatenate(ue, 0))


def _save_split(scenario_name, split_name, X, y, cls, sinr_raw, serving_now, ue):
    d = ensure_dir(CFG.data_dir / scenario_name)
    save_arrays(d / f"{split_name}.npz", X=X, y=y, cls=cls,
                sinr_raw=sinr_raw, serving_now=serving_now, ue=ue,
                num_cells=np.array([CFG.network.num_cells]))


def build_all():
    """Generate and save all scenario datasets (source split by UE + shifted tests)."""
    rng = np.random.default_rng(CFG.seed)
    cell_positions = build_cell_positions()
    task = CFG.task

    # -------- SOURCE scenario: split BY UE (disjoint) --------
    src = _gen_scenario_ues(SCENARIOS[SOURCE_SCENARIO], CFG.mobility.num_ues,
                            cell_positions, rng, SOURCE_SCENARIO)
    n = len(src)
    order = rng.permutation(n)
    n_tr = int(task.train_ue_frac * n)
    n_va = int(task.val_ue_frac * n)
    n_ca = int(task.calib_ue_frac * n)
    idx_tr = order[:n_tr]
    idx_va = order[n_tr:n_tr + n_va]
    idx_ca = order[n_tr + n_va:n_tr + n_va + n_ca]
    idx_te = order[n_tr + n_va + n_ca:]
    for split, idxs in [("train", idx_tr), ("val", idx_va),
                        ("calib", idx_ca), ("test", idx_te)]:
        arrays = _concat(src, idxs)
        _save_split(SOURCE_SCENARIO, split, *arrays)
        print(f"  {SOURCE_SCENARIO}/{split}: {len(arrays[1])} samples from {len(idxs)} UEs")

    # -------- SHIFTED scenarios: all UEs -> test --------
    for name in SHIFT_SCENARIOS:
        ues = _gen_scenario_ues(SCENARIOS[name], CFG.mobility.num_ues_shift,
                                cell_positions, rng, name)
        arrays = _concat(ues, list(range(len(ues))))
        _save_split(name, "test", *arrays)
        print(f"  {name}/test: {len(arrays[1])} samples from {len(ues)} UEs")

    _write_data_dictionary()
    print("\nAll scenario datasets saved under:", CFG.data_dir)


def build_source_only():
    """Generate ONLY the SOURCE scenario (train/val/calib/test) into CFG.data_dir.

    Used by the load-sensitivity sweep (experiments/run_load_sweep.py), which varies
    CFG.channel.load_factor and needs a fresh in-distribution dataset per load
    WITHOUT the shifted tests. Uses the exact same UE-disjoint split as build_all(),
    so the source dataset is identical in construction -- only the load differs.
    """
    rng = np.random.default_rng(CFG.seed)
    cell_positions = build_cell_positions()
    task = CFG.task
    src = _gen_scenario_ues(SCENARIOS[SOURCE_SCENARIO], CFG.mobility.num_ues,
                            cell_positions, rng, SOURCE_SCENARIO)
    n = len(src)
    order = rng.permutation(n)
    n_tr = int(task.train_ue_frac * n)
    n_va = int(task.val_ue_frac * n)
    n_ca = int(task.calib_ue_frac * n)
    idx_tr = order[:n_tr]
    idx_va = order[n_tr:n_tr + n_va]
    idx_ca = order[n_tr + n_va:n_tr + n_va + n_ca]
    idx_te = order[n_tr + n_va + n_ca:]
    for split, idxs in [("train", idx_tr), ("val", idx_va),
                        ("calib", idx_ca), ("test", idx_te)]:
        arrays = _concat(src, idxs)
        _save_split(SOURCE_SCENARIO, split, *arrays)
        print(f"  {SOURCE_SCENARIO}/{split}: {len(arrays[1])} samples from {len(idxs)} UEs")


def _write_data_dictionary():
    d = ensure_dir(CFG.data_dir)
    K = CFG.network.num_cells
    txt = f"""# Data Dictionary - UA-PHM datasets

Organized by scenario and split. Each `.npz` holds:
- `X`: float32 (N, {CFG.task.window_len}, {3*K}) - window of features per sample.
       Per step: [RSRP x{K}, SINR x{K}, current-serving one-hot x{K}] (RSRP/SINR past-only/causal z-normalized per UE).
- `y`: int64 (N,) - label = serving cell index ({K} classes) {CFG.task.horizon} steps ahead.
- `cls`: int64 (N,) - service class (0=URLLC, 1=eMBB, 2=mMTC).
- `sinr_raw`: float32 (N, {K}) - RAW SINR (dB) per cell at the decision step (for HO-execution KPIs).
- `serving_now`: int64 (N,) - current serving cell at the decision step.
- `ue`: int64 (N,) - UE id within the scenario/split (samples ordered in time within a UE).
- `num_cells`: [{K}].

## Scenarios
- `{SOURCE_SCENARIO}/` (SOURCE): split BY UE into train/val/calib/test (disjoint users -> no leakage).
- {", ".join(SHIFT_SCENARIOS)}: shifted held-out tests (genuinely out-of-source; unseen in training).

## Protocol
Train on {SOURCE_SCENARIO}/train; tune thresholds/temperature on val; calibrate conformal on calib;
evaluate on {SOURCE_SCENARIO}/test (in-distribution) AND the shifted tests. 3GPP UMi 19-site
hexagonal layout (ISD 200 m, TR 38.901); Gauss-Markov mobility (alpha={CFG.mobility.gauss_markov_alpha});
AR(1) shadowing (rho={CFG.channel.shadowing_time_corr}); 3GPP A3 hysteresis + time-to-trigger.
"""
    (d / "data_dictionary.md").write_text(txt, encoding="utf-8")
    print("  wrote data/data_dictionary.md")
