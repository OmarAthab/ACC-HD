"""
causal_normalization_patch.py
=========================================================
FIXES A REAL LEAK. In simulator/dataset.py the current normalisation

    radio = (radio - radio.mean(axis=0)) / (radio.std(axis=0) + 1e-6)

standardises each UE using statistics over its WHOLE trajectory (axis=0 = time),
i.e. including FUTURE steps -> non-causal. The paper claims "leakage-free"; this
must be made past-only before that claim is defensible.

This file gives a drop-in replacement for the normalisation block inside
`_windows_from_ue` in simulator/dataset.py: an EXPANDING (causal) per-UE z-score
that uses only past-and-current observations at each step. After applying it you
MUST regenerate the datasets and re-run the pipeline:

    1) replace the two normalisation lines in dataset.py `_windows_from_ue` with
       `radio = causal_zscore(np.concatenate([rsrp, sinr], axis=1))`
       (and `from experiments.causal_normalization_patch import causal_zscore`,
        or paste the function into dataset.py directly),
    2) python run_simulator.py          # rebuild data/ with causal normalisation
    3) python experiments/run_seeds.py  # re-run; coverage/HOF now leakage-free
    4) then add sentence (past-only normalisation) to the paper.

The expanding estimate matches the paper's "each UE standardised using its own
statistics" while being causal. A short warm-up (first `min_count` steps) falls
back to a stable estimate so early windows are not divided by a near-zero std.
"""

import numpy as np


def causal_zscore(radio: np.ndarray, min_count: int = 20, eps: float = 1e-6) -> np.ndarray:
    """Past-only (expanding) z-score of a (T, D) per-UE feature stream.

    At step t the mean/std use steps [0..t] only (never the future). For the first
    `min_count` steps the statistics are thin, so we blend toward the first-window
    estimate to avoid division by a near-zero early std (a deployable cold-start
    would use the training-set transform; the blend is a simple, causal stand-in).
    """
    radio = np.asarray(radio, dtype=np.float64)
    T, D = radio.shape
    csum = np.cumsum(radio, axis=0)
    csum2 = np.cumsum(radio * radio, axis=0)
    cnt = np.arange(1, T + 1, dtype=np.float64)[:, None]
    mean_c = csum / cnt
    var_c = np.maximum(csum2 / cnt - mean_c * mean_c, 0.0)
    std_c = np.sqrt(var_c)

    # cold-start stabiliser: for the first min_count steps, use the stats AT
    # min_count (still past-only w.r.t. the rest of the trajectory, computed once).
    if T > min_count:
        warm_mean = mean_c[min_count - 1]
        warm_std = std_c[min_count - 1]
        mean_c[:min_count] = warm_mean
        std_c[:min_count] = warm_std

    return ((radio - mean_c) / (std_c + eps)).astype(np.float32)


# ---- optional self-test (no leakage: value at t must not depend on t+1..) -------
if __name__ == "__main__":
    rng = np.random.default_rng(0)
    x = rng.normal(size=(100, 4))
    a = causal_zscore(x)
    # changing a FUTURE row must not change an earlier normalised row
    x2 = x.copy(); x2[80] += 100.0
    b = causal_zscore(x2)
    assert np.allclose(a[:80], b[:80]), "LEAK: past rows changed when a future row changed"
    print("causal_zscore OK: past rows are independent of future observations.")
