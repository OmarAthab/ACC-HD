# ACC-HD — reproducible code

Reproducible code for *"Adaptive Conformal Coverage for Uncertainty-Aware Handover Decisions under Distribution Shift in 5G-Advanced Networks"* (framework: **ACC-HD**). The pipeline provides: **Bayesian hyper-parameter optimisation (Optuna/TPE)** on the validation split, a training loop (early stopping + LR scheduling + gradient clipping + best-checkpoint), an evidential ablation, an ordered drifting-stream experiment for adaptive conformal (ACI), and stride=2 for cleaner conformal exchangeability. It includes the methodology package: Gauss-Markov mobility, isolated scenario datasets with UE-disjoint (leakage-free) splits, a distribution-shift protocol, full-test-set metrics (accuracy/F1/precision/recall), validation-tuned thresholds, focal+temperature calibration, and a coverage-vs-set-size trade-off.

## Hyper-parameter selection (new, principled)
Hyper-parameters are NOT hand-picked. `models/hpo.py` runs Optuna's TPE (Bayesian) search with median pruning over hidden size, dropout, learning rate, weight decay, batch size and focal gamma, optimising validation macro-F1. The search never touches the test set; the same winning backbone config is applied to all GRU-based methods; the search space and selected values are written to `results/metrics/hpo_best.json`. Epoch count is decided by early stopping (best-val checkpoint), not searched. Toggle via `config.py HPOConfig.enabled`.

## How it works
1. `python run_simulator.py`
   Builds isolated scenario datasets under `data/`:
   - `data/mixed/` (SOURCE) → split BY UE into `train / val / calib / test` (no leakage).
   - `data/vehicular_fast/test`, `data/high_blockage/test`, `data/non_stationary/test` → shifted, unseen tests.
   - `data/data_dictionary.md` explains every file/field.
2. `python experiments/run_all.py`
   - Trains ACC-HD (focal) + baselines on `mixed/train`; tunes temperature + policy thresholds on `mixed/val`; calibrates conformal on `mixed/calib`.
   - Evaluates on `mixed/test` (in-distribution) and the shifted tests.
   - Writes `results/metrics/results_indist.csv`, `results_shift.csv`, `coverage_vs_size.csv`, and 300-dpi figures in `results/figures/`.

## Key outputs
- `results_indist.csv` — full-test-set metrics per method (accuracy, F1, precision, recall, ECE, coverage, set size, handover KPIs).
- `results_shift.csv` — coverage under distribution shift: static conformal (degrades) vs adaptive/ACI (holds).
- `coverage_vs_size.csv` + `fig_coverage_vs_setsize.png` — the trade-off curve (stricter coverage → larger set). The set size is the uncertainty signal that feeds the confidence-aware policy (larger set → more deferral to safe CHO).
- `fig_coverage_under_shift.png`, reliability diagram, per-method bars, radar, KPI heatmap, loss curve.

## Per-class operating points (config.py SERVICE_CLASSES)
URLLC alpha=0.02, eMBB alpha=0.05, mMTC alpha=0.10 — ordered by strictness, informative sets. The full sweep (ALPHA_SWEEP, incl. the 99% extreme) is reported as the trade-off curve.

## Profiles (config.py)
- LIGHT (default): num_ues=24, num_ues_shift=12, window_stride=5 — a quick smoke test only. Too small for reliable per-class conformal (a class may get few calibration samples).
- FULL (for real results, on a GPU): num_ues≈90, num_ues_shift≈30, window_stride=2. Set `UA_PROFILE=full` to switch.

## Run on a GPU (recommended for the FULL profile)
On any machine with a CUDA-capable GPU:
```
pip install -r requirements.txt
python run_simulator.py
UA_PROFILE=full UA_HP_JSON=results/metrics/hpo_best.json python -m experiments.run_all
```
This builds `data/`, runs the full pipeline, and writes results and figures under `results/`.

## Reproducibility
Fixed random seeds: the headline tables (4, 7, 9) use 5 seeds (0–4) via `UA_SEEDS="0,1,2,3,4" python experiments/run_seeds.py`; `run_all.py` performs a single-seed base run from `config.py SEED`, and the remaining single-seed tables are labeled as such. UE-disjoint splits; all parameters in `config.py`; every result maps to a script. Reference hardware: a single NVIDIA T4 GPU. Gauss-Markov mobility (Liang & Haas 1999); AR(1) shadowing (3GPP TR 38.901 style); 3GPP A3 hysteresis + time-to-trigger handover.

## Naming
The method is **ACC-HD** (Adaptive Conformal Coverage for Handover Decisions) in the paper. Some internal
modules/identifiers retain the historical development name **CoCA-HD** (e.g. `models/coca_hd.py`); they refer
to the same method. Result files are written under the label "ACC-HD (proposed)".

## Reproducing the paper numbers
The config defaults are a quick SMOKE profile. For the paper (90/30 UEs, 600 s, stride 2) set `UA_PROFILE=full`
(and point `UA_HP_JSON` at `results/metrics/hpo_best.json` for the HPO-selected backbone). Example:
`UA_PROFILE=full UA_HP_JSON=results/metrics/hpo_best.json python -m experiments.run_all`
