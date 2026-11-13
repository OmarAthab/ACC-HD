"""
config.py
=========
ALL adjustable parameters for the whole project live in this one file.

Beginner tip: to change the experiment (make it bigger/smaller, faster/slower),
edit the numbers here and re-run. You should not need to edit any other file
to change the size or settings of a run.

Everything is grouped into small dataclasses by topic.
"""

from dataclasses import dataclass, field
from pathlib import Path


# ---------------------------------------------------------------------------
# Where results are written (created automatically).
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent      # the UA-PHM-code/ folder
RESULTS_DIR = PROJECT_ROOT / "results"
DATASET_DIR = RESULTS_DIR / "dataset"
DATA_DIR = PROJECT_ROOT / "data"        # organized per-scenario datasets (train/val/calib/test)


# ---------------------------------------------------------------------------
# Global reproducibility
# ---------------------------------------------------------------------------
SEED = 42                       # default base seed for run_all.py; the 5-seed protocol uses run_seeds.py


# ---------------------------------------------------------------------------
# Progress printing (requirement: show progress while long code runs)
# ---------------------------------------------------------------------------
@dataclass
class ProgressConfig:
    print_every_seconds: float = 30.0   # print a progress line at most this often
    show_eta: bool = True               # show estimated time remaining


# ---------------------------------------------------------------------------
# Network layout (the cells / base stations)
# ---------------------------------------------------------------------------
@dataclass
class NetworkConfig:
    # Standard 3GPP hexagonal 19-site cluster with the Urban Micro inter-site
    # distance (TR 38.901): ISD = 200 m. The 19-cell cluster spans ~1.2 km; the
    # coverage area gives the UEs room to roam around it. Single-sector (omni) cells.
    num_cells: int = 19             # 3GPP 19-site hexagonal cluster (centre + rings 6, 12)
    area_size_m: float = 1200.0     # square coverage area side length, in metres
    inter_site_distance_m: float = 200.0   # 3GPP Urban Micro ISD (TR 38.901)
    cell_tx_power_dbm: float = 46.0        # base-station transmit power (dBm)


# ---------------------------------------------------------------------------
# Handover-execution thresholds (used by the temporal HO state machine that
# turns per-decision predictions into executed handovers and standard KPIs).
# ---------------------------------------------------------------------------
@dataclass
class HandoverExecConfig:
    q_out_sinr_db: float = -8.0     # radio-link "out-of-sync" SINR threshold (3GPP-style)
    q_in_sinr_db: float = -4.0      # radio-link "in-sync" recovery threshold
    t310_steps: int = 5            # steps below q_out before a Radio-Link Failure (T310)
    tcrit_steps: int = 10           # min time-of-stay; a return within this = ping-pong
    # Handover interruption-time (HIT) bands, from 5G-Advanced literature:
    #   - proactive make-before-break (LTM / early CHO): ~10-20 ms  (Ericsson LTM; IET 2025)
    #   - late / reactive L3 handover:                    50-90 ms  (Ericsson: "50-90 ms")
    #   - a failed handover attempt:                       ~90 ms
    # RLF recovery (RRC re-establishment) is a SEPARATE event, reported on its own
    # and NOT mixed into the handover interruption time.
    hit_ltm_ms: float = 15.0        # proactive make-before-break (source still comfortable at execution)
    hit_legacy_ms: float = 70.0     # late/reactive handover (source already near the edge)
    hit_failed_ms: float = 90.0     # failed handover attempt
    rlf_recovery_ms: float = 150.0  # RRC re-establishment time after an RLF (reported separately)
    proactive_margin_db: float = -4.0  # source SINR at execution >= this => make-before-break (LTM band)
    ttt_steps: int = 3              # time-to-trigger: consecutive steps a candidate must lead (A3)
    a3_hysteresis: float = 0.05     # min score margin (candidate over serving) to arm a handover (A3)
    arrival_window_steps: int = 5   # decision->execution arrival window (~ horizon/stride); the target
                                    # must become servable (SINR >= q_out) within it or the HO fails


# ---------------------------------------------------------------------------
# UE mobility (how users move). Non-stationary = the movement statistics change.
# ---------------------------------------------------------------------------
@dataclass
class MobilityConfig:
    # ---- SAMPLE-SIZE RATIONALE (statistical, not hardware-driven) ----
    # num_ues is chosen so that AFTER the UE-disjoint split (train/val/calib/test)
    # every split — especially the conformal CALIBRATION split — retains enough
    # distinct users and samples for a valid, low-variance estimate:
    #   * Split conformal at miscoverage alpha needs a calibration set large enough
    #     that ceil((n+1)(1-alpha)) sits well inside n; per service class we want
    #     >= a few hundred calibration samples (n_calib UEs * windows/UE).
    #   * Coverage is a proportion, so its estimate variance ~ p(1-p)/N; tens of
    #     thousands of test samples give a tight (<~0.5%) confidence interval.
    # LIGHT defaults below are a SMOKE TEST only. The FULL profile
    # (UA_PROFILE=full) uses num_ues=90, num_ues_shift=30, sim_duration_s=600 —
    # comfortably above the minimum for stable per-class conformal. This is NOT a
    # hardware ceiling; raise it further on a stronger machine if desired.
    num_ues: int = 24               # UEs for the SOURCE (mixed) scenario (smoke: 24; FULL: 90)
    num_ues_shift: int = 12         # UEs for EACH shifted test scenario (smoke: 12; FULL: 30)
    sim_duration_s: float = 300.0   # simulated seconds per UE (smoke: 300; FULL: 600)
    time_step_s: float = 0.1        # measurement interval (Delta t): 100 ms

    # Gauss-Markov mobility model memory parameter (Liang & Haas, 1999).
    # alpha->1: smooth/correlated; alpha->0: random walk. Sweep it for sensitivity.
    gauss_markov_alpha: float = 0.85
    # regime speeds (m/s), kept for reference; scenarios.py sets per-scenario speeds.
    pedestrian_speed_ms: float = 1.4
    vehicular_speed_ms: float = 15.0
    stationary_speed_ms: float = 0.0
    regime_switch_prob: float = 0.002
    direction_change_std_rad: float = 0.05   # (unused by Gauss-Markov; kept for reference)


# ---------------------------------------------------------------------------
# Radio channel (turns distance + environment into signal strength)
# ---------------------------------------------------------------------------
@dataclass
class ChannelConfig:
    # ---- 3GPP TR 38.901 Urban Micro (UMi) Street-Canyon channel ----
    carrier_freq_ghz: float = 3.5        # 5G-Advanced mid-band (fc)
    bs_height_m: float = 10.0            # UMi base-station height (TR 38.901)
    ue_height_m: float = 1.5             # UE height (TR 38.901 default)
    env_height_m: float = 1.0            # effective environment height h_E (UMi)
    # 3GPP antenna element / 3-sector site (TR 38.901 Table 7.3-1)
    num_sectors: int = 3                 # sectors per site (boresights 0/120/240 deg)
    ant_gain_max_dbi: float = 8.0        # max element gain
    ant_hpbw_deg: float = 65.0           # 3 dB beamwidth (H and V)
    ant_front_back_db: float = 30.0      # side-lobe / front-to-back floor (A_m, SLA_V)
    ant_downtilt_deg: float = 10.0       # mechanical downtilt
    # shadow fading: scenario-correct std (TR 38.901 Table 7.4.1-1)
    sf_los_db: float = 4.0
    sf_nlos_db: float = 7.82
    shadowing_time_corr: float = 0.98    # AR(1) temporal correlation (smooth, realistic)
    fast_fading_std_db: float = 1.0      # residual small-scale (L3 filtering averages most out)
    noise_floor_dbm: float = -104.0      # thermal noise + NF over the operating bandwidth
    # average co-channel sector activity (network load): fraction of interfering
    # sectors active on the same resource. NOT a fudge for SINR level (that comes
    # from the 38.901 pathloss + antenna geometry) -- it is the standard
    # full-buffer-vs-partial-load OPERATING POINT. 1.0 = full buffer (worst case);
    # 0.5 = a typical average cell resource-utilisation / activity factor, and is
    # the operating point reported in the main results. Full-buffer (1.0) is kept
    # as a stress point in the load-sensitivity sweep (experiments/run_load_sweep.py).
    # Overridable at run time with env UA_LOAD_FACTOR (applied at the bottom of this
    # module) so the simulator and the trainer always agree on one value per run.
    load_factor: float = 0.5
    blockage_prob: float = 0.01          # chance per step a link becomes blocked (dynamic blockage)
    blockage_extra_loss_db: float = 20.0 # extra loss while blocked


# ---------------------------------------------------------------------------
# Service classes (5G-Advanced): URLLC is the primary scenario.
# Each class gets its own conformal risk/coverage level alpha.
# ---------------------------------------------------------------------------
@dataclass
class ServiceClassConfig:
    name: str
    fraction_of_ues: float      # share of UEs assigned to this class
    alpha: float                # miscoverage / risk level (smaller = stricter)


SERVICE_CLASSES = [
    # Per-class operating points (miscoverage alpha), ORDERED by strictness.
    # These are points on the reported coverage-vs-set-size trade-off curve (the
    # alpha sweep); the set size itself is the uncertainty signal that feeds the
    # confidence-aware policy (larger set -> more deferral).
    #
    # v5 CHANGE: aligned to KPI Table 2 "Confidence Reliability" targets so each
    # service class MEETS its column (URLLC 95-98%, eMBB 90-95%, mMTC 88-92%):
    #   URLLC alpha=0.02 -> coverage ~97.9%   (target 95-98%)
    #   eMBB  alpha=0.05 -> coverage ~94.8%   (target 90-95%)
    #   mMTC  alpha=0.10 -> coverage ~90.1%   (target 88-92%)
    # (v4 used 0.05/0.10/0.15, which left mMTC at ~84.7%, below its 88% floor.)
    ServiceClassConfig(name="URLLC", fraction_of_ues=0.34, alpha=0.02),  # strictest
    ServiceClassConfig(name="eMBB",  fraction_of_ues=0.33, alpha=0.05),
    ServiceClassConfig(name="mMTC",  fraction_of_ues=0.33, alpha=0.10),
]

# Alpha values swept to produce the coverage-vs-set-size trade-off curve.
ALPHA_SWEEP = [0.01, 0.02, 0.05, 0.10, 0.15, 0.20]


# ---------------------------------------------------------------------------
# Prediction task (how the dataset is shaped)
# ---------------------------------------------------------------------------
@dataclass
class TaskConfig:
    window_len: int = 20            # W: how many past measurement steps per sample
    # window_stride: take one window every `stride` steps. FULL profile uses
    # stride=2 (not 1): overlapping windows are strongly auto-correlated, which
    # weakens the exchangeability assumption behind split conformal and inflates
    # the effective-vs-nominal sample gap; stride=2 halves that autocorrelation
    # (and memory) with negligible information loss. smoke: 5; FULL: 2.
    window_stride: int = 5
    horizon: int = 10               # H: predict the serving cell H steps ahead
    top_k_neighbours: int = 6       # number of strongest neighbour cells kept as features
    # SOURCE-scenario split is done by UE (disjoint users -> no leakage):
    train_ue_frac: float = 0.5      # UEs used for training
    val_ue_frac: float = 0.2        # UEs used for validation (tuning thresholds/temperature)
    calib_ue_frac: float = 0.15     # UEs used for conformal calibration
    # remaining ~0.15 of UEs -> in-distribution test


# ---------------------------------------------------------------------------
# Training procedure (shared by every neural model, so comparisons are fair).
# Epochs are NOT a fixed hand-picked number: training runs up to `max_epochs`
# and stops early on the validation macro-F1, returning the BEST checkpoint.
# The default hyper-parameters below are only a FALLBACK used when HPO is off or
# Optuna is unavailable; otherwise the Bayesian search (HPOConfig) sets them.
# ---------------------------------------------------------------------------
@dataclass
class TrainConfig:
    max_epochs: int = 120           # upper bound; early stopping usually stops earlier
    patience: int = 15              # stop if val macro-F1 doesn't improve this many epochs
    min_delta: float = 1e-4         # minimum val improvement that counts as progress
    grad_clip: float = 5.0          # max global grad-norm (0 disables) — stabilises the GRU
    lr_schedule_patience: int = 4   # ReduceLROnPlateau patience (epochs)
    lr_schedule_factor: float = 0.5 # LR *= factor when val stalls
    # fallback hyper-parameters (used only if HPO disabled/unavailable):
    hidden_size: int = 128
    dropout: float = 0.2
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 128
    focal_gamma: float = 2.0
    # evaluation/ensemble knobs:
    ensemble_members: int = 3
    mc_dropout_passes: int = 20
    target_commit: float = 0.70     # policy-threshold tuning target on validation


# ---------------------------------------------------------------------------
# Bayesian hyper-parameter optimisation (Optuna TPE) on the VALIDATION split.
# The search never sees the test set. Selected values + the search space are
# saved to results/metrics/hpo_best.json for the paper's reproducibility table.
# ---------------------------------------------------------------------------
@dataclass
class HPOConfig:
    enabled: bool = True
    n_trials: int = 30              # Bayesian (TPE) trials
    max_epochs_per_trial: int = 40  # shorter budget during search (pruned earlier)
    patience_per_trial: int = 6
    timeout_seconds: int = 0        # 0 = no wall-clock cap (don't limit by wall-clock time)
    hidden_choices: tuple = (64, 96, 128, 192, 256)
    dropout_range: tuple = (0.1, 0.5)
    lr_range: tuple = (1e-4, 5e-3)          # log-uniform
    weight_decay_range: tuple = (1e-6, 1e-3)  # log-uniform
    batch_choices: tuple = (64, 128, 256)
    focal_gamma_range: tuple = (0.5, 3.0)


# ---------------------------------------------------------------------------
# One object that bundles everything, imported everywhere as `CFG`.
# ---------------------------------------------------------------------------
@dataclass
class Config:
    seed: int = SEED
    progress: ProgressConfig = field(default_factory=ProgressConfig)
    network: NetworkConfig = field(default_factory=NetworkConfig)
    mobility: MobilityConfig = field(default_factory=MobilityConfig)
    channel: ChannelConfig = field(default_factory=ChannelConfig)
    task: TaskConfig = field(default_factory=TaskConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    hpo: HPOConfig = field(default_factory=HPOConfig)
    hexec: HandoverExecConfig = field(default_factory=HandoverExecConfig)
    results_dir: Path = RESULTS_DIR
    dataset_dir: Path = DATASET_DIR
    data_dir: Path = DATA_DIR


CFG = Config()   # <-- import this: `from config.config import CFG, SERVICE_CLASSES`

# --- run-time channel-load override (env) -----------------------------------
# UA_LOAD_FACTOR lets the simulator AND the trainer share one network-load
# operating point per run (the SINR feature is baked in at simulation time, so
# both processes must read the same value). Inert when the env var is unset.
import os as _os_cfg
_lf = _os_cfg.environ.get("UA_LOAD_FACTOR")
if _lf:
    try:
        CFG.channel.load_factor = float(_lf)
    except ValueError:
        pass

# ---------------------------------------------------------------------------
# Reproducibility profile. The dataclass defaults above are a SMOKE TEST (24
# UEs, 300 s, stride 5) for quick checks. The paper's results use the FULL
# profile (90/30 UEs, 600 s, stride 2); set UA_PROFILE=full to reproduce them,
# or override any field with UA_NUM_UES / UA_NUM_UES_SHIFT / UA_SIM_S /
# UA_STRIDE. (Backbone size comes from HPO via UA_HP_JSON.)
# ---------------------------------------------------------------------------
if _os_cfg.environ.get("UA_PROFILE", "").lower() == "full":
    CFG.mobility.num_ues = 90
    CFG.mobility.num_ues_shift = 30
    CFG.mobility.sim_duration_s = 600.0
    CFG.task.window_stride = 2
for _ev, _obj, _attr, _cast in [
    ("UA_NUM_UES", CFG.mobility, "num_ues", int),
    ("UA_NUM_UES_SHIFT", CFG.mobility, "num_ues_shift", int),
    ("UA_SIM_S", CFG.mobility, "sim_duration_s", float),
    ("UA_STRIDE", CFG.task, "window_stride", int),
]:
    _v = _os_cfg.environ.get(_ev)
    if _v:
        try:
            setattr(_obj, _attr, _cast(_v))
        except ValueError:
            pass
