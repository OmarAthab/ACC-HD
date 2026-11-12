"""
make_figures.py
===============
Renders the paper figures from the saved results at 300 dpi (Times New Roman).
Each figure is isolated in its own try/except so one failure never blocks the rest.

Figures (results/figures/):
  fig_topology.png           19-site UMi hexagonal layout + example UE path
  fig_coverage_vs_setsize.png, fig_coverage_under_shift.png, fig_coverage_drift.png
  fig_risk_coverage.png, fig_loss_convergence.png, fig_reliability.png
  fig_ece_bar.png, fig_accuracy_bar.png
  fig_ho_kpis.png            standard handover KPIs (RLF/HOF/PPR/HIT) across methods
  fig_radar.png              proposed vs baselines on normalised metrics
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.config import CFG
from common.io_utils import load_arrays, ensure_dir
from models.calibration import reliability_curve

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 10, "axes.titlesize": 10, "axes.labelsize": 10,
    "savefig.dpi": 300, "figure.dpi": 300,
})
PALETTE = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9",
           "#F0E442", "#999999", "#000000", "#8C564B"]

METRICS_DIR = CFG.results_dir / "metrics"
FIGDATA_DIR = CFG.results_dir / "figure_data"
FIG_DIR = CFG.results_dir / "figures"


def _save(fig, name):
    ensure_dir(FIG_DIR)
    fig.savefig(FIG_DIR / name, bbox_inches="tight")
    plt.close(fig)
    print(f"  figure -> {FIG_DIR / name}")


def _try(name, fn):
    try:
        fn()
    except Exception as e:
        print(f"  (figure {name} skipped: {e})")


def _decode(arr):
    return [s.decode() if isinstance(s, bytes) else str(s) for s in arr]


def make_all_figures():
    ensure_dir(FIG_DIR)
    df = pd.read_csv(METRICS_DIR / "results_indist.csv")
    fig = load_arrays(FIGDATA_DIR / "figdata.npz")
    ho = None
    try:
        ho = pd.read_csv(METRICS_DIR / "results_ho.csv")
    except Exception:
        pass

    _try("topology", lambda: _topology())

    def _sweep():
        tgt = 1 - fig["sweep_alpha"]
        f, ax = plt.subplots(figsize=(4.5, 3))
        ax.plot(tgt, fig["sweep_cov"], marker="o", color=PALETTE[0], label="coverage")
        ax.plot(tgt, tgt, "--", color="gray", lw=1)
        ax.set_xlabel("target coverage (1 - alpha)"); ax.set_ylabel("coverage", color=PALETTE[0])
        ax2 = ax.twinx()
        ax2.plot(tgt, fig["sweep_size"], marker="s", color=PALETTE[1])
        ax2.set_ylabel("avg set size (cells)", color=PALETTE[1])
        ax.set_title("Coverage vs set-size trade-off")
        _save(f, "fig_coverage_vs_setsize.png")
    if "sweep_alpha" in fig:
        _try("sweep", _sweep)

    def _shift():
        names = _decode(fig["shift_scenarios"])
        x = np.arange(len(names)); w = 0.38
        f, ax = plt.subplots(figsize=(5.2, 3))
        ax.bar(x - w/2, fig["shift_static"], w, label="static conformal", color=PALETTE[3])
        ax.bar(x + w/2, fig["shift_aci"], w, label="adaptive conformal (ours)", color=PALETTE[0])
        ax.set_xticks(x); ax.set_xticklabels(names, rotation=20, ha="right")
        ax.set_ylabel("coverage"); ax.set_ylim(0.7, 1.02)
        ax.set_title("Coverage under distribution shift"); ax.legend(fontsize=7)
        _save(f, "fig_coverage_under_shift.png")
    if "shift_scenarios" in fig:
        _try("shift", _shift)

    def _drift():
        st = fig["drift_static_run"]; ac = fig["drift_aci_run"]
        tgt = float(fig["drift_target"][0]) if "drift_target" in fig else 0.90
        x = np.arange(1, len(st) + 1)
        f, ax = plt.subplots(figsize=(5.4, 3))
        ax.plot(x, st, color=PALETTE[3], lw=1.4, label="static conformal")
        ax.plot(x, ac, color=PALETTE[0], lw=1.4, label="adaptive conformal (ACI, ours)")
        ax.axhline(tgt, ls="--", color="gray", lw=1, label=f"target {tgt:.2f}")
        if "drift_bounds" in fig:
            names = _decode(fig.get("drift_seg_names", []))
            b = list(fig["drift_bounds"]); starts = [0] + b[:-1]
            for xb in b[:-1]:
                ax.axvline(xb, color="k", lw=0.6, alpha=0.3)
            for i, nm in enumerate(names):
                mid = (starts[i] + b[i]) / 2 if i < len(b) else starts[i]
                ax.text(mid, 0.53, nm, ha="center", fontsize=6)
        ax.set_xlabel("stream index  (easy -> hard shift)")
        ax.set_ylabel("running coverage"); ax.set_ylim(0.5, 1.02)
        ax.set_title("Coverage on a drifting stream"); ax.legend(fontsize=7, loc="lower left")
        _save(f, "fig_coverage_drift.png")
    if "drift_static_run" in fig:
        _try("drift", _drift)

    def _rc():
        f, ax = plt.subplots(figsize=(4.5, 3))
        ax.plot(fig["rc_coverage"] * 100, fig["rc_accuracy"] * 100, marker="o", color=PALETTE[0])
        ax.set_xlabel("committed fraction (%)"); ax.set_ylabel("accuracy on committed (%)")
        ax.set_title("Risk-coverage (selective prediction)"); ax.grid(alpha=0.3)
        _save(f, "fig_risk_coverage.png")
    if "rc_coverage" in fig:
        _try("risk_coverage", _rc)

    def _loss():
        f, ax = plt.subplots(figsize=(4, 3))
        ax.plot(np.arange(1, len(fig["coca_loss"]) + 1), fig["coca_loss"],
                color=PALETTE[0], marker="o", ms=3)
        ax.set_xlabel("epoch"); ax.set_ylabel("training loss")
        ax.set_title("ACC-HD training convergence"); ax.grid(alpha=0.3)
        _save(f, "fig_loss_convergence.png")
    if "coca_loss" in fig:
        _try("loss", _loss)

    def _rel():
        xs, ys, _ = reliability_curve(fig["coca_probs"], fig["y_test"])
        f, ax = plt.subplots(figsize=(4, 4))
        ax.plot([0, 1], [0, 1], "--", color="gray", label="perfect")
        ax.plot(xs, ys, marker="o", color=PALETTE[2], label="ACC-HD")
        ax.set_xlabel("confidence"); ax.set_ylabel("accuracy")
        ax.set_title("Reliability diagram"); ax.legend(); ax.grid(alpha=0.3)
        _save(f, "fig_reliability.png")
    if "coca_probs" in fig:
        _try("reliability", _rel)

    def _ece_bar():
        f, ax = plt.subplots(figsize=(6, 3))
        ax.bar(df["method"], df["ece"], color=PALETTE[:len(df)])
        ax.set_ylabel("ECE (lower better)"); ax.set_title("Calibration error by method")
        plt.xticks(rotation=40, ha="right"); _save(f, "fig_ece_bar.png")
    _try("ece_bar", _ece_bar)

    def _acc_bar():
        f, ax = plt.subplots(figsize=(6, 3))
        ax.bar(df["method"], df["accuracy"] * 100, color=PALETTE[:len(df)])
        ax.set_ylabel("accuracy (%)"); ax.set_title("Prediction accuracy by method")
        plt.xticks(rotation=40, ha="right"); _save(f, "fig_accuracy_bar.png")
    _try("accuracy_bar", _acc_bar)

    if ho is not None:
        _try("ho_kpis", lambda: _ho_kpis(ho))
        _try("radar", lambda: _radar(df, ho))

    print("All figures written to", FIG_DIR)


def _topology():
    from simulator.network import build_cell_positions
    pos = build_cell_positions()
    f, ax = plt.subplots(figsize=(4, 4))
    ax.scatter(pos[:, 0], pos[:, 1], c=PALETTE[0], s=60, marker="^", label="cell (gNB)")
    for i, (x, y) in enumerate(pos):
        ax.text(x, y + 15, str(i), ha="center", fontsize=6)
    # a simple example UE path across the cluster
    t = np.linspace(0, 1, 50)
    px = pos[:, 0].min() + t * (pos[:, 0].max() - pos[:, 0].min())
    py = pos[:, 1].mean() + 120 * np.sin(2 * np.pi * t)
    ax.plot(px, py, color=PALETTE[3], lw=1.2, ls="--", label="example UE path")
    ax.set_aspect("equal"); ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    ax.set_title("3GPP UMi 19-cell hexagonal layout (ISD 200 m)")
    ax.legend(fontsize=7)
    _save(f, "fig_topology.png")


def _ho_kpis(ho):
    cols = [c for c in ["hof_rate", "ping_pong_rate", "rlf_per_ue", "mean_hit_ms"]
            if c in ho.columns]
    if not cols:
        return
    mat = ho[cols].astype(float).values
    # normalise each column to [0,1] for a comparable heatmap
    norm = mat.copy()
    for j in range(mat.shape[1]):
        c = mat[:, j]; rng = (np.nanmax(c) - np.nanmin(c)) or 1.0
        norm[:, j] = (c - np.nanmin(c)) / rng
    f, ax = plt.subplots(figsize=(5.5, 0.5 * len(ho) + 1))
    im = ax.imshow(norm, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols, rotation=30, ha="right")
    ax.set_yticks(range(len(ho))); ax.set_yticklabels(ho["method"], fontsize=7)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", color="white", fontsize=6)
    f.colorbar(im, ax=ax, fraction=0.046)
    ax.set_title("Standard handover KPIs (raw values; colour = normalised)")
    _save(f, "fig_ho_kpis.png")


def _radar(df, ho):
    keep = ["ACC-HD (proposed)", "B1: Deep hybrid", "B2: DDQN-RL"]
    m = df.merge(ho, on="method", how="left", suffixes=("", "_ho"))
    sub = m[m["method"].isin(keep)].copy()
    # metrics: higher better after inversion of the "lower-better" ones
    spec = [("accuracy", False), ("macro_f1", False), ("ece", True),
            ("hof_rate", True), ("ping_pong_rate", True), ("mean_hit_ms", True)]
    cols = [c for c, _ in spec if c in sub.columns]
    spec = [(c, inv) for c, inv in spec if c in sub.columns]
    if sub.empty or len(cols) < 3:
        return
    norm = sub.copy()
    for c, inv in spec:
        col = sub[c].astype(float); rng = (col.max() - col.min()) or 1.0
        v = (col - col.min()) / rng
        norm[c] = (1 - v) if inv else v
    labels = [c.replace("_", " ") for c in cols]
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist(); angles += angles[:1]
    f, ax = plt.subplots(figsize=(4, 4), subplot_kw=dict(polar=True))
    for i, (_, r) in enumerate(norm.iterrows()):
        vals = [r[c] for c in cols] + [r[cols[0]]]
        ax.plot(angles, vals, color=PALETTE[i], label=r["method"])
        ax.fill(angles, vals, color=PALETTE[i], alpha=0.1)
    ax.set_xticks(angles[:-1]); ax.set_xticklabels(labels, fontsize=7)
    ax.set_yticklabels([]); ax.set_title("Method comparison (radar)")
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=7)
    _save(f, "fig_radar.png")


if __name__ == "__main__":
    make_all_figures()
