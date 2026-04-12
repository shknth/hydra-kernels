"""
Generate all 4 report figures from experiment results.

Outputs (saved to same directory as this script):
    fig1_reproduction_scatter.pdf  - Paper vs our accuracy scatter
    fig2_hyperparam_heatmap.pdf    - Track A k/g accuracy heatmap
    fig3_variant_bars.pdf          - Track B variant win-rate + delta bar chart
    fig4_timing_breakdown.pdf      - Track C stacked timing bar chart

Usage (from repo root, inside venv):
    python figures/generate_figures.py
"""

import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR   = Path(__file__).resolve().parent

# Consistent style
plt.rcParams.update({
    "font.family":     "serif",
    "font.size":       8,
    "axes.titlesize":  9,
    "axes.labelsize":  8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "figure.dpi":      150,
    "savefig.dpi":     300,
    "savefig.bbox":    "tight",
    "savefig.pad_inches": 0.05,
})

GREY   = "#555555"
BLUE   = "#2166AC"
RED    = "#D6604D"
GREEN  = "#4DAC26"
ORANGE = "#F4A582"


# ---------------------------------------------------------------------------
# Fig 1 — Reproduction scatter: paper acc vs our acc
# ---------------------------------------------------------------------------

def fig1_scatter():
    paper = pd.read_csv(REPO_ROOT / "results" / "results_ucr112_hydra.csv")[["dataset", "accuracy"]]
    paper = paper.rename(columns={"accuracy": "paper_acc"})

    log = pd.read_csv(REPO_ROOT / "reproduction_results" / "run_log.csv")
    ours = log.groupby("dataset")["accuracy"].mean().reset_index()
    ours = ours.rename(columns={"accuracy": "our_acc"})

    merged = paper.merge(ours, on="dataset")
    delta  = (merged["our_acc"] - merged["paper_acc"]).abs()

    fig, ax = plt.subplots(figsize=(3.4, 3.1))

    # Diagonal perfect-reproduction line
    lims = [0.28, 1.02]
    ax.plot(lims, lims, "--", color=GREY, linewidth=0.8, label="Perfect reproduction", zorder=1)

    # ±1% band
    ax.fill_between(lims, [l - 0.01 for l in lims], [l + 0.01 for l in lims],
                    color=BLUE, alpha=0.10, zorder=0, label="±1% band")

    flagged = delta > 0.01
    ax.scatter(merged.loc[~flagged, "paper_acc"], merged.loc[~flagged, "our_acc"],
               s=12, color=BLUE, alpha=0.65, linewidths=0, zorder=2,
               label=f"Within 1% ({(~flagged).sum()} datasets)")
    ax.scatter(merged.loc[flagged, "paper_acc"], merged.loc[flagged, "our_acc"],
               s=14, color=RED, alpha=0.80, linewidths=0, marker="^", zorder=3,
               label=f">1% delta ({flagged.sum()} datasets)")

    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("Published accuracy")
    ax.set_ylabel("Reproduced accuracy (mean / 30 resamples)")
    ax.set_title("Reproduction vs. Published Results (UCR-112)")
    ax.legend(loc="upper left", framealpha=0.9)

    # Annotation
    mean_d = delta.mean()
    ax.text(0.98, 0.04, f"Mean |Δ| = {mean_d:.4f}",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=7, color=GREY)

    fig.tight_layout()
    path = OUT_DIR / "fig1_reproduction_scatter.pdf"
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved: {path}")


# ---------------------------------------------------------------------------
# Fig 2 — Hyperparameter heatmap: mean acc by (k, g) averaged across datasets
# ---------------------------------------------------------------------------

def fig2_heatmap():
    df = pd.read_csv(REPO_ROOT / "improvements" / "analysis" / "track_a" / "track_a_summary.csv")

    pivot = df.groupby(["k", "g"])["mean_accuracy"].mean().reset_index()
    pivot = pivot.pivot(index="k", columns="g", values="mean_accuracy")
    pivot = pivot.sort_index(ascending=True)
    pivot = pivot.reindex(sorted(pivot.columns), axis=1)

    fig, ax = plt.subplots(figsize=(3.1, 2.4))

    im = ax.imshow(pivot.values, cmap="Blues", aspect="auto",
                   vmin=pivot.values.min() - 0.005,
                   vmax=pivot.values.max() + 0.005)

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"g={c}" for c in pivot.columns])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([f"k={r}" for r in pivot.index])

    # Annotate cells
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            ax.text(j, i, f"{val:.4f}", ha="center", va="center",
                    fontsize=7,
                    color="white" if val > (pivot.values.max() - 0.003) else "black")

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Mean accuracy", fontsize=7)

    ax.set_title("Track A: Accuracy by Hyperparameter (k, g)\nAveraged over 50 datasets")
    ax.set_xlabel("Number of groups (g)")
    ax.set_ylabel("Kernels per group (k)")

    fig.tight_layout()
    path = OUT_DIR / "fig2_hyperparam_heatmap.pdf"
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved: {path}")


# ---------------------------------------------------------------------------
# Fig 3 — Variant analysis: win rate + mean delta bar chart
# ---------------------------------------------------------------------------

def fig3_variants():
    df = pd.read_csv(REPO_ROOT / "improvements" / "analysis" / "track_b" / "variant_summary.csv")

    labels = {
        "Hydra+MiniRocket":  "H+MiniRocket",
        "Hydra+MultiRocket": "H+MultiRocket",
        "Hydra+Rocket":      "H+Rocket",
    }
    df["label"] = df["variant"].map(labels)
    df = df.sort_values("win_rate", ascending=True)

    colors = [BLUE, GREEN, ORANGE]
    x = np.arange(len(df))
    width = 0.35

    fig, ax1 = plt.subplots(figsize=(3.4, 2.6))
    ax2 = ax1.twinx()

    bars1 = ax1.barh(x - width/2, df["win_rate"] * 100, width,
                     color=BLUE, alpha=0.80, label="Win rate (%)")
    bars2 = ax2.barh(x + width/2, df["mean_delta"] * 100, width,
                     color=GREEN, alpha=0.80, label="Mean Δ acc (%)")

    ax1.set_yticks(x)
    ax1.set_yticklabels(df["label"].tolist())
    ax1.set_xlabel("Win rate (%)")
    ax2.set_xlabel("Mean Δ accuracy (pp)", color=GREEN)
    ax2.tick_params(axis="x", colors=GREEN)

    ax1.axvline(50, color=GREY, linestyle="--", linewidth=0.7)

    # Combined legend
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="lower right", framealpha=0.9)

    ax1.set_title("Track B: Hydra Variant Analysis (UCR-112)")

    fig.tight_layout()
    path = OUT_DIR / "fig3_variant_bars.pdf"
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved: {path}")


# ---------------------------------------------------------------------------
# Fig 4 — Timing breakdown: stacked bar per dataset (Track C)
# ---------------------------------------------------------------------------

def fig4_timing():
    df = pd.read_csv(REPO_ROOT / "improvements" / "analysis" / "track_c" / "track_c_summary.csv")
    df = df.sort_values("mean_total_time_sec", ascending=True)

    fig, ax = plt.subplots(figsize=(3.4, 2.8))

    x = np.arange(len(df))
    width = 0.6

    p1 = ax.bar(x, df["mean_transform_time_sec"], width, label="Transform", color=BLUE, alpha=0.85)
    p2 = ax.bar(x, df["mean_fit_time_sec"], width,
                bottom=df["mean_transform_time_sec"],
                label="Ridge fit", color=GREEN, alpha=0.85)
    p3 = ax.bar(x, df["mean_predict_time_sec"], width,
                bottom=df["mean_transform_time_sec"] + df["mean_fit_time_sec"],
                label="Predict", color=ORANGE, alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(df["dataset"].tolist(), rotation=40, ha="right")
    ax.set_ylabel("Time (seconds)")
    ax.set_title("Track C: Timing Breakdown per Dataset")
    ax.legend(loc="upper left", framealpha=0.9)

    fig.tight_layout()
    path = OUT_DIR / "fig4_timing_breakdown.pdf"
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved: {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Generating report figures...")
    fig1_scatter()
    fig2_heatmap()
    fig3_variants()
    fig4_timing()
    print("All figures saved.")


if __name__ == "__main__":
    main()
