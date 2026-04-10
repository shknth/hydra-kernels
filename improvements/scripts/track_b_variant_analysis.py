"""
Track B: Analyze Hydra variants against baseline Hydra.

Reads:
- results/results_ucr112_hydra.csv
- results/results_ucr112_variants.csv

Writes:
- improvements/analysis/track_b/variant_deltas.csv
- improvements/analysis/track_b/variant_summary.csv
- improvements/analysis/track_b/best_method_per_dataset.csv

Usage:
    python improvements/scripts/track_b_variant_analysis.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import pandas as pd


def load_manifest(manifest_path: Path) -> Dict:
    with manifest_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_baseline(hydra_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(hydra_csv)
    required = {"dataset", "accuracy"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in baseline CSV: {sorted(missing)}")
    return df[["dataset", "accuracy"]].rename(columns={"accuracy": "Hydra_baseline"})


def read_variants(variants_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(variants_csv)
    if "dataset" not in df.columns or "Hydra" not in df.columns:
        raise ValueError("Variants CSV must include columns: dataset, Hydra")
    return df


def ensure_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def build_variant_delta_table(merged: pd.DataFrame, variant_columns: List[str]) -> pd.DataFrame:
    out = merged[["dataset"]].copy()
    out["Hydra_baseline"] = merged["Hydra_baseline"]

    for col in variant_columns:
        out[col] = merged[col]
        out[f"delta_{col}"] = merged[col] - merged["Hydra_baseline"]

    return out


def summarize_variants(delta_df: pd.DataFrame, variant_columns: List[str]) -> pd.DataFrame:
    rows = []
    for col in variant_columns:
        delta_col = f"delta_{col}"
        series = delta_df[delta_col]

        wins = int((series > 0).sum())
        ties = int((series == 0).sum())
        losses = int((series < 0).sum())

        rows.append(
            {
                "variant": col,
                "mean_delta": float(series.mean()),
                "median_delta": float(series.median()),
                "std_delta": float(series.std(ddof=0)),
                "max_gain": float(series.max()),
                "max_drop": float(series.min()),
                "wins": wins,
                "ties": ties,
                "losses": losses,
                "win_rate": float(wins / len(series)),
            }
        )

    summary = pd.DataFrame(rows)
    return summary.sort_values("mean_delta", ascending=False)


def best_method_per_dataset(merged: pd.DataFrame, method_columns: List[str]) -> pd.DataFrame:
    best_col = merged[method_columns].idxmax(axis=1)
    best_val = merged[method_columns].max(axis=1)

    out = merged[["dataset", "Hydra_baseline"]].copy()
    out["best_method"] = best_col
    out["best_accuracy"] = best_val
    out["hydra_gap_to_best"] = out["best_accuracy"] - out["Hydra_baseline"]
    return out.sort_values("hydra_gap_to_best", ascending=False)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    manifest_path = repo_root / "improvements" / "configs" / "experiment_manifest.json"

    manifest = load_manifest(manifest_path)
    track_cfg = manifest["tracks"]["track_b_variant_analysis"]
    source_files = track_cfg["source_files"]

    hydra_csv = repo_root / source_files["hydra"]
    variants_csv = repo_root / source_files["variants"]

    baseline_df = read_baseline(hydra_csv)
    variants_df = read_variants(variants_csv)

    merged = baseline_df.merge(variants_df, on="dataset", how="inner", validate="one_to_one")

    variant_columns = [
        c
        for c in merged.columns
        if c not in {"dataset", "Hydra_baseline", "Hydra"}
    ]

    if not variant_columns:
        raise ValueError("No variant columns found in variants CSV.")

    analysis_dir = repo_root / "improvements" / "analysis" / "track_b"
    ensure_output_dir(analysis_dir)

    delta_df = build_variant_delta_table(merged, variant_columns)
    summary_df = summarize_variants(delta_df, variant_columns)
    best_df = best_method_per_dataset(merged, ["Hydra_baseline"] + variant_columns)

    delta_path = analysis_dir / "variant_deltas.csv"
    summary_path = analysis_dir / "variant_summary.csv"
    best_path = analysis_dir / "best_method_per_dataset.csv"

    delta_df.to_csv(delta_path, index=False, float_format="%.6f")
    summary_df.to_csv(summary_path, index=False, float_format="%.6f")
    best_df.to_csv(best_path, index=False, float_format="%.6f")

    print("Track B analysis complete.")
    print(f"Merged datasets: {len(merged)}")
    print(f"Variant columns: {', '.join(variant_columns)}")
    print(f"Saved: {delta_path}")
    print(f"Saved: {summary_path}")
    print(f"Saved: {best_path}")


if __name__ == "__main__":
    main()
