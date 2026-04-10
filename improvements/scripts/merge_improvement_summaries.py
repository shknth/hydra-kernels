"""
Merge per-track analysis outputs into one master summary table.

Expected inputs (if present):
- improvements/analysis/track_a/track_a_summary.csv
- improvements/analysis/track_b/variant_summary.csv
- improvements/analysis/track_c/track_c_summary.csv

Output:
- improvements/analysis/improvements_master_summary.csv

Usage:
    python improvements/scripts/merge_improvement_summaries.py
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = REPO_ROOT / "improvements" / "configs" / "experiment_manifest.json"


def load_manifest(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_optional_csv(path: Path, track_name: str) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["track", "metric", "value", "context"])

    df = pd.read_csv(path)
    if df.empty:
        return pd.DataFrame(columns=["track", "metric", "value", "context"])

    rows = []
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    for col in numeric_cols:
        rows.append(
            {
                "track": track_name,
                "metric": f"mean_{col}",
                "value": float(df[col].mean()),
                "context": path.as_posix(),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    manifest = load_manifest(MANIFEST_PATH)

    master_rel = manifest["output_policy"]["master_summary"]
    master_path = REPO_ROOT / master_rel
    master_path.parent.mkdir(parents=True, exist_ok=True)

    track_a_path = REPO_ROOT / "improvements" / "analysis" / "track_a" / "track_a_summary.csv"
    track_b_path = REPO_ROOT / "improvements" / "analysis" / "track_b" / "variant_summary.csv"
    track_c_path = REPO_ROOT / "improvements" / "analysis" / "track_c" / "track_c_summary.csv"

    merged = pd.concat(
        [
            load_optional_csv(track_a_path, "track_a_hyperparam_sensitivity"),
            load_optional_csv(track_b_path, "track_b_variant_analysis"),
            load_optional_csv(track_c_path, "track_c_timing_quality"),
        ],
        ignore_index=True,
    )

    merged.to_csv(master_path, index=False, float_format="%.6f")
    print(f"Saved merged summary: {master_path}")
    print(f"Rows: {len(merged)}")


if __name__ == "__main__":
    main()
