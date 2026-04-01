"""
Compare our reproduction results against the paper's published results.

Aggregates all resample_*.json files under ../reproduction_results/ into
per-dataset mean accuracy, then diffs against results_ucr112_hydra.csv.

Outputs:
    ../reproduction_results/comparison_summary.csv

Usage (from repo root, inside the venv):
    python reproduction/compare_results.py
"""

import json
import os
import sys

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR  = os.path.join(REPO_ROOT, "reproduction_results")
PAPER_CSV   = os.path.join(REPO_ROOT, "results", "results_ucr112_hydra.csv")
SUMMARY_CSV = os.path.join(OUTPUT_DIR, "comparison_summary.csv")

DELTA_THRESHOLD = 0.01   # flag datasets where |our_acc - paper_acc| > this


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate_results() -> pd.DataFrame:
    """
    Walk reproduction_results/<dataset>/resample_*.json and compute per-dataset
    mean accuracy and resample count.
    """
    rows = []
    if not os.path.isdir(OUTPUT_DIR):
        return pd.DataFrame()

    for dataset in sorted(os.listdir(OUTPUT_DIR)):
        dataset_dir = os.path.join(OUTPUT_DIR, dataset)
        if not os.path.isdir(dataset_dir):
            continue

        accuracies = []
        cv_modes   = []
        for fname in sorted(os.listdir(dataset_dir)):
            if not (fname.startswith("resample_") and fname.endswith(".json")):
                continue
            with open(os.path.join(dataset_dir, fname)) as f:
                data = json.load(f)
            accuracies.append(data["accuracy"])
            cv_modes.append(data.get("cv_mode", "unknown"))

        if not accuracies:
            continue

        rows.append({
            "dataset":         dataset,
            "our_acc":         float(np.mean(accuracies)),
            "our_acc_std":     float(np.std(accuracies)),
            "resamples_done":  len(accuracies),
            "cv_mode":         "mixed" if len(set(cv_modes)) > 1 else cv_modes[0],
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # Load paper results
    paper = pd.read_csv(PAPER_CSV)[["dataset", "accuracy"]].rename(
        columns={"accuracy": "paper_acc"}
    )

    # Load our results
    ours = aggregate_results()
    if ours.empty:
        print("[WARN] No reproduction results found. Run run_reproduction.py first.")
        sys.exit(0)

    # Merge
    merged = paper.merge(ours, on="dataset", how="left")
    merged["delta"]     = merged["our_acc"] - merged["paper_acc"]
    merged["abs_delta"] = merged["delta"].abs()
    merged["flag"]      = merged["abs_delta"] > DELTA_THRESHOLD

    # Sort: biggest deviations first
    merged = merged.sort_values("abs_delta", ascending=False)

    # Save
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    merged.to_csv(SUMMARY_CSV, index=False, float_format="%.6f")

    # Print report
    done = merged["resamples_done"].notna().sum()
    flagged = merged["flag"].sum()

    print(f"=== Reproduction vs Paper ===")
    print(f"Datasets with results : {int(done)}/112")
    print(f"Mean |delta|          : {merged['abs_delta'].mean():.4f}")
    print(f"Max  |delta|          : {merged['abs_delta'].max():.4f}")
    print(f"Flagged (>1% delta)   : {int(flagged)}")
    print(f"\nSummary saved to: {SUMMARY_CSV}\n")

    if flagged > 0:
        print(f"--- Flagged datasets (|delta| > {DELTA_THRESHOLD}) ---")
        cols = ["dataset", "paper_acc", "our_acc", "delta", "resamples_done", "cv_mode"]
        print(merged[merged["flag"]][cols].to_string(index=False))
    else:
        print("All datasets within 1% of paper results.")

    # Datasets not yet run
    missing = merged[merged["resamples_done"].isna()]["dataset"].tolist()
    if missing:
        print(f"\n--- Not yet run ({len(missing)}) ---")
        for d in missing:
            print(f"  {d}")


if __name__ == "__main__":
    main()
