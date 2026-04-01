"""
Checkpointed reproduction runner for the Hydra paper.

Runs Hydra on all 112 UCR datasets × 30 resamples.
Each (dataset, resample) result is saved as a JSON file under
../reproduction_results/<dataset>/resample_<r>.json immediately after
completion, so the run can be interrupted and resumed at any time.

Usage (from repo root, inside the venv):
    python reproduction/run_reproduction.py

Colab usage:
    Set BASE_DIR = "/content/drive/MyDrive/hydra" below so outputs persist
    across session resets.

Resample convention (matches TSC community standard):
    resample 0  = default train/test split from UCR archive
    resamples 1-29 = stratified resample preserving original train/test ratio,
                     seeded by resample index for reproducibility
"""

import csv
import json
import os
import sys
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import torch

# ---------------------------------------------------------------------------
# Paths — change BASE_DIR here for Colab
# ---------------------------------------------------------------------------
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE_DIR = REPO_ROOT                                   # Colab: "/content/drive/MyDrive/hydra"
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "reproduction_results")
RESULTS_CSV = os.path.join(REPO_ROOT, "results", "results_ucr112_hydra.csv")
RUN_LOG = os.path.join(OUTPUT_DIR, "run_log.csv")
ERROR_LOG = os.path.join(OUTPUT_DIR, "errors.log")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Add original code to path
sys.path.insert(0, os.path.join(REPO_ROOT, "code"))

# ---------------------------------------------------------------------------
# Imports — fail fast with clear messages
# ---------------------------------------------------------------------------
try:
    from hydra import Hydra, SparseScaler
except ImportError:
    print("[ERROR] Cannot import hydra.py — ensure code/ is in the repo root.")
    sys.exit(1)

try:
    from aeon.datasets import load_classification
except ImportError:
    print("[ERROR] aeon not installed. Run: uv pip install -r reproduction/requirements.txt")
    sys.exit(1)

try:
    from sklearn.linear_model import RidgeClassifierCV
    from sklearn.model_selection import StratifiedShuffleSplit
    from sklearn.preprocessing import LabelEncoder
except ImportError:
    print("[ERROR] scikit-learn not installed.")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
NUM_RESAMPLES = 30
LARGE_DATASET_THRESHOLD = 2000   # use cv=5 instead of LOO above this
RUN_LOG_FIELDS = ["dataset", "resample", "accuracy",
                  "train_time_sec", "test_time_sec", "timestamp"]

# ---------------------------------------------------------------------------
# Resampling
# ---------------------------------------------------------------------------

def get_resample(X_train, y_train, X_test, y_test, resample_idx: int):
    """
    resample_idx == 0 : return the default split unchanged
    resample_idx  > 0 : stratified resample seeded by resample_idx,
                        preserving the original train/test ratio
    """
    if resample_idx == 0:
        return X_train, y_train, X_test, y_test

    X_all = np.concatenate([X_train, X_test], axis=0)
    y_all = np.concatenate([y_train, y_test], axis=0)
    train_ratio = len(X_train) / len(X_all)

    sss = StratifiedShuffleSplit(
        n_splits=1,
        train_size=train_ratio,
        random_state=resample_idx,
    )
    train_idx, test_idx = next(sss.split(X_all, y_all))
    return X_all[train_idx], y_all[train_idx], X_all[test_idx], y_all[test_idx]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_dataset(name: str):
    """
    Load train and test splits from aeon.
    Returns (X_train, y_train, X_test, y_test) as numpy float32 arrays.
    X shape: (n_samples, 1, n_timepoints)
    """
    X_tr, y_tr = load_classification(name, split="train", extract_path=DATA_DIR)
    X_te, y_te = load_classification(name, split="test",  extract_path=DATA_DIR)

    # Shape normalisation: aeon may return (n, 1, L) or (n, L)
    if X_tr.ndim == 2:
        X_tr = X_tr[:, np.newaxis, :]
        X_te = X_te[:, np.newaxis, :]

    # Encode string labels to integers
    le = LabelEncoder()
    y_tr = le.fit_transform(y_tr)
    y_te = le.transform(y_te)

    return (X_tr.astype(np.float32),
            y_tr,
            X_te.astype(np.float32),
            y_te)


# ---------------------------------------------------------------------------
# Hydra pipeline
# ---------------------------------------------------------------------------

def run_hydra(X_train: np.ndarray, y_train: np.ndarray,
              X_test: np.ndarray,  y_test: np.ndarray,
              dataset_name: str):
    """
    Run the standard Hydra pipeline (transform → scale → RidgeCV).
    Returns (accuracy, train_time_sec, test_time_sec).
    """
    X_tr_t = torch.FloatTensor(X_train)   # (N, 1, L)
    X_te_t = torch.FloatTensor(X_test)

    transform = Hydra(X_tr_t.shape[-1])

    # Always use .batch() to avoid OOM on large series
    t0 = time.perf_counter()
    X_tr_h = transform.batch(X_tr_t)
    X_te_h = transform.batch(X_te_t)

    scaler  = SparseScaler()
    X_tr_h  = scaler.fit_transform(X_tr_h)
    X_te_h  = scaler.transform(X_te_h)

    # Large-dataset guard: LOO-CV is O(n²) — use k-fold above threshold
    n_train = X_tr_h.shape[0]
    if n_train > LARGE_DATASET_THRESHOLD:
        clf = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10), cv=5)
    else:
        clf = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10))

    clf.fit(X_tr_h.numpy(), y_train)
    train_time = time.perf_counter() - t0

    t1 = time.perf_counter()
    acc = clf.score(X_te_h.numpy(), y_test)
    test_time = time.perf_counter() - t1

    return float(acc), train_time, test_time


# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------

def log_error(dataset: str, resample: int, msg: str):
    with open(ERROR_LOG, "a") as f:
        ts = datetime.now(timezone.utc).isoformat()
        f.write(f"{ts}\t{dataset}\tresample_{resample}\t{msg}\n")
    print(f"  [ERROR] {dataset} resample {resample}: {msg}")


def append_run_log(row: dict):
    write_header = not os.path.exists(RUN_LOG)
    with open(RUN_LOG, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RUN_LOG_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


# ---------------------------------------------------------------------------
# Progress helpers
# ---------------------------------------------------------------------------

def count_completed(dataset: str) -> int:
    d = os.path.join(OUTPUT_DIR, dataset)
    if not os.path.isdir(d):
        return 0
    return sum(1 for f in os.listdir(d) if f.startswith("resample_") and f.endswith(".json"))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    df = pd.read_csv(RESULTS_CSV)
    datasets = df["dataset"].tolist()
    total = len(datasets)

    print(f"Hydra Reproduction — {total} datasets × {NUM_RESAMPLES} resamples")
    print(f"Output dir : {OUTPUT_DIR}")
    print(f"Resume     : completed runs are skipped automatically\n")

    # Count already-done work for progress display
    total_done_before = sum(count_completed(d) for d in datasets)
    total_jobs = total * NUM_RESAMPLES
    print(f"Progress   : {total_done_before}/{total_jobs} already complete\n")

    for i, dataset in enumerate(datasets, 1):
        # Quick check: if all 30 resamples exist, skip the dataset entirely
        if count_completed(dataset) == NUM_RESAMPLES:
            print(f"[{i:3d}/{total}] {dataset:<40s} all 30 resamples done — skip")
            continue

        # Load dataset once per dataset (not per resample)
        try:
            X_tr0, y_tr0, X_te0, y_te0 = load_dataset(dataset)
        except Exception as e:
            log_error(dataset, -1, f"load failed: {e}")
            continue

        dataset_dir = os.path.join(OUTPUT_DIR, dataset)
        os.makedirs(dataset_dir, exist_ok=True)

        for r in range(NUM_RESAMPLES):
            result_path = os.path.join(dataset_dir, f"resample_{r}.json")
            if os.path.exists(result_path):
                continue   # already done — resume

            print(f"[{i:3d}/{total}] {dataset:<40s} resample {r:2d}/29",
                  end=" ... ", flush=True)

            try:
                X_tr, y_tr, X_te, y_te = get_resample(
                    X_tr0, y_tr0, X_te0, y_te0, r
                )
                acc, t_train, t_test = run_hydra(X_tr, y_tr, X_te, y_te, dataset)
            except Exception as e:
                log_error(dataset, r, str(e))
                print("ERROR — see errors.log")
                continue

            result = {
                "dataset":        dataset,
                "resample":       r,
                "accuracy":       round(acc, 6),
                "train_time_sec": round(t_train, 4),
                "test_time_sec":  round(t_test, 4),
                "timestamp":      datetime.now(timezone.utc).isoformat(),
                "n_train":        int(X_tr.shape[0]),
                "cv_mode":        "cv5" if X_tr.shape[0] > LARGE_DATASET_THRESHOLD else "loo",
            }

            with open(result_path, "w") as f:
                json.dump(result, f, indent=2)

            append_run_log({k: result[k] for k in RUN_LOG_FIELDS})

            print(f"acc={acc:.4f}  train={t_train:.1f}s  test={t_test:.1f}s")

    print("\n--- All runs complete ---")
    total_done_after = sum(count_completed(d) for d in datasets)
    print(f"Total completed: {total_done_after}/{total_jobs}")
    if total_done_after < total_jobs:
        print(f"Incomplete: {total_jobs - total_done_after} runs — check {ERROR_LOG}")


if __name__ == "__main__":
    main()
