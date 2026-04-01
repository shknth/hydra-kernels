"""
Download all 112 UCR datasets used in the Hydra paper.

Dataset names are read directly from the paper's results CSV so there is no
hardcoded list.  Each dataset is downloaded via aeon, verified with a SHA-256
checksum, and saved under ../data/.  Failed downloads are written to
../data/failed_downloads.txt for manual follow-up.

Usage (from repo root, inside the venv):
    python reproduction/download_datasets.py

Colab usage:
    BASE_DIR = "/content/drive/MyDrive/hydra"
    Change DATA_DIR below to f"{BASE_DIR}/data" before running.
"""

import hashlib
import json
import os
import time
import sys

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths — adjust BASE_DIR for Colab if needed
# ---------------------------------------------------------------------------
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(REPO_ROOT, "data")
RESULTS_CSV = os.path.join(REPO_ROOT, "results", "results_ucr112_hydra.csv")
CHECKSUMS_PATH = os.path.join(DATA_DIR, "checksums.json")
FAILED_LOG = os.path.join(DATA_DIR, "failed_downloads.txt")

os.makedirs(DATA_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# aeon import — validate API at startup
# ---------------------------------------------------------------------------
try:
    from aeon.datasets import load_classification
except ImportError as e:
    print(f"[ERROR] aeon not installed or import failed: {e}")
    print("Run: uv pip install -r reproduction/requirements.txt")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def sha256_array(arr: np.ndarray) -> str:
    """Return hex SHA-256 of a numpy array's raw bytes."""
    return hashlib.sha256(arr.tobytes()).hexdigest()


def load_checksums() -> dict:
    if os.path.exists(CHECKSUMS_PATH):
        with open(CHECKSUMS_PATH) as f:
            return json.load(f)
    return {}


def save_checksums(cs: dict):
    with open(CHECKSUMS_PATH, "w") as f:
        json.dump(cs, f, indent=2)


def log_failure(name: str, reason: str):
    with open(FAILED_LOG, "a") as f:
        f.write(f"{name}\t{reason}\n")
    print(f"  [FAIL] {name}: {reason}")


def download_with_retry(name: str, retries: int = 3, delay: int = 5):
    """
    Download train and test splits for a UCR dataset via aeon.
    Returns (X_train, y_train, X_test, y_test) as numpy arrays, or None on failure.

    aeon returns X with shape (n_samples, n_channels, n_timepoints).
    For univariate datasets n_channels == 1.
    """
    for attempt in range(1, retries + 1):
        try:
            X_tr, y_tr = load_classification(name, split="train",
                                             extract_path=DATA_DIR)
            X_te, y_te = load_classification(name, split="test",
                                             extract_path=DATA_DIR)

            # Shape guard: ensure 3D (n, channels, length)
            if X_tr.ndim == 2:
                X_tr = X_tr[:, np.newaxis, :]
                X_te = X_te[:, np.newaxis, :]
            assert X_tr.ndim == 3, f"Unexpected shape {X_tr.shape}"

            return X_tr, y_tr, X_te, y_te

        except Exception as e:
            print(f"  [attempt {attempt}/{retries}] {name}: {e}")
            if attempt < retries:
                time.sleep(delay)

    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # Read the exact 112 dataset names from the paper's results file
    df = pd.read_csv(RESULTS_CSV)
    datasets = df["dataset"].tolist()
    print(f"Found {len(datasets)} datasets in results CSV.\n")

    checksums = load_checksums()
    failed = []

    for i, name in enumerate(datasets, 1):
        print(f"[{i:3d}/{len(datasets)}] {name}", end=" ... ", flush=True)

        # Skip if already verified
        if name in checksums:
            print("already downloaded (checksum ok)")
            continue

        result = download_with_retry(name)
        if result is None:
            log_failure(name, "download failed after retries")
            failed.append(name)
            continue

        X_tr, y_tr, X_te, y_te = result

        # Store checksum over concatenated train+test data
        combined = np.concatenate([X_tr, X_te], axis=0)
        checksums[name] = {
            "sha256": sha256_array(combined),
            "train_shape": list(X_tr.shape),
            "test_shape": list(X_te.shape),
            "n_classes": int(len(np.unique(y_tr))),
        }
        save_checksums(checksums)
        print(f"ok  train={X_tr.shape}  test={X_te.shape}")

    # Summary
    print(f"\n--- Download complete ---")
    print(f"  Succeeded : {len(datasets) - len(failed)}/{len(datasets)}")
    if failed:
        print(f"  Failed    : {len(failed)} — see {FAILED_LOG}")
        for name in failed:
            print(f"    - {name}")
    else:
        print("  All datasets downloaded successfully.")


if __name__ == "__main__":
    main()
