"""
Smoke test — run before the full reproduction to validate:
  1. aeon loads data correctly with expected shape
  2. Hydra transform runs without error
  3. RidgeClassifierCV fits and scores
  4. Resampling produces consistent results

Tests 3 datasets: one small (GunPoint), one medium (Adiac),
one large (ElectricDevices — triggers the cv=5 guard).

Usage:
    python reproduction/smoke_test.py
"""

import os
import sys
import time

import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR  = os.path.join(REPO_ROOT, "data")
sys.path.insert(0, os.path.join(REPO_ROOT, "code"))

SMOKE_DATASETS = ["GunPoint", "Adiac", "ElectricDevices"]
LARGE_THRESHOLD = 2000

PASS = "[PASS]"
FAIL = "[FAIL]"
WARN = "[WARN]"

errors = []


def check(condition: bool, label: str, detail: str = ""):
    sym = PASS if condition else FAIL
    print(f"  {sym} {label}" + (f" — {detail}" if detail else ""))
    if not condition:
        errors.append(label)


def run_smoke(name: str):
    print(f"\n{'='*60}")
    print(f"Dataset: {name}")
    print(f"{'='*60}")

    # --- 1. aeon import ---
    try:
        from aeon.datasets import load_classification
        check(True, "aeon import")
    except ImportError as e:
        check(False, "aeon import", str(e))
        return

    # --- 2. Data download / load ---
    try:
        t0 = time.perf_counter()
        X_tr, y_tr = load_classification(name, split="train", extract_path=DATA_DIR)
        X_te, y_te = load_classification(name, split="test",  extract_path=DATA_DIR)
        elapsed = time.perf_counter() - t0
        check(True, "load_classification", f"{elapsed:.1f}s")
    except Exception as e:
        check(False, "load_classification", str(e))
        return

    # --- 3. Shape check ---
    check(X_tr.ndim in (2, 3), "X_train ndim", f"got {X_tr.ndim}")
    if X_tr.ndim == 2:
        X_tr = X_tr[:, np.newaxis, :]
        X_te = X_te[:, np.newaxis, :]
        print(f"  {WARN} 2D array reshaped to 3D — note in notes.md")
    check(X_tr.ndim == 3, "X_train is 3D after normalisation", str(X_tr.shape))
    check(X_te.ndim == 3, "X_test  is 3D after normalisation", str(X_te.shape))

    n_train, n_ch, length = X_tr.shape
    print(f"       train={X_tr.shape}  test={X_te.shape}  classes={len(np.unique(y_tr))}")

    # --- 4. Label encoding ---
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    y_tr_enc = le.fit_transform(y_tr)
    y_te_enc = le.transform(y_te)
    check(y_tr_enc.min() == 0, "labels start at 0")

    # --- 5. Hydra transform ---
    try:
        from hydra import Hydra, SparseScaler
        check(True, "hydra.py import")
    except ImportError as e:
        check(False, "hydra.py import", str(e))
        return

    X_tr_t = torch.FloatTensor(X_tr.astype(np.float32))
    X_te_t = torch.FloatTensor(X_te.astype(np.float32))

    try:
        transform = Hydra(length)
        t0 = time.perf_counter()
        X_tr_h = transform.batch(X_tr_t)
        X_te_h = transform.batch(X_te_t)
        elapsed = time.perf_counter() - t0
        check(True, "Hydra.batch()", f"{elapsed:.2f}s  features={X_tr_h.shape[1]}")
    except Exception as e:
        check(False, "Hydra.batch()", str(e))
        return

    # --- 6. SparseScaler ---
    try:
        scaler  = SparseScaler()
        X_tr_h  = scaler.fit_transform(X_tr_h)
        X_te_h  = scaler.transform(X_te_h)
        check(True, "SparseScaler")
    except Exception as e:
        check(False, "SparseScaler", str(e))
        return

    # --- 7. RidgeClassifierCV ---
    from sklearn.linear_model import RidgeClassifierCV
    cv = 5 if n_train > LARGE_THRESHOLD else None
    cv_label = f"cv=5 (large dataset guard)" if cv else "cv=LOO"
    try:
        t0  = time.perf_counter()
        clf = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10), cv=cv)
        clf.fit(X_tr_h.numpy(), y_tr_enc)
        acc = clf.score(X_te_h.numpy(), y_te_enc)
        elapsed = time.perf_counter() - t0
        check(True, f"RidgeClassifierCV ({cv_label})",
              f"acc={acc:.4f}  time={elapsed:.2f}s")
    except Exception as e:
        check(False, f"RidgeClassifierCV ({cv_label})", str(e))
        return

    # --- 8. Resampling sanity ---
    from sklearn.model_selection import StratifiedShuffleSplit
    X_all = np.concatenate([X_tr, X_te], axis=0)
    y_all = np.concatenate([y_tr_enc, y_te_enc], axis=0)
    train_ratio = n_train / len(X_all)
    sss = StratifiedShuffleSplit(n_splits=1, train_size=train_ratio, random_state=1)
    tr_idx, te_idx = next(sss.split(X_all, y_all))
    expected_train = round(train_ratio * len(X_all))
    check(
        abs(len(tr_idx) - expected_train) <= 1,
        "Resampling preserves train ratio",
        f"got {len(tr_idx)} train (expected ~{expected_train})"
    )


def main():
    print("Hydra Smoke Test")
    print(f"DATA_DIR : {DATA_DIR}\n")

    for ds in SMOKE_DATASETS:
        run_smoke(ds)

    print(f"\n{'='*60}")
    if errors:
        print(f"SMOKE TEST FAILED — {len(errors)} issue(s):")
        for e in errors:
            print(f"  - {e}")
        print("\nFix these before running the full reproduction.")
        sys.exit(1)
    else:
        print("SMOKE TEST PASSED — safe to run download_datasets.py and run_reproduction.py")


if __name__ == "__main__":
    main()
