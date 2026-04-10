"""
Track A: Hydra hyperparameter sensitivity (k, g).

This script runs a small pilot by default using the dataset subset from
improvements/configs/experiment_manifest.json.

Outputs:
- improvements/outputs/track_a/<dataset>_k<k>_g<g>_r<resample>.json
- improvements/analysis/track_a/track_a_summary.csv

Usage:
    python improvements/scripts/track_a_hyperparam_sensitivity.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from sklearn.linear_model import RidgeClassifierCV
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"

sys.path.insert(0, str(REPO_ROOT / "code"))
from hydra import Hydra, SparseScaler  # noqa: E402


def load_manifest() -> dict:
    manifest_path = REPO_ROOT / "improvements" / "configs" / "experiment_manifest.json"
    with manifest_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_dataset(name: str):
    from aeon.datasets import load_classification

    X_tr, y_tr = load_classification(name, split="train", extract_path=str(DATA_DIR))
    X_te, y_te = load_classification(name, split="test", extract_path=str(DATA_DIR))

    if X_tr.ndim == 2:
        X_tr = X_tr[:, np.newaxis, :]
        X_te = X_te[:, np.newaxis, :]

    le = LabelEncoder()
    y_tr = le.fit_transform(y_tr)
    y_te = le.transform(y_te)

    return X_tr.astype(np.float32), y_tr, X_te.astype(np.float32), y_te


def get_resample(X_train, y_train, X_test, y_test, resample_idx: int):
    if resample_idx == 0:
        return X_train, y_train, X_test, y_test

    X_all = np.concatenate([X_train, X_test], axis=0)
    y_all = np.concatenate([y_train, y_test], axis=0)
    train_ratio = len(X_train) / len(X_all)

    sss = StratifiedShuffleSplit(n_splits=1, train_size=train_ratio, random_state=resample_idx)
    train_idx, test_idx = next(sss.split(X_all, y_all))
    return X_all[train_idx], y_all[train_idx], X_all[test_idx], y_all[test_idx]


def run_once(X_train, y_train, X_test, y_test, k: int, g: int):
    X_tr_t = torch.FloatTensor(X_train)
    X_te_t = torch.FloatTensor(X_test)

    t0 = time.perf_counter()
    transform = Hydra(X_tr_t.shape[-1], k=k, g=g)
    X_tr_h = transform.batch(X_tr_t)
    X_te_h = transform.batch(X_te_t)
    transform_time = time.perf_counter() - t0

    scaler = SparseScaler()
    X_tr_h = scaler.fit_transform(X_tr_h)
    X_te_h = scaler.transform(X_te_h)

    # Keep CV behavior consistent with reproduction runner.
    cv = 5 if X_tr_h.shape[0] > 2000 else None

    t1 = time.perf_counter()
    clf = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10), cv=cv)
    clf.fit(X_tr_h.numpy(), y_train)
    fit_time = time.perf_counter() - t1

    t2 = time.perf_counter()
    acc = clf.score(X_te_h.numpy(), y_test)
    predict_time = time.perf_counter() - t2

    return float(acc), float(transform_time), float(fit_time), float(predict_time), ("cv5" if cv else "loo")


def main():
    manifest = load_manifest()
    cfg = manifest["tracks"]["track_a_hyperparam_sensitivity"]

    datasets = manifest["dataset_subsets"]["local_pilot"][: cfg["max_local_datasets"]]
    k_values = cfg["k_values"]
    g_values = cfg["g_values"]
    resamples = int(cfg["resamples"])

    raw_dir = REPO_ROOT / "improvements" / "outputs" / "track_a"
    analysis_dir = REPO_ROOT / "improvements" / "analysis" / "track_a"
    raw_dir.mkdir(parents=True, exist_ok=True)
    analysis_dir.mkdir(parents=True, exist_ok=True)

    rows = []

    for dataset in datasets:
        X_tr0, y_tr0, X_te0, y_te0 = load_dataset(dataset)

        for r in range(resamples):
            X_tr, y_tr, X_te, y_te = get_resample(X_tr0, y_tr0, X_te0, y_te0, r)

            for k in k_values:
                for g in g_values:
                    acc, t_transform, t_fit, t_predict, cv_mode = run_once(
                        X_tr, y_tr, X_te, y_te, k, g
                    )

                    record = {
                        "dataset": dataset,
                        "resample": r,
                        "k": k,
                        "g": g,
                        "accuracy": round(acc, 6),
                        "transform_time_sec": round(t_transform, 4),
                        "fit_time_sec": round(t_fit, 4),
                        "predict_time_sec": round(t_predict, 4),
                        "cv_mode": cv_mode,
                    }
                    rows.append(record)

                    out_path = raw_dir / f"{dataset}_k{k}_g{g}_r{r}.json"
                    with out_path.open("w", encoding="utf-8") as f:
                        json.dump(record, f, indent=2)

                    print(
                        f"{dataset} r={r} k={k} g={g} "
                        f"acc={acc:.4f} t={t_transform+t_fit+t_predict:.2f}s"
                    )

    df = pd.DataFrame(rows)
    summary = (
        df.groupby(["dataset", "k", "g"], as_index=False)
        .agg(
            mean_accuracy=("accuracy", "mean"),
            std_accuracy=("accuracy", "std"),
            mean_transform_time_sec=("transform_time_sec", "mean"),
            mean_fit_time_sec=("fit_time_sec", "mean"),
            mean_predict_time_sec=("predict_time_sec", "mean"),
        )
        .sort_values(["dataset", "mean_accuracy"], ascending=[True, False])
    )

    summary_path = analysis_dir / "track_a_summary.csv"
    summary.to_csv(summary_path, index=False, float_format="%.6f")
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    os.environ.setdefault("PYTHONWARNINGS", "ignore")
    main()
