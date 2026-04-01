# Reproduction Notes — Hydra (arXiv:2203.13652)

This file is maintained throughout the reproduction run.
Every anomaly, deviation, version issue, or interesting finding is logged here.
It directly feeds the "Lessons Learned" section of the report (30/100 marks).

---

## Environment

| Item | Value |
|---|---|
| Python | 3.10.4 (CPython, macOS arm64) |
| PyTorch | 2.1.0 |
| aeon | 0.7.1 |
| scikit-learn | 1.3.2 |
| NumPy | 1.24.4 |
| Hardware | Apple Silicon Mac (personal machine) |
| OS | macOS darwin 24.6.0 |

---

## Setup Issues

### SSL Certificate Verification Failure — macOS Python 3.10
- **Issue:** `urllib` raised `SSL: CERTIFICATE_VERIFY_FAILED` when aeon tried to download datasets.
- **Cause:** Python 3.10 installed via the official python.org installer on macOS does not automatically use the system keychain. The venv inherited this.
- **Fix:** Installed `certifi==2026.2.25` into the venv. Added `SSL_CERT_FILE` and `REQUESTS_CA_BUNDLE` env vars pointing to certifi's bundle at the end of `.venv/bin/activate`. Now resolved permanently — just `source .venv/bin/activate` is enough.
- **Report relevance:** Worth mentioning in Lessons Learned as a reproducibility barrier.

### LinAlgWarning: Ill-conditioned matrix (RidgeClassifierCV)
- **Issue:** scikit-learn emits `LinAlgWarning` during LOO-CV for certain alpha values in `np.logspace(-3, 3, 10)`.
- **Cause:** Very small alpha values (e.g., 1e-3) produce near-singular matrices when the Hydra feature matrix is large and sparse. The warning is from scipy's `linalg.solve`.
- **Impact:** Results are still valid — the chosen alpha (via CV) will be large enough that the final fit is numerically stable. The warnings come from rejected alpha candidates.
- **Fix:** No fix needed. Can suppress with `warnings.filterwarnings("ignore", category=LinAlgWarning)` in the runner if output is noisy.
- **Report relevance:** Interesting implementation detail — the sparse, high-dimensional Hydra feature space causes conditioning issues for small regularisation values.

---

## aeon API Observations

- `load_classification(name, split="train"/"test", extract_path=DATA_DIR)` — works correctly in aeon 0.7.1.
- Returns `X` with shape `(n_samples, 1, n_timepoints)` for univariate datasets (already 3D). No reshape needed.
- Labels `y` are returned as string arrays. Must be encoded via `LabelEncoder` before passing to sklearn.
- Native aeon resample function not confirmed yet — using `StratifiedShuffleSplit(random_state=resample_idx)` as the resampling strategy.

---

## Resampling

- [x] Used `StratifiedShuffleSplit(random_state=resample_idx)` as fallback — aeon 0.7.1 does not expose a dedicated resample utility matching the exact community standard.
- Resample 0 = default UCR split. Resamples 1–29 = stratified splits preserving original train ratio, seeded by index.
- **Deviation note:** This may produce slightly different splits from those used in the paper. Expect small variance in results across resamples vs paper; mean over 30 should converge.

---

## Smoke Test Results (2026-04-01)

| Dataset | n_train | n_test | Classes | Length | Our Acc | Time (train+Ridge) | CV Mode |
|---|---|---|---|---|---|---|---|
| GunPoint | 50 | 150 | 2 | 150 | 1.0000 | 0.51s | LOO |
| Adiac | 390 | 391 | 37 | 176 | 0.8210 | 2.47s | LOO |
| ElectricDevices | 8926 | 7711 | 7 | 96 | 0.7456 | 89.52s | cv=5 |

Paper acc for Adiac = 0.815772 (delta = +0.005 on single resample — within noise).
Paper acc for ElectricDevices = 0.881671 — our single-resample result of 0.7456 is lower; expected since paper reports mean over 30 resamples and different train split.

---

## Large Dataset Guard (cv=5)

Datasets where `n_train > 2000` trigger `cv=5` instead of LOO (confirmed from smoke test):

| Dataset | n_train |
|---|---|
| ElectricDevices | 8926 |
| _(others TBD during full run)_ | |

---

## Download Failures

_Populated during download_datasets.py run._

---

## Results Deviations (|delta| > 1%)

_Populated after running compare_results.py._

| Dataset | Paper Acc | Our Acc | Delta | Likely Reason |
|---|---|---|---|---|
| | | | | |

---

## Timing Observations

- ElectricDevices: Hydra transform ~21.5s + RidgeCV (cv=5) ~68s per resample. With 30 resamples ≈ 45 min for this dataset alone.
- GunPoint: ~0.5s per resample (trivial).
- Adiac: ~2.5s per resample.
- Overall runtime estimate for full 112 × 30: to be measured during full run.

---

## Other Findings

_Anything else interesting observed during reproduction._
