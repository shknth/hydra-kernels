# Hydra: Reproduction and Improvement Study

Reproduction and extension of:

> **Hydra: Competing Convolutional Kernels for Fast and Accurate Time Series Classification**
> Angus Dempster, Daniel F. Schmidt, Geoffrey I. Webb
> *Data Mining and Knowledge Discovery*, 2023.
> DOI: [10.1007/s10618-023-00939-3](https://doi.org/10.1007/s10618-023-00939-3) · arXiv: [2203.13652](https://arxiv.org/abs/2203.13652)

This repository reproduces the published Hydra results on the UCR-112 benchmark and evaluates three improvement tracks: hyperparameter sensitivity, variant combinations, and timing profiling.

---

## Repository Structure

```
hydra/
├── code/                        # Original paper code (unmodified)
│   ├── hydra.py                 # Hydra transform + SparseScaler
│   ├── hydra_multivariate.py    # Experimental multivariate extension
│   └── softmax.py               # SGD variant for large datasets
├── results/                     # Paper's published results
│   ├── results_ucr112_hydra.csv         # Hydra standalone (112 datasets, 30 resamples)
│   └── results_ucr112_variants.csv      # Hydra + MiniRocket/MultiRocket/Rocket
├── reproduction/                # Reproduction pipeline
│   ├── requirements.txt         # Pinned dependencies
│   ├── smoke_test.py            # Environment validation (run first)
│   ├── download_datasets.py     # Download all 112 UCR datasets
│   ├── run_reproduction.py      # Checkpointed baseline runner
│   ├── compare_results.py       # Diff our results vs paper CSV
│   └── notes.md                 # Documented findings and deviations
├── improvements/                # Improvement experiments
│   ├── configs/
│   │   └── experiment_manifest.json    # Toggle pilot / full run mode
│   ├── scripts/
│   │   ├── track_a_hyperparam_sensitivity.py
│   │   ├── track_b_variant_analysis.py
│   │   ├── track_c_timing_quality.py
│   │   ├── run_improvements.py          # Orchestrator for all tracks
│   │   └── merge_improvement_summaries.py
│   ├── outputs/                 # Raw per-run JSON results
│   └── analysis/                # Summary CSVs per track
├── figures/
│   ├── generate_figures.py      # Generate all 4 report figures
│   ├── fig1_reproduction_scatter.pdf
│   ├── fig2_hyperparam_heatmap.pdf
│   ├── fig3_variant_bars.pdf
│   └── fig4_timing_breakdown.pdf
├── reproduction_results/        # Baseline run outputs (gitignored)
├── report.tex                   # Full IEEE LaTeX report
└── data/                        # Downloaded datasets (gitignored)
```

---

## Quickstart

### 1. Set Up the Environment

**macOS / Linux (using [uv](https://github.com/astral-sh/uv)):**

```bash
# Install uv if not present
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env

# Create venv and install dependencies
uv venv .venv --python 3.10
source .venv/bin/activate
uv pip install -r reproduction/requirements.txt
```

> **macOS SSL note:** Python 3.10 on macOS requires a certifi fix. The
> activation script sets `SSL_CERT_FILE` automatically after running
> the above commands. If you see `CERTIFICATE_VERIFY_FAILED`, run:
> `open "/Applications/Python 3.10/Install Certificates.command"`

**Google Colab:**

```python
from google.colab import drive
drive.mount('/content/drive')
# Set BASE_DIR in run_reproduction.py to "/content/drive/MyDrive/hydra"
!pip install -r reproduction/requirements.txt
```

### 2. Validate the Environment

```bash
python reproduction/smoke_test.py
```

Expected output: `SMOKE TEST PASSED` on all 3 test datasets.

### 3. Download All 112 UCR Datasets

```bash
python reproduction/download_datasets.py
```

Downloads to `data/`. Retries on failure. Saves SHA-256 checksums to
`data/checksums.json`. Failed downloads are logged to
`data/failed_downloads.txt`.

### 4. Run the Baseline Reproduction

```bash
python reproduction/run_reproduction.py
```

- Runs 112 datasets × 30 resamples = **3,360 jobs**
- Each result is saved to `reproduction_results/<dataset>/resample_<r>.json`
- **Fully resumable:** re-running the same command continues from where it stopped
- Progress is printed per run; errors go to `reproduction_results/errors.log`

### 5. Compare Against Published Results

```bash
python reproduction/compare_results.py
```

Outputs `reproduction_results/comparison_summary.csv` with per-dataset
accuracy delta vs the paper's published values.

---

## Improvement Experiments

All three improvement tracks are controlled by
`improvements/configs/experiment_manifest.json`.

Set `"full_run": false` for a quick pilot (3 datasets), or
`"full_run": true` for the full experiment matrix.

### Run all tracks (orchestrated):

```bash
python improvements/scripts/run_improvements.py
python improvements/scripts/merge_improvement_summaries.py
```

### Run tracks individually:

```bash
# Track A: Hyperparameter sensitivity (k, g sweep)
python improvements/scripts/track_a_hyperparam_sensitivity.py

# Track B: Variant analysis (Hydra vs Hydra+MiniRocket/MultiRocket/Rocket)
python improvements/scripts/track_b_variant_analysis.py

# Track C: Timing profiling (transform / fit / predict breakdown)
python improvements/scripts/track_c_timing_quality.py
```

### Generate report figures:

```bash
python figures/generate_figures.py
```

Saves 4 PDF figures to `figures/`.

---

## Key Results

### Baseline Reproduction (UCR-112, 30 resamples each)

| Metric | Value |
|---|---|
| Total runs | 3,360 / 3,360 |
| Failures | 0 |
| Mean \|Δ\| accuracy vs paper | 0.74% |
| Datasets within 1% of paper | 88 / 112 |
| Wall-clock duration | 17.4 hours (Apple M-series CPU) |

### Improvement Track B — Variant Summary

| Variant | Win Rate | Mean Δ | Max Gain |
|---|---|---|---|
| Hydra + MiniRocket | 61.6% | +0.80% | +9.6% |
| Hydra + MultiRocket | 58.0% | +1.04% | +27.9% |
| Hydra + Rocket | 54.5% | +0.68% | +16.7% |

---

## Dependencies

| Package | Version |
|---|---|
| Python | 3.10.4 |
| PyTorch | 2.1.0 |
| NumPy | 1.24.4 |
| scikit-learn | 1.3.2 |
| aeon | 0.7.1 |
| pandas | 2.0.3 |
| certifi | 2026.2.25 |
| matplotlib | latest |

---

## Documented Reproducibility Findings

Seven findings are documented in `reproduction/notes.md`:

1. SSL certificate failure on macOS Python 3.10 — fix included
2. `LinAlgWarning` from RidgeClassifierCV on small alpha values (benign)
3. Resampling deviation: `StratifiedShuffleSplit` vs community-standard seeds
4. cv=5 guard for 4 large datasets (ElectricDevices, Crop, FordA, FordB)
5. Timing measurement bug — test transform included in train time
6. Random kernel initialisation per resample (seed=None by default)
7. SparseScaler clamps negative Hydra features to zero before scaling

---

## Citation

If you use this reproduction, please also cite the original paper:

```bibtex
@article{dempster_etal_2023,
  author  = {Dempster, Angus and Schmidt, Daniel F and Webb, Geoffrey I},
  title   = {Hydra: Competing Convolutional Kernels for Fast and Accurate Time Series Classification},
  year    = {2023},
  journal = {Data Mining and Knowledge Discovery},
  volume  = {37},
  pages   = {1779--1805},
}
```
