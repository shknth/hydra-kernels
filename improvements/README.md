# Improvements Workspace

This folder contains all artifacts related to project improvements beyond baseline Hydra reproduction.

## Goals

We evaluate three improvement tracks:

1. Hyperparameter sensitivity for Hydra (`k`, `g`)
2. Variant analysis using existing ROCKET-family results
3. Runtime-quality profiling (transform/fit/predict timing)

## Planned Structure

- `configs/`: experiment manifests and dataset subsets
- `scripts/`: runnable scripts for each improvement track
- `outputs/`: intermediate run logs and raw result files
- `analysis/`: summary tables and report-ready plots

## Execution Policy

- Keep the original repository structure untouched.
- Store new improvement outputs separately from baseline reproduction outputs.
- Use local machine for quick validation only, then run full experiments on higher-end hardware.

## Naming Convention

- Raw outputs: `outputs/<track_name>/<dataset>_<resample>.json`
- Summaries: `analysis/<track_name>_summary.csv`
- Final merged file for report: `analysis/improvements_master_summary.csv`

## Run Commands

Run from repository root after installing dependencies from `reproduction/requirements.txt`.

### 1) Local Pilot (validation run)

```bash
python improvements/scripts/track_b_variant_analysis.py
python improvements/scripts/track_a_hyperparam_sensitivity.py
python improvements/scripts/track_c_timing_quality.py
python improvements/scripts/merge_improvement_summaries.py
```

### 2) Full Improvement Run

Use the same scripts with a larger dataset subset and/or resample count by editing `improvements/configs/experiment_manifest.json`.

Optional orchestrated run:

```bash
python improvements/scripts/run_improvements.py
python improvements/scripts/merge_improvement_summaries.py
```

## Expected Outputs

- Track A summary: `improvements/analysis/track_a/track_a_summary.csv`
- Track B summaries:
  - `improvements/analysis/track_b/variant_deltas.csv`
  - `improvements/analysis/track_b/variant_summary.csv`
  - `improvements/analysis/track_b/best_method_per_dataset.csv`
- Track C summary: `improvements/analysis/track_c/track_c_summary.csv`
- Master summary: `improvements/analysis/improvements_master_summary.csv`
