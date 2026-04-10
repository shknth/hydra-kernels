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

## Mode Toggle (Pilot vs Full)

Use `improvements/configs/experiment_manifest.json`:

```json
"execution_mode": {
  "full_run": false
}
```

- Set `full_run` to `false` for local pilot runs.
- Set `full_run` to `true` for full runs on a higher-end machine.

When running, scripts print selected mode at startup:

- `[MODE] ... running in PILOT mode`
- `[MODE] ... running in FULL mode`

Track A and Track C also print selected subset/resamples (and Track A prints k/g values), so you can verify config before long runs.

### 1) Local Pilot (validation run)

```bash
python improvements/scripts/track_b_variant_analysis.py
python improvements/scripts/track_a_hyperparam_sensitivity.py
python improvements/scripts/track_c_timing_quality.py
python improvements/scripts/merge_improvement_summaries.py
```

### 2) Full Improvement Run

Set `execution_mode.full_run=true` in the manifest, then run the same commands.

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
