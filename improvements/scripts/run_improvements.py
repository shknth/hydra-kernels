"""
Run enabled improvement tracks in sequence.

Tracks are controlled by:
- improvements/configs/experiment_manifest.json

Usage:
    python improvements/scripts/run_improvements.py
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO_ROOT / "improvements" / "scripts"
MANIFEST_PATH = REPO_ROOT / "improvements" / "configs" / "experiment_manifest.json"


TRACK_SCRIPTS = {
    "track_a_hyperparam_sensitivity": "track_a_hyperparam_sensitivity.py",
    "track_b_variant_analysis": "track_b_variant_analysis.py",
    "track_c_timing_quality": "track_c_timing_quality.py",
}


def load_manifest(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def run_script(script_name: str) -> int:
    script_path = SCRIPTS_DIR / script_name
    cmd = [sys.executable, str(script_path)]
    print(f"\n[RUN] {script_name}")
    result = subprocess.run(cmd, cwd=str(REPO_ROOT))
    return int(result.returncode)


def main() -> int:
    manifest = load_manifest(MANIFEST_PATH)
    tracks_cfg = manifest.get("tracks", {})

    enabled_tracks = [
        track_name
        for track_name, cfg in tracks_cfg.items()
        if cfg.get("enabled", False)
    ]

    if not enabled_tracks:
        print("No enabled tracks in manifest. Nothing to run.")
        return 0

    print("Enabled tracks:")
    for t in enabled_tracks:
        print(f"- {t}")

    for track_name in enabled_tracks:
        script_name = TRACK_SCRIPTS.get(track_name)
        if not script_name:
            print(f"[WARN] No script mapping found for {track_name}; skipping.")
            continue

        rc = run_script(script_name)
        if rc != 0:
            print(f"[ERROR] {track_name} failed with exit code {rc}.")
            return rc

    print("\nAll enabled improvement tracks completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
