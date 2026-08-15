#!/usr/bin/env python3
"""Run the repository's data/model/reporting scientific invariant suite."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SMOKE_TESTS = (
    "Code/apply_manual_reviews_smoke_test.py",
    "Code/dataset_qa_smoke_test.py",
    "tests/integration/static_robot_anchor_smoke.py",
    "Training/smoke_test.py",
    "Training/ablation_smoke_test.py",
    "Training/clip_alignment_smoke_test.py",
    "Training/dataset_snapshot_smoke_test.py",
    "Training/export_predictions_smoke_test.py",
    "Training/hyperparameter_search_smoke_test.py",
    "Training/live_decision_smoke_test.py",
    "Training/live_validation_smoke_test.py",
    "Training/pose_baselines_smoke_test.py",
    "Training/visual_embedding_smoke_test.py",
    "Training/endpose_smoke_test.py",
    "Training/endpose_v2_smoke_test.py",
    "Training/inference_decision_smoke_test.py",
    "Training/batch_replay_validation_smoke_test.py",
    "Training/run_discovery_smoke_test.py",
    "Training/run_layout_smoke_test.py",
    "Training/run_registry_smoke_test.py",
    "Training/standard_training_smoke_test.py",
    "Training/residual_smoke_test.py",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--unit-only", action="store_true", help="Run pytest without training smokes"
    )
    return parser.parse_args()


def run(command: list[str]) -> None:
    print("RUN " + " ".join(command), flush=True)
    completed = subprocess.run(command, cwd=PROJECT_ROOT, check=False)
    if completed.returncode:
        raise RuntimeError(
            f"Scientific invariant command failed ({completed.returncode}): "
            + " ".join(command)
        )


def main() -> int:
    args = parse_args()
    try:
        run([sys.executable, "-m", "pytest", "-q", "tests"])
        if not args.unit_only:
            for script in SMOKE_TESTS:
                run([sys.executable, script])
    except (OSError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print("All local scientific invariants passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
