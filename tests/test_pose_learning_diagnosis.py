from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EVALUATION = ROOT / "Training" / "evaluation"
if str(EVALUATION) not in sys.path:
    sys.path.insert(0, str(EVALUATION))

from diagnose_pose_learning_curves import (  # noqa: E402
    EXPECTED_DATASET,
    build_report,
    trajectory_summary,
)


def test_trajectory_summary_preserves_nonmonotonic_behavior() -> None:
    result = trajectory_summary([10.0, 8.0, 9.0, 7.0], [1, 2, 3, 4])
    assert result["first"] == 10.0
    assert result["last"] == 7.0
    assert result["minimum"] == 7.0
    assert result["minimum_epoch"] == 4
    assert result["decreasing_step_fraction"] == 2 / 3
    assert result["least_squares_slope_per_epoch"] < 0


def _metric_block(position_loss: float, position_error_cm: float, macro_f1: float) -> dict:
    return {
        "loss": {"position": position_loss, "orientation": 0.1},
        "pose_oracle": {
            "position_mean_euclidean_error_cm": position_error_cm,
        },
        "intention": {"macro_f1": macro_f1},
    }


def _make_run(root: Path, seed: int, errors: list[float]) -> Path:
    run = root / f"seed_{seed}"
    run.mkdir()
    config = {
        "training": {
            "seed": seed,
            "pose_loss_weight": 1.0,
            "orientation_loss_weight": 0.25,
        }
    }
    history = []
    for epoch, error in enumerate(errors, start=1):
        history.append(
            {
                "epoch": epoch,
                "train": _metric_block(error / 1000, error, 0.7 + epoch / 100),
                "validation": _metric_block(
                    error / 2000,
                    error + (0.5 if epoch == 1 else -0.5),
                    0.8 if epoch == 1 else 0.79,
                ),
            }
        )
    metrics = {
        "run_context": {
            "dataset_tag": EXPECTED_DATASET,
            "experiment_tag": "thesis_final_v2_validation",
        },
        "architecture": {"fusion_mode": "temporal_channel_gated"},
        "checkpoints": {
            "best_intention": {"epoch": 1},
            "best_pose": {
                "epoch": len(errors),
                "selection_value": errors[-1] - 0.5,
            },
        },
        "history": history,
    }
    (run / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (run / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    return run


def test_primary_curves_reject_underfitting_trigger_and_find_selection_lag(
    tmp_path: Path,
) -> None:
    runs = [
        _make_run(tmp_path, 42, [9.5, 9.0, 8.8]),
        _make_run(tmp_path, 43, [9.4, 9.2, 8.9]),
        _make_run(tmp_path, 44, [9.6, 9.3, 9.0]),
    ]
    report, rows = build_report(runs, expected_dataset=EXPECTED_DATASET)
    assert len(rows) == 9
    assert report["scope"]["target"] == "receiving-wrist pose at t+1 s"
    assert report["scope"]["endpose_included"] is False
    assert report["decision"]["checklist_case_a_underfitting_signal"] is False
    assert (
        report["decision"]["normalized_smooth_l1_sensitivity_run_recommended"]
        is False
    )
    assert report["decision"]["selection_timing_seeds"] == [42, 43, 44]
    assert all(
        run["position_loss"]["type"] == "smooth_l1_meters"
        and run["position_loss"]["explicit_config_block"] is False
        for run in report["runs"]
    )
