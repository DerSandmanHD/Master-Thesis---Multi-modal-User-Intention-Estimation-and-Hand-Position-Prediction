#!/usr/bin/env python3
"""Fast checks for canonical run, report, and live-session paths."""

from __future__ import annotations

from pathlib import Path

from run_layout import (
    build_run_context,
    experiment_report_directory,
    live_session_directory,
    training_run_directory,
    validate_run_context,
    validate_tag,
)


def main() -> int:
    assert build_run_context(
        dataset_tag=None,
        experiment_tag=None,
        model_tag="transformer_v1",
    ) == {
        "dataset_tag": "development",
        "experiment_tag": "manual",
        "model_tag": "transformer_v1",
    }
    run = training_run_directory(
        dataset_tag="dataset_v2_20260815_n180_ab12cd34",
        experiment_tag="benchmark_v2",
        model_tag="residual_v2",
        seed=44,
        timestamp="20260815_121314",
    )
    assert run == Path(
        "Training/runs/dataset_v2_20260815_n180_ab12cd34/"
        "benchmark_v2/residual_v2/20260815_121314_seed_44"
    )
    assert experiment_report_directory(
        "dataset_v2_20260815_n180_ab12cd34",
        "benchmark_v2",
    ) == Path(
        "Training/reports/dataset_v2_20260815_n180_ab12cd34/benchmark_v2"
    )
    assert live_session_directory(
        "dataset_v2_20260815_n180_ab12cd34",
        "live_validation_01",
    ) == Path(
        "Training/live_runs/dataset_v2_20260815_n180_ab12cd34/"
        "live_validation_01"
    )
    valid_context = {
        "run_context": {
            "dataset_tag": "dataset_v2_20260815_n180_ab12cd34",
            "experiment_tag": "benchmark_v2",
            "model_tag": "residual_v2",
        }
    }
    assert validate_run_context(
        valid_context,
        dataset_tag="dataset_v2_20260815_n180_ab12cd34",
        experiment_tag="benchmark_v2",
    ) == valid_context["run_context"]
    assert validate_run_context(
        {},
        experiment_tag="final_clean_v1",
    ) == {}
    for kwargs in (
        {
            "dataset_tag": "dataset_v3_20260901_n200_cd34ef56",
            "experiment_tag": "benchmark_v2",
        },
        {
            "dataset_tag": "dataset_v2_20260815_n180_ab12cd34",
            "experiment_tag": "benchmark_v3",
        },
    ):
        try:
            validate_run_context(valid_context, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(
                f"Mismatched run context was accepted: {kwargs}"
            )
    for invalid in ("", "Ohne Titel", "../escape", "Final_V2", "tag/child"):
        try:
            validate_tag(invalid, "test_tag")
        except ValueError:
            pass
        else:
            raise AssertionError(f"Invalid tag was accepted: {invalid!r}")
    print("run layout smoke test passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
