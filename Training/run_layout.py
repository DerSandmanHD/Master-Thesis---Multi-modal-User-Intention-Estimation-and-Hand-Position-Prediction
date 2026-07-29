#!/usr/bin/env python3
"""Canonical paths and safe tags for future training artifacts."""

from __future__ import annotations

import re
from pathlib import Path


TAG_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]*$")
DEFAULT_RUNS_ROOT = Path("Training/runs")
DEFAULT_REPORTS_ROOT = Path("Training/reports")
DEFAULT_LIVE_ROOT = Path("Training/live_runs")


def validate_tag(value: str, field: str) -> str:
    tag = str(value).strip()
    if not TAG_PATTERN.fullmatch(tag):
        raise ValueError(
            f"{field} must match {TAG_PATTERN.pattern!r}, got {value!r}"
        )
    return tag


def build_run_context(
    *,
    dataset_tag: str | None,
    experiment_tag: str | None,
    model_tag: str,
) -> dict:
    return {
        "dataset_tag": validate_tag(
            dataset_tag or "development",
            "dataset_tag",
        ),
        "experiment_tag": validate_tag(
            experiment_tag or "manual",
            "experiment_tag",
        ),
        "model_tag": validate_tag(model_tag, "model_tag"),
    }


def validate_run_context(
    config: dict,
    *,
    experiment_tag: str,
    dataset_tag: str | None = None,
    source: str = "run",
) -> dict:
    expected_experiment = validate_tag(
        experiment_tag,
        "experiment_tag",
    )
    expected_dataset = (
        validate_tag(dataset_tag, "dataset_tag")
        if dataset_tag is not None
        else None
    )
    run_context = config.get("run_context", {})
    if not isinstance(run_context, dict):
        raise ValueError(f"Invalid run_context in {source}")
    if (
        expected_dataset is not None
        and run_context.get("dataset_tag") != expected_dataset
    ):
        raise ValueError(
            f"Dataset tag mismatch in {source}: "
            f"{run_context.get('dataset_tag')!r} != "
            f"{expected_dataset!r}"
        )
    if (
        run_context
        and run_context.get("experiment_tag") != expected_experiment
    ):
        raise ValueError(
            f"Experiment tag mismatch in {source}: "
            f"{run_context.get('experiment_tag')!r} != "
            f"{expected_experiment!r}"
        )
    return run_context


def training_run_directory(
    *,
    dataset_tag: str,
    experiment_tag: str,
    model_tag: str,
    seed: int,
    timestamp: str,
    runs_root: Path = DEFAULT_RUNS_ROOT,
) -> Path:
    dataset = validate_tag(dataset_tag, "dataset_tag")
    experiment = validate_tag(experiment_tag, "experiment_tag")
    model = validate_tag(model_tag, "model_tag")
    run_id = validate_tag(f"{timestamp}_seed_{int(seed)}", "run_id")
    return Path(runs_root) / dataset / experiment / model / run_id


def experiment_report_directory(
    dataset_tag: str,
    experiment_tag: str,
    reports_root: Path = DEFAULT_REPORTS_ROOT,
) -> Path:
    return (
        Path(reports_root)
        / validate_tag(dataset_tag, "dataset_tag")
        / validate_tag(experiment_tag, "experiment_tag")
    )


def live_session_directory(
    dataset_tag: str,
    session_tag: str,
    live_root: Path = DEFAULT_LIVE_ROOT,
) -> Path:
    return (
        Path(live_root)
        / validate_tag(dataset_tag, "dataset_tag")
        / validate_tag(session_tag, "session_tag")
    )
