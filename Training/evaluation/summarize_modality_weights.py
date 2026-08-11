#!/usr/bin/env python3
"""Validate and summarize per-window learned modality-fusion weights."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def resolve(path: Path) -> Path:
    path = path.expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def modality_names(frame: pd.DataFrame) -> list[str]:
    names = []
    for column in frame.columns:
        if column.startswith("modality_") and column.endswith("_weight"):
            names.append(column.removeprefix("modality_").removesuffix("_weight"))
    if not names:
        raise ValueError("Prediction export contains no modality-weight columns")
    for name in names:
        availability = f"modality_{name}_available"
        if availability not in frame:
            raise ValueError(f"Missing availability column: {availability}")
    return names


def summarize(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    required = {
        "sample_key",
        "participant",
        "sequence_id",
        "endpoint_timestamp_ns",
        "target_intention",
        "predicted_intention",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError("Prediction export lacks columns: " + ", ".join(missing))
    names = modality_names(frame)
    weight_columns = [f"modality_{name}_weight" for name in names]
    availability_columns = [f"modality_{name}_available" for name in names]
    weights = frame[weight_columns].apply(pd.to_numeric, errors="raise").to_numpy(
        np.float64
    )
    available = frame[availability_columns].astype(bool).to_numpy()
    if not np.isfinite(weights).all() or np.any(weights < -1e-7):
        raise ValueError("Modality weights must be finite and non-negative")
    if np.any(np.abs(weights[~available]) > 1e-6):
        raise ValueError("Unavailable modalities must have exactly zero weight")
    sums = weights.sum(axis=1)
    expected = available.any(axis=1).astype(np.float64)
    if not np.allclose(sums, expected, atol=1e-5, rtol=0.0):
        raise ValueError("Available modality weights must sum to one per window")

    columns = [
        "sample_key",
        "participant",
        "sequence_id",
        "endpoint_timestamp_ns",
        "target_intention",
        "predicted_intention",
        *weight_columns,
        *availability_columns,
    ]
    windows = frame[columns].copy()
    grouped = []
    for (target, prediction), group in frame.groupby(
        ["target_intention", "predicted_intention"], dropna=False
    ):
        row = {
            "target_intention": target,
            "predicted_intention": prediction,
            "windows": int(len(group)),
        }
        for name in names:
            subset = group.loc[group[f"modality_{name}_available"].astype(bool)]
            row[f"{name}_available_windows"] = int(len(subset))
            row[f"{name}_mean_weight_when_available"] = (
                float(subset[f"modality_{name}_weight"].mean())
                if len(subset)
                else None
            )
        grouped.append(row)
    report = {
        "schema_version": 1,
        "rows": int(len(frame)),
        "modality_names": names,
        "weight_semantics": (
            "learned internal per-window conditioning weights; not causal modality effects"
        ),
        "invariants": {
            "unavailable_weight_zero": True,
            "available_weights_sum_to_one": True,
        },
        "by_target_and_prediction": grouped,
    }
    return windows, report


def main() -> int:
    args = parse_args()
    predictions = resolve(args.predictions)
    output_dir = resolve(args.output_dir)
    frame = pd.read_csv(predictions)
    windows, report = summarize(frame)
    output_dir.mkdir(parents=True, exist_ok=True)
    windows.to_csv(output_dir / "modality_weights_by_window.csv", index=False)
    (output_dir / "summary.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Windows: {len(windows)}; report: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
