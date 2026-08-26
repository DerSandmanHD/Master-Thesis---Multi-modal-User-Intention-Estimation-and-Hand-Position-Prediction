#!/usr/bin/env python3
"""Audit empirical master sampling and actual thesis-window durations."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


TRAINING_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = TRAINING_DIR.parent
if str(TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINING_DIR))

from artifact_freeze import canonical_json_hash, sha256_file  # noqa: E402
from data import DataBundle, WindowDataset, prepare_data  # noqa: E402


SCHEMA_VERSION = "empirical_sampling_window_audit_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("Training/configs/models/residual_transformer_v2.json"),
    )
    parser.add_argument(
        "--dataset-descriptor",
        type=Path,
        default=Path(
            "Training/datasets/"
            "dataset_v3_causal_20260815_n214_5d136a34.json"
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _resolve(path: Path) -> Path:
    value = path.expanduser()
    return value.resolve() if value.is_absolute() else (PROJECT_ROOT / value).resolve()


def _relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path.resolve())


def _identity(path: Path) -> dict[str, Any]:
    return {
        "path": _relative(path),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def distribution(values: Sequence[float] | np.ndarray) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or not len(array) or not np.isfinite(array).all():
        raise ValueError("Distribution requires finite one-dimensional values")
    q05, q25, q50, q75, q95 = np.quantile(
        array, [0.05, 0.25, 0.5, 0.75, 0.95]
    )
    return {
        "count": int(len(array)),
        "minimum": float(array.min()),
        "percentile_05": float(q05),
        "quartile_25": float(q25),
        "median": float(q50),
        "quartile_75": float(q75),
        "percentile_95": float(q95),
        "maximum": float(array.max()),
        "iqr": float(q75 - q25),
        "mean": float(array.mean()),
        "population_std": float(array.std(ddof=0)),
    }


def interval_arrays(bundle: DataBundle) -> tuple[np.ndarray, list[dict[str, Any]]]:
    intervals: list[np.ndarray] = []
    per_sequence: list[dict[str, Any]] = []
    records = [
        record
        for split in (bundle.train, bundle.validation, bundle.test)
        for record in split.records
    ]
    if len({record.sequence_id for record in records}) != len(records):
        raise ValueError("A sequence occurs in more than one participant split")
    for record in records:
        delta = np.diff(record.timestamps_ns.astype(np.int64)) / 1e9
        if not len(delta):
            raise ValueError(f"Sequence has fewer than two rows: {record.sequence_id}")
        intervals.append(delta)
        positive = delta[delta > 0]
        per_sequence.append(
            {
                "sequence_id": record.sequence_id,
                "participant": record.participant,
                "rows": int(len(record.timestamps_ns)),
                "intervals": int(len(delta)),
                "nonpositive_intervals": int((delta <= 0).sum()),
                "median_delta_seconds": (
                    float(np.median(positive)) if len(positive) else None
                ),
            }
        )
    return np.concatenate(intervals), per_sequence


def window_durations(dataset: WindowDataset) -> np.ndarray:
    values = []
    for record_index, endpoint in dataset.indices:
        record = dataset.records[record_index]
        start = endpoint - dataset.window_size + 1
        values.append(
            (int(record.timestamps_ns[endpoint]) - int(record.timestamps_ns[start]))
            / 1e9
        )
    return np.asarray(values, dtype=np.float64)


def build_audit(
    bundle: DataBundle,
    *,
    descriptor: Mapping[str, Any],
    max_timestamp_gap_seconds: float,
    configured_stride_rows: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if bundle.provenance.get("dataset_content_fingerprint") != descriptor.get(
        "dataset_content_fingerprint"
    ):
        raise ValueError("Dataset content fingerprint differs from descriptor")
    if bundle.provenance.get("source_content_fingerprint") != descriptor.get(
        "source_content_fingerprint"
    ):
        raise ValueError("Source content fingerprint differs from descriptor")
    intervals, per_sequence = interval_arrays(bundle)
    positive = intervals[intervals > 0]
    eligible = positive[positive <= max_timestamp_gap_seconds]
    if not len(eligible):
        raise ValueError("No positive intervals pass the configured gap limit")
    split_durations = {
        split: window_durations(getattr(bundle, split))
        for split in ("train", "validation", "test")
    }
    all_durations = np.concatenate(list(split_durations.values()))
    interval_stats = distribution(positive)
    eligible_stats = distribution(eligible)
    audit = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "report_fingerprint": None,
        "dataset": {
            "identifier": descriptor["dataset_tag"],
            "selected_sequences": descriptor["selected_sequences"],
            "sequence_fingerprint": descriptor["sequence_fingerprint"],
            "dataset_content_fingerprint": descriptor[
                "dataset_content_fingerprint"
            ],
            "source_content_fingerprint": descriptor[
                "source_content_fingerprint"
            ],
        },
        "sampling_intervals_seconds": {
            "definition": (
                "Within-sequence timestamp_ns[i+1] - timestamp_ns[i], pooled "
                "across all selected causal master CSVs; no cross-sequence delta."
            ),
            "raw_positive": interval_stats,
            "nonpositive_count": int((intervals <= 0).sum()),
            "configured_window_gap_limit_seconds": max_timestamp_gap_seconds,
            "within_configured_window_gap_limit": eligible_stats,
            "above_configured_window_gap_limit_count": int(
                (positive > max_timestamp_gap_seconds).sum()
            ),
            "empirical_median_sampling_hz": (
                1.0 / eligible_stats["median"]
                if eligible_stats["median"] > 0
                else None
            ),
        },
        "observation_window": {
            "samples": int(bundle.train.window_size),
            "intervals_per_window": int(bundle.train.window_size - 1),
            "configured_stride_rows": int(configured_stride_rows),
            "actual_duration_seconds_all_valid_windows": distribution(
                all_durations
            ),
            "actual_duration_seconds_by_split": {
                split: distribution(values)
                for split, values in split_durations.items()
            },
            "window_counts": {
                split: int(len(values))
                for split, values in split_durations.items()
            },
            "median_delta_duration_estimate_seconds": float(
                (bundle.train.window_size - 1) * eligible_stats["median"]
            ),
            "duration_definition": (
                "endpoint timestamp minus first timestamp of each accepted "
                "60-sample window; 60 samples span 59 intervals."
            ),
        },
        "split": {
            "participants": bundle.split_metadata["participants"],
            "sequence_counts": {
                split: len(bundle.split_metadata["sequences"][split])
                for split in ("train", "validation", "test")
            },
        },
    }
    audit["report_fingerprint"] = canonical_json_hash(audit)
    return audit, per_sequence


def _markdown(report: Mapping[str, Any]) -> str:
    delta = report["sampling_intervals_seconds"][
        "within_configured_window_gap_limit"
    ]
    window = report["observation_window"][
        "actual_duration_seconds_all_valid_windows"
    ]
    return "\n".join(
        [
            "# Empirical sampling and observation-window duration",
            "",
            (
                f"Dataset: `{report['dataset']['identifier']}` "
                f"({report['dataset']['selected_sequences']} sequences)."
            ),
            "",
            "| Quantity | Median | IQR | 5th–95th percentile | Min–max |",
            "|---|---:|---:|---:|---:|",
            (
                "| Positive Δt within configured gap limit (s) | "
                f"{delta['median']:.6f} | {delta['iqr']:.6f} | "
                f"{delta['percentile_05']:.6f}–{delta['percentile_95']:.6f} | "
                f"{delta['minimum']:.6f}–{delta['maximum']:.6f} |"
            ),
            (
                "| Actual valid 60-sample window duration (s) | "
                f"{window['median']:.6f} | {window['iqr']:.6f} | "
                f"{window['percentile_05']:.6f}–{window['percentile_95']:.6f} | "
                f"{window['minimum']:.6f}–{window['maximum']:.6f} |"
            ),
            "",
            (
                "A 60-sample window spans 59 timestamp intervals. The reported "
                "window distribution is measured directly for every accepted "
                "train/validation/test window."
            ),
            "",
        ]
    )


def _write_sequence_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    try:
        config_path = _resolve(args.config)
        descriptor_path = _resolve(args.dataset_descriptor)
        output_dir = _resolve(args.output_dir)
        if output_dir.exists() and any(output_dir.iterdir()):
            raise FileExistsError(
                f"Refusing to overwrite non-empty sampling audit: {output_dir}"
            )
        config = _read_object(config_path)
        descriptor = _read_object(descriptor_path)
        data_config = dict(config["data"])
        master_dir = Path(data_config["master_dir"]).expanduser()
        if not master_dir.is_absolute():
            master_dir = (PROJECT_ROOT / master_dir).resolve()
        data_config["master_dir"] = str(master_dir)
        bundle = prepare_data(data_config, seed=42)
        report, sequence_rows = build_audit(
            bundle,
            descriptor=descriptor,
            max_timestamp_gap_seconds=float(
                data_config["max_timestamp_gap_seconds"]
            ),
            configured_stride_rows=int(data_config["stride"]),
        )
        report["inputs"] = {
            "config": _identity(config_path),
            "dataset_descriptor": _identity(descriptor_path),
        }
        report["report_fingerprint"] = canonical_json_hash(
            {**report, "report_fingerprint": None}
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        report_path = output_dir / "sampling_window_audit.json"
        sequence_path = output_dir / "sampling_by_sequence.csv"
        markdown_path = output_dir / "SAMPLING_WINDOW_AUDIT.md"
        report_path.write_text(
            json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        _write_sequence_csv(sequence_path, sequence_rows)
        markdown_path.write_text(_markdown(report), encoding="utf-8")
        manifest = {
            "schema_version": "empirical_sampling_window_artifacts_v1",
            "manifest_fingerprint": None,
            "report_fingerprint": report["report_fingerprint"],
            "inputs": report["inputs"],
            "outputs": {
                path.name: _identity(path)
                for path in (report_path, sequence_path, markdown_path)
            },
        }
        manifest["manifest_fingerprint"] = canonical_json_hash(manifest)
        manifest_path = output_dir / "artifact_manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False)
            + "\n",
            encoding="utf-8",
        )
    except (FileExistsError, KeyError, OSError, TypeError, ValueError) as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}")
        return 2
    print(f"Empirical sampling/window audit: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
