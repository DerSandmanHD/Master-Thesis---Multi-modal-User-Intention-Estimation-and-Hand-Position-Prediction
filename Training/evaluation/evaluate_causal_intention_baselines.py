#!/usr/bin/env python3
"""Evaluate deterministic causal intention baselines on frozen window splits.

The elapsed-time model uses only ``t - t_START``. The last-frame model uses
the final causally aligned, train-normalized sensor row of the same 60-sample
window as the thesis models. Neither model reads the sequence end or future
events, and no test metric is used for fitting or hyperparameter selection.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import torch
from scipy.optimize import minimize


TRAINING_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = TRAINING_DIR.parent
if str(TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINING_DIR))

from artifact_freeze import canonical_json_hash, sha256_file  # noqa: E402
from data import INTENTION_NAMES, DataBundle, WindowDataset, prepare_data  # noqa: E402
from metrics import classification_metrics  # noqa: E402


SCHEMA_VERSION = "causal_intention_baselines_v1"
MODEL_PROTOCOL = "deterministic_train_only_multinomial_logistic_v1"
DEFAULT_L2 = 1e-4
DEFAULT_MAX_ITERATIONS = 500


@dataclass(frozen=True)
class BaselineArrays:
    sample_keys: list[str]
    sequence_ids: list[str]
    participants: list[str]
    endpoint_timestamps_ns: np.ndarray
    elapsed_time_s: np.ndarray
    last_sensor_frame: np.ndarray
    targets: np.ndarray


@dataclass(frozen=True)
class SoftmaxFit:
    weights: np.ndarray
    objective: float
    iterations: int
    gradient_max_abs: float
    optimizer_message: str


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
    parser.add_argument("--prediction-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--l2", type=float, default=DEFAULT_L2)
    parser.add_argument(
        "--max-iterations", type=int, default=DEFAULT_MAX_ITERATIONS
    )
    return parser.parse_args()


def _resolve(path: Path) -> Path:
    value = path.expanduser()
    return value.resolve() if value.is_absolute() else (PROJECT_ROOT / value).resolve()


def _relative(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(resolved)


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _input_identity(path: Path) -> dict[str, Any]:
    return {
        "path": _relative(path),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def load_elapsed_times(
    dataset: WindowDataset, *, master_dir: Path
) -> dict[str, np.ndarray]:
    """Load only START-relative time and verify its device-time row binding."""

    elapsed: dict[str, np.ndarray] = {}
    for record in dataset.records:
        path = master_dir / f"{record.sequence_id}_master.csv"
        frame = pd.read_csv(path, usecols=["timestamp_ns", "time_since_start_s"])
        timestamps = pd.to_numeric(frame["timestamp_ns"], errors="raise").to_numpy(
            dtype=np.int64
        )
        values = pd.to_numeric(
            frame["time_since_start_s"], errors="raise"
        ).to_numpy(dtype=np.float64)
        if not np.array_equal(timestamps, record.timestamps_ns):
            raise ValueError(
                f"Elapsed-time rows are not timestamp-bound for {record.sequence_id}"
            )
        if not np.isfinite(values).all() or np.any(np.diff(values) < 0):
            raise ValueError(
                f"Elapsed time is non-finite or non-monotonic for {record.sequence_id}"
            )
        elapsed[record.sequence_id] = values
    return elapsed


def collect_arrays(
    dataset: WindowDataset,
    *,
    split: str,
    elapsed_by_sequence: Mapping[str, np.ndarray],
) -> BaselineArrays:
    sample_keys: list[str] = []
    sequence_ids: list[str] = []
    participants: list[str] = []
    timestamps: list[int] = []
    elapsed: list[float] = []
    frames: list[np.ndarray] = []
    targets: list[int] = []
    for record_index, endpoint in dataset.indices:
        record = dataset.records[record_index]
        timestamp = int(record.timestamps_ns[endpoint])
        sample_keys.append(f"{split}|{record.sequence_id}|{timestamp}")
        sequence_ids.append(record.sequence_id)
        participants.append(record.participant)
        timestamps.append(timestamp)
        elapsed.append(float(elapsed_by_sequence[record.sequence_id][endpoint]))
        frames.append(record.features[endpoint].astype(np.float64, copy=False))
        targets.append(int(record.intentions[endpoint]))
    return BaselineArrays(
        sample_keys=sample_keys,
        sequence_ids=sequence_ids,
        participants=participants,
        endpoint_timestamps_ns=np.asarray(timestamps, dtype=np.int64),
        elapsed_time_s=np.asarray(elapsed, dtype=np.float64)[:, None],
        last_sensor_frame=np.stack(frames).astype(np.float64, copy=False),
        targets=np.asarray(targets, dtype=np.int64),
    )


def fit_softmax_regression(
    features: np.ndarray,
    targets: np.ndarray,
    *,
    class_count: int,
    l2: float = DEFAULT_L2,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
) -> SoftmaxFit:
    """Fit a full-batch multinomial logistic model with an unpenalized bias."""

    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(targets, dtype=np.int64)
    if x.ndim != 2 or len(x) != len(y) or not len(x):
        raise ValueError("Softmax regression requires a non-empty 2D design matrix")
    if not np.isfinite(x).all():
        raise ValueError("Softmax regression features must be finite")
    if set(np.unique(y)) != set(range(class_count)):
        raise ValueError("Every intention class must occur in the training split")
    if not math.isfinite(l2) or l2 < 0:
        raise ValueError("l2 must be finite and non-negative")
    if max_iterations <= 0:
        raise ValueError("max_iterations must be positive")

    design = np.concatenate((x, np.ones((len(x), 1), dtype=np.float64)), axis=1)
    one_hot = np.eye(class_count, dtype=np.float64)[y]
    shape = (design.shape[1], class_count)

    def objective(flat: np.ndarray) -> tuple[float, np.ndarray]:
        weights = flat.reshape(shape)
        logits = design @ weights
        logits -= logits.max(axis=1, keepdims=True)
        exp_logits = np.exp(logits)
        probabilities = exp_logits / exp_logits.sum(axis=1, keepdims=True)
        negative_log_likelihood = -np.log(
            probabilities[np.arange(len(y)), y].clip(min=1e-300)
        ).mean()
        penalty = 0.5 * l2 * np.square(weights[:-1]).sum()
        gradient = design.T @ (probabilities - one_hot) / len(y)
        gradient[:-1] += l2 * weights[:-1]
        return float(negative_log_likelihood + penalty), gradient.ravel()

    result = minimize(
        objective,
        np.zeros(shape, dtype=np.float64).ravel(),
        method="L-BFGS-B",
        jac=True,
        options={
            "maxiter": int(max_iterations),
            "ftol": 1e-12,
            "gtol": 1e-8,
            "maxls": 50,
        },
    )
    if not result.success:
        raise RuntimeError(
            "Multinomial logistic optimization failed: " + str(result.message)
        )
    gradient = np.asarray(result.jac, dtype=np.float64)
    return SoftmaxFit(
        weights=np.asarray(result.x, dtype=np.float64).reshape(shape),
        objective=float(result.fun),
        iterations=int(result.nit),
        gradient_max_abs=float(np.abs(gradient).max()),
        optimizer_message=str(result.message),
    )


def predict_softmax(features: np.ndarray, fit: SoftmaxFit) -> np.ndarray:
    x = np.asarray(features, dtype=np.float64)
    design = np.concatenate((x, np.ones((len(x), 1), dtype=np.float64)), axis=1)
    return np.argmax(design @ fit.weights, axis=1).astype(np.int64)


def _metrics(predictions: np.ndarray, targets: np.ndarray) -> dict[str, Any]:
    result = classification_metrics(
        torch.from_numpy(predictions),
        torch.from_numpy(targets),
        len(INTENTION_NAMES),
    )
    result["class_names"] = list(INTENTION_NAMES)
    return result


def _normalization(features: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    mean = features.mean(axis=0)
    std = features.std(axis=0)
    std = np.where(std < 1e-12, 1.0, std)
    return (features - mean) / std, {
        "mean": mean.tolist(),
        "std": std.tolist(),
        "fit_split": "train",
    }


def _apply_normalization(
    features: np.ndarray, metadata: Mapping[str, Any]
) -> np.ndarray:
    mean = np.asarray(metadata["mean"], dtype=np.float64)
    std = np.asarray(metadata["std"], dtype=np.float64)
    return (features - mean) / std


def evaluate_baselines(
    arrays: Mapping[str, BaselineArrays],
    *,
    feature_names: list[str],
    l2: float,
    max_iterations: int,
) -> tuple[dict[str, Any], dict[str, dict[str, np.ndarray]]]:
    train = arrays["train"]
    majority_class = int(np.bincount(train.targets, minlength=3).argmax())
    elapsed_train, elapsed_normalizer = _normalization(train.elapsed_time_s)
    elapsed_fit = fit_softmax_regression(
        elapsed_train,
        train.targets,
        class_count=3,
        l2=l2,
        max_iterations=max_iterations,
    )
    sensor_fit = fit_softmax_regression(
        train.last_sensor_frame,
        train.targets,
        class_count=3,
        l2=l2,
        max_iterations=max_iterations,
    )

    predictions: dict[str, dict[str, np.ndarray]] = {}
    methods: dict[str, Any] = {
        "majority_class": {
            "definition": "Always predict the majority intention from train windows.",
            "fit_split": "train",
            "selected_class_id": majority_class,
            "selected_class": INTENTION_NAMES[majority_class],
            "train_class_counts": np.bincount(
                train.targets, minlength=3
            ).tolist(),
        },
        "elapsed_time_since_start_logistic": {
            "definition": (
                "Multinomial logistic regression using only t - t_START; no "
                "sequence-end or future-event information."
            ),
            "fit_split": "train",
            "input_features": ["time_since_start_s"],
            "normalizer": elapsed_normalizer,
            "optimizer": {
                "protocol": MODEL_PROTOCOL,
                "l2": l2,
                "iterations": elapsed_fit.iterations,
                "objective": elapsed_fit.objective,
                "gradient_max_abs": elapsed_fit.gradient_max_abs,
                "message": elapsed_fit.optimizer_message,
            },
        },
        "last_sensor_frame_logistic": {
            "definition": (
                "Multinomial logistic regression on the final causally aligned "
                "sensor row and observation masks of each thesis window."
            ),
            "fit_split": "train",
            "input_features": feature_names,
            "input_feature_count": len(feature_names),
            "input_feature_fingerprint": canonical_json_hash(feature_names),
            "normalizer": (
                "The raw channels use the thesis train-only normalizer; observed "
                "mask channels remain binary."
            ),
            "optimizer": {
                "protocol": MODEL_PROTOCOL,
                "l2": l2,
                "iterations": sensor_fit.iterations,
                "objective": sensor_fit.objective,
                "gradient_max_abs": sensor_fit.gradient_max_abs,
                "message": sensor_fit.optimizer_message,
            },
        },
    }
    for split, values in arrays.items():
        split_predictions = {
            "majority_class": np.full(
                len(values.targets), majority_class, dtype=np.int64
            ),
            "elapsed_time_since_start_logistic": predict_softmax(
                _apply_normalization(values.elapsed_time_s, elapsed_normalizer),
                elapsed_fit,
            ),
            "last_sensor_frame_logistic": predict_softmax(
                values.last_sensor_frame, sensor_fit
            ),
        }
        predictions[split] = split_predictions
        for name, predicted in split_predictions.items():
            methods[name].setdefault("metrics", {})[split] = _metrics(
                predicted, values.targets
            )
    return methods, predictions


def _validate_frozen_bindings(
    *,
    bundle: DataBundle,
    descriptor: Mapping[str, Any],
    prediction_report: Mapping[str, Any],
) -> None:
    checks = {
        "dataset tag": (
            descriptor.get("dataset_tag"),
            prediction_report.get("dataset_identifier")
            or descriptor.get("dataset_tag"),
        ),
        "dataset content fingerprint": (
            bundle.provenance.get("dataset_content_fingerprint"),
            descriptor.get("dataset_content_fingerprint"),
        ),
        "prediction dataset content fingerprint": (
            bundle.provenance.get("dataset_content_fingerprint"),
            prediction_report.get("dataset_content_fingerprint"),
        ),
        "source content fingerprint": (
            bundle.provenance.get("source_content_fingerprint"),
            descriptor.get("source_content_fingerprint"),
        ),
        "prediction source content fingerprint": (
            bundle.provenance.get("source_content_fingerprint"),
            prediction_report.get("source_content_fingerprint"),
        ),
        "test endpoint fingerprint": (
            bundle.test.endpoint_fingerprint(),
            prediction_report.get("frozen_split_endpoint_fingerprint"),
        ),
        "test endpoint count": (
            len(bundle.test),
            prediction_report.get("frozen_split_endpoint_count"),
        ),
    }
    for label, (actual, expected) in checks.items():
        if str(actual) != str(expected):
            raise ValueError(
                f"Frozen {label} mismatch: actual={actual!r}, expected={expected!r}"
            )
    if prediction_report.get("split") != "test" or prediction_report.get(
        "full_split_export"
    ) is not True:
        raise ValueError("Prediction binding is not the complete frozen test split")
    if prediction_report.get("report_fingerprint") != canonical_json_hash(
        {**prediction_report, "report_fingerprint": None}
    ):
        raise ValueError("Prediction report fingerprint is invalid")


def _markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# Causal intention baselines",
        "",
        (
            "Retrospective descriptive baselines on the frozen v3 split. Fitting "
            "uses train windows only; no test metric selects a feature, parameter, "
            "or hyperparameter."
        ),
        "",
        "| Method | Test accuracy | Test macro-F1 | Continue F1 | Fetch F1 | Handover F1 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, method in report["methods"].items():
        metric = method["metrics"]["test"]
        f1 = metric["per_class_f1"]
        lines.append(
            f"| {name} | {metric['accuracy']:.4f} | {metric['macro_f1']:.4f} "
            f"| {f1[0]:.4f} | {f1[1]:.4f} | {f1[2]:.4f} |"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    try:
        config_path = _resolve(args.config)
        descriptor_path = _resolve(args.dataset_descriptor)
        prediction_report_path = _resolve(args.prediction_report)
        output_dir = _resolve(args.output_dir)
        if output_dir.exists() and any(output_dir.iterdir()):
            raise FileExistsError(
                f"Refusing to overwrite non-empty baseline directory: {output_dir}"
            )

        config = _read_object(config_path)
        descriptor = _read_object(descriptor_path)
        prediction_report = _read_object(prediction_report_path)
        data_config = dict(config["data"])
        master_dir = Path(data_config["master_dir"]).expanduser()
        if not master_dir.is_absolute():
            master_dir = (PROJECT_ROOT / master_dir).resolve()
        data_config["master_dir"] = str(master_dir)
        bundle = prepare_data(data_config, seed=42)
        _validate_frozen_bindings(
            bundle=bundle,
            descriptor=descriptor,
            prediction_report=prediction_report,
        )

        arrays: dict[str, BaselineArrays] = {}
        for split in ("train", "validation", "test"):
            dataset = getattr(bundle, split)
            arrays[split] = collect_arrays(
                dataset,
                split=split,
                elapsed_by_sequence=load_elapsed_times(
                    dataset, master_dir=master_dir
                ),
            )
        methods, predictions = evaluate_baselines(
            arrays,
            feature_names=bundle.normalizer.output_feature_names,
            l2=float(args.l2),
            max_iterations=int(args.max_iterations),
        )

        test = arrays["test"]
        prediction_frame = pd.DataFrame(
            {
                "sample_key": test.sample_keys,
                "sequence_id": test.sequence_ids,
                "participant": test.participants,
                "endpoint_timestamp_ns": test.endpoint_timestamps_ns,
                "time_since_start_s": test.elapsed_time_s[:, 0],
                "target_intention_id": test.targets,
                "target_intention": [INTENTION_NAMES[value] for value in test.targets],
                **{
                    f"{name}_prediction_id": values
                    for name, values in predictions["test"].items()
                },
            }
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        predictions_path = output_dir / "test_predictions.csv"
        prediction_frame.to_csv(predictions_path, index=False)
        report = {
            "schema_version": SCHEMA_VERSION,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "report_fingerprint": None,
            "dataset": {
                "identifier": descriptor["dataset_tag"],
                "selected_sequences": descriptor["selected_sequences"],
                "sequence_fingerprint": descriptor["sequence_fingerprint"],
                "dataset_content_fingerprint": bundle.provenance[
                    "dataset_content_fingerprint"
                ],
                "source_content_fingerprint": bundle.provenance[
                    "source_content_fingerprint"
                ],
                "required_observation_alignment_version": descriptor[
                    "required_observation_alignment_version"
                ],
            },
            "split": {
                "participants": bundle.split_metadata["participants"],
                "sequence_counts": {
                    name: len(bundle.split_metadata["sequences"][name])
                    for name in ("train", "validation", "test")
                },
                "window_counts": {
                    name: len(getattr(bundle, name))
                    for name in ("train", "validation", "test")
                },
                "endpoint_fingerprints": bundle.split_metadata[
                    "window_eligibility"
                ]["endpoint_fingerprints"],
            },
            "scientific_policy": {
                "analysis_timing": (
                    "retrospective descriptive baseline added after the primary "
                    "test split had already been observed"
                ),
                "fit_split": "train",
                "validation_role": "report_only_no_selection",
                "test_role": "single_frozen_evaluation_no_selection",
                "future_information_used": False,
                "forbidden_inputs": [
                    "sequence_end_time",
                    "normalized_sequence_progress",
                    "future_event_timestamp",
                    "future_sensor_observation",
                ],
            },
            "methods": methods,
            "inputs": {
                "config": _input_identity(config_path),
                "dataset_descriptor": _input_identity(descriptor_path),
                "frozen_prediction_report": _input_identity(
                    prediction_report_path
                ),
            },
            "test_predictions": {
                "path": predictions_path.name,
                "sha256": sha256_file(predictions_path),
                "rows": len(prediction_frame),
            },
        }
        report["report_fingerprint"] = canonical_json_hash(report)
        report_path = output_dir / "intention_baselines.json"
        report_path.write_text(
            json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        markdown_path = output_dir / "INTENTION_BASELINES.md"
        markdown_path.write_text(_markdown(report), encoding="utf-8")
        artifact_manifest = {
            "schema_version": "causal_intention_baseline_artifacts_v1",
            "manifest_fingerprint": None,
            "report_fingerprint": report["report_fingerprint"],
            "inputs": report["inputs"],
            "outputs": {
                path.name: _input_identity(path)
                for path in (predictions_path, report_path, markdown_path)
            },
        }
        artifact_manifest["manifest_fingerprint"] = canonical_json_hash(
            artifact_manifest
        )
        manifest_path = output_dir / "artifact_manifest.json"
        manifest_path.write_text(
            json.dumps(
                artifact_manifest,
                indent=2,
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
    except (FileExistsError, KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}")
        return 2
    print(f"Causal intention baselines: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
