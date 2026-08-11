#!/usr/bin/env python3
"""Evaluate one validation-frozen checkpoint on test without retraining it.

This is the only intended transition from validation screening to final test.
The command refuses runs that already contain test metrics and validates the
hash-bound artifact manifest before loading the selected checkpoint.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch import nn

from artifact_freeze import (
    MANIFEST_NAME,
    canonical_json_hash,
    sha256_file,
    validate_artifact_freeze,
)
from data import DataBundle, prepare_data
from endpose_v2 import DUAL_HORIZON_MODEL_TYPE, wrap_endpose_v2_bundle
from train import (
    build_model as build_standard_model,
    class_weights as standard_class_weights,
    make_loader as make_standard_loader,
    run_epoch as run_standard_epoch,
)
from train_residual import (
    RESIDUAL_V2_MODEL_TYPE,
    build_residual_model,
    class_weights as residual_class_weights,
    make_loader as make_residual_loader,
    run_epoch as run_residual_epoch,
)
from select_matrix_checkpoints import validate_final_test_authorization


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", default="best_intention")
    parser.add_argument("--master-dir", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--selection-file",
        type=Path,
        default=None,
        help="Completed validation-selection manifest authorizing this matrix run.",
    )
    parser.add_argument(
        "--experiment-id",
        default=None,
        help="Matrix experiment ID; required together with --selection-file.",
    )
    parser.add_argument(
        "--device", choices=("auto", "cpu", "cuda", "mps"), default="auto"
    )
    return parser.parse_args()


def resolve(path: Path) -> Path:
    value = Path(path).expanduser()
    return value.resolve() if value.is_absolute() else (PROJECT_ROOT / value).resolve()


def choose_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected an object in {path}")
    return value


def validate_validation_only_source(metrics: dict, checkpoint: str) -> dict:
    if metrics.get("test_evaluation_skipped") is not True:
        raise ValueError("Source run was not validation-only")
    if "test" in metrics or "test_by_checkpoint" in metrics:
        raise ValueError("Source run already exposes test metrics")
    metadata = metrics.get("checkpoints", {}).get(checkpoint)
    if not isinstance(metadata, dict):
        raise ValueError(f"Unknown checkpoint {checkpoint!r}")
    selection_metric = str(metadata.get("selection_metric", ""))
    if not selection_metric.startswith("validation_"):
        raise ValueError("Checkpoint was not selected using validation")
    if checkpoint != "best_intention":
        raise ValueError(
            "Final main evaluation must use best_intention; pose-selected checkpoints "
            "are diagnostic-only"
        )
    return metadata


def prepare_frozen_bundle(
    config: dict,
    metadata: dict,
    *,
    master_dir: Path | None,
) -> DataBundle:
    data_config = dict(config["data"])
    if master_dir is not None:
        data_config["master_dir"] = str(resolve(master_dir))
    bundle = prepare_data(data_config, int(config["training"]["seed"]))
    model_type = str(config.get("model_type", ""))
    if model_type == DUAL_HORIZON_MODEL_TYPE:
        bundle = wrap_endpose_v2_bundle(bundle, data_config)
    expected_provenance = metadata["provenance"]
    checks = {
        "dataset_content_fingerprint": (
            bundle.provenance.get("dataset_content_fingerprint"),
            expected_provenance.get("dataset_content_fingerprint"),
        ),
        "schema_fingerprint": (
            bundle.provenance.get("schema", {}).get("fingerprint"),
            expected_provenance.get("schema", {}).get("fingerprint"),
        ),
        "split_sequences": (
            bundle.split_metadata.get("sequences"),
            metadata.get("split", {}).get("sequences"),
        ),
    }
    mismatches = [name for name, (actual, expected) in checks.items() if actual != expected]
    if mismatches:
        raise ValueError(
            "Frozen dataset reconstruction mismatch: " + ", ".join(mismatches)
        )
    return bundle


def evaluate(
    *,
    config: dict,
    bundle: DataBundle,
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[str, dict, int]:
    model_type = str(config.get("model_type", ""))
    training_config = config["training"]
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    residual = model_type in {RESIDUAL_V2_MODEL_TYPE, DUAL_HORIZON_MODEL_TYPE}
    if residual:
        model = build_residual_model(
            model_type,
            input_dim=len(bundle.normalizer.output_feature_names),
            window_size=int(config["data"]["window_size"]),
            model_config=config["model"],
        ).to(device)
        loader = make_residual_loader(
            bundle.test, training_config, shuffle=False, device=device
        )
        counts = bundle.train.intention_counts()
        assistance = nn.CrossEntropyLoss(
            weight=residual_class_weights(
                [counts[0], counts[1] + counts[2]], device
            )
        )
        assistance_type = nn.CrossEntropyLoss(
            weight=residual_class_weights(counts[1:3], device)
        )
        receiving_hand = nn.CrossEntropyLoss(
            weight=residual_class_weights(
                bundle.train.receiving_hand_counts(), device
            )
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        metrics = run_residual_epoch(
            model,
            loader,
            device,
            assistance,
            assistance_type,
            receiving_hand,
            training_config,
        )
    else:
        model, resolved_type = build_standard_model(
            config,
            input_dim=len(bundle.normalizer.output_feature_names),
            window_size=int(config["data"]["window_size"]),
        )
        if resolved_type != model_type:
            raise ValueError("Resolved standard model type changed")
        model = model.to(device)
        loader = make_standard_loader(
            bundle.test, training_config, shuffle=False, device=device
        )
        counts = bundle.train.intention_counts()
        assistance = nn.CrossEntropyLoss(
            weight=standard_class_weights(
                [counts[0], counts[1] + counts[2]], device
            )
        )
        assistance_type = nn.CrossEntropyLoss(
            weight=standard_class_weights(counts[1:3], device)
        )
        receiving_hand = nn.CrossEntropyLoss(
            weight=standard_class_weights(
                bundle.train.receiving_hand_counts(), device
            )
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        metrics = run_standard_epoch(
            model,
            loader,
            device,
            assistance,
            assistance_type,
            receiving_hand,
            training_config,
        )
    parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    return model_type, metrics, int(parameters)


def main() -> int:
    args = parse_args()
    started = datetime.now(timezone.utc)
    started_monotonic = time.perf_counter()
    try:
        run_dir = resolve(args.run_dir)
        config = read_json(run_dir / "config.json")
        metadata = read_json(run_dir / "data_metadata.json")
        source_metrics = read_json(run_dir / "metrics.json")
        freeze = validate_artifact_freeze(run_dir / MANIFEST_NAME)
        checkpoint_metadata = validate_validation_only_source(
            source_metrics, args.checkpoint
        )
        checkpoint_path = Path(checkpoint_metadata["path"])
        if not checkpoint_path.is_absolute():
            checkpoint_path = (run_dir / checkpoint_path).resolve()
        if not checkpoint_path.is_file():
            portable_identity = freeze.get("output_artifacts", {}).get(
                "checkpoints", {}
            ).get(args.checkpoint)
            if portable_identity is None:
                portable_identity = freeze.get("output_artifacts", {}).get(
                    "checkpoints", {}
                ).get(f"{args.checkpoint}_diagnostic")
            if not portable_identity:
                raise FileNotFoundError(checkpoint_path)
            checkpoint_path = (run_dir / portable_identity["path"]).resolve()
        expected_hash = checkpoint_metadata.get("sha256")
        actual_hash = sha256_file(checkpoint_path)
        if expected_hash and expected_hash != actual_hash:
            raise ValueError("Selected checkpoint hash differs from metrics metadata")
        if (args.selection_file is None) != (args.experiment_id is None):
            raise ValueError(
                "--selection-file and --experiment-id must be provided together"
            )
        matrix_authorization = None
        if args.selection_file is not None:
            selection_path = resolve(args.selection_file)
            selection = read_json(selection_path)
            try:
                portable_run = run_dir.relative_to(PROJECT_ROOT).as_posix()
            except ValueError:
                portable_run = str(run_dir)
            authorized = validate_final_test_authorization(
                selection,
                experiment_id=str(args.experiment_id),
                seed=int(config["training"]["seed"]),
                run_dir=portable_run,
                checkpoint_sha256=actual_hash,
                artifact_manifest_fingerprint=freeze["manifest_fingerprint"],
            )
            matrix_authorization = {
                "selection_file": str(selection_path),
                "selection_file_sha256": sha256_file(selection_path),
                "matrix_id": selection.get("matrix_id"),
                "experiment_id": args.experiment_id,
                "seed": int(config["training"]["seed"]),
                "authorized_checkpoint_sha256": authorized[
                    "checkpoint_sha256"
                ],
                "test_metrics_read_during_authorization": False,
            }
        bundle = prepare_frozen_bundle(
            config, metadata, master_dir=args.master_dir
        )
        device = choose_device(args.device)
        model_type, test_metrics, parameters = evaluate(
            config=config,
            bundle=bundle,
            checkpoint_path=checkpoint_path,
            device=device,
        )
        output = (
            resolve(args.output)
            if args.output is not None
            else run_dir / f"final_test_{args.checkpoint}.json"
        )
        if output.exists():
            raise FileExistsError(
                f"Final test output already exists and will not be overwritten: {output}"
            )
        output.parent.mkdir(parents=True, exist_ok=True)
        source_multitask = source_metrics.get("multitask")
        if isinstance(source_multitask, dict):
            future_pose_loss_enabled = bool(
                source_multitask.get("future_pose_loss_enabled")
            )
            auxiliary_pose_loss_enabled = bool(
                source_multitask.get("auxiliary_pose_loss_enabled", False)
            )
        else:
            future_pose_loss_enabled = (
                float(config["training"].get("pose_loss_weight", 0.0)) > 0.0
            )
            auxiliary_pose_loss_enabled = (
                float(
                    config["training"].get("auxiliary_pose_loss_weight", 0.0)
                )
                > 0.0
            )
        report = {
            "schema_version": 2,
            "evaluation_protocol": "validation_frozen_checkpoint_single_test_v2",
            "report_fingerprint": None,
            "split": "test",
            "source_run": str(run_dir),
            "source_artifact_manifest": str(run_dir / MANIFEST_NAME),
            "source_artifact_manifest_fingerprint": freeze[
                "manifest_fingerprint"
            ],
            "dataset_identifier": freeze["dataset"]["identifier"],
            "dataset_content_fingerprint": freeze["dataset"][
                "dataset_content_fingerprint"
            ],
            "source_content_fingerprint": freeze["dataset"][
                "source_content_fingerprint"
            ],
            "model_type": model_type,
            "trainable_parameters": parameters,
            "training_task_semantics": {
                "future_pose_loss_enabled": future_pose_loss_enabled,
                "future_pose_loss_weight": float(
                    config["training"].get("pose_loss_weight", 0.0)
                ),
                "auxiliary_pose_loss_enabled": auxiliary_pose_loss_enabled,
                "auxiliary_pose_loss_weight": float(
                    config["training"].get("auxiliary_pose_loss_weight", 0.0)
                ),
                "pose_metrics_role": (
                    "main_learned_output"
                    if future_pose_loss_enabled
                    else "untrained_pose_head_diagnostic_only"
                ),
            },
            "checkpoint": {
                "name": args.checkpoint,
                "path": str(checkpoint_path),
                "sha256": actual_hash,
                "epoch": int(checkpoint_metadata["epoch"]),
                "selection_split": "validation",
                "selection_metric": checkpoint_metadata["selection_metric"],
                "selection_value": float(checkpoint_metadata["selection_value"]),
            },
            "matrix_authorization": matrix_authorization,
            "test_metrics": test_metrics,
            "test_used_for_model_or_checkpoint_selection": False,
            "runtime": {
                "started_at": started.isoformat(),
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "wall_seconds": time.perf_counter() - started_monotonic,
                "device": str(device),
                "command": [sys.executable, *sys.argv],
            },
        }
        report["report_fingerprint"] = canonical_json_hash(report)
        output.write_text(
            json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        print(f"Final frozen test evaluation: {output}")
        print(f"Output SHA-256: {sha256_file(output)}")
    except (
        FileExistsError,
        FileNotFoundError,
        KeyError,
        OSError,
        RuntimeError,
        ValueError,
    ) as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
