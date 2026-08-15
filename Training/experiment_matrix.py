#!/usr/bin/env python3
"""Validate and emit the minimal predeclared thesis experiment matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MATRIX = Path("Training/configs/experiment_matrix_v2.json")
EXPECTED_SEEDS = (42, 43, 44)
REQUIRED_OBSERVATION_ALIGNMENT_VERSION = "causal_backward_device_time_v1"
REQUIRED_DATASET_CONTRACT = {
    "expected_selected_sequences": 214,
    "expected_sequence_fingerprint": (
        "5d136a34b915f4e6a81fda70d34c959be48b4be79f0f7922decfdaae65ad12cd"
    ),
}


class ExperimentMatrixError(ValueError):
    """Raised when matrix entries could not form a fair executable protocol."""


def resolve(path: Path) -> Path:
    value = Path(path).expanduser()
    return value.resolve() if value.is_absolute() else (PROJECT_ROOT / value).resolve()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ExperimentMatrixError(f"Expected JSON object: {path}")
    return value


def _nested(value: dict, *keys: str) -> Any:
    for key in keys:
        value = value[key]
    return value


def validate_matrix(matrix_path: Path) -> dict[str, Any]:
    path = resolve(matrix_path)
    matrix = read_json(path)
    if int(matrix.get("schema_version", -1)) != 1:
        raise ExperimentMatrixError("Unsupported experiment-matrix schema")
    if tuple(matrix.get("seeds", [])) != EXPECTED_SEEDS:
        raise ExperimentMatrixError("Final matrix must use seeds 42, 43 and 44")
    policy = matrix.get("policy", {})
    if policy.get("selection_split") != "validation":
        raise ExperimentMatrixError("Experiment selection must be validation-only")
    if not policy.get("test_already_observed"):
        raise ExperimentMatrixError("Observed historical test status must be explicit")

    entries = matrix.get("training_experiments", [])
    ids = [str(entry.get("id", "")) for entry in entries]
    if not ids or any(not value for value in ids) or len(ids) != len(set(ids)):
        raise ExperimentMatrixError("Training experiment IDs must be nonempty and unique")
    configs: dict[str, dict] = {}
    for entry in entries:
        config_path = resolve(Path(entry["config"]))
        entrypoint = resolve(Path(entry["entrypoint"]))
        if not config_path.is_file() or not entrypoint.is_file():
            raise ExperimentMatrixError(f"Missing executable/config for {entry['id']}")
        config = read_json(config_path)
        configs[entry["id"]] = config
        if float(_nested(config, "data", "future_horizon_seconds")) != 1.0:
            raise ExperimentMatrixError(f"{entry['id']} is not a t+1 data protocol")
        if config["data"].get("required_observation_alignment_version") != (
            REQUIRED_OBSERVATION_ALIGNMENT_VERSION
        ):
            raise ExperimentMatrixError(
                f"{entry['id']} does not require the causal master alignment"
            )
        if config["data"].get("dataset_contract") != REQUIRED_DATASET_CONTRACT:
            raise ExperimentMatrixError(
                f"{entry['id']} does not bind the active n214 sequence set"
            )
        if int(_nested(config, "training", "seed")) not in EXPECTED_SEEDS:
            raise ExperimentMatrixError(f"Unexpected base seed in {entry['id']}")

    expected_modes = {
        "residual_current_gate": ("temporal_channel_gated", "hierarchical"),
        "residual_simple_gate": ("temporal_channel_simple", "hierarchical"),
        "residual_modality_gated": ("modality_gated", "hierarchical"),
        "residual_temporal_only": ("temporal_only", "hierarchical"),
        "residual_flat": ("temporal_channel_gated", "flat"),
    }
    for experiment_id, (fusion, head) in expected_modes.items():
        config = configs[experiment_id]
        actual = (
            _nested(config, "model", "fusion_mode"),
            _nested(config, "model", "intention_head_mode"),
        )
        if actual != (fusion, head):
            raise ExperimentMatrixError(
                f"Architecture mode mismatch for {experiment_id}: {actual}"
            )
    if float(_nested(configs["residual_without_pose_aux"], "training", "pose_loss_weight")) != 0.0:
        raise ExperimentMatrixError("Auxiliary-pose-off config has nonzero pose loss")

    fair_ids = tuple(expected_modes)
    reference = configs["residual_current_gate"]
    data_reference = reference["data"]
    training_reference = reference["training"]
    capacity_keys = ("d_model", "nhead", "num_layers", "dim_feedforward", "dropout")
    for experiment_id in fair_ids[1:]:
        config = configs[experiment_id]
        if config["data"] != data_reference:
            raise ExperimentMatrixError(
                f"Architecture ablation changes input data: {experiment_id}"
            )
        comparable_training = {
            key: value
            for key, value in config["training"].items()
            if key != "intention_loss_weight"
        }
        if comparable_training != training_reference:
            raise ExperimentMatrixError(
                f"Architecture ablation changes training budget: {experiment_id}"
            )
        for key in capacity_keys:
            if config["model"][key] != reference["model"][key]:
                raise ExperimentMatrixError(
                    f"Architecture ablation changes {key}: {experiment_id}"
                )

    for experiment_id in (
        "visual_corrected_clip_current_gate",
        "visual_corrected_clip_modality_gate",
    ):
        visual = _nested(configs[experiment_id], "data", "visual_embeddings")
        if not visual.get("enabled"):
            raise ExperimentMatrixError(f"Visual input disabled in {experiment_id}")
        if "device_time_v2" not in str(visual.get("cache_dir")):
            raise ExperimentMatrixError(f"Obsolete CLIP cache path in {experiment_id}")
        if "device_time_v2" not in str(visual.get("projection_path")):
            raise ExperimentMatrixError(f"Obsolete visual projection in {experiment_id}")
        if not visual.get("verify_cache_hashes"):
            raise ExperimentMatrixError(f"Cache hashes disabled in {experiment_id}")

        sensor_data = dict(data_reference)
        visual_data = dict(configs[experiment_id]["data"])
        visual_data.pop("visual_embeddings", None)
        if visual_data != sensor_data:
            raise ExperimentMatrixError(
                f"Visual comparison changes non-visual data: {experiment_id}"
            )
        if configs[experiment_id]["training"] != training_reference:
            raise ExperimentMatrixError(
                f"Visual comparison changes training budget: {experiment_id}"
            )
        visual_model = dict(configs[experiment_id]["model"])
        expected_fusion = (
            "temporal_channel_gated"
            if experiment_id == "visual_corrected_clip_current_gate"
            else "modality_gated"
        )
        if visual_model.pop("fusion_mode") != expected_fusion:
            raise ExperimentMatrixError(
                f"Unexpected visual fusion mode: {experiment_id}"
            )
        sensor_model = dict(reference["model"])
        sensor_model.pop("fusion_mode")
        if visual_model != sensor_model:
            raise ExperimentMatrixError(
                f"Visual comparison changes backbone capacity: {experiment_id}"
            )

    terminal_target = _nested(
        configs["terminal_endpose_learned"], "data", "pose_target"
    )
    if terminal_target.get("target_definition_version") != (
        "terminal_endpose_unique_hand_capture_v2"
    ):
        raise ExperimentMatrixError("Terminal experiment uses a stale target")

    alias_ids = [entry["id"] for entry in matrix.get("aliases_without_training", [])]
    if set(ids) & set(alias_ids) or len(alias_ids) != len(set(alias_ids)):
        raise ExperimentMatrixError("Alias IDs collide with trained experiment IDs")
    known = set(ids)
    for alias in matrix.get("aliases_without_training", []):
        if alias.get("source_experiment") not in known:
            raise ExperimentMatrixError(f"Unknown alias source: {alias}")
    evaluation = {
        entry.get("id"): entry for entry in matrix.get("evaluation_only", [])
    }
    t1 = evaluation.get("t1_pose_baselines", {})
    if "Training/export_residual_predictions.py" not in str(t1.get("command", "")):
        raise ExperimentMatrixError(
            "The authoritative t+1 comparison must use a checkpoint-bound export"
        )
    if tuple(t1.get("methods", ())) != (
        "persistence",
        "constant_velocity",
        "learned_model_oracle_hand",
    ):
        raise ExperimentMatrixError("The t+1 fair comparison methods are incomplete")
    if t1.get("primary_comparison") != (
        "test_predictions.json "
        "pose_comparison.methods.<method>.fair_common_metrics"
    ):
        raise ExperimentMatrixError(
            "The t+1 primary comparison points to a stale report field"
        )
    postprocessing = matrix.get("postprocessing", {})
    required_t1 = postprocessing.get("required_t1_experiments")
    if required_t1 != ["residual_current_gate"]:
        raise ExperimentMatrixError(
            "The primary t+1 baseline comparison must be predeclared"
        )
    if postprocessing.get("seed_policy") != "all_matrix_seeds":
        raise ExperimentMatrixError("t+1 postprocessing must cover every matrix seed")
    if postprocessing.get("require_grouped_report_in_authoritative_summary") is not True:
        raise ExperimentMatrixError(
            "Authoritative reporting must require the t+1 grouped artifacts"
        )
    for experiment_id in required_t1:
        entry = next((entry for entry in entries if entry["id"] == experiment_id), None)
        if entry is None or entry.get("entrypoint") != "Training/train_residual.py":
            raise ExperimentMatrixError(
                f"t+1 postprocessing target is not a residual model: {experiment_id}"
            )
        if float(_nested(configs[experiment_id], "training", "pose_loss_weight")) <= 0.0:
            raise ExperimentMatrixError(
                f"t+1 postprocessing target has no learned pose task: {experiment_id}"
            )
    matrix["_path"] = str(path)
    return matrix


def run_directory(matrix: dict, experiment_id: str, seed: int) -> str:
    dataset = matrix["dataset_tag"]
    experiment = matrix["validation_experiment_tag"]
    return (
        f"Training/runs/{dataset}/{experiment}/{experiment_id}/"
        f"{experiment_id}_seed{seed}"
    )


def validation_commands(matrix: dict) -> list[dict[str, str | int]]:
    rows = []
    for entry in matrix["training_experiments"]:
        for seed in matrix["seeds"]:
            run_dir = run_directory(matrix, entry["id"], int(seed))
            command = (
                f"python3 {entry['entrypoint']} --config {entry['config']} "
                f"--dataset-tag {matrix['dataset_tag']} "
                f"--experiment-tag {matrix['validation_experiment_tag']} "
                f"--seed {seed} --run-dir {run_dir} --skip-test-evaluation"
            )
            rows.append(
                {
                    "stage": "validation",
                    "experiment_id": entry["id"],
                    "seed": int(seed),
                    "run_dir": run_dir,
                    "command": command,
                }
            )
    return rows


def final_test_commands(matrix: dict) -> list[dict[str, str | int]]:
    rows = []
    report_root = (
        f"Training/reports/{matrix['dataset_tag']}/"
        f"{matrix['matrix_id']}/final_test"
    )
    selection_file = (
        f"Training/reports/{matrix['dataset_tag']}/"
        f"{matrix['matrix_id']}/validation_selection.json"
    )
    for entry in matrix["training_experiments"]:
        for seed in matrix["seeds"]:
            run_dir = run_directory(matrix, entry["id"], int(seed))
            output = f"{report_root}/{entry['id']}_seed{seed}.json"
            command = (
                "python3 Training/evaluate_frozen_run.py "
                f"--run-dir {run_dir} --checkpoint best_intention "
                f"--selection-file {selection_file} "
                f"--experiment-id {entry['id']} --output {output}"
            )
            rows.append(
                {
                    "stage": "final_test",
                    "experiment_id": entry["id"],
                    "seed": int(seed),
                    "run_dir": run_dir,
                    "output": output,
                    "command": command,
                }
            )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, default=DEFAULT_MATRIX)
    parser.add_argument(
        "--stage", choices=("validation", "final-test", "all"), default="validation"
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON rather than commands")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        matrix = validate_matrix(args.matrix)
        rows = []
        if args.stage in {"validation", "all"}:
            rows.extend(validation_commands(matrix))
        if args.stage in {"final-test", "all"}:
            rows.extend(final_test_commands(matrix))
    except (ExperimentMatrixError, FileNotFoundError, KeyError, TypeError, ValueError) as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}")
        return 2
    if args.json:
        print(json.dumps(rows, indent=2, ensure_ascii=False))
    else:
        for row in rows:
            print(row["command"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
