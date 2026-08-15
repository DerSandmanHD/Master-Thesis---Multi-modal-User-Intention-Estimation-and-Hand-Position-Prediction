#!/usr/bin/env python3
"""Materialize executable nested participant Group-CV configs and commands."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

from artifact_freeze import canonical_json_hash, sha256_file


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def resolve(path: Path) -> Path:
    value = Path(path).expanduser()
    return value.resolve() if value.is_absolute() else (PROJECT_ROOT / value).resolve()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def portable(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path.resolve())


def infer_entrypoint(config: dict[str, Any]) -> str:
    model_type = str(config.get("model_type", ""))
    return (
        "Training/train_residual.py"
        if "residual_pose_transformer" in model_type
        else "Training/train.py"
    )


def build_group_cv_plan(
    *,
    audit: dict[str, Any],
    base_config: dict[str, Any],
    base_config_path: Path,
    output_dir: Path,
    dataset_tag: str,
    experiment_tag: str,
    seeds: list[int],
    entrypoint: str | None = None,
) -> dict[str, Any]:
    cv = audit.get("participant_group_cv")
    if not isinstance(cv, dict) or not cv.get("folds"):
        raise ValueError("Split audit has no participant_group_cv folds")
    if cv.get("execution_protocol") is None:
        raise ValueError("Group-CV audit predates the nested executable protocol")
    if cv.get("participant_balanced_aggregation_identifiable") is not True:
        raise ValueError(
            "Executable thesis Group-CV requires leave-one-participant-out folds"
        )
    executable = entrypoint or infer_entrypoint(base_config)
    if not (PROJECT_ROOT / executable).is_file():
        raise FileNotFoundError(executable)
    output_dir.mkdir(parents=True, exist_ok=True)
    configs_dir = output_dir / "configs"
    results_dir = output_dir / "outer_evaluation"
    configs_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for fold in cv["folds"]:
        train = sorted(str(value) for value in fold["train_participants"])
        validation = sorted(
            str(value) for value in fold["validation_participants"]
        )
        test = sorted(str(value) for value in fold["test_participants"])
        if len(test) != 1:
            raise ValueError(
                f"Fold {fold['fold']} must contain exactly one outer participant"
            )
        sets = tuple(map(set, (train, validation, test)))
        if any(sets[left] & sets[right] for left, right in ((0, 1), (0, 2), (1, 2))):
            raise ValueError(f"Fold {fold['fold']} is not participant-disjoint")
        if not all(sets):
            raise ValueError(f"Fold {fold['fold']} has an empty partition")
        for seed in seeds:
            config = copy.deepcopy(base_config)
            config["run_name"] = f"group_cv_fold{int(fold['fold']):02d}"
            config["training"]["seed"] = int(seed)
            config["data"]["train_participants"] = train
            config["data"]["validation_participants"] = validation
            config["data"]["test_participants"] = test
            config["group_cv"] = {
                "protocol": cv["execution_protocol"],
                "split_fingerprint_sha256": cv["split_fingerprint_sha256"],
                "fold": int(fold["fold"]),
                "train_participants": train,
                "validation_participants": validation,
                "outer_evaluation_participants": test,
                "checkpoint_selection_split": "validation",
                "outer_evaluation_used_for_selection": False,
            }
            config_name = f"fold_{int(fold['fold']):02d}_seed{int(seed)}.json"
            config_path = configs_dir / config_name
            config_path.write_text(
                json.dumps(config, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            run_id = f"fold_{int(fold['fold']):02d}_seed{int(seed)}"
            run_dir = (
                PROJECT_ROOT
                / "Training/runs"
                / dataset_tag
                / experiment_tag
                / run_id
            )
            result = results_dir / f"{run_id}.json"
            portable_config = portable(config_path)
            portable_run = portable(run_dir)
            portable_result = portable(result)
            train_command = (
                f"python3 {executable} --config {portable_config} "
                f"--dataset-tag {dataset_tag} --experiment-tag {experiment_tag} "
                f"--seed {int(seed)} --run-dir {portable_run} "
                "--skip-test-evaluation"
            )
            evaluate_command = (
                "python3 Training/evaluate_frozen_run.py "
                f"--run-dir {portable_run} --checkpoint best_intention "
                f"--output {portable_result}"
            )
            rows.append(
                {
                    "fold": int(fold["fold"]),
                    "seed": int(seed),
                    "config": portable_config,
                    "config_sha256": sha256_file(config_path),
                    "run_dir": portable_run,
                    "outer_evaluation_output": portable_result,
                    "train_participants": train,
                    "validation_participants": validation,
                    "outer_evaluation_participants": test,
                    "validation_command": train_command,
                    "outer_evaluation_command": evaluate_command,
                }
            )
    plan = {
        "schema_version": 1,
        "protocol": "nested_participant_group_cv_executable_v1",
        "plan_fingerprint": None,
        "dataset_tag": dataset_tag,
        "experiment_tag": experiment_tag,
        "base_config": portable(base_config_path),
        "base_config_sha256": sha256_file(base_config_path),
        "split_fingerprint_sha256": cv["split_fingerprint_sha256"],
        "fold_count": int(cv["fold_count"]),
        "seeds": [int(seed) for seed in seeds],
        "checkpoint_selection_split": "inner_validation",
        "outer_evaluation_used_for_selection": False,
        "runs": rows,
    }
    plan["plan_fingerprint"] = canonical_json_hash(plan)
    return plan


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-audit", type=Path, required=True)
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset-tag", required=True)
    parser.add_argument("--experiment-tag", default="thesis_v2_group_cv")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--entrypoint", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        audit_path = resolve(args.split_audit)
        base_path = resolve(args.base_config)
        output_dir = resolve(args.output_dir)
        plan = build_group_cv_plan(
            audit=read_json(audit_path),
            base_config=read_json(base_path),
            base_config_path=base_path,
            output_dir=output_dir,
            dataset_tag=args.dataset_tag,
            experiment_tag=args.experiment_tag,
            seeds=list(dict.fromkeys(args.seeds)),
            entrypoint=args.entrypoint,
        )
        plan_path = output_dir / "group_cv_plan.json"
        plan_path.write_text(
            json.dumps(plan, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    except (FileNotFoundError, KeyError, OSError, TypeError, ValueError) as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}")
        return 2
    print(f"Executable Group-CV plan: {plan_path}")
    for row in plan["runs"]:
        print(row["validation_command"])
        print(row["outer_evaluation_command"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
