#!/usr/bin/env python3
"""Generate deterministic Residual-v2 random-search configurations."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import random
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BASE_CONFIG = Path("Training/configs/models/residual_transformer_v2.json")
DEFAULT_OUTPUT_DIR = Path("Training/configs/hyperparameter_search_v1")
SEARCH_VERSION = "residual_v2_random_search_v1"
SELECTION_RULE = (
    "maximize validation intention macro-F1; retain trials within 0.005 of "
    "the best; then minimize validation pose MAE; then maximize validation "
    "receiving-hand macro-F1; then minimize trainable parameters"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config", type=Path, default=DEFAULT_BASE_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--trials", type=int, default=24)
    parser.add_argument("--search-seed", type=int, default=20260808)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def project_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def relative_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path.resolve())


def sample_parameters(rng: random.Random) -> dict:
    d_model = rng.choice((32, 64, 128))
    nhead = rng.choice(tuple(value for value in (2, 4, 8) if d_model % value == 0))
    dim_feedforward = rng.choice(
        tuple(value for value in (64, 128, 256) if value >= d_model)
    )
    return {
        "learning_rate": float(f"{10 ** rng.uniform(-5.0, -3.0):.12g}"),
        "weight_decay": rng.choice((0.0, 1e-5, 1e-4, 1e-3)),
        "dropout": rng.choice((0.05, 0.15, 0.30)),
        "d_model": d_model,
        "nhead": nhead,
        "num_layers": rng.choice((1, 2, 3)),
        "dim_feedforward": dim_feedforward,
        "batch_size": rng.choice((16, 32, 64)),
        "orientation_loss_weight": rng.choice((0.10, 0.25, 0.50)),
        "receiving_hand_loss_weight": rng.choice((0.5, 1.0, 2.0)),
    }


def parameter_key(parameters: dict) -> str:
    return json.dumps(parameters, sort_keys=True, separators=(",", ":"))


def generate_parameter_sets(count: int, search_seed: int) -> list[dict]:
    if count <= 0:
        raise ValueError("trials must be positive")
    rng = random.Random(search_seed)
    parameter_sets: list[dict] = []
    seen: set[str] = set()
    max_attempts = count * 100
    for _ in range(max_attempts):
        parameters = sample_parameters(rng)
        key = parameter_key(parameters)
        if key in seen:
            continue
        seen.add(key)
        parameter_sets.append(parameters)
        if len(parameter_sets) == count:
            return parameter_sets
    raise RuntimeError(f"Could not generate {count} unique trials")


def build_trial_config(
    base_config: dict,
    *,
    trial_index: int,
    parameters: dict,
    search_seed: int,
    base_config_path: Path,
    base_config_sha256: str,
) -> dict:
    config = copy.deepcopy(base_config)
    trial_tag = f"trial_{trial_index:03d}"
    config["run_name"] = trial_tag
    config["model"].update(
        {
            "d_model": parameters["d_model"],
            "nhead": parameters["nhead"],
            "num_layers": parameters["num_layers"],
            "dim_feedforward": parameters["dim_feedforward"],
            "dropout": parameters["dropout"],
        }
    )
    config["training"].update(
        {
            "seed": 42,
            "learning_rate": parameters["learning_rate"],
            "weight_decay": parameters["weight_decay"],
            "batch_size": parameters["batch_size"],
            "orientation_loss_weight": parameters[
                "orientation_loss_weight"
            ],
            "receiving_hand_loss_weight": parameters[
                "receiving_hand_loss_weight"
            ],
        }
    )
    config["hyperparameter_search"] = {
        "schema_version": 1,
        "search_version": SEARCH_VERSION,
        "stage": "A",
        "trial_index": trial_index,
        "trial_tag": trial_tag,
        "search_seed": search_seed,
        "training_seed": 42,
        "selection_split": "validation",
        "test_evaluation": "disabled",
        "selection_rule": SELECTION_RULE,
        "base_config": relative_path(base_config_path),
        "base_config_sha256": base_config_sha256,
        "parameters": parameters,
    }
    return config


def write_json(path: Path, value: dict) -> str:
    encoded = (json.dumps(value, indent=2, ensure_ascii=False) + "\n").encode(
        "utf-8"
    )
    path.write_bytes(encoded)
    return sha256_bytes(encoded)


def main() -> int:
    args = parse_args()
    base_config_path = project_path(args.base_config)
    output_dir = project_path(args.output_dir)
    if not base_config_path.is_file():
        raise FileNotFoundError(base_config_path)
    if args.trials != 24:
        print(
            "WARNING: the pre-registered Stage-A protocol specifies 24 trials; "
            f"generating {args.trials}"
        )
    base_bytes = base_config_path.read_bytes()
    base_config = json.loads(base_bytes)
    if base_config.get("model_type") != (
        "hierarchical_residual_pose_transformer_v2"
    ):
        raise ValueError("Base config is not Residual Transformer v2")
    output_dir.mkdir(parents=True, exist_ok=True)
    expected_paths = [
        output_dir / f"trial_{index:03d}.json" for index in range(args.trials)
    ]
    existing = [path for path in expected_paths if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            f"{len(existing)} trial configs already exist; use --overwrite"
        )

    parameter_sets = generate_parameter_sets(args.trials, args.search_seed)
    base_sha256 = sha256_bytes(base_bytes)
    records = []
    for index, (path, parameters) in enumerate(
        zip(expected_paths, parameter_sets)
    ):
        config = build_trial_config(
            base_config,
            trial_index=index,
            parameters=parameters,
            search_seed=args.search_seed,
            base_config_path=base_config_path,
            base_config_sha256=base_sha256,
        )
        config_sha256 = write_json(path, config)
        records.append(
            {
                "trial_index": index,
                "trial_tag": f"trial_{index:03d}",
                "config": relative_path(path),
                "config_sha256": config_sha256,
                **parameters,
            }
        )

    manifest = {
        "schema_version": 1,
        "search_version": SEARCH_VERSION,
        "stage": "A",
        "search_seed": args.search_seed,
        "training_seed": 42,
        "trial_count": args.trials,
        "selection_split": "validation",
        "test_evaluation": "disabled",
        "selection_rule": SELECTION_RULE,
        "base_config": relative_path(base_config_path),
        "base_config_sha256": base_sha256,
        "trials": records,
    }
    write_json(output_dir / "manifest.json", manifest)
    with (output_dir / "trials.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    print(f"Generated {len(records)} deterministic trials in {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
