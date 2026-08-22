#!/usr/bin/env python3
"""Generate deterministic validation-only terminal end-pose v2 trials."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import random
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BASE = Path("Training/configs/models/residual_transformer_endpose_v2.json")
DEFAULT_OUTPUT = Path("Training/configs/endpose_v2_search")
SEARCH_VERSION = "terminal_endpose_v2_random_search_v1"
SELECTION_RULE = (
    "minimize mean validation terminal position error across confirmation seeds; "
    "retain candidates within 0.5 cm; then minimize validation orientation error; "
    "then maximize validation intent and receiving-hand macro-F1; never use test"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--trials", type=int, default=12)
    parser.add_argument("--search-seed", type=int, default=20260810)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def relative(path: Path) -> str:
    return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sample(rng: random.Random) -> dict:
    d_model = rng.choice((32, 64))
    return {
        "learning_rate": float(f"{10 ** rng.uniform(-4.2, -3.0):.12g}"),
        "pose_loss_weight": rng.choice((1.0, 2.0, 4.0, 8.0)),
        "orientation_loss_weight": rng.choice((0.1, 0.25, 0.5, 1.0)),
        "auxiliary_pose_loss_weight": rng.choice((0.0, 0.1, 0.25, 0.5, 1.0)),
        "dropout": rng.choice((0.05, 0.15, 0.30)),
        "d_model": d_model,
        "nhead": rng.choice((4, 8)),
        "num_layers": rng.choice((1, 2)),
        "dim_feedforward": rng.choice((128, 256)),
        "batch_size": rng.choice((32, 64)),
    }


def main() -> int:
    args = parse_args()
    if args.trials <= 0:
        raise ValueError("trials must be positive")
    base_path = resolve(args.base_config)
    output = resolve(args.output_dir)
    base_bytes = base_path.read_bytes()
    base = json.loads(base_bytes)
    output.mkdir(parents=True, exist_ok=True)
    expected = [output / f"trial_{index:03d}.json" for index in range(args.trials)]
    existing = [path for path in expected if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(f"{len(existing)} trial configs already exist")

    rng = random.Random(args.search_seed)
    paired_anchor = {
        "learning_rate": 0.000381605632501,
        "pose_loss_weight": 2.0,
        "orientation_loss_weight": 0.25,
        "auxiliary_pose_loss_weight": 0.0,
        "dropout": 0.15,
        "d_model": 32,
        "nhead": 8,
        "num_layers": 1,
        "dim_feedforward": 256,
        "batch_size": 64,
    }
    values = [
        paired_anchor,
        {**paired_anchor, "auxiliary_pose_loss_weight": 0.25},
    ][: args.trials]
    seen = {json.dumps(value, sort_keys=True) for value in values}
    while len(values) < args.trials:
        parameters = sample(rng)
        key = json.dumps(parameters, sort_keys=True)
        if key not in seen:
            seen.add(key)
            values.append(parameters)

    records = []
    for index, (path, parameters) in enumerate(zip(expected, values)):
        trial_tag = f"trial_{index:03d}"
        config = copy.deepcopy(base)
        config["run_name"] = trial_tag
        for key in ("dropout", "d_model", "nhead", "num_layers", "dim_feedforward"):
            config["model"][key] = parameters[key]
        for key in (
            "learning_rate",
            "pose_loss_weight",
            "orientation_loss_weight",
            "auxiliary_pose_loss_weight",
            "batch_size",
        ):
            config["training"][key] = parameters[key]
        config["hyperparameter_search"] = {
            "schema_version": 1,
            "search_version": SEARCH_VERSION,
            "stage": "A",
            "trial_index": index,
            "trial_tag": trial_tag,
            "search_seed": args.search_seed,
            "training_seed": 42,
            "selection_split": "validation",
            "test_evaluation": "disabled",
            "selection_rule": SELECTION_RULE,
            "base_config": relative(base_path),
            "base_config_sha256": digest(base_bytes),
            "parameters": parameters,
        }
        encoded = (json.dumps(config, indent=2, ensure_ascii=False) + "\n").encode()
        path.write_bytes(encoded)
        records.append(
            {
                "trial_index": index,
                "trial_tag": trial_tag,
                "config": relative(path),
                "config_sha256": digest(encoded),
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
        "position_tolerance_cm": 0.5,
        "selection_rule": SELECTION_RULE,
        "base_config": relative(base_path),
        "base_config_sha256": digest(base_bytes),
        "trials": records,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"
    )
    with (output / "trials.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(records[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(records)
    print(f"Generated {len(records)} endpose-v2 trials in {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
