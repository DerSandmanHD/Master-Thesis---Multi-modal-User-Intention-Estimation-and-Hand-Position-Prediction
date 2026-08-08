#!/usr/bin/env python3
"""Freeze CLIP experiment configs on the validation-selected sensor settings."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from data import sha256_file


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_NAMES = (
    "residual_v2_clip_only.json",
    "residual_v2_sensor_plus_clip.json",
    "residual_v2_sensor_plus_random.json",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selected-config", type=Path, required=True)
    parser.add_argument(
        "--visual-config-dir",
        type=Path,
        default=Path("Training/configs/visual"),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve(path: Path) -> Path:
    path = path.expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def relative_or_absolute(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def main() -> int:
    args = parse_args()
    selected_path = resolve(args.selected_config).resolve()
    config_dir = resolve(args.visual_config_dir).resolve()
    selected = json.loads(selected_path.read_text(encoding="utf-8"))
    provenance = selected.get("hyperparameter_selection", {})
    if provenance.get("selection_split") != "validation":
        raise ValueError("Selected config was not frozen on validation")
    if provenance.get("test_evaluation_used_for_selection") is not False:
        raise ValueError("Selected config does not prove test-independent selection")

    selected_hash = sha256_file(selected_path)
    for name in CONFIG_NAMES:
        target = config_dir / name
        template = json.loads(target.read_text(encoding="utf-8"))
        if target.exists() and not args.overwrite:
            raise FileExistsError(f"Pass --overwrite to update {target}")
        visual_embeddings = copy.deepcopy(
            template["data"]["visual_embeddings"]
        )
        output = copy.deepcopy(selected)
        output["run_name"] = template["run_name"]
        output["data"] = copy.deepcopy(selected["data"])
        output["data"]["visual_embeddings"] = visual_embeddings
        output.pop("hyperparameter_search", None)
        visual_experiment = copy.deepcopy(template["visual_experiment"])
        visual_experiment.update(
            {
                "sensor_hyperparameters_frozen": True,
                "sensor_hyperparameter_config": relative_or_absolute(
                    selected_path
                ),
                "sensor_hyperparameter_config_sha256": selected_hash,
                "sensor_hyperparameter_source_trial": provenance.get(
                    "source_trial"
                ),
            }
        )
        output["visual_experiment"] = visual_experiment
        target.write_text(
            json.dumps(output, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(f"Frozen visual config: {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
