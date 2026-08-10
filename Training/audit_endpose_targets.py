#!/usr/bin/env python3
"""Audit robust terminal receiving-hand targets before end-pose training."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from data import INTENTION_TO_ID, prepare_data, sha256_file


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SPLITS = ("train", "validation", "test")
POSE_COMPONENTS = ("x_m", "y_m", "z_m", "qx", "qy", "qz", "qw")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("Training/configs/models/residual_transformer_endpose_v1.json"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--require-sufficient",
        action="store_true",
        help="Exit with status 2 when the predeclared audit requirements fail.",
    )
    return parser.parse_args()


def resolve(path: Path) -> Path:
    path = path.expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def sequence_row(split_name: str, record) -> dict:
    metadata = dict(record.pose_target_metadata or {})
    pose = metadata.pop("pose", None)
    row = {
        "split": split_name,
        "participant": record.participant,
        "sequence_id": record.sequence_id,
        **metadata,
    }
    row["reasons"] = ";".join(row.get("reasons") or [])
    for component, value in zip(POSE_COMPONENTS, pose or [None] * 7):
        row[f"target_{component}"] = value
    return row


def split_summary(dataset) -> dict:
    handover_id = INTENTION_TO_ID["handover"]
    handover_windows = sum(
        int(dataset.records[record_index].intentions[endpoint]) == handover_id
        for record_index, endpoint in dataset.indices
    )
    target_windows = sum(
        bool(dataset.records[record_index].pose_valid[endpoint])
        for record_index, endpoint in dataset.indices
    )
    residual_windows = dataset.residual_pose_count()
    audit = dataset.pose_target_sequence_audit()
    return {
        **audit,
        "handover_windows": int(handover_windows),
        "target_windows": int(target_windows),
        "target_window_coverage": (
            target_windows / handover_windows if handover_windows else None
        ),
        "residual_pose_windows": int(residual_windows),
        "residual_reference_coverage": (
            residual_windows / target_windows if target_windows else None
        ),
    }


def main() -> int:
    args = parse_args()
    config_path = resolve(args.config).resolve()
    output_dir = resolve(args.output_dir).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    data_config = dict(config["data"])
    master_dir = resolve(Path(data_config["master_dir"])).resolve()
    data_config["master_dir"] = str(master_dir)
    seed = int(config["training"]["seed"])
    bundle = prepare_data(data_config, seed=seed)

    rows = []
    summaries = {}
    for split_name in SPLITS:
        dataset = getattr(bundle, split_name)
        summaries[split_name] = split_summary(dataset)
        rows.extend(sequence_row(split_name, record) for record in dataset.records)
    details = pd.DataFrame(rows).sort_values(
        ["split", "participant", "sequence_id"]
    )
    applicable = details.loc[details["status"] != "not_applicable"]
    accepted = applicable.loc[applicable["eligible"].astype(bool)]
    coverage = len(accepted) / len(applicable) if len(applicable) else 0.0
    rejection_reasons = Counter()
    for value in applicable.loc[~applicable["eligible"].astype(bool), "reasons"]:
        rejection_reasons.update(part for part in str(value).split(";") if part)

    requirements = dict(config.get("audit_requirements", {}))
    checks = {
        "overall_target_sequence_coverage": {
            "actual": coverage,
            "minimum": float(
                requirements.get("minimum_overall_target_sequence_coverage", 0.0)
            ),
        }
    }
    for split_name in SPLITS:
        key = f"minimum_accepted_{split_name}_sequences"
        checks[f"accepted_{split_name}_sequences"] = {
            "actual": summaries[split_name]["accepted_handover_sequences"],
            "minimum": int(requirements.get(key, 1)),
        }
    for check in checks.values():
        check["passed"] = check["actual"] >= check["minimum"]
    sufficient = all(check["passed"] for check in checks.values())

    report = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "sufficient" if sufficient else "insufficient",
        "training_authorized_by_audit": sufficient,
        "config": str(config_path),
        "config_sha256": sha256_file(config_path),
        "dataset_tag": config.get("source_hyperparameters", {}).get("dataset_tag"),
        "dataset_contract": bundle.provenance.get("dataset_contract"),
        "dataset_content_fingerprint": bundle.provenance[
            "dataset_content_fingerprint"
        ],
        "source_content_fingerprint": bundle.provenance[
            "source_content_fingerprint"
        ],
        "sequence_fingerprint": bundle.split_metadata["dataset_filter"][
            "sequence_fingerprint"
        ],
        "pose_target_definition": bundle.split_metadata["pose_target"],
        "split_participants": bundle.split_metadata["participants"],
        "selected_sequences": int(len(details)),
        "handover_sequences": int(len(applicable)),
        "accepted_handover_sequences": int(len(accepted)),
        "rejected_handover_sequences": int(len(applicable) - len(accepted)),
        "target_sequence_coverage": coverage,
        "rejection_reason_counts": dict(sorted(rejection_reasons.items())),
        "splits": summaries,
        "requirements": requirements,
        "requirement_checks": checks,
        "test_labels_used_for_threshold_tuning": False,
        "notes": [
            "Quality thresholds were fixed in the config before this audit.",
            "Rejected sequences remain available for intention and receiving-hand classification but contribute no pose loss or pose metric.",
            "No model training or checkpoint selection is performed by this audit.",
        ],
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    details_path = output_dir / "endpose_target_audit.csv"
    report_path = output_dir / "endpose_target_audit.json"
    details.to_csv(details_path, index=False)
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(
        f"End-pose targets: accepted={len(accepted)}/{len(applicable)} "
        f"({coverage:.1%}); status={report['status']}"
    )
    for split_name in SPLITS:
        split = summaries[split_name]
        print(
            f"  {split_name}: sequences="
            f"{split['accepted_handover_sequences']}/{split['handover_sequences']}, "
            f"target windows={split['target_windows']}/{split['handover_windows']}, "
            f"residual windows={split['residual_pose_windows']}"
        )
    if rejection_reasons:
        print(f"Rejections: {dict(sorted(rejection_reasons.items()))}")
    print(f"Report:  {report_path}")
    print(f"Details: {details_path}")
    return 2 if args.require_sufficient and not sufficient else 0


if __name__ == "__main__":
    raise SystemExit(main())
