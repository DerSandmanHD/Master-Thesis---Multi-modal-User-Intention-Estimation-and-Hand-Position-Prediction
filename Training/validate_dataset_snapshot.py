#!/usr/bin/env python3
"""Validate and fingerprint a frozen master-dataset snapshot.

The validator is read-only with respect to master CSVs. It writes only the
requested JSON report and never changes training data or checkpoints.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MASTER_SUFFIX = "_master.csv"
ARTIFACT_FILES = (
    "config.json",
    "data_metadata.json",
    "best_intention_model.pt",
    "best_pose_model.pt",
)


def resolve_path(path: Path) -> Path:
    path = path.expanduser()
    return path.resolve() if path.is_absolute() else (Path.cwd() / path).resolve()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def truthy(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def sequence_id_from_path(path: Path) -> str:
    if not path.name.endswith(MASTER_SUFFIX):
        raise ValueError(f"Not a master CSV: {path}")
    return path.name[: -len(MASTER_SUFFIX)]


def read_manifest(
    path: Path,
    *,
    allowed_statuses: set[str],
    allowed_next_actions: set[str],
) -> tuple[list[dict[str, str]], set[str], list[str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    required = {
        "sequence_id",
        "include_in_training",
        "status",
        "next_action",
        "master_csv_exists",
    }
    available = set(rows[0]) if rows else set()
    missing = sorted(required - available)
    if missing:
        raise ValueError(
            f"Manifest {path} is missing columns: {', '.join(missing)}"
        )

    eligible: set[str] = set()
    seen: set[str] = set()
    duplicates: list[str] = []
    for row in rows:
        sequence_id = row["sequence_id"].strip()
        if not sequence_id:
            raise ValueError(f"Manifest {path} contains an empty sequence_id")
        if sequence_id in seen:
            duplicates.append(sequence_id)
        seen.add(sequence_id)
        if (
            truthy(row["include_in_training"])
            and row["status"].strip() in allowed_statuses
            and row["next_action"].strip() in allowed_next_actions
            and truthy(row["master_csv_exists"])
        ):
            eligible.add(sequence_id)
    return rows, eligible, sorted(set(duplicates))


def first_row_and_header(path: Path) -> tuple[set[str], dict[str, str] | None]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        header = set(reader.fieldnames or [])
        return header, next(reader, None)


def relative_file_records(
    paths: Iterable[Path],
    root: Path,
    *,
    calculate_hashes: bool,
) -> list[dict[str, object]]:
    records = []
    for path in sorted(paths):
        record: dict[str, object] = {
            "path": str(path.relative_to(root)),
            "size_bytes": path.stat().st_size,
        }
        if calculate_hashes:
            record["sha256"] = sha256_file(path)
        records.append(record)
    return records


def validate_snapshot(
    snapshot_dir: Path,
    artifacts_dir: Path,
    *,
    calculate_hashes: bool = True,
) -> dict:
    snapshot_dir = resolve_path(snapshot_dir)
    artifacts_dir = resolve_path(artifacts_dir)
    master_dir = snapshot_dir / "master_datasets"
    manifest_path = snapshot_dir / "dataset_manifest.csv"
    metadata_path = artifacts_dir / "data_metadata.json"
    config_path = artifacts_dir / "config.json"

    required = [master_dir, manifest_path, metadata_path, config_path]
    missing_required = [str(path) for path in required if not path.exists()]
    if missing_required:
        raise FileNotFoundError(
            "Snapshot validation inputs are missing: "
            + ", ".join(missing_required)
        )

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    config = json.loads(config_path.read_text(encoding="utf-8"))
    filter_config = config.get("data", {}).get("manifest_filter", {})
    allowed_statuses = {
        str(value).strip()
        for value in filter_config.get("allowed_statuses", ["valid"])
    }
    allowed_actions = {
        str(value).strip()
        for value in filter_config.get(
            "allowed_next_actions", ["ready_for_master_merge"]
        )
    }

    manifest_rows, eligible_ids, duplicate_manifest_ids = read_manifest(
        manifest_path,
        allowed_statuses=allowed_statuses,
        allowed_next_actions=allowed_actions,
    )
    master_paths = sorted(master_dir.glob(f"*{MASTER_SUFFIX}"))
    master_by_id = {sequence_id_from_path(path): path for path in master_paths}
    duplicate_master_ids = (
        []
        if len(master_by_id) == len(master_paths)
        else ["duplicate_filename_or_sequence_id"]
    )
    master_ids = set(master_by_id)

    split_sequences = metadata.get("split", {}).get("sequences", {})
    expected_by_split = {
        split: [str(value) for value in split_sequences.get(split, [])]
        for split in ("train", "validation", "test")
    }
    expected_ids = {
        sequence_id
        for sequence_ids in expected_by_split.values()
        for sequence_id in sequence_ids
    }

    feature_columns = set(metadata.get("feature_columns", []))
    schema_issues: dict[str, list[str]] = {}
    identity_issues: dict[str, str] = {}
    empty_master_ids: list[str] = []
    for sequence_id in sorted(expected_ids & master_ids):
        header, first_row = first_row_and_header(master_by_id[sequence_id])
        missing_columns = sorted(
            {"timestamp_ns", "sequence_id", "participant", *feature_columns}
            - header
        )
        if missing_columns:
            schema_issues[sequence_id] = missing_columns
        if first_row is None:
            empty_master_ids.append(sequence_id)
        elif first_row.get("sequence_id", "").strip() != sequence_id:
            identity_issues[sequence_id] = first_row.get("sequence_id", "")

    split_report = {}
    for split, sequence_ids in expected_by_split.items():
        missing = sorted(set(sequence_ids) - master_ids)
        ineligible = sorted(set(sequence_ids) - eligible_ids)
        split_report[split] = {
            "expected_sequences": len(sequence_ids),
            "present_sequences": len(sequence_ids) - len(missing),
            "eligible_sequences": len(sequence_ids) - len(ineligible),
            "missing_sequence_ids": missing,
            "ineligible_sequence_ids": ineligible,
        }

    unexpected_eligible = sorted(eligible_ids - expected_ids)
    expected_not_eligible = sorted(expected_ids - eligible_ids)
    eligible_without_master = sorted(eligible_ids - master_ids)
    masters_not_in_manifest = sorted(
        master_ids
        - {
            row["sequence_id"].strip()
            for row in manifest_rows
            if row.get("sequence_id")
        }
    )

    artifact_paths = [
        artifacts_dir / name
        for name in ARTIFACT_FILES
        if (artifacts_dir / name).is_file()
    ]
    snapshot_files = [
        path for path in snapshot_dir.rglob("*") if path.is_file()
    ]
    # Do not make a newly generated report part of its own fingerprint.
    snapshot_files = [
        path
        for path in snapshot_files
        if path.name != "snapshot_validation.json"
    ]

    blocking_issues = {
        "duplicate_manifest_sequence_ids": duplicate_manifest_ids,
        "duplicate_master_sequence_ids": duplicate_master_ids,
        "eligible_without_master": eligible_without_master,
        "masters_not_in_manifest": masters_not_in_manifest,
        "expected_not_eligible": expected_not_eligible,
        "unexpected_eligible_vs_artifact_split": unexpected_eligible,
        "schema_issues": schema_issues,
        "identity_issues": identity_issues,
        "empty_master_sequence_ids": empty_master_ids,
    }
    valid = not any(bool(value) for value in blocking_issues.values())
    sequence_fingerprint = hashlib.sha256(
        "\n".join(sorted(eligible_ids)).encode("utf-8")
    ).hexdigest()

    return {
        "status": "valid" if valid else "invalid",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "snapshot_dir": str(snapshot_dir),
        "artifacts_dir": str(artifacts_dir),
        "hashes_calculated": calculate_hashes,
        "manifest": {
            "path": str(manifest_path),
            "rows": len(manifest_rows),
            "sha256": (
                sha256_file(manifest_path) if calculate_hashes else None
            ),
            "allowed_statuses": sorted(allowed_statuses),
            "allowed_next_actions": sorted(allowed_actions),
            "eligible_sequences": len(eligible_ids),
            "sequence_fingerprint": sequence_fingerprint,
        },
        "masters": {
            "files_found": len(master_paths),
            "total_size_bytes": sum(path.stat().st_size for path in master_paths),
            "eligible_files": len(eligible_ids & master_ids),
            "excluded_files": len(master_ids - eligible_ids),
        },
        "artifact_split": {
            "total_sequences": len(expected_ids),
            "splits": split_report,
        },
        "blocking_issues": blocking_issues,
        "snapshot_files": relative_file_records(
            snapshot_files,
            snapshot_dir,
            calculate_hashes=calculate_hashes,
        ),
        "deployment_artifacts": relative_file_records(
            artifact_paths,
            artifacts_dir,
            calculate_hashes=calculate_hashes,
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--snapshot-dir",
        type=Path,
        default=Path(
            "Data_collection/final_dataset_snapshot_20260729"
        ),
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=Path("Training/final_clean_v1_residual_v2_seed44"),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Defaults to <snapshot-dir>/snapshot_validation.json.",
    )
    parser.add_argument(
        "--skip-file-hashes",
        action="store_true",
        help="Run structural checks without calculating SHA-256 per file.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    snapshot_dir = resolve_path(args.snapshot_dir)
    output_path = (
        resolve_path(args.output_json)
        if args.output_json is not None
        else snapshot_dir / "snapshot_validation.json"
    )
    report = validate_snapshot(
        snapshot_dir,
        args.artifacts_dir,
        calculate_hashes=not args.skip_file_hashes,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(
        f"Snapshot {report['status']}: "
        f"masters={report['masters']['files_found']}, "
        f"eligible={report['manifest']['eligible_sequences']}, "
        f"artifact_sequences={report['artifact_split']['total_sequences']}"
    )
    print(f"Report: {output_path}")
    return 0 if report["status"] == "valid" else 1


if __name__ == "__main__":
    raise SystemExit(main())
