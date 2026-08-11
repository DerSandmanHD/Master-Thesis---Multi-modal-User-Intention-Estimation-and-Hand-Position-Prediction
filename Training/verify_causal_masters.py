#!/usr/bin/env python3
"""Fail fast unless every selected master uses the causal observation join."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any


EXPECTED_ALIGNMENT_VERSION = "causal_backward_device_time_v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def truthy(value: Any) -> bool:
    return str(value).strip().casefold() == "true"


def sequence_fingerprint(sequence_ids: list[str]) -> str:
    return hashlib.sha256(
        "\n".join(sorted(sequence_ids)).encode("utf-8")
    ).hexdigest()


def selected_sequence_ids(manifest: Path) -> list[str]:
    with manifest.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    selected = []
    for row in rows:
        if (
            truthy(row.get("include_in_training"))
            and str(row.get("status", "")).strip() == "valid"
            and str(row.get("next_action", "")).strip()
            == "ready_for_master_merge"
            and truthy(row.get("master_csv_exists"))
        ):
            sequence_id = str(row.get("sequence_id", "")).strip()
            if not sequence_id:
                raise ValueError("Selected manifest row has no sequence_id")
            selected.append(sequence_id)
    if len(selected) != len(set(selected)):
        raise ValueError("Selected manifest contains duplicate sequence IDs")
    if not selected:
        raise ValueError("Manifest selects no causal master datasets")
    return sorted(selected)


def verify_dataset(
    master_dir: Path,
    manifest: Path,
    *,
    expected_alignment_version: str = EXPECTED_ALIGNMENT_VERSION,
    expected_sequence_fingerprint: str | None = None,
) -> dict[str, Any]:
    master_dir = master_dir.expanduser().resolve()
    manifest = manifest.expanduser().resolve()
    sequence_ids = selected_sequence_ids(manifest)
    fingerprint = sequence_fingerprint(sequence_ids)
    if (
        expected_sequence_fingerprint
        and fingerprint != expected_sequence_fingerprint
    ):
        raise ValueError(
            "Selected sequence fingerprint differs from the frozen protocol: "
            f"{fingerprint} != {expected_sequence_fingerprint}"
        )
    identities = []
    for sequence_id in sequence_ids:
        master = master_dir / f"{sequence_id}_master.csv"
        report_path = master_dir / f"{sequence_id}_master_report.json"
        if not master.is_file() or not report_path.is_file():
            raise FileNotFoundError(
                f"Missing causal master/report pair for {sequence_id}"
            )
        row_count = 0
        versions: set[str] = set()
        with master.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if "observation_alignment_version" not in (reader.fieldnames or []):
                raise ValueError(
                    f"{master.name} predates causal observation alignment"
                )
            for row in reader:
                row_count += 1
                versions.add(
                    str(row.get("observation_alignment_version", "")).strip()
                )
        if row_count == 0 or versions != {expected_alignment_version}:
            raise ValueError(
                f"{master.name} has invalid observation alignment values: "
                f"{sorted(versions)}"
            )
        report = json.loads(report_path.read_text(encoding="utf-8"))
        alignment = report.get("observation_alignment")
        if not isinstance(alignment, dict) or alignment.get("version") != (
            expected_alignment_version
        ):
            raise ValueError(f"{report_path.name} has stale alignment metadata")
        if alignment.get("future_source_captures_allowed") is not False:
            raise ValueError(
                f"{report_path.name} does not forbid future source captures"
            )
        identities.append(
            {
                "sequence_id": sequence_id,
                "rows": row_count,
                "master_sha256": sha256_file(master),
                "master_report_sha256": sha256_file(report_path),
            }
        )
    source_payload = {
        "master_files": [
            {
                "sequence_id": row["sequence_id"],
                "sha256": row["master_sha256"],
            }
            for row in identities
        ],
        "manifest_sha256": sha256_file(manifest),
    }
    source_content_fingerprint = hashlib.sha256(
        json.dumps(
            source_payload, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()
    return {
        "schema_version": 1,
        "verification_protocol": "causal_master_preflight_v1",
        "passed": True,
        "observation_alignment_version": expected_alignment_version,
        "future_source_captures_allowed": False,
        "master_dir": str(master_dir),
        "manifest": str(manifest),
        "manifest_sha256": sha256_file(manifest),
        "selected_sequences": len(sequence_ids),
        "sequence_ids": sequence_ids,
        "sequence_fingerprint": fingerprint,
        "source_content_fingerprint": source_content_fingerprint,
        "masters": identities,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--master-dir", type=Path, default=Path("Data_collection/master_datasets")
    )
    parser.add_argument(
        "--manifest", type=Path, default=Path("Data_collection/dataset_manifest.csv")
    )
    parser.add_argument(
        "--expected-alignment-version", default=EXPECTED_ALIGNMENT_VERSION
    )
    parser.add_argument("--expected-sequence-fingerprint", default=None)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        report = verify_dataset(
            args.master_dir,
            args.manifest,
            expected_alignment_version=args.expected_alignment_version,
            expected_sequence_fingerprint=args.expected_sequence_fingerprint,
        )
        if args.output is not None:
            output = args.output.expanduser().resolve()
            output.parent.mkdir(parents=True, exist_ok=True)
            if output.exists():
                raise FileExistsError(
                    f"Refusing to overwrite causal-master report: {output}"
                )
            output.write_text(
                json.dumps(report, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
    except (FileNotFoundError, OSError, TypeError, ValueError) as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}")
        return 2
    print(
        "Causal masters verified: "
        f"{report['selected_sequences']} sequences, "
        f"source={report['source_content_fingerprint']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
