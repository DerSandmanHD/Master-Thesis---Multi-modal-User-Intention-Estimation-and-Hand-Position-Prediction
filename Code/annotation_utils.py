#!/usr/bin/env python3
"""Shared schema and validation helpers for manual sequence annotations."""

from __future__ import annotations

import csv
import tempfile
from pathlib import Path


COMMANDS = ("START", "SECOND", "DONE", "THIRD")
VALID_DECISIONS = {"", "accept_auto", "manual_fix", "exclude", "uncertain"}
VALID_RECEIVING_HANDS = {"", "left", "right", "both", "uncertain"}
VALID_ANNOTATION_CONFIDENCE = {"", "certain", "uncertain"}
OBJECT_MARKER_IDS = set(range(6, 15))

REVIEW_FIELDS = [
    "sequence_id",
    "decision",
    "auto_start_s",
    "auto_second_s",
    "auto_done_s",
    "auto_third_s",
    "manual_start_s",
    "manual_second_s",
    "manual_done_s",
    "manual_third_s",
    "target_object_id",
    "receiving_hand",
    "annotation_confidence",
    "missing_commands",
    "status",
    "next_action",
    "notes",
]


def clean_text(value) -> str:
    return "" if value is None else str(value).strip()


def normalize_decision(value) -> str:
    decision = clean_text(value).lower()
    if decision not in VALID_DECISIONS:
        raise ValueError(f"invalid review decision: {value!r}")
    return decision


def parse_target_object_id(value) -> int | None:
    text = clean_text(value)
    if not text:
        return None
    try:
        marker_id = int(text)
    except ValueError as exc:
        raise ValueError(f"target_object_id must be an integer from 6 through 14: {value!r}") from exc
    if marker_id not in OBJECT_MARKER_IDS:
        raise ValueError(f"target_object_id must be an object marker ID from 6 through 14: {marker_id}")
    return marker_id


def normalize_receiving_hand(value) -> str:
    hand = clean_text(value).lower()
    if hand not in VALID_RECEIVING_HANDS:
        raise ValueError(
            "receiving_hand must be left, right, both, uncertain, or empty: "
            f"{value!r}"
        )
    return hand


def normalize_annotation_confidence(value) -> str:
    confidence = clean_text(value).lower()
    if confidence not in VALID_ANNOTATION_CONFIDENCE:
        raise ValueError(
            "annotation_confidence must be certain, uncertain, or empty: "
            f"{value!r}"
        )
    return confidence


def normalize_review_row(row: dict) -> dict:
    normalized = {field: clean_text(row.get(field, "")) for field in REVIEW_FIELDS}
    normalized["sequence_id"] = clean_text(row.get("sequence_id"))
    normalized["decision"] = normalize_decision(row.get("decision"))
    target_object_id = parse_target_object_id(row.get("target_object_id"))
    normalized["target_object_id"] = "" if target_object_id is None else str(target_object_id)
    normalized["receiving_hand"] = normalize_receiving_hand(row.get("receiving_hand"))
    normalized["annotation_confidence"] = normalize_annotation_confidence(
        row.get("annotation_confidence")
    )
    return normalized


def read_review_rows(path: Path) -> dict[str, dict]:
    path = Path(path)
    if not path.exists():
        return {}
    rows = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row_number, row in enumerate(csv.DictReader(handle), start=2):
            try:
                normalized = normalize_review_row(row)
            except ValueError as exc:
                raise ValueError(f"{path}:{row_number}: {exc}") from exc
            sequence_id = normalized["sequence_id"]
            if not sequence_id:
                continue
            if sequence_id in rows:
                raise ValueError(f"{path}:{row_number}: duplicate sequence_id: {sequence_id}")
            rows[sequence_id] = normalized
    return rows


def write_review_rows(path: Path, rows_by_sequence: dict[str, dict]) -> None:
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as handle:
            temp_path = Path(handle.name)
            writer = csv.DictWriter(handle, fieldnames=REVIEW_FIELDS)
            writer.writeheader()
            for sequence_id in sorted(rows_by_sequence):
                row = dict(rows_by_sequence[sequence_id])
                row["sequence_id"] = sequence_id
                writer.writerow(normalize_review_row(row))
        temp_path.replace(path)
        path.chmod(0o644)
    except Exception:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
        raise
