#!/usr/bin/env python3
"""Participant-disjoint split summaries, diagnostics, and deterministic plans.

This module deliberately consumes dataset labels and counts only.  It does not
accept model predictions, losses, validation scores, or test metrics as inputs
to split generation.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence


SPLIT_NAMES = ("train", "validation", "test")
PHASE_NAMES = ("continue", "fetch", "transition", "handover")
HAND_NAMES = ("left", "right", "both", "unknown")
OBJECTIVE_INPUTS = (
    "participant_count",
    "sequence_count",
    "receiving_hand_sequence_counts",
    "target_object_sequence_counts",
    "intention_or_phase_distribution",
)


@dataclass(frozen=True)
class SequenceSummary:
    sequence_id: str
    participant: str
    receiving_hand: str
    target_object_id: int | None
    phase_distribution: dict[str, float]
    phase_unit: str
    phase_scope: str = "unknown"
    row_count: int | None = None
    source: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def canonical_participant(value: object) -> str:
    participant = str(value).strip()
    if not participant:
        raise ValueError("Participant name must not be empty")
    return participant.casefold().capitalize()


def normalize_split_name(value: object) -> str:
    normalized = str(value).strip().casefold()
    aliases = {"val": "validation", "valid": "validation"}
    normalized = aliases.get(normalized, normalized)
    if normalized not in SPLIT_NAMES:
        raise ValueError(f"Unknown split name: {value!r}")
    return normalized


def normalize_hand(value: object, receiving_hand_id: object = None) -> str:
    normalized = str(value or "").strip().casefold()
    aliases = {"0": "left", "1": "right", "2": "both", "-1": "unknown"}
    normalized = aliases.get(normalized, normalized)
    if normalized in HAND_NAMES:
        return normalized
    id_value = str(receiving_hand_id or "").strip()
    return aliases.get(id_value, "unknown")


def parse_target_object(value: object) -> int | None:
    text = str(value or "").strip()
    if not text or text.casefold() in {"nan", "none", "unknown"}:
        return None
    try:
        parsed = int(float(text))
    except ValueError:
        return None
    return parsed if parsed >= 0 else None


def truthy(value: object) -> bool:
    return str(value).strip().casefold() in {"1", "true", "yes", "y"}


def _finite_nonnegative(value: object) -> float | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        parsed = float(text)
    except ValueError:
        return None
    return parsed if math.isfinite(parsed) and parsed >= 0.0 else None


def sequence_id_from_master(path: Path) -> str:
    suffix = "_master.csv"
    if not path.name.endswith(suffix):
        raise ValueError(f"Not a master CSV: {path}")
    return path.name[: -len(suffix)]


def eligible_master_paths(
    master_dir: Path,
    *,
    manifest_path: Path | None = None,
    allowed_statuses: Iterable[str] = ("valid",),
    allowed_next_actions: Iterable[str] = ("ready_for_master_merge",),
    strict: bool = True,
) -> tuple[list[Path], dict]:
    """Select master files with the same eligibility fields used by training."""

    master_dir = Path(master_dir).expanduser().resolve()
    files = sorted(master_dir.glob("*_master.csv"))
    by_sequence = {sequence_id_from_master(path): path for path in files}
    if len(by_sequence) != len(files):
        raise ValueError("Duplicate master sequence IDs")
    if manifest_path is None:
        return files, {
            "manifest_enabled": False,
            "master_files_found": len(files),
            "selected_sequences": len(files),
            "sequence_ids": sorted(by_sequence),
        }

    manifest_path = Path(manifest_path).expanduser().resolve()
    with manifest_path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    required = {
        "sequence_id",
        "include_in_training",
        "status",
        "next_action",
        "master_csv_exists",
    }
    columns = set(rows[0]) if rows else set()
    missing = sorted(required - columns)
    if missing:
        raise ValueError(
            f"Manifest {manifest_path} is missing columns: {', '.join(missing)}"
        )

    statuses = {str(value).strip() for value in allowed_statuses}
    actions = {str(value).strip() for value in allowed_next_actions}
    manifest_ids: set[str] = set()
    eligible_ids: set[str] = set()
    rejected = Counter()
    for row in rows:
        sequence_id = str(row["sequence_id"]).strip()
        if not sequence_id or sequence_id in manifest_ids:
            raise ValueError(f"Empty or duplicate manifest sequence ID: {sequence_id!r}")
        manifest_ids.add(sequence_id)
        reasons = []
        if not truthy(row["include_in_training"]):
            reasons.append("excluded_from_training")
        if str(row["status"]).strip() not in statuses:
            reasons.append(f"status:{str(row['status']).strip() or 'empty'}")
        if str(row["next_action"]).strip() not in actions:
            reasons.append(
                f"next_action:{str(row['next_action']).strip() or 'empty'}"
            )
        if not truthy(row["master_csv_exists"]):
            reasons.append("manifest_master_csv_missing")
        if reasons:
            rejected.update(reasons)
        else:
            eligible_ids.add(sequence_id)

    unlisted = sorted(set(by_sequence) - manifest_ids)
    missing_masters = sorted(eligible_ids - set(by_sequence))
    if strict and unlisted:
        raise ValueError(
            f"{len(unlisted)} master files are absent from {manifest_path}: "
            + ", ".join(unlisted[:10])
        )
    if strict and missing_masters:
        raise ValueError(
            f"{len(missing_masters)} eligible sequences have no master CSV: "
            + ", ".join(missing_masters[:10])
        )
    selected_ids = sorted(eligible_ids & set(by_sequence))
    return [by_sequence[value] for value in selected_ids], {
        "manifest_enabled": True,
        "manifest_path": str(manifest_path),
        "manifest_rows": len(rows),
        "master_files_found": len(files),
        "selected_sequences": len(selected_ids),
        "sequence_ids": selected_ids,
        "unlisted_master_sequence_ids": unlisted,
        "eligible_without_master_sequence_ids": missing_masters,
        "rejected_reason_counts": dict(sorted(rejected.items())),
    }


def summarize_master_csv(path: Path) -> SequenceSummary:
    path = Path(path).expanduser().resolve()
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        columns = set(reader.fieldnames or [])
        required = {"sequence_id", "participant", "receiving_hand", "target_object_id"}
        missing = sorted(required - columns)
        if missing:
            raise ValueError(f"{path} is missing columns: {', '.join(missing)}")
        if "intent_label" not in columns and "intent_id" not in columns:
            raise ValueError(f"{path} has neither intent_label nor intent_id")

        sequence_ids: set[str] = set()
        participants: set[str] = set()
        hands: set[str] = set()
        objects: set[int] = set()
        phases: Counter[str] = Counter()
        rows = 0
        for row in reader:
            rows += 1
            sequence_ids.add(str(row["sequence_id"]).strip())
            participants.add(canonical_participant(row["participant"]))
            hand = normalize_hand(row.get("receiving_hand"), row.get("receiving_hand_id"))
            if hand != "unknown":
                hands.add(hand)
            object_id = parse_target_object(row.get("target_object_id"))
            if object_id is not None:
                objects.add(object_id)
            label = str(row.get("intent_label", "")).strip().casefold()
            if not label:
                label = {"-1": "transition", "0": "continue", "1": "fetch", "2": "handover"}.get(
                    str(row.get("intent_id", "")).strip(), "unknown"
                )
            phases[label or "unknown"] += 1

    if rows == 0:
        raise ValueError(f"Empty master CSV: {path}")
    if len(sequence_ids) != 1 or "" in sequence_ids:
        raise ValueError(f"Mixed or empty sequence IDs in {path}: {sorted(sequence_ids)}")
    if len(participants) != 1:
        raise ValueError(f"Mixed participants in {path}: {sorted(participants)}")
    if len(hands) > 1:
        raise ValueError(f"Mixed receiving-hand annotations in {path}: {sorted(hands)}")
    if len(objects) > 1:
        raise ValueError(f"Mixed target-object annotations in {path}: {sorted(objects)}")
    return SequenceSummary(
        sequence_id=next(iter(sequence_ids)),
        participant=next(iter(participants)),
        receiving_hand=next(iter(hands), "unknown"),
        target_object_id=next(iter(objects), None),
        phase_distribution={key: float(value) for key, value in sorted(phases.items())},
        phase_unit="rows",
        phase_scope="master_rows_before_windowing",
        row_count=rows,
        source=str(path),
    )


def summarize_master_files(paths: Iterable[Path]) -> list[SequenceSummary]:
    summaries = [summarize_master_csv(path) for path in sorted(paths)]
    _validate_unique_summaries(summaries)
    return summaries


def _validate_unique_summaries(summaries: Sequence[SequenceSummary]) -> None:
    identifiers = [item.sequence_id for item in summaries]
    duplicates = sorted(
        identifier for identifier, count in Counter(identifiers).items() if count > 1
    )
    if duplicates:
        raise ValueError(f"Duplicate sequence summaries: {duplicates}")


def _read_rows(path: Path | None) -> list[dict[str, str]]:
    if path is None:
        return []
    with Path(path).expanduser().resolve().open(
        newline="", encoding="utf-8-sig"
    ) as handle:
        return list(csv.DictReader(handle))


def _effective_annotation_time(row: Mapping[str, object], name: str) -> float | None:
    manual = _finite_nonnegative(row.get(f"manual_{name}_s"))
    return manual if manual is not None else _finite_nonnegative(row.get(f"auto_{name}_s"))


def _phase_distribution_from_static_row(
    row: Mapping[str, object], annotation: Mapping[str, object]
) -> tuple[dict[str, float], str]:
    direct: dict[str, float] = {}
    direct_units = ""
    for phase in PHASE_NAMES:
        for key, unit in (
            (f"intent_{phase}", "counts"),
            (f"phase_{phase}", "counts"),
            (f"{phase}_rows", "rows"),
            (f"{phase}_count", "counts"),
            (f"{phase}_duration_s", "seconds"),
            (f"{phase}_duration_seconds", "seconds"),
        ):
            value = _finite_nonnegative(row.get(key))
            if value is not None:
                direct[phase] = value
                direct_units = unit
                break
    times = {
        name: _effective_annotation_time(annotation, name)
        for name in ("start", "second", "done", "third")
    }
    derived_durations: dict[str, float] = {}
    if all(value is not None for value in times.values()):
        start = float(times["start"])
        second = float(times["second"])
        done = float(times["done"])
        third = float(times["third"])
        if start <= second <= done <= third:
            derived_durations = {
                "continue": second - start,
                "fetch": done - second,
                "transition": third - done,
            }
            handover = _finite_nonnegative(row.get("handover_duration_seconds"))
            if handover is None:
                handover = _finite_nonnegative(row.get("handover_duration_s"))
            if handover is not None:
                derived_durations["handover"] = handover
    if direct:
        if direct_units == "seconds":
            for phase, value in derived_durations.items():
                direct.setdefault(phase, value)
        return direct, direct_units
    if derived_durations:
        return derived_durations, "seconds"
    return {}, "unavailable"


def load_static_sequence_summaries(
    summary_csv: Path,
    *,
    annotation_csv: Path | None = None,
) -> tuple[list[SequenceSummary], dict[str, list[str]] | None, dict]:
    """Load one-row-per-sequence metadata, optionally joined to annotations.

    Existing audit CSVs are supported: a ``split`` column becomes an explicit
    historical split, while annotation timestamps can supply phase durations.
    """

    summary_csv = Path(summary_csv).expanduser().resolve()
    rows = _read_rows(summary_csv)
    if not rows:
        raise ValueError(f"No rows in sequence-summary CSV: {summary_csv}")
    if "sequence_id" not in rows[0]:
        raise ValueError(f"{summary_csv} is missing sequence_id")

    annotation_rows = _read_rows(annotation_csv)
    annotations: dict[str, dict[str, str]] = {}
    for row in annotation_rows:
        sequence_id = str(row.get("sequence_id", "")).strip()
        if not sequence_id or sequence_id in annotations:
            raise ValueError(f"Empty or duplicate annotation sequence: {sequence_id!r}")
        annotations[sequence_id] = row

    historical: dict[str, set[str]] = {name: set() for name in SPLIT_NAMES}
    split_seen = False
    joined_annotations = 0
    phase_available = 0
    summaries = []
    for row in rows:
        sequence_id = str(row.get("sequence_id", "")).strip()
        if not sequence_id:
            raise ValueError(f"Empty sequence ID in {summary_csv}")
        annotation = annotations.get(sequence_id, {})
        if annotation:
            joined_annotations += 1
        participant_value = row.get("participant") or sequence_id.split("_", 1)[0]
        hand = normalize_hand(
            row.get("receiving_hand") or annotation.get("receiving_hand"),
            row.get("receiving_hand_id"),
        )
        target_object = parse_target_object(row.get("target_object_id"))
        if target_object is None:
            target_object = parse_target_object(annotation.get("target_object_id"))
        phases, phase_unit = _phase_distribution_from_static_row(row, annotation)
        phase_available += int(bool(phases))
        summary = SequenceSummary(
            sequence_id=sequence_id,
            participant=canonical_participant(participant_value),
            receiving_hand=hand,
            target_object_id=target_object,
            phase_distribution=phases,
            phase_unit=phase_unit,
            phase_scope=(
                "annotation_phase_durations"
                if phase_unit == "seconds"
                else "precomputed_sequence_phase_distribution"
                if phases
                else "unavailable"
            ),
            row_count=None,
            source=str(summary_csv),
        )
        summaries.append(summary)
        if str(row.get("split", "")).strip():
            split_seen = True
            historical[normalize_split_name(row["split"])].add(summary.participant)

    _validate_unique_summaries(summaries)
    historical_output = (
        {name: sorted(values) for name, values in historical.items()}
        if split_seen
        else None
    )
    return summaries, historical_output, {
        "summary_csv": str(summary_csv),
        "annotation_csv": str(Path(annotation_csv).resolve())
        if annotation_csv is not None
        else None,
        "sequences": len(summaries),
        "annotations_joined": joined_annotations,
        "sequences_with_target_object": sum(
            item.target_object_id is not None for item in summaries
        ),
        "sequences_with_known_receiving_hand": sum(
            item.receiving_hand in {"left", "right"} for item in summaries
        ),
        "sequences_with_phase_distribution": phase_available,
        "historical_split_present": split_seen,
    }


def _participant_profiles(
    summaries: Sequence[SequenceSummary],
) -> dict[str, dict[str, Counter[str]]]:
    profiles: dict[str, dict[str, Counter[str]]] = {}
    for summary in summaries:
        profile = profiles.setdefault(
            summary.participant,
            {
                "sequences": Counter(),
                "hands": Counter(),
                "objects": Counter(),
                "phases": Counter(),
            },
        )
        profile["sequences"]["all"] += 1.0
        profile["hands"][summary.receiving_hand] += 1.0
        object_name = (
            str(summary.target_object_id)
            if summary.target_object_id is not None
            else "unknown"
        )
        profile["objects"][object_name] += 1.0
        profile["phases"].update(summary.phase_distribution)
    return profiles


def participant_hand_diagnostics(
    summaries: Sequence[SequenceSummary],
    *,
    required_partitions: int = 3,
    partition_capacities: Sequence[int] | None = None,
) -> dict:
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    for item in summaries:
        counts[item.participant][item.receiving_hand] += 1

    side_participants = {"left": set(), "right": set()}
    pure_participants = []
    mixed_participants = []
    known_participants = []
    for participant, hand_counts in sorted(counts.items()):
        sides = {
            side
            for side in ("left", "right")
            if hand_counts[side] > 0 or hand_counts["both"] > 0
        }
        for side in sides:
            side_participants[side].add(participant)
        if sides:
            known_participants.append(participant)
        if len(sides) == 1:
            pure_participants.append(participant)
        elif len(sides) == 2:
            mixed_participants.append(participant)

    left_right_rows = [
        item for item in summaries if item.receiving_hand in {"left", "right"}
    ]
    global_counts = Counter(item.receiving_hand for item in left_right_rows)
    participant_correct = sum(
        max(hand_counts["left"], hand_counts["right"])
        for hand_counts in counts.values()
    )
    total_known = sum(global_counts.values())
    participant_majority_accuracy = (
        participant_correct / total_known if total_known else None
    )
    global_majority_accuracy = (
        max(global_counts.values()) / total_known if total_known else None
    )

    cramers_v = None
    if total_known and all(global_counts[side] > 0 for side in ("left", "right")):
        chi_square = 0.0
        for hand_counts in counts.values():
            row_total = hand_counts["left"] + hand_counts["right"]
            if not row_total:
                continue
            for side in ("left", "right"):
                expected = row_total * global_counts[side] / total_known
                chi_square += (hand_counts[side] - expected) ** 2 / expected
        cramers_v = math.sqrt(chi_square / total_known)

    necessary = all(
        len(side_participants[side]) >= required_partitions
        for side in ("left", "right")
    )
    exact_feasible = None
    if partition_capacities is not None:
        exact_feasible = _both_hands_partition_feasible(
            counts, tuple(int(value) for value in partition_capacities)
        )

    all_known_are_pure = bool(known_participants) and not mixed_participants
    warnings = []
    for side in ("left", "right"):
        if not side_participants[side]:
            warnings.append(f"No participant has a known {side} receiving-hand sequence")
        elif len(side_participants[side]) < required_partitions:
            warnings.append(
                f"Only {len(side_participants[side])} participants cover {side}; "
                f"{required_partitions} partitions cannot all contain that hand"
            )
    if all_known_are_pure:
        warnings.append(
            "Every participant with a known hand is single-hand; participant-hand "
            "coupling cannot be removed by participant reassignment alone"
        )
    if participant_majority_accuracy == 1.0:
        warnings.append(
            "Receiving hand is perfectly determined by participant within this subset"
        )
    if exact_feasible is False:
        warnings.append(
            "Both receiving hands in every requested partition are infeasible for "
            "the requested participant capacities"
        )

    return {
        "sequence_counts": dict(sorted(global_counts.items())),
        "participant_hand_sequence_counts": {
            participant: {
                hand: int(hand_counts[hand])
                for hand in HAND_NAMES
                if hand_counts[hand]
            }
            for participant, hand_counts in sorted(counts.items())
        },
        "participants_by_hand": {
            side: sorted(values) for side, values in side_participants.items()
        },
        "pure_hand_participants": pure_participants,
        "mixed_hand_participants": mixed_participants,
        "pure_hand_participant_fraction": (
            len(pure_participants) / len(known_participants)
            if known_participants
            else None
        ),
        "participant_majority_hand_accuracy": participant_majority_accuracy,
        "global_majority_hand_accuracy": global_majority_accuracy,
        "participant_majority_excess_accuracy": (
            participant_majority_accuracy - global_majority_accuracy
            if participant_majority_accuracy is not None
            and global_majority_accuracy is not None
            else None
        ),
        "cramers_v_participant_by_hand": cramers_v,
        "all_known_participants_are_hand_pure": all_known_are_pure,
        "reassignment_alone_can_remove_participant_hand_coupling": not all_known_are_pure,
        "both_hands_each_partition_necessary_condition": necessary,
        "both_hands_each_partition_exactly_feasible": exact_feasible,
        "required_partitions": required_partitions,
        "warnings": warnings,
    }


def _both_hands_partition_feasible(
    participant_counts: Mapping[str, Counter[str]], capacities: tuple[int, ...]
) -> bool:
    participants = []
    for participant in sorted(participant_counts):
        counts = participant_counts[participant]
        mask = 0
        if counts["left"] or counts["both"]:
            mask |= 1
        if counts["right"] or counts["both"]:
            mask |= 2
        participants.append(mask)
    if sum(capacities) != len(participants) or any(value <= 0 for value in capacities):
        raise ValueError("Partition capacities must be positive and cover participants")
    states = {(tuple(0 for _ in capacities), tuple(0 for _ in capacities))}
    for participant_mask in participants:
        next_states = set()
        for used, coverage in states:
            for index, capacity in enumerate(capacities):
                if used[index] >= capacity:
                    continue
                next_used = list(used)
                next_coverage = list(coverage)
                next_used[index] += 1
                next_coverage[index] |= participant_mask
                next_states.add((tuple(next_used), tuple(next_coverage)))
        states = next_states
        if not states:
            return False
    return any(
        used == capacities and all(mask == 3 for mask in coverage)
        for used, coverage in states
    )


def _normalize_partition(
    summaries: Sequence[SequenceSummary],
    partition: Mapping[str, Iterable[str]],
    names: Sequence[str],
    *,
    derive_first_remainder: bool = False,
) -> dict[str, tuple[str, ...]]:
    known = {item.participant for item in summaries}
    normalized: dict[str, set[str]] = {}
    for name in names:
        values = partition.get(name, ())
        normalized[name] = {canonical_participant(value) for value in values}
    if derive_first_remainder and not normalized[names[0]]:
        normalized[names[0]] = known - set().union(
            *(normalized[name] for name in names[1:])
        )
    unknown = set().union(*normalized.values()) - known
    if unknown:
        raise ValueError(f"Split contains unknown participants: {sorted(unknown)}")
    for index, left_name in enumerate(names):
        if not normalized[left_name]:
            raise ValueError(f"Split {left_name} contains no participants")
        for right_name in names[index + 1 :]:
            overlap = normalized[left_name] & normalized[right_name]
            if overlap:
                raise ValueError(
                    f"Participant overlap between {left_name} and {right_name}: "
                    f"{sorted(overlap)}"
                )
    assigned = set().union(*normalized.values())
    missing = known - assigned
    if missing:
        raise ValueError(f"Participants are unassigned: {sorted(missing)}")
    return {name: tuple(sorted(normalized[name])) for name in names}


def split_summary_table(
    summaries: Sequence[SequenceSummary],
    partition: Mapping[str, Iterable[str]],
) -> list[dict]:
    rows = []
    for split_name, values in partition.items():
        participants = {canonical_participant(value) for value in values}
        selected = [item for item in summaries if item.participant in participants]
        hands = Counter(item.receiving_hand for item in selected)
        objects = Counter(
            str(item.target_object_id)
            if item.target_object_id is not None
            else "unknown"
            for item in selected
        )
        phases: Counter[str] = Counter()
        units = Counter()
        for item in selected:
            phases.update(item.phase_distribution)
            units[item.phase_unit] += 1
        rows.append(
            {
                "split": split_name,
                "participant_count": len(participants),
                "participants": sorted(participants),
                "sequence_count": len(selected),
                "receiving_hand_sequence_counts": dict(sorted(hands.items())),
                "target_object_sequence_counts": dict(sorted(objects.items())),
                "phase_distribution": dict(sorted(phases.items())),
                "phase_units": dict(sorted(units.items())),
            }
        )
    return rows


def split_participant_summary_table(
    summaries: Sequence[SequenceSummary],
    partition: Mapping[str, Iterable[str]],
) -> list[dict]:
    """Return one transparent summary row for every split-participant pair."""

    rows = []
    for split_name, values in partition.items():
        for participant in sorted(canonical_participant(value) for value in values):
            selected = [
                item for item in summaries if item.participant == participant
            ]
            hands = Counter(item.receiving_hand for item in selected)
            objects = Counter(
                str(item.target_object_id)
                if item.target_object_id is not None
                else "unknown"
                for item in selected
            )
            phases: Counter[str] = Counter()
            units = Counter()
            for item in selected:
                phases.update(item.phase_distribution)
                units[item.phase_unit] += 1
            rows.append(
                {
                    "split": split_name,
                    "participant": participant,
                    "sequence_count": len(selected),
                    "receiving_hand_sequence_counts": dict(sorted(hands.items())),
                    "target_object_sequence_counts": dict(sorted(objects.items())),
                    "phase_distribution": dict(sorted(phases.items())),
                    "phase_units": dict(sorted(units.items())),
                    "sequence_ids": sorted(item.sequence_id for item in selected),
                }
            )
    return rows


def split_fingerprint(
    partition: Mapping[str, Iterable[str]], names: Sequence[str]
) -> str:
    payload = [
        f"{name}:{canonical_participant(participant)}"
        for name in names
        for participant in sorted(
            canonical_participant(value) for value in partition[name]
        )
    ]
    return hashlib.sha256("\n".join(payload).encode("utf-8")).hexdigest()


def participant_hand_diagnostics_by_partition(
    summaries: Sequence[SequenceSummary],
    partition: Mapping[str, Iterable[str]],
) -> dict[str, dict]:
    diagnostics = {}
    for name, values in partition.items():
        participants = {canonical_participant(value) for value in values}
        selected = [
            item for item in summaries if item.participant in participants
        ]
        diagnostics[name] = participant_hand_diagnostics(
            selected, required_partitions=1
        )
    return diagnostics


def _capacities_for_three_way_split(
    participant_count: int, validation_fraction: float, test_fraction: float
) -> tuple[int, int, int]:
    if participant_count < 3:
        raise ValueError("At least three participants are required")
    if not 0.0 < validation_fraction < 1.0 or not 0.0 < test_fraction < 1.0:
        raise ValueError("Validation and test fractions must be between zero and one")
    if validation_fraction + test_fraction >= 1.0:
        raise ValueError("Validation and test fractions must sum to less than one")
    validation = max(1, round(participant_count * validation_fraction))
    test = max(1, round(participant_count * test_fraction))
    while validation + test >= participant_count:
        if validation >= test and validation > 1:
            validation -= 1
        elif test > 1:
            test -= 1
        else:
            raise ValueError("Not enough participants for a three-way split")
    return participant_count - validation - test, validation, test


def _objective_score(
    partition: Mapping[str, Sequence[str]],
    profiles: Mapping[str, Mapping[str, Counter[str]]],
    capacities: Mapping[str, int],
) -> float:
    total_participants = sum(capacities.values())
    fractions = {
        name: capacities[name] / total_participants for name in partition
    }
    group_weights = {"sequences": 2.0, "hands": 8.0, "objects": 1.0, "phases": 1.0}
    total_score = 0.0
    for group, group_weight in group_weights.items():
        dimensions = sorted(
            {
                dimension
                for profile in profiles.values()
                for dimension in profile[group]
            }
        )
        if not dimensions:
            continue
        group_error = 0.0
        terms = 0
        totals = {
            dimension: sum(profile[group][dimension] for profile in profiles.values())
            for dimension in dimensions
        }
        for split_name, participants in partition.items():
            for dimension in dimensions:
                target = totals[dimension] * fractions[split_name]
                actual = sum(
                    profiles[participant][group][dimension]
                    for participant in participants
                )
                group_error += ((actual - target) / max(1.0, target)) ** 2
                terms += 1
        total_score += group_weight * group_error / max(1, terms)

    globally_present = {
        side: sum(profile["hands"][side] for profile in profiles.values()) > 0
        for side in ("left", "right")
    }
    for participants in partition.values():
        for side in ("left", "right"):
            if globally_present[side] and not any(
                profiles[participant]["hands"][side]
                or profiles[participant]["hands"]["both"]
                for participant in participants
            ):
                total_score += 100.0

    # Marginally balanced hands do not remove confounding if each evaluation
    # participant is still hand-pure. Prefer mixed-hand participants in both
    # validation and test, and reduce the excess accuracy obtained merely by
    # knowing participant identity. This uses annotations only, never metrics.
    for split_name in ("validation", "test"):
        if split_name not in partition:
            continue
        participants = partition[split_name]
        left_total = sum(
            profiles[participant]["hands"]["left"]
            for participant in participants
        )
        right_total = sum(
            profiles[participant]["hands"]["right"]
            for participant in participants
        )
        known_total = left_total + right_total
        if not left_total or not right_total or not known_total:
            continue
        mixed_count = sum(
            bool(profiles[participant]["hands"]["left"])
            and bool(profiles[participant]["hands"]["right"])
            for participant in participants
        )
        participant_majority_accuracy = sum(
            max(
                profiles[participant]["hands"]["left"],
                profiles[participant]["hands"]["right"],
            )
            for participant in participants
        ) / known_total
        marginal_majority_accuracy = max(left_total, right_total) / known_total
        excess_identity_predictability = max(
            0.0,
            participant_majority_accuracy - marginal_majority_accuracy,
        )
        total_score += 20.0 * excess_identity_predictability**2
        if not mixed_count:
            total_score += 25.0
    return total_score


def _partition_signature(
    partition: Mapping[str, Sequence[str]], names: Sequence[str]
) -> tuple:
    return tuple((name, *sorted(partition[name])) for name in names)


def _local_swap_improvement(
    partition: dict[str, tuple[str, ...]],
    *,
    names: Sequence[str],
    profiles: Mapping[str, Mapping[str, Counter[str]]],
    capacities: Mapping[str, int],
) -> tuple[dict[str, tuple[str, ...]], float]:
    current = {name: tuple(sorted(partition[name])) for name in names}
    current_score = _objective_score(current, profiles, capacities)
    for _ in range(50):
        best = current
        best_score = current_score
        best_signature = _partition_signature(best, names)
        for left_index, left_name in enumerate(names):
            for right_name in names[left_index + 1 :]:
                for left_participant in current[left_name]:
                    for right_participant in current[right_name]:
                        candidate = {name: list(current[name]) for name in names}
                        candidate[left_name].remove(left_participant)
                        candidate[right_name].remove(right_participant)
                        candidate[left_name].append(right_participant)
                        candidate[right_name].append(left_participant)
                        normalized = {
                            name: tuple(sorted(candidate[name])) for name in names
                        }
                        score = _objective_score(normalized, profiles, capacities)
                        signature = _partition_signature(normalized, names)
                        if score < best_score - 1e-12 or (
                            abs(score - best_score) <= 1e-12
                            and signature < best_signature
                        ):
                            best = normalized
                            best_score = score
                            best_signature = signature
        if best is current or best_score >= current_score - 1e-12:
            break
        current, current_score = best, best_score
    return current, current_score


def _balanced_partition(
    summaries: Sequence[SequenceSummary],
    *,
    names: Sequence[str],
    capacities: Sequence[int],
    seed: int,
    restarts: int,
) -> tuple[dict[str, tuple[str, ...]], float]:
    participants = sorted({item.participant for item in summaries})
    if len(names) != len(capacities) or sum(capacities) != len(participants):
        raise ValueError("Partition names/capacities do not cover all participants")
    if any(capacity <= 0 for capacity in capacities):
        raise ValueError("Every partition requires at least one participant")
    if restarts <= 0:
        raise ValueError("restarts must be positive")
    profiles = _participant_profiles(summaries)
    capacity_by_name = dict(zip(names, capacities))
    rng = random.Random(seed)
    candidates: dict[tuple, tuple[float, dict[str, tuple[str, ...]]]] = {}

    for restart in range(max(1, restarts)):
        order = participants.copy()
        if restart:
            rng.shuffle(order)
        partition: dict[str, tuple[str, ...]] = {}
        offset = 0
        for name, capacity in zip(names, capacities):
            partition[name] = tuple(sorted(order[offset : offset + capacity]))
            offset += capacity
        signature = _partition_signature(partition, names)
        score = _objective_score(partition, profiles, capacity_by_name)
        candidates[signature] = (score, partition)

    ranked = sorted(
        candidates.values(),
        key=lambda item: (item[0], _partition_signature(item[1], names)),
    )
    improved = [
        _local_swap_improvement(
            partition,
            names=names,
            profiles=profiles,
            capacities=capacity_by_name,
        )
        for _, partition in ranked[: min(16, len(ranked))]
    ]
    best_partition, best_score = min(
        improved,
        key=lambda item: (item[1], _partition_signature(item[0], names)),
    )
    return best_partition, best_score


def _partition_hand_warnings(
    summaries: Sequence[SequenceSummary], partition: Mapping[str, Iterable[str]]
) -> list[str]:
    warnings = []
    for name, participants in partition.items():
        participant_set = set(participants)
        sides = {
            item.receiving_hand
            for item in summaries
            if item.participant in participant_set
            and item.receiving_hand in {"left", "right"}
        }
        for side in ("left", "right"):
            if side not in sides:
                warnings.append(f"{name} contains no {side} receiving-hand sequence")
    return warnings


def generate_balanced_participant_split(
    summaries: Sequence[SequenceSummary],
    *,
    seed: int,
    validation_fraction: float,
    test_fraction: float,
    historical_split: Mapping[str, Iterable[str]] | None = None,
    restarts: int = 256,
) -> dict:
    """Build or preserve a participant-disjoint train/validation/test split."""

    _validate_unique_summaries(summaries)
    participant_count = len({item.participant for item in summaries})
    capacities = _capacities_for_three_way_split(
        participant_count, validation_fraction, test_fraction
    )
    if historical_split is not None:
        partition = _normalize_partition(
            summaries,
            historical_split,
            SPLIT_NAMES,
            derive_first_remainder=True,
        )
        strategy = "explicit_historical_participants"
        optimized = False
        objective_score = None
        effective_capacities = tuple(len(partition[name]) for name in SPLIT_NAMES)
    else:
        partition, objective_score = _balanced_partition(
            summaries,
            names=SPLIT_NAMES,
            capacities=capacities,
            seed=seed,
            restarts=restarts,
        )
        strategy = "deterministic_label_balanced_participant_split"
        optimized = True
        effective_capacities = capacities

    partition = _normalize_partition(summaries, partition, SPLIT_NAMES)
    diagnostics = participant_hand_diagnostics(
        summaries,
        required_partitions=3,
        partition_capacities=effective_capacities,
    )
    split_diagnostics = participant_hand_diagnostics_by_partition(
        summaries, partition
    )
    warnings = list(diagnostics["warnings"])
    for split_name, values in split_diagnostics.items():
        warnings.extend(
            f"{split_name}: {warning}" for warning in values["warnings"]
        )
    warnings.extend(_partition_hand_warnings(summaries, partition))
    return {
        "strategy": strategy,
        "historical_split_preserved": historical_split is not None,
        "selection_optimized": optimized,
        "seed": seed,
        "participants": {name: list(partition[name]) for name in SPLIT_NAMES},
        "split_fingerprint_sha256": split_fingerprint(partition, SPLIT_NAMES),
        "objective_score": objective_score,
        "objective": {
            "uses_only_dataset_labels_and_counts": True,
            "model_performance_metrics_used": False,
            "inputs": list(OBJECTIVE_INPUTS),
            "group_weights": {
                "sequence_count": 2.0,
                "receiving_hand": 8.0,
                "target_object": 1.0,
                "phase_distribution": 1.0,
            },
            "missing_known_hand_penalty": 100.0,
            "participant_hand_coupling_penalty": {
                "excess_identity_predictability_weight": 20.0,
                "no_mixed_participant_in_validation_or_test": 25.0,
            },
            "historical_preservation_precedes_optimization": True,
            "random_restarts": restarts if optimized else 0,
            "tie_break": (
                "lowest objective, then lexicographically smallest canonical "
                "participant assignment"
                if optimized
                else "not applicable: explicit historical assignment preserved"
            ),
        },
        "table": split_summary_table(summaries, partition),
        "participant_table": split_participant_summary_table(
            summaries, partition
        ),
        "participant_hand_diagnostics": diagnostics,
        "participant_hand_diagnostics_by_split": split_diagnostics,
        "warnings": sorted(set(warnings)),
    }


def generate_participant_group_cv(
    summaries: Sequence[SequenceSummary],
    *,
    folds: int,
    seed: int,
    restarts: int = 256,
) -> dict:
    """Generate deterministic participant-wise Group-CV validation folds."""

    _validate_unique_summaries(summaries)
    participants = sorted({item.participant for item in summaries})
    if folds < 3 or folds > len(participants):
        raise ValueError(
            "executable nested Group-CV requires between 3 folds and the "
            "participant count"
        )
    base, remainder = divmod(len(participants), folds)
    capacities = tuple(base + int(index < remainder) for index in range(folds))
    names = tuple(f"fold_{index}" for index in range(folds))
    partition, objective_score = _balanced_partition(
        summaries,
        names=names,
        capacities=capacities,
        seed=seed,
        restarts=restarts,
    )
    fold_rows = []
    all_participants = set(participants)
    for index, name in enumerate(names):
        outer_evaluation = set(partition[name])
        inner_name = names[(index + 1) % len(names)]
        inner_validation = set(partition[inner_name])
        train = all_participants - outer_evaluation - inner_validation
        fold_rows.append(
            {
                "fold": index,
                "protocol": "nested_participant_disjoint_train_validation_outer_evaluation",
                "train_participants": sorted(train),
                "validation_participants": sorted(inner_validation),
                "test_participants": sorted(outer_evaluation),
                "outer_evaluation_participants": sorted(outer_evaluation),
                "inner_validation_partition": inner_name,
                "table": split_summary_table(
                    summaries,
                    {
                        "train": sorted(train),
                        "validation": sorted(inner_validation),
                        "test": sorted(outer_evaluation),
                    },
                ),
                "validation_participant_hand_diagnostics": (
                    participant_hand_diagnostics(
                        [
                            item
                            for item in summaries
                            if item.participant in inner_validation
                        ],
                        required_partitions=1,
                    )
                ),
                "outer_evaluation_participant_hand_diagnostics": (
                    participant_hand_diagnostics(
                        [
                            item
                            for item in summaries
                            if item.participant in outer_evaluation
                        ],
                        required_partitions=1,
                    )
                ),
            }
        )
    diagnostics = participant_hand_diagnostics(
        summaries,
        required_partitions=folds,
        partition_capacities=capacities if folds <= 4 else None,
    )
    warnings = list(diagnostics["warnings"])
    warnings.extend(_partition_hand_warnings(summaries, partition))
    return {
        "strategy": "deterministic_label_balanced_participant_group_cv",
        "execution_protocol": (
            "Each fold uses the held-out partition as outer evaluation, the next "
            "partition as inner validation for checkpoint selection, and all "
            "remaining participants for training. Outer metrics are never used "
            "for checkpoint or architecture selection."
        ),
        "seed": seed,
        "fold_count": folds,
        "objective_score": objective_score,
        "split_fingerprint_sha256": split_fingerprint(partition, names),
        "objective": {
            "uses_only_dataset_labels_and_counts": True,
            "model_performance_metrics_used": False,
            "inputs": list(OBJECTIVE_INPUTS),
            "group_weights": {
                "sequence_count": 2.0,
                "receiving_hand": 8.0,
                "target_object": 1.0,
                "phase_distribution": 1.0,
            },
            "missing_known_hand_penalty": 100.0,
            "random_restarts": restarts,
            "tie_break": (
                "lowest objective, then lexicographically smallest canonical "
                "participant assignment"
            ),
        },
        "validation_partition": {
            name: list(partition[name]) for name in names
        },
        "folds": fold_rows,
        "participant_hand_diagnostics": diagnostics,
        "warnings": sorted(set(warnings)),
    }


def load_historical_split_from_config(path: Path) -> dict[str, list[str]] | None:
    with Path(path).expanduser().resolve().open(encoding="utf-8") as handle:
        config = json.load(handle)
    data = config.get("data", config)
    validation = list(data.get("validation_participants", []))
    test = list(data.get("test_participants", []))
    if not validation and not test:
        return None
    if not validation or not test:
        raise ValueError(
            "Historical config must define both validation_participants and "
            "test_participants"
        )
    return {"validation": validation, "test": test}
