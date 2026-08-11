#!/usr/bin/env python3
"""Audit or propose participant-disjoint dataset splits without model metrics."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from participant_splits import (
    eligible_master_paths,
    generate_balanced_participant_split,
    generate_participant_group_cv,
    load_historical_split_from_config,
    load_static_sequence_summaries,
    participant_hand_diagnostics,
    summarize_master_files,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--master-dir",
        type=Path,
        help="Directory containing eligible *_master.csv files.",
    )
    source.add_argument(
        "--summary-csv",
        type=Path,
        help=(
            "One-row-per-sequence CSV. Existing audit CSVs with participant, "
            "sequence_id, receiving_hand, and split columns are supported."
        ),
    )
    parser.add_argument(
        "--annotations",
        type=Path,
        default=None,
        help=(
            "Optional annotation CSV joined by sequence_id. It supplies target "
            "object, receiving hand, and phase times when absent from summary CSV."
        ),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Eligibility manifest for --master-dir. Auto-detected beside masters.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help=(
            "Config containing explicit validation/test participants. These are "
            "audited and preserved, never silently rebalanced."
        ),
    )
    parser.add_argument(
        "--balanced-candidate",
        action="store_true",
        help=(
            "Also generate a label/count-balanced candidate. Historical output "
            "remains unchanged and is reported separately."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation-fraction", type=float, default=0.12)
    parser.add_argument("--test-fraction", type=float, default=0.12)
    parser.add_argument("--group-cv-folds", type=int, default=5)
    parser.add_argument("--restarts", type=int, default=256)
    parser.add_argument(
        "--strict-manifest",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--output-json", type=Path, default=None)
    return parser.parse_args()


def _resolve(path: Path | None) -> Path | None:
    if path is None:
        return None
    expanded = path.expanduser()
    return (
        expanded.resolve()
        if expanded.is_absolute()
        else (PROJECT_ROOT / expanded).resolve()
    )


def load_inputs(args: argparse.Namespace) -> tuple[list, dict | None, dict]:
    summary_path = _resolve(args.summary_csv)
    annotation_path = _resolve(args.annotations)
    if summary_path is not None:
        return load_static_sequence_summaries(
            summary_path, annotation_csv=annotation_path
        )

    master_dir = _resolve(args.master_dir) or (
        PROJECT_ROOT / "Data_collection" / "master_datasets"
    )
    manifest_path = _resolve(args.manifest)
    if manifest_path is None:
        candidate = master_dir.parent / "dataset_manifest.csv"
        manifest_path = candidate if candidate.is_file() else None
    paths, eligibility = eligible_master_paths(
        master_dir,
        manifest_path=manifest_path,
        strict=args.strict_manifest,
    )
    if not paths:
        raise FileNotFoundError(
            f"No eligible master CSVs in {master_dir}. Use --summary-csv for "
            "tracked audit metadata when master files are unavailable."
        )
    return summarize_master_files(paths), None, {"eligibility": eligibility}


def _historical_split(args: argparse.Namespace, embedded: dict | None) -> dict | None:
    config_split = (
        load_historical_split_from_config(_resolve(args.config))
        if args.config is not None
        else None
    )
    if embedded is not None and config_split is not None:
        for name in ("validation", "test"):
            embedded_values = {value.casefold() for value in embedded.get(name, [])}
            config_values = {value.casefold() for value in config_split.get(name, [])}
            if embedded_values != config_values:
                raise ValueError(
                    f"Embedded and configured historical {name} participants differ"
                )
    return embedded if embedded is not None else config_split


def build_report(args: argparse.Namespace) -> dict:
    summaries, embedded_historical, source_metadata = load_inputs(args)
    historical = _historical_split(args, embedded_historical)
    test_named_sequences = [
        item for item in summaries if item.participant.casefold() == "test"
    ]
    source_is_static = args.summary_csv is not None
    limitations = [
        "Phase distributions describe pre-window sequence rows or durations; "
        "they are not sliding-window endpoint target counts.",
        "The transition phase is context-only and must not be interpreted as a "
        "trainable window-end intention class.",
    ]
    if source_is_static:
        limitations.append(
            "Static audit/annotation input can reconstruct sequence-level phase "
            "durations, but cannot verify master-row or post-window endpoint "
            "distributions while committed master CSVs are unavailable."
        )
    identity_flags = []
    if test_named_sequences:
        historical_locations = []
        if historical is not None:
            for split_name in ("validation", "test"):
                if any(
                    str(value).strip().casefold() == "test"
                    for value in historical.get(split_name, [])
                ):
                    historical_locations.append(split_name)
            if not historical_locations:
                historical_locations.append("train")
        identity_flags.append(
            {
                "participant": "Test",
                "sequence_count": len(test_named_sequences),
                "historical_splits": historical_locations,
                "sequence_ids": sorted(
                    item.sequence_id for item in test_named_sequences
                ),
                "action": (
                    "Confirm manually that 'Test' is an intentional participant "
                    "pseudonym rather than technical test data."
                ),
            }
        )
    report = {
        "schema_version": "participant_split_audit_v1",
        "source": source_metadata,
        "sequence_count": len(summaries),
        "participant_count": len({item.participant for item in summaries}),
        "data_completeness": {
            "known_target_object_sequences": sum(
                item.target_object_id is not None for item in summaries
            ),
            "known_receiving_hand_sequences": sum(
                item.receiving_hand in {"left", "right"} for item in summaries
            ),
            "phase_distribution_sequences": sum(
                bool(item.phase_distribution) for item in summaries
            ),
            "phase_scope_counts": dict(
                sorted(
                    Counter(item.phase_scope for item in summaries).items()
                )
            ),
        },
        "sequence_summaries": [item.to_dict() for item in summaries],
        "distribution_semantics": {
            "scope": "pre_window_sequence_phase_distribution",
            "is_window_endpoint_target_distribution": False,
            "transition_is_context_only": True,
        },
        "limitations": limitations,
        "identity_provenance_flags": identity_flags,
        "global_participant_hand_diagnostics": participant_hand_diagnostics(
            summaries
        ),
    }
    if historical is not None:
        report["historical_split"] = generate_balanced_participant_split(
            summaries,
            seed=args.seed,
            validation_fraction=args.validation_fraction,
            test_fraction=args.test_fraction,
            historical_split=historical,
            restarts=args.restarts,
        )
    if historical is None or args.balanced_candidate:
        report["balanced_candidate"] = generate_balanced_participant_split(
            summaries,
            seed=args.seed,
            validation_fraction=args.validation_fraction,
            test_fraction=args.test_fraction,
            restarts=args.restarts,
        )
    if args.group_cv_folds:
        report["participant_group_cv"] = generate_participant_group_cv(
            summaries,
            folds=args.group_cv_folds,
            seed=args.seed,
            restarts=args.restarts,
        )
    return report


def print_split_table(title: str, plan: dict) -> None:
    print(f"\n{title}")
    print("split       participants  sequences  left  right  objects  p->hand")
    for row in plan["table"]:
        hands = row["receiving_hand_sequence_counts"]
        coupling = plan["participant_hand_diagnostics_by_split"][row["split"]][
            "participant_majority_hand_accuracy"
        ]
        coupling_text = f"{coupling:.3f}" if coupling is not None else "n/a"
        known_objects = {
            key: value
            for key, value in row["target_object_sequence_counts"].items()
            if key != "unknown" and value
        }
        print(
            f"{row['split']:<11} {row['participant_count']:>12} "
            f"{row['sequence_count']:>10} {hands.get('left', 0):>5} "
            f"{hands.get('right', 0):>6} {len(known_objects):>8} "
            f"{coupling_text:>8}"
        )
    for warning in plan.get("warnings", []):
        print(f"WARNING: {warning}")


def main() -> int:
    args = parse_args()
    try:
        report = build_report(args)
    except (FileNotFoundError, KeyError, OSError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(
        f"Sequences: {report['sequence_count']} | "
        f"Participants: {report['participant_count']} | "
        f"Objects known: "
        f"{report['data_completeness']['known_target_object_sequences']}/"
        f"{report['sequence_count']}"
    )
    for limitation in report["limitations"]:
        print(f"LIMITATION: {limitation}")
    for flag in report["identity_provenance_flags"]:
        print(
            f"IDENTITY CHECK: participant={flag['participant']} "
            f"sequences={flag['sequence_count']} "
            f"historical_splits={','.join(flag['historical_splits']) or 'unknown'} "
            f"- {flag['action']}"
        )
    if "historical_split" in report:
        print_split_table("Historical split (preserved)", report["historical_split"])
    if "balanced_candidate" in report:
        print_split_table("Balanced candidate", report["balanced_candidate"])
    cv = report.get("participant_group_cv")
    if cv:
        print(f"\nParticipant Group-CV folds: {cv['fold_count']}")
        for fold in cv["folds"]:
            print(
                f"fold {fold['fold']}: validation="
                f"{','.join(fold['validation_participants'])}"
            )

    output_path = _resolve(args.output_json)
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"\nJSON: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
