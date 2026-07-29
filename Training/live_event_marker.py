#!/usr/bin/env python3
"""Record monotonic event boundaries for a controlled Aria live test.

Run this helper in a second terminal on the same computer as
aria_live_inference.py. It records annotations only and has no device or robot
interface.
"""

from __future__ import annotations

import argparse
import json
import shlex
import time
from datetime import datetime, timezone
from pathlib import Path


INTENTIONS = {"continue", "fetch", "handover"}


def parse_start_command(command: str) -> dict:
    parts = shlex.split(command)
    if len(parts) != 4 or parts[0].lower() != "start":
        raise ValueError(
            "Use: start <scenario_id> "
            "<continue|fetch|handover|unscored> <true|false|any>"
        )
    scenario_id = parts[1].strip()
    if not scenario_id:
        raise ValueError("scenario_id must not be empty")
    intention_value = parts[2].lower()
    if intention_value not in {*INTENTIONS, "unscored"}:
        raise ValueError(f"Invalid expected intention: {parts[2]}")
    quality_value = parts[3].lower()
    quality_mapping = {"true": True, "false": False, "any": None}
    if quality_value not in quality_mapping:
        raise ValueError(f"Invalid expected quality: {parts[3]}")
    return {
        "scenario_id": scenario_id,
        "expected_intention": (
            None if intention_value == "unscored" else intention_value
        ),
        "expected_quality_ok": quality_mapping[quality_value],
    }


def event_record(event: str, context: dict, note: str | None = None) -> dict:
    record = {
        "event": event,
        **context,
        "host_monotonic_ns": time.monotonic_ns(),
        "wall_time_utc": datetime.now(timezone.utc).isoformat(),
    }
    if note is not None:
        record["note"] = note
    return record


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append to an existing event file; a new file is safer by default.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_path = args.output_jsonl.expanduser().resolve()
    if output_path.exists() and output_path.stat().st_size and not args.append:
        raise FileExistsError(
            f"Event file already contains data: {output_path}. "
            "Use a new filename or explicitly pass --append."
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if args.append else "w"

    print("Live event marker (annotation only; no hardware commands).")
    print(
        "Start: start <scenario> "
        "<continue|fetch|handover|unscored> <true|false|any>"
    )
    print("End:   end")
    print("Note:  note <text>")
    print("Stop:  quit")

    active: dict | None = None
    with output_path.open(mode, encoding="utf-8") as handle:
        while True:
            try:
                command = input("event> ").strip()
            except EOFError:
                command = "quit"
            if not command:
                continue
            keyword = command.split(maxsplit=1)[0].lower()
            try:
                if keyword == "start":
                    if active is not None:
                        raise ValueError(
                            f"Scenario {active['scenario_id']!r} is still active"
                        )
                    active = parse_start_command(command)
                    record = event_record("start", active)
                elif keyword == "end":
                    if command.lower() != "end":
                        raise ValueError("Use exactly: end")
                    if active is None:
                        raise ValueError("No scenario is active")
                    record = event_record("end", active)
                    active = None
                elif keyword == "note":
                    note = command.partition(" ")[2].strip()
                    if not note:
                        raise ValueError("A note must contain text")
                    record = event_record("note", active or {}, note)
                elif keyword in {"quit", "exit"}:
                    if active is not None:
                        print(
                            "End the active scenario first, or press Ctrl-C "
                            "if it should remain incomplete."
                        )
                        continue
                    break
                else:
                    raise ValueError(f"Unknown command: {keyword}")
            except ValueError as error:
                print(f"ERROR: {error}")
                continue

            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            print(
                f"saved {record['event']}: "
                f"{record.get('scenario_id', record.get('note', ''))}"
            )

    print(f"Events: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
