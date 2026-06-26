#!/usr/bin/env python3
"""Batch-convert Project Aria VRS recordings to MP4."""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VRS_DIR = REPO_ROOT / "Data_collection" / "Data_vrs"
DEFAULT_MP4_DIR = REPO_ROOT / "Data_collection" / "Data_mp4"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert missing .vrs recordings to .mp4 with vrs_to_mp4."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_VRS_DIR,
        help=f"Directory containing .vrs files. Default: {DEFAULT_VRS_DIR}",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_MP4_DIR,
        help=f"Directory for .mp4 outputs. Default: {DEFAULT_MP4_DIR}",
    )
    parser.add_argument(
        "--pattern",
        default="*.vrs",
        help="Glob pattern for input recordings. Default: *.vrs",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recreate MP4s even when output files already exist.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print what would be converted.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Convert at most N recordings. Useful for small test runs.",
    )
    parser.add_argument(
        "--tool",
        default="vrs_to_mp4",
        help="Conversion command to call. Default: vrs_to_mp4",
    )
    parser.add_argument(
        "--show-tool-output",
        action="store_true",
        help="Forward vrs_to_mp4 output instead of keeping the terminal concise.",
    )
    return parser.parse_args()


def discover_vrs_files(input_dir: Path, pattern: str) -> list[Path]:
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input path is not a directory: {input_dir}")
    return sorted(path for path in input_dir.glob(pattern) if path.is_file() and path.suffix == ".vrs")


def planned_conversions(vrs_files: list[Path], output_dir: Path, overwrite: bool) -> tuple[list[tuple[Path, Path]], list[Path]]:
    to_convert = []
    skipped = []
    for vrs_path in vrs_files:
        mp4_path = output_dir / f"{vrs_path.stem}.mp4"
        if mp4_path.exists() and not overwrite:
            skipped.append(vrs_path)
        else:
            to_convert.append((vrs_path, mp4_path))
    return to_convert, skipped


def run_conversion(tool: str, vrs_path: Path, mp4_path: Path, show_tool_output: bool) -> subprocess.CompletedProcess:
    command = [tool, "--vrs", str(vrs_path), "--output_video", str(mp4_path)]
    stdout = None if show_tool_output else subprocess.DEVNULL
    stderr = None if show_tool_output else subprocess.PIPE
    return subprocess.run(command, check=False, stdout=stdout, stderr=stderr, text=True)


def print_plan(
    total: int,
    to_convert: list[tuple[Path, Path]],
    skipped: list[Path],
    dry_run: bool,
    total_to_convert: int,
) -> None:
    mode = "DRY RUN" if dry_run else "RUN"
    print(f"{mode}: found {total} VRS files")
    if len(to_convert) == total_to_convert:
        print(f"  to convert: {len(to_convert)}")
    else:
        print(f"  to convert: {len(to_convert)} shown by --limit, {total_to_convert} missing total")
    print(f"  skipped existing: {len(skipped)}")
    if dry_run and to_convert:
        print("\nPlanned conversions:")
        for vrs_path, mp4_path in to_convert:
            print(f"  {vrs_path.name} -> {mp4_path}")


def main() -> int:
    args = parse_args()
    input_dir = args.input_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    try:
        vrs_files = discover_vrs_files(input_dir, args.pattern)
    except (FileNotFoundError, NotADirectoryError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    if not vrs_files:
        print(f"No VRS files found in {input_dir} with pattern {args.pattern}")
        return 0

    to_convert, skipped = planned_conversions(vrs_files, output_dir, args.overwrite)
    total_to_convert = len(to_convert)
    if args.limit is not None:
        to_convert = to_convert[: max(args.limit, 0)]

    print_plan(len(vrs_files), to_convert, skipped, args.dry_run, total_to_convert)

    if args.dry_run:
        return 0

    if not to_convert:
        print("Nothing to convert.")
        return 0

    if shutil.which(args.tool) is None:
        print(
            f"Error: conversion tool '{args.tool}' was not found. "
            "Activate the aria_conda environment or pass --tool.",
            file=sys.stderr,
        )
        return 2

    output_dir.mkdir(parents=True, exist_ok=True)
    converted = []
    failed = []

    for index, (vrs_path, mp4_path) in enumerate(to_convert, start=1):
        print(f"[{index}/{len(to_convert)}] Converting {vrs_path.name} ... ", end="", flush=True)
        result = run_conversion(args.tool, vrs_path, mp4_path, args.show_tool_output)
        if result.returncode == 0 and mp4_path.exists():
            converted.append(mp4_path)
            print("ok")
        else:
            error_text = (result.stderr or "").strip()
            failed.append({
                "vrs": str(vrs_path),
                "mp4": str(mp4_path),
                "returncode": result.returncode,
                "stderr": error_text,
            })
            print("failed")

    print("\nSummary")
    print(f"  converted: {len(converted)}")
    print(f"  skipped existing: {len(skipped)}")
    print(f"  failed: {len(failed)}")

    if failed:
        print("\nFailures:")
        for item in failed:
            print(f"  {item['vrs']} -> {item['mp4']} (returncode={item['returncode']})")
            if item["stderr"]:
                print(f"    {item['stderr'].splitlines()[-1]}")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
