#!/usr/bin/env python3
"""Shared recursive discovery for nested training-run directories."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable


DEFAULT_REQUIRED_ARTIFACTS = (
    "config.json",
    "data_metadata.json",
    "metrics.json",
)


def _has_artifacts(path: Path, required_artifacts: Iterable[str]) -> bool:
    return all((path / name).is_file() for name in required_artifacts)


def discover_run_directories(
    runs_root: Path,
    *,
    name_prefix: str | None = None,
    required_artifacts: Iterable[str] = DEFAULT_REQUIRED_ARTIFACTS,
) -> list[Path]:
    """Find valid runs at any nesting depth without returning duplicates."""

    runs_root = runs_root.expanduser().resolve()
    if not runs_root.is_dir():
        raise FileNotFoundError(f"Runs root does not exist: {runs_root}")
    required = tuple(required_artifacts)
    candidates: set[Path] = set()
    anchors = (required[0],) if required else ("config.json",)
    for anchor in anchors:
        for artifact in runs_root.rglob(anchor):
            run_dir = artifact.parent.resolve()
            if (
                (name_prefix is None or run_dir.name.startswith(name_prefix))
                and _has_artifacts(run_dir, required)
            ):
                candidates.add(run_dir)
    return sorted(candidates)


def resolve_run_directory(
    requested: Path,
    *,
    runs_root: Path,
    required_artifacts: Iterable[str] = DEFAULT_REQUIRED_ARTIFACTS,
) -> Path:
    """Resolve an explicit directory or one unique nested run basename."""

    requested = requested.expanduser()
    required = tuple(required_artifacts)
    if requested.is_dir():
        resolved = requested.resolve()
        if not _has_artifacts(resolved, required):
            missing = [
                name for name in required if not (resolved / name).is_file()
            ]
            raise FileNotFoundError(
                f"Run directory lacks required artifacts {missing}: {resolved}"
            )
        return resolved

    matches = [
        path
        for path in discover_run_directories(
            runs_root,
            required_artifacts=required,
        )
        if path.name == requested.name
    ]
    if not matches:
        raise FileNotFoundError(
            f"Run {requested} was not found recursively below "
            f"{runs_root.expanduser().resolve()}"
        )
    if len(matches) > 1:
        raise ValueError(
            f"Run basename {requested.name!r} is ambiguous: "
            + ", ".join(str(path) for path in matches)
        )
    return matches[0]
