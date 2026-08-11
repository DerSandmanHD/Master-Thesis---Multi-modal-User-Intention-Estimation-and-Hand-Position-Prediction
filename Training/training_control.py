"""Small, testable controls shared by the training entry points."""

from __future__ import annotations

import math
from pathlib import Path


POSE_DIAGNOSTIC_UNAVAILABLE_REASON = (
    "No finite validation pose metric was observed; no pose-selected "
    "diagnostic checkpoint was created."
)


def next_primary_patience(
    current: int,
    *,
    primary_improved: bool,
    diagnostic_improved: bool = False,
) -> int:
    """Advance patience using only the predeclared primary validation rule.

    ``diagnostic_improved`` is accepted deliberately so callers make the
    distinction explicit.  A diagnostic checkpoint must never prolong model
    selection based on the primary intention metric.
    """

    if current < 0:
        raise ValueError("Early-stopping patience counter cannot be negative")
    del diagnostic_improved
    return 0 if primary_improved else current + 1


def finite_diagnostic_improved(candidate: float, best: float) -> bool:
    """A pose diagnostic is checkpointable only when its metric is finite."""

    return math.isfinite(candidate) and candidate < best


def available_validation_checkpoints(
    primary_path: Path,
    pose_diagnostic_path: Path,
) -> tuple[tuple[tuple[str, Path], ...], dict[str, object]]:
    """Return executable checkpoints and explicit optional-pose status."""

    primary_path = Path(primary_path)
    pose_diagnostic_path = Path(pose_diagnostic_path)
    if not primary_path.is_file():
        raise FileNotFoundError(
            f"Primary intention checkpoint was not created: {primary_path}"
        )
    checkpoints: list[tuple[str, Path]] = [("best_intention", primary_path)]
    pose_available = pose_diagnostic_path.is_file()
    if pose_available:
        checkpoints.append(("best_pose", pose_diagnostic_path))
    status: dict[str, object] = {
        "available": pose_available,
        "path": str(pose_diagnostic_path) if pose_available else None,
        "role": "diagnostic_only",
        "reason": None if pose_available else POSE_DIAGNOSTIC_UNAVAILABLE_REASON,
    }
    return tuple(checkpoints), status
