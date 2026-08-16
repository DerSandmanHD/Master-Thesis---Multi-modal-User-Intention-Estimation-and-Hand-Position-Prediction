#!/usr/bin/env python3
"""Hash-bound provenance manifests for thesis training runs.

The manifest is deliberately separate from metrics: it records the exact code,
dataset, split, feature schema, normalization, visual artifacts, command and
selected checkpoints needed to reconstruct a run.  Validation recomputes file
hashes, so a derived artifact is never accepted merely because its name exists.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch


PROJECT_ROOT = Path(__file__).resolve().parent.parent
ARTIFACT_FREEZE_SCHEMA_VERSION = 2
ARTIFACT_FREEZE_PROTOCOL = "thesis_artifact_freeze_hash_bound_v2"
MANIFEST_NAME = "artifact_manifest.json"
PACKAGE_NAMES = (
    "projectaria-tools",
    "projectaria-mps",
    "torch",
    "torchvision",
    "open-clip-torch",
    "numpy",
    "pandas",
    "scipy",
    "scikit-learn",
)
SOURCE_SUFFIXES = {
    ".py",
    ".json",
    ".md",
    ".sbatch",
    ".sh",
    ".yaml",
    ".yml",
    ".toml",
    ".txt",
    ".recipe",
}
GENERATED_PREFIXES = (
    "Data_collection/",
    "Training/runs/",
    "Training/reports/",
    "Training/slurm_logs/",
    "Training/Outputs/",
)


class ArtifactFreezeError(ValueError):
    """Raised when a frozen run cannot prove its artifact identities."""


def canonical_json_hash(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _portable_path(path: Path, *, base: Path = PROJECT_ROOT) -> str:
    resolved = Path(path).expanduser().resolve()
    try:
        return resolved.relative_to(base.resolve()).as_posix()
    except ValueError:
        return str(resolved)


def file_identity(
    path: Path,
    *,
    base: Path = PROJECT_ROOT,
    required: bool = True,
) -> dict[str, Any] | None:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        if required:
            raise FileNotFoundError(resolved)
        return None
    stat = resolved.stat()
    return {
        "path": _portable_path(resolved, base=base),
        "size_bytes": int(stat.st_size),
        "sha256": sha256_file(resolved),
    }


def _run_git(*arguments: str, binary: bool = False) -> bytes | str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(PROJECT_ROOT), *arguments],
            check=True,
            capture_output=True,
            text=not binary,
            timeout=30,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return None
    return result.stdout if binary else result.stdout.strip()


def _changed_worktree_file_hashes() -> dict[str, str]:
    """Hash changed and untracked source files, including untracked code.

    ``git diff`` binds tracked edits but deliberately omits untracked files.
    Training from a dirty research checkout therefore also records source
    content omitted by ``git diff``. Generated datasets, runs and reports are
    bound separately and are excluded here to avoid hashing large checkpoints.
    The NUL-delimited form avoids quoting and whitespace ambiguities.
    """

    raw = _run_git(
        "status",
        "--porcelain=v1",
        "-z",
        "--untracked-files=all",
        binary=True,
    )
    if not isinstance(raw, bytes):
        return {}
    records = raw.split(b"\0")
    hashes: dict[str, str] = {}
    index = 0
    while index < len(records):
        record = records[index]
        index += 1
        if not record:
            continue
        status = record[:2].decode("ascii", errors="replace")
        relative = record[3:].decode("utf-8", errors="surrogateescape")
        candidates = [relative]
        if status[:1] in {"R", "C"} or status[1:2] in {"R", "C"}:
            if index < len(records) and records[index]:
                candidates.append(
                    records[index].decode("utf-8", errors="surrogateescape")
                )
                index += 1
        for candidate in candidates:
            portable_candidate = Path(candidate).as_posix()
            if portable_candidate.startswith(GENERATED_PREFIXES) or (
                Path(portable_candidate).suffix.casefold() not in SOURCE_SUFFIXES
            ):
                continue
            path = (PROJECT_ROOT / candidate).resolve()
            if path.is_file():
                hashes[_portable_path(path, base=PROJECT_ROOT)] = sha256_file(path)
    return dict(sorted(hashes.items()))


def git_snapshot() -> dict[str, Any]:
    commit = _run_git("rev-parse", "HEAD")
    status = _run_git("status", "--porcelain=v1", "--untracked-files=all")
    diff = _run_git("diff", "--binary", "HEAD", binary=True)
    changed_file_hashes = _changed_worktree_file_hashes()
    status_text = status if isinstance(status, str) else None
    diff_bytes = diff if isinstance(diff, bytes) else None
    changed_payload = json.dumps(
        changed_file_hashes,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    payload = (
        (status_text or "").encode("utf-8")
        + b"\0"
        + (diff_bytes or b"")
        + b"\0"
        + changed_payload
    )
    return {
        "commit": commit,
        "dirty": bool(status_text) if status_text is not None else None,
        "status_available": status_text is not None,
        "status_porcelain": [] if not status_text else status_text.splitlines(),
        "tracked_diff_sha256": (
            hashlib.sha256(diff_bytes).hexdigest()
            if diff_bytes is not None
            else None
        ),
        "changed_file_sha256": changed_file_hashes,
        "worktree_state_sha256": (
            hashlib.sha256(payload).hexdigest()
            if status_text is not None and diff_bytes is not None
            else None
        ),
    }


def _package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in PACKAGE_NAMES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def runtime_environment() -> dict[str, Any]:
    variables = (
        "APPTAINER_CONTAINER",
        "APPTAINER_NAME",
        "CONTAINER_IMAGE_DIGEST",
        "SINGULARITY_CONTAINER",
        "SINGULARITY_NAME",
        "SLURM_JOB_ID",
        "SLURM_ARRAY_JOB_ID",
        "SLURM_ARRAY_TASK_ID",
        "SLURM_JOB_PARTITION",
        "CUDA_VISIBLE_DEVICES",
    )
    environment = {
        name: os.environ[name] for name in variables if os.environ.get(name)
    }
    return {
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "packages": _package_versions(),
        "torch": torch.__version__,
        "torch_cuda_runtime": torch.version.cuda,
        "torch_cudnn": (
            torch.backends.cudnn.version()
            if torch.backends.cudnn.is_available()
            else None
        ),
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_names": [
            torch.cuda.get_device_name(index)
            for index in range(torch.cuda.device_count())
        ],
        "execution_environment": environment,
    }


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ArtifactFreezeError(f"Expected a JSON object in {path}")
    return value


def _source_master_reports(data_metadata: Mapping[str, Any]) -> list[dict[str, Any]]:
    reports: list[dict[str, Any]] = []
    provenance = data_metadata.get("provenance", {})
    for master in provenance.get("master_files", []):
        stored = master.get("master_report")
        if isinstance(stored, Mapping) and stored.get("sha256"):
            reports.append(dict(stored))
            continue
        master_path_value = master.get("absolute_path")
        if master_path_value:
            master_path = Path(master_path_value)
        else:
            master_dir = provenance.get("master_dir")
            if not master_dir:
                continue
            master_path = Path(master_dir) / str(master.get("relative_path", ""))
        report_path = master_path.with_name(
            master_path.name.replace("_master.csv", "_master_report.json")
        )
        identity = file_identity(report_path, required=False)
        if identity is not None:
            reports.append(identity)
    return reports


def _checked_identity(
    path: Path,
    *,
    expected_sha256: str | None,
    label: str,
) -> dict[str, Any]:
    identity = file_identity(path)
    if expected_sha256 and identity["sha256"] != expected_sha256:
        raise ArtifactFreezeError(
            f"Current {label} differs from dataset provenance"
        )
    return identity


def _dataset_input_identities(
    provenance: Mapping[str, Any], visual: Mapping[str, Any]
) -> dict[str, Any]:
    master_dir_value = provenance.get("master_dir")
    if not master_dir_value:
        raise ArtifactFreezeError("Dataset provenance has no master_dir")
    master_dir = Path(str(master_dir_value)).expanduser().resolve()
    masters: dict[str, Any] = {}
    reports: dict[str, Any] = {}
    for index, stored in enumerate(provenance.get("master_files", [])):
        relative = stored.get("relative_path") or stored.get("file_name")
        if not relative:
            raise ArtifactFreezeError("Master provenance entry has no path")
        key = str(stored.get("sequence_id") or index)
        master_path = master_dir / str(relative)
        masters[key] = _checked_identity(
            master_path,
            expected_sha256=stored.get("sha256"),
            label=f"master {key}",
        )
        stored_report = stored.get("master_report")
        if isinstance(stored_report, Mapping) and stored_report.get("file_name"):
            report_path = master_path.with_name(str(stored_report["file_name"]))
            reports[key] = _checked_identity(
                report_path,
                expected_sha256=stored_report.get("sha256"),
                label=f"master report {key}",
            )

    manifest_identity = None
    stored_manifest = provenance.get("manifest")
    if isinstance(stored_manifest, Mapping) and stored_manifest.get("source_path"):
        manifest_identity = _checked_identity(
            Path(str(stored_manifest["source_path"])),
            expected_sha256=stored_manifest.get("sha256"),
            label="dataset manifest",
        )

    builders: dict[str, Any] = {}
    for relative, expected in provenance.get("builder_file_sha256", {}).items():
        builders[str(relative)] = _checked_identity(
            PROJECT_ROOT / str(relative),
            expected_sha256=str(expected),
            label=f"builder source {relative}",
        )

    visual_inputs: dict[str, Any] = {}
    if visual.get("enabled"):
        required = {
            "cache_manifest": (
                visual.get("cache_manifest_path"),
                visual.get("cache_manifest_sha256"),
            ),
            "projection": (
                visual.get("projection_path"),
                visual.get("projection_sha256"),
            ),
            "projection_metadata": (
                visual.get("projection_metadata_path"),
                visual.get("projection_metadata_sha256"),
            ),
        }
        for name, (path_value, expected) in required.items():
            if not path_value or not expected:
                raise ArtifactFreezeError(
                    f"Visual provenance lacks path/hash for {name}"
                )
            visual_inputs[name] = _checked_identity(
                Path(str(path_value)),
                expected_sha256=str(expected),
                label=f"visual {name}",
            )
    return {
        "master_files": masters,
        "master_reports": reports,
        "dataset_manifest_source": manifest_identity,
        "builder_sources": builders,
        "visual": visual_inputs,
    }


def _command_record(argv: Sequence[str] | None) -> dict[str, Any]:
    arguments = [str(value) for value in (argv if argv is not None else sys.argv)]
    return {
        "python_executable": sys.executable,
        "argv": arguments,
        "shell_command": shlex.join([sys.executable, *arguments]),
        "working_directory": _portable_path(Path.cwd()),
    }


def start_artifact_freeze(
    *,
    run_dir: Path,
    source_config_path: Path,
    run_context: Mapping[str, Any],
    seed: int,
    selection_policy: Mapping[str, Any],
    started_at: str,
    argv: Sequence[str] | None = None,
) -> Path:
    """Write the immutable-input portion of a run manifest."""

    run_dir = Path(run_dir).expanduser().resolve()
    config_path = run_dir / "config.json"
    metadata_path = run_dir / "data_metadata.json"
    provenance_path = run_dir / "dataset_provenance.json"
    for path in (config_path, metadata_path, provenance_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    config = _read_json(config_path)
    metadata = _read_json(metadata_path)
    provenance = metadata.get("provenance", {})
    split = metadata.get("split", {})
    visual = provenance.get("schema", {}).get("visual_features", {"enabled": False})
    frozen_dataset_inputs = _dataset_input_identities(
        provenance,
        visual if isinstance(visual, Mapping) else {"enabled": False},
    )
    manifest = {
        "schema_version": ARTIFACT_FREEZE_SCHEMA_VERSION,
        "protocol": ARTIFACT_FREEZE_PROTOCOL,
        "status": "running",
        "started_at": str(started_at),
        "completed_at": None,
        "run_context": dict(run_context),
        "seed": int(seed),
        "command": _command_record(argv),
        "git": git_snapshot(),
        "runtime": runtime_environment(),
        "selection_policy": dict(selection_policy),
        "configuration": {
            "source": file_identity(source_config_path),
            "resolved": file_identity(config_path, base=run_dir),
            "resolved_fingerprint": canonical_json_hash(config),
        },
        "run_local_inputs": {
            "data_metadata": file_identity(metadata_path, base=run_dir),
            "dataset_provenance": file_identity(provenance_path, base=run_dir),
        },
        "dataset": {
            "identifier": run_context.get("dataset_tag"),
            "builder_version": provenance.get("builder_version"),
            "dataset_content_fingerprint": provenance.get(
                "dataset_content_fingerprint"
            ),
            "source_content_fingerprint": provenance.get(
                "source_content_fingerprint"
            ),
            "dataset_contract": provenance.get("dataset_contract"),
            "master_files": provenance.get("master_files", []),
            "master_reports": _source_master_reports(metadata),
            "manifest": provenance.get("manifest"),
            "sequences": split.get("sequences", {}),
            "participants": split.get("participants", {}),
            "split_strategy": split.get("strategy"),
            "window_eligibility": split.get("window_eligibility", {}),
        },
        "features": {
            "schema": provenance.get("schema", {}),
            "schema_fingerprint": provenance.get("schema", {}).get(
                "fingerprint"
            ),
            "modality_schema": metadata.get("modality_schema", {}),
            "normalizer_fingerprint": canonical_json_hash(
                metadata.get("normalizer", {})
            ),
            "normalizer_fit_split": "train",
            "feature_columns": metadata.get("feature_columns", []),
            "model_feature_columns": metadata.get("model_feature_columns", []),
        },
        "visual": {
            **(visual if isinstance(visual, dict) else {"enabled": False}),
            "required_alignment_version": (
                visual.get("alignment_version")
                if isinstance(visual, dict) and visual.get("enabled")
                else None
            ),
        },
        "pose_target": metadata.get("pose_target", {}),
        "input_artifacts": {
            "data_metadata": file_identity(metadata_path, base=run_dir),
            "dataset_provenance": file_identity(provenance_path, base=run_dir),
            "dataset_manifest_snapshot": file_identity(
                run_dir / "dataset_manifest_snapshot.csv",
                base=run_dir,
                required=False,
            ),
            **frozen_dataset_inputs,
        },
        "output_artifacts": {},
        "manifest_fingerprint": None,
    }
    manifest["manifest_fingerprint"] = canonical_json_hash(
        {**manifest, "manifest_fingerprint": None}
    )
    path = run_dir / MANIFEST_NAME
    path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return path


def finalize_artifact_freeze(
    manifest_path: Path,
    *,
    checkpoint_paths: Mapping[str, Path],
    metrics_path: Path,
    completed_at: str | None = None,
) -> dict[str, Any]:
    """Bind final metrics and every executable/diagnostic checkpoint by hash."""

    manifest_path = Path(manifest_path).expanduser().resolve()
    run_dir = manifest_path.parent
    manifest = _read_json(manifest_path)
    if manifest.get("protocol") != ARTIFACT_FREEZE_PROTOCOL:
        raise ArtifactFreezeError("Cannot finalize an unknown artifact protocol")
    manifest["status"] = "complete"
    manifest["completed_at"] = completed_at or datetime.now(timezone.utc).isoformat()
    manifest["output_artifacts"] = {
        "metrics": file_identity(metrics_path, base=run_dir),
        "checkpoints": {
            str(name): file_identity(path, base=run_dir)
            for name, path in checkpoint_paths.items()
        },
    }
    manifest["manifest_fingerprint"] = None
    manifest["manifest_fingerprint"] = canonical_json_hash(manifest)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    validate_artifact_freeze(manifest_path, require_complete=True)
    return manifest


def _resolve_identity_path(
    identity: Mapping[str, Any], *, run_dir: Path
) -> Path:
    value = Path(str(identity["path"])).expanduser()
    if value.is_absolute():
        return value.resolve()
    # Run-local paths (config.json/checkpoints) take precedence; project paths
    # are used for source configs and repository artifacts.
    local = (run_dir / value).resolve()
    if local.is_file():
        return local
    return (PROJECT_ROOT / value).resolve()


def _validate_identity(
    identity: Mapping[str, Any] | None,
    *,
    run_dir: Path,
    label: str,
) -> None:
    if identity is None:
        return
    if not isinstance(identity, Mapping) or not identity.get("path"):
        raise ArtifactFreezeError(f"Malformed identity for {label}")
    path = _resolve_identity_path(identity, run_dir=run_dir)
    if not path.is_file():
        raise ArtifactFreezeError(f"Frozen artifact is missing for {label}: {path}")
    if int(identity.get("size_bytes", -1)) != int(path.stat().st_size):
        raise ArtifactFreezeError(f"Frozen artifact size changed for {label}")
    if identity.get("sha256") != sha256_file(path):
        raise ArtifactFreezeError(f"Frozen artifact hash changed for {label}")


def _validate_identity_tree(
    value: Any,
    *,
    run_dir: Path,
    label: str,
) -> None:
    if value is None:
        return
    if isinstance(value, Mapping) and value.get("path"):
        _validate_identity(value, run_dir=run_dir, label=label)
        return
    if isinstance(value, Mapping):
        for name, child in value.items():
            _validate_identity_tree(
                child,
                run_dir=run_dir,
                label=f"{label}.{name}",
            )
        return
    if isinstance(value, list):
        for index, child in enumerate(value):
            _validate_identity_tree(
                child,
                run_dir=run_dir,
                label=f"{label}[{index}]",
            )
        return
    raise ArtifactFreezeError(f"Malformed frozen identity tree for {label}")


def _validate_git_state(stored: Mapping[str, Any]) -> None:
    current = git_snapshot()
    for field, description in (
        ("commit", "Git commit"),
        ("tracked_diff_sha256", "tracked source diff"),
    ):
        expected = stored.get(field)
        actual = current.get(field)
        if expected is not None and actual != expected:
            raise ArtifactFreezeError(f"{description} changed since training")
    expected_sources = stored.get("changed_file_sha256", {})
    actual_sources = current.get("changed_file_sha256", {})
    if expected_sources != actual_sources:
        raise ArtifactFreezeError(
            "Dirty/untracked source content changed since training"
        )


def validate_artifact_freeze(
    manifest_path: Path,
    *,
    require_complete: bool = True,
    require_current_git_state: bool = True,
) -> dict[str, Any]:
    """Recompute frozen hashes and validate scientific fields.

    ``require_current_git_state=False`` is intended for retrospective readers
    of immutable run artifacts.  The training commit remains recorded and
    fingerprint-bound in the manifest, while a later reporting-only checkout
    is allowed to validate the run-local inputs and outputs.
    """

    manifest_path = Path(manifest_path).expanduser().resolve()
    manifest = _read_json(manifest_path)
    if manifest.get("schema_version") != ARTIFACT_FREEZE_SCHEMA_VERSION:
        raise ArtifactFreezeError("Unsupported artifact manifest schema")
    if manifest.get("protocol") != ARTIFACT_FREEZE_PROTOCOL:
        raise ArtifactFreezeError("Unsupported artifact freeze protocol")
    if require_complete and manifest.get("status") != "complete":
        raise ArtifactFreezeError("Artifact manifest is not complete")
    fingerprint = manifest.get("manifest_fingerprint")
    expected = canonical_json_hash({**manifest, "manifest_fingerprint": None})
    if fingerprint != expected:
        raise ArtifactFreezeError("Artifact manifest fingerprint mismatch")
    required_values = {
        "dataset.identifier": manifest.get("dataset", {}).get("identifier"),
        "dataset.dataset_content_fingerprint": manifest.get("dataset", {}).get(
            "dataset_content_fingerprint"
        ),
        "dataset.source_content_fingerprint": manifest.get("dataset", {}).get(
            "source_content_fingerprint"
        ),
        "features.schema_fingerprint": manifest.get("features", {}).get(
            "schema_fingerprint"
        ),
        "features.normalizer_fingerprint": manifest.get("features", {}).get(
            "normalizer_fingerprint"
        ),
        "selection_policy.primary_checkpoint": manifest.get(
            "selection_policy", {}
        ).get("primary_checkpoint"),
        "selection_policy.selection_split": manifest.get(
            "selection_policy", {}
        ).get("selection_split"),
        "command.argv": manifest.get("command", {}).get("argv"),
    }
    missing = [key for key, value in required_values.items() if value in (None, "", [])]
    if missing:
        raise ArtifactFreezeError("Required freeze fields are missing: " + ", ".join(missing))
    if required_values["selection_policy.selection_split"] != "validation":
        raise ArtifactFreezeError("Checkpoint selection is not validation-only")

    run_dir = manifest_path.parent
    if require_current_git_state:
        _validate_git_state(manifest.get("git", {}))
    configuration = manifest.get("configuration", {})
    _validate_identity(configuration.get("source"), run_dir=run_dir, label="source config")
    _validate_identity(configuration.get("resolved"), run_dir=run_dir, label="resolved config")
    run_local_inputs = manifest.get("run_local_inputs")
    if not isinstance(run_local_inputs, Mapping):
        raise ArtifactFreezeError("Run-local data inputs are not frozen")
    for name in ("data_metadata", "dataset_provenance"):
        _validate_identity(
            run_local_inputs.get(name),
            run_dir=run_dir,
            label=f"run-local {name}",
        )
    for name, identity in manifest.get("input_artifacts", {}).items():
        _validate_identity_tree(
            identity,
            run_dir=run_dir,
            label=f"input {name}",
        )
    outputs = manifest.get("output_artifacts", {})
    _validate_identity(outputs.get("metrics"), run_dir=run_dir, label="metrics")
    for name, identity in outputs.get("checkpoints", {}).items():
        _validate_identity(identity, run_dir=run_dir, label=f"checkpoint {name}")
    if require_complete and not outputs.get("checkpoints"):
        raise ArtifactFreezeError("Complete manifest has no checkpoint identities")

    visual = manifest.get("visual", {})
    if visual.get("enabled"):
        if not all(
            visual.get(key)
            for key in (
                "cache_manifest_sha256",
                "projection_sha256",
                "alignment_version",
                "alignment_fingerprint",
            )
        ):
            raise ArtifactFreezeError(
                "Visual run lacks hash-bound alignment/projection provenance"
            )
        if visual.get("projection_fit_split") != "train_only":
            raise ArtifactFreezeError(
                "Visual projection was not fitted on train_only"
            )
        split_binding = visual.get("projection_split_binding")
        if not isinstance(split_binding, Mapping) or split_binding.get(
            "verified"
        ) is not True:
            raise ArtifactFreezeError(
                "Visual projection is not verified against the active split"
            )
        for key in (
            "train_sequence_fingerprint",
            "selected_sequence_fingerprint",
        ):
            if not split_binding.get(key):
                raise ArtifactFreezeError(
                    f"Visual projection split binding lacks {key}"
                )
    return manifest


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--allow-running", action="store_true")
    args = parser.parse_args()
    try:
        manifest = validate_artifact_freeze(
            args.manifest, require_complete=not args.allow_running
        )
    except (ArtifactFreezeError, FileNotFoundError, OSError, ValueError) as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}")
        return 2
    print(
        f"Artifact freeze valid: status={manifest['status']} "
        f"fingerprint={manifest['manifest_fingerprint']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
