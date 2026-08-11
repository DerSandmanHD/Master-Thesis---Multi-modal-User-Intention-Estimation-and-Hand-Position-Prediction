from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
TRAINING = ROOT / "Training"
if str(TRAINING) not in sys.path:
    sys.path.insert(0, str(TRAINING))

import artifact_freeze  # noqa: E402
from artifact_freeze import (  # noqa: E402
    ARTIFACT_FREEZE_PROTOCOL,
    ArtifactFreezeError,
    finalize_artifact_freeze,
    start_artifact_freeze,
    validate_artifact_freeze,
)


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def prepared_run(tmp_path: Path) -> tuple[Path, Path]:
    source_config = tmp_path / "source_config.json"
    write_json(source_config, {"training": {"seed": 42}})
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    write_json(
        run_dir / "config.json",
        {
            "training": {"seed": 42},
            "run_context": {
                "dataset_tag": "dataset_x",
                "experiment_tag": "validation_x",
                "model_tag": "model_x",
            },
        },
    )
    master = tmp_path / "P1_1_master.csv"
    master.write_text("timestamp_ns,value\n1,2\n", encoding="utf-8")
    provenance = {
        "builder_version": "builder_test",
        "master_dir": str(tmp_path),
        "dataset_content_fingerprint": "dataset-sha",
        "source_content_fingerprint": "source-sha",
        "master_files": [
            {
                "sequence_id": "P1_1",
                "file_name": master.name,
                "relative_path": master.name,
                "size_bytes": master.stat().st_size,
                "sha256": artifact_freeze.sha256_file(master),
                "master_report": None,
            }
        ],
        "manifest": None,
        "schema": {
            "fingerprint": "schema-sha",
            "modality_schema": {"fingerprint": "modality-sha"},
        },
    }
    write_json(run_dir / "dataset_provenance.json", provenance)
    write_json(
        run_dir / "data_metadata.json",
        {
            "feature_columns": ["gaze_valid"],
            "model_feature_columns": ["gaze_valid", "gaze_valid__observed"],
            "normalizer": {"mean": [0.0], "std": [1.0]},
            "modality_schema": {"fingerprint": "modality-sha"},
            "pose_target": {
                "mode": "future_offset",
                "future_horizon_seconds": 1.0,
            },
            "split": {
                "strategy": "participant_disjoint",
                "sequences": {
                    "train": ["P1_1"],
                    "validation": ["P2_1"],
                    "test": ["P3_1"],
                },
                "participants": {
                    "train": ["P1"],
                    "validation": ["P2"],
                    "test": ["P3"],
                },
            },
            "provenance": provenance,
        },
    )
    return run_dir, source_config


def selection_policy() -> dict:
    return {
        "selection_split": "validation",
        "primary_checkpoint": "best_intention",
        "primary_checkpoint_rule": "maximize validation intention macro-F1",
        "test_used_for_selection": False,
    }


def test_complete_freeze_binds_inputs_command_and_checkpoints(tmp_path: Path) -> None:
    run_dir, source_config = prepared_run(tmp_path)
    manifest_path = start_artifact_freeze(
        run_dir=run_dir,
        source_config_path=source_config,
        run_context={
            "dataset_tag": "dataset_x",
            "experiment_tag": "validation_x",
            "model_tag": "model_x",
        },
        seed=42,
        selection_policy=selection_policy(),
        started_at="2026-08-11T00:00:00+00:00",
        argv=["Training/train_residual.py", "--seed", "42"],
    )
    checkpoint = run_dir / "best_intention_model.pt"
    checkpoint.write_bytes(b"checkpoint")
    metrics = run_dir / "metrics.json"
    write_json(metrics, {"result": "validation selected"})
    finalize_artifact_freeze(
        manifest_path,
        checkpoint_paths={"best_intention": checkpoint},
        metrics_path=metrics,
        completed_at="2026-08-11T00:01:00+00:00",
    )
    manifest = validate_artifact_freeze(manifest_path)
    assert manifest["protocol"] == ARTIFACT_FREEZE_PROTOCOL
    assert manifest["status"] == "complete"
    assert manifest["dataset"]["sequences"]["test"] == ["P3_1"]
    assert manifest["features"]["normalizer_fit_split"] == "train"
    assert manifest["selection_policy"]["test_used_for_selection"] is False
    assert "--seed 42" in manifest["command"]["shell_command"]
    assert manifest["output_artifacts"]["checkpoints"]["best_intention"][
        "sha256"
    ]


def test_modified_checkpoint_invalidates_freeze(tmp_path: Path) -> None:
    run_dir, source_config = prepared_run(tmp_path)
    manifest_path = start_artifact_freeze(
        run_dir=run_dir,
        source_config_path=source_config,
        run_context={
            "dataset_tag": "dataset_x",
            "experiment_tag": "validation_x",
            "model_tag": "model_x",
        },
        seed=42,
        selection_policy=selection_policy(),
        started_at="2026-08-11T00:00:00+00:00",
    )
    checkpoint = run_dir / "best_intention_model.pt"
    checkpoint.write_bytes(b"checkpoint")
    metrics = run_dir / "metrics.json"
    metrics.write_text("{}", encoding="utf-8")
    finalize_artifact_freeze(
        manifest_path,
        checkpoint_paths={"best_intention": checkpoint},
        metrics_path=metrics,
    )
    checkpoint.write_bytes(b"changed checkpoint")
    with pytest.raises(ArtifactFreezeError, match="hash changed|size changed"):
        validate_artifact_freeze(manifest_path)


def test_running_manifest_cannot_be_claimed_as_complete(tmp_path: Path) -> None:
    run_dir, source_config = prepared_run(tmp_path)
    manifest_path = start_artifact_freeze(
        run_dir=run_dir,
        source_config_path=source_config,
        run_context={
            "dataset_tag": "dataset_x",
            "experiment_tag": "validation_x",
            "model_tag": "model_x",
        },
        seed=42,
        selection_policy=selection_policy(),
        started_at="2026-08-11T00:00:00+00:00",
    )
    with pytest.raises(ArtifactFreezeError, match="not complete"):
        validate_artifact_freeze(manifest_path)


def test_visual_freeze_requires_train_only_active_split_binding(
    tmp_path: Path,
) -> None:
    run_dir, source_config = prepared_run(tmp_path)
    manifest_path = start_artifact_freeze(
        run_dir=run_dir,
        source_config_path=source_config,
        run_context={
            "dataset_tag": "dataset_x",
            "experiment_tag": "validation_x",
            "model_tag": "model_x",
        },
        seed=42,
        selection_policy=selection_policy(),
        started_at="2026-08-11T00:00:00+00:00",
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["visual"] = {
        "enabled": True,
        "cache_manifest_sha256": "a" * 64,
        "projection_sha256": "b" * 64,
        "alignment_version": "vrs_rgb_device_time_v2",
        "alignment_fingerprint": "c" * 64,
        "projection_fit_split": "all_splits",
        "projection_split_binding": {"verified": False},
    }
    manifest["manifest_fingerprint"] = artifact_freeze.canonical_json_hash(
        {**manifest, "manifest_fingerprint": None}
    )
    write_json(manifest_path, manifest)
    with pytest.raises(ArtifactFreezeError, match="train_only"):
        validate_artifact_freeze(manifest_path, require_complete=False)

    manifest["visual"]["projection_fit_split"] = "train_only"
    manifest["manifest_fingerprint"] = artifact_freeze.canonical_json_hash(
        {**manifest, "manifest_fingerprint": None}
    )
    write_json(manifest_path, manifest)
    with pytest.raises(ArtifactFreezeError, match="active split"):
        validate_artifact_freeze(manifest_path, require_complete=False)


def test_test_based_selection_is_rejected(tmp_path: Path) -> None:
    run_dir, source_config = prepared_run(tmp_path)
    policy = selection_policy()
    policy["selection_split"] = "test"
    manifest_path = start_artifact_freeze(
        run_dir=run_dir,
        source_config_path=source_config,
        run_context={
            "dataset_tag": "dataset_x",
            "experiment_tag": "validation_x",
            "model_tag": "model_x",
        },
        seed=42,
        selection_policy=policy,
        started_at="2026-08-11T00:00:00+00:00",
    )
    with pytest.raises(ArtifactFreezeError, match="validation-only"):
        validate_artifact_freeze(manifest_path, require_complete=False)


def test_modified_master_invalidates_freeze(tmp_path: Path) -> None:
    run_dir, source_config = prepared_run(tmp_path)
    manifest_path = start_artifact_freeze(
        run_dir=run_dir,
        source_config_path=source_config,
        run_context={
            "dataset_tag": "dataset_x",
            "experiment_tag": "validation_x",
            "model_tag": "model_x",
        },
        seed=42,
        selection_policy=selection_policy(),
        started_at="2026-08-11T00:00:00+00:00",
    )
    (tmp_path / "P1_1_master.csv").write_text(
        "timestamp_ns,value\n1,changed\n", encoding="utf-8"
    )
    with pytest.raises(ArtifactFreezeError, match="input master_files"):
        validate_artifact_freeze(manifest_path, require_complete=False)


def test_modified_normalizer_metadata_invalidates_freeze(tmp_path: Path) -> None:
    run_dir, source_config = prepared_run(tmp_path)
    manifest_path = start_artifact_freeze(
        run_dir=run_dir,
        source_config_path=source_config,
        run_context={
            "dataset_tag": "dataset_x",
            "experiment_tag": "validation_x",
            "model_tag": "model_x",
        },
        seed=42,
        selection_policy=selection_policy(),
        started_at="2026-08-11T00:00:00+00:00",
    )
    metadata_path = run_dir / "data_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["normalizer"]["mean"] = [123.0]
    write_json(metadata_path, metadata)
    with pytest.raises(ArtifactFreezeError, match="run-local data_metadata"):
        validate_artifact_freeze(manifest_path, require_complete=False)


def test_changed_git_source_state_invalidates_freeze(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir, source_config = prepared_run(tmp_path)
    manifest_path = start_artifact_freeze(
        run_dir=run_dir,
        source_config_path=source_config,
        run_context={
            "dataset_tag": "dataset_x",
            "experiment_tag": "validation_x",
            "model_tag": "model_x",
        },
        seed=42,
        selection_policy=selection_policy(),
        started_at="2026-08-11T00:00:00+00:00",
    )
    stored = json.loads(manifest_path.read_text(encoding="utf-8"))["git"]
    changed = {**stored, "tracked_diff_sha256": "0" * 64}
    monkeypatch.setattr(artifact_freeze, "git_snapshot", lambda: changed)
    with pytest.raises(ArtifactFreezeError, match="tracked source diff"):
        validate_artifact_freeze(manifest_path, require_complete=False)


def test_dirty_snapshot_hashes_untracked_source_content(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "new_source.py"
    source.write_text("VALUE = 1\n", encoding="utf-8")

    def fake_git(*arguments: str, binary: bool = False):
        assert binary is True
        assert "--porcelain=v1" in arguments
        return b"?? new_source.py\0"

    monkeypatch.setattr(artifact_freeze, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(artifact_freeze, "_run_git", fake_git)
    first = artifact_freeze._changed_worktree_file_hashes()
    source.write_text("VALUE = 2\n", encoding="utf-8")
    second = artifact_freeze._changed_worktree_file_hashes()

    assert set(first) == {"new_source.py"}
    assert first["new_source.py"] != second["new_source.py"]
