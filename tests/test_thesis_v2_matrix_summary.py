from __future__ import annotations

import json
import hashlib
import sys
from pathlib import Path

import pytest
import pandas as pd
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
TRAINING = ROOT / "Training"
EVALUATION = TRAINING / "evaluation"
for path in (TRAINING, EVALUATION):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

TEST_ENDPOINT_FINGERPRINT = hashlib.sha256(
    "\n".join(f"S1:{index}" for index in range(1, 6)).encode("utf-8")
).hexdigest()

from artifact_freeze import canonical_json_hash, sha256_file  # noqa: E402
from experiment_matrix import run_directory  # noqa: E402
from summarize_thesis_v2_matrix import (  # noqa: E402
    FINAL_TEST_PROTOCOL,
    MatrixSummaryError,
    _classification_fields,
    build_matrix_summary,
    validate_historical_artifact_freeze,
    write_outputs,
)
from grouped_metrics import (  # noqa: E402
    discover_pose_methods,
    prepare_prediction_frame,
    summarize_windows,
)
from pose_baselines import sample_key_fingerprint  # noqa: E402


def write_final_report(path: Path, report: dict) -> None:
    report["report_fingerprint"] = None
    report["report_fingerprint"] = canonical_json_hash(report)
    path.write_text(json.dumps(report), encoding="utf-8")


def classification(confusion: list[list[int]], names: list[str]) -> dict:
    samples = sum(sum(row) for row in confusion)
    per_precision = []
    per_recall = []
    per_f1 = []
    support = []
    for index in range(len(names)):
        true_positive = confusion[index][index]
        actual = sum(confusion[index])
        predicted = sum(row[index] for row in confusion)
        precision = true_positive / predicted if predicted else 0.0
        recall = true_positive / actual if actual else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        per_precision.append(precision)
        per_recall.append(recall)
        per_f1.append(f1)
        support.append(actual)
    return {
        "accuracy": sum(confusion[i][i] for i in range(len(names))) / samples,
        "macro_f1": sum(per_f1) / len(names),
        "macro_f1_supported": sum(per_f1) / len(names),
        "per_class_precision": per_precision,
        "per_class_recall": per_recall,
        "per_class_f1": per_f1,
        "support": support,
        "confusion_matrix": confusion,
        "samples": samples,
        "class_names": names,
    }


def pose(mean: float, samples: int) -> dict:
    return {
        "position_mean_euclidean_error_cm": mean,
        "position_mae_cm": mean,
        "position_root_mean_square_euclidean_error_cm": mean + 0.5,
        "position_rmse_cm": mean + 0.5,
        "position_median_cm": mean - 0.5,
        "orientation_mean_deg": mean + 10.0,
        "orientation_median_deg": mean + 9.0,
        "samples": samples,
    }


def sample_metrics(*, terminal: bool, offset: float) -> dict:
    values = {
        "assistance": classification(
            [[8, 2], [1, 9]], ["continue", "assistance"]
        ),
        "intention": classification(
            [[7, 1, 0], [1, 6, 1], [0, 1, 7]],
            ["continue", "fetch", "handover"],
        ),
        "assistance_type": classification(
            [[6, 1], [1, 7]], ["fetch", "handover"]
        ),
        "receiving_hand": classification([[4, 1], [1, 4]], ["left", "right"]),
        "pose_end_to_end": pose(7.0 + offset, 8),
        "pose_oracle": pose(6.0 + offset, 9),
        "pose_coverage": {
            "pose_targets": 10,
            "oracle_reference_valid": 9,
            "predicted_reference_valid": 8,
        },
        "pose_fixed_both_references": {
            **pose(7.0 + offset, 7),
            "cohort_definition": (
                "pose_target_valid_and_both_hand_references_valid"
            ),
            "cohort_model_dependent": False,
            "coverage_denominator_pose_targets": 10,
            "sample_key_fingerprint": "f" * 64,
        },
    }
    if terminal:
        values["pose_fair_common"] = {
            "comparison_role": "paired terminal comparison",
            "receiving_hand_context": {
                "learned_end_to_end": "predicted receiving hand",
                "persistence": "ground-truth receiving hand",
            },
            "shared_samples": 7,
            "coverage_denominator_pose_targets": 10,
            "sample_key_fingerprint": "f" * 64,
            "methods": {
                "learned_oracle_hand": pose(6.5 + offset, 7),
                "learned_end_to_end": pose(7.5 + offset, 7),
                "persistence": pose(8.5 + offset, 7),
            },
        }
        values["pose_by_terminal_target_regime"] = {
            "strictly_before_aggregation": {
                "interpretation": "pure future terminal forecast",
                "shared_samples": 4,
                "coverage_denominator_pose_targets": 6,
                "methods": {
                    "learned_oracle_hand": pose(6.0 + offset, 4),
                    "learned_end_to_end": pose(7.0 + offset, 4),
                    "persistence": pose(8.0 + offset, 4),
                },
            },
            "partially_overlapping_aggregation": {
                "interpretation": "partial target evidence",
                "shared_samples": 3,
                "coverage_denominator_pose_targets": 4,
                "methods": {
                    "learned_oracle_hand": pose(5.0 + offset, 3),
                    "learned_end_to_end": pose(6.0 + offset, 3),
                    "persistence": pose(7.0 + offset, 3),
                },
            },
        }
    return values


def build_fixture(tmp_path: Path) -> dict:
    tmp_path.mkdir(parents=True, exist_ok=True)
    matrix = {
        "matrix_id": "mini_thesis_v2",
        "dataset_tag": "mini_dataset",
        "validation_experiment_tag": "mini_validation",
        "seeds": [42, 43],
        "training_experiments": [
            {
                "id": "primary_model",
                "family": "primary",
                "factor": "fusion",
                "variant": "current",
            },
            {
                "id": "terminal_model",
                "family": "secondary_endpose",
                "factor": "terminal_pose_method",
                "variant": "learned",
            },
        ],
    }
    matrix_path = tmp_path / "matrix.json"
    matrix_path.write_text(json.dumps(matrix), encoding="utf-8")
    selection_path = tmp_path / "validation_selection.json"
    final_dir = tmp_path / "final_test"
    final_dir.mkdir()
    authorizations = []
    identities = {}
    index = 1
    for experiment in matrix["training_experiments"]:
        for seed in matrix["seeds"]:
            run_relative = run_directory(matrix, experiment["id"], seed)
            run_path = tmp_path / run_relative
            run_path.mkdir(parents=True, exist_ok=True)
            checkpoint_path = run_path / "best_intention_model.pt"
            checkpoint_path.write_bytes(
                f"{experiment['id']}:{seed}:checkpoint".encode("utf-8")
            )
            checkpoint_hash = sha256_file(checkpoint_path)
            dataset_content = (
                "terminal-derived-target-data"
                if experiment["family"] == "secondary_endpose"
                else "dataset-content"
            )
            manifest = {
                "status": "complete",
                "dataset": {
                    "identifier": matrix["dataset_tag"],
                    "dataset_content_fingerprint": dataset_content,
                    "source_content_fingerprint": "shared-source-content",
                    "window_eligibility": {
                        "endpoint_fingerprints": {
                            "test": TEST_ENDPOINT_FINGERPRINT,
                        },
                        "endpoint_counts": {"test": 5},
                    },
                },
                "output_artifacts": {
                    "checkpoints": {
                        "best_intention": {"sha256": checkpoint_hash}
                    }
                },
                "manifest_fingerprint": None,
            }
            fingerprint = canonical_json_hash(manifest)
            manifest["manifest_fingerprint"] = fingerprint
            (run_path / "artifact_manifest.json").write_text(
                json.dumps(manifest), encoding="utf-8"
            )
            authorizations.append(
                {
                    "experiment_id": experiment["id"],
                    "seed": seed,
                    "run_dir": run_relative,
                    "checkpoint_name": "best_intention",
                    "checkpoint_path": "best_intention_model.pt",
                    "checkpoint_sha256": checkpoint_hash,
                    "checkpoint_epoch": index,
                    "checkpoint_selection_metric": "validation_intention_macro_f1",
                    "checkpoint_selection_value": 0.8 + index / 100.0,
                    "artifact_manifest_fingerprint": fingerprint,
                }
            )
            identities[(experiment["id"], seed)] = {
                "hash": checkpoint_hash,
                "fingerprint": fingerprint,
                "run_relative": run_relative,
                "epoch": index,
                "value": 0.8 + index / 100.0,
                "dataset_content": dataset_content,
            }
            index += 1
    selection = {
        "schema_version": 2,
        "matrix_id": matrix["matrix_id"],
        "matrix_file": str(matrix_path),
        "matrix_sha256": sha256_file(matrix_path),
        "dataset_tag": matrix["dataset_tag"],
        "complete": True,
        "selection_split": "validation",
        "test_metrics_read": False,
        "final_test_runs": authorizations,
    }
    selection_path.write_text(json.dumps(selection), encoding="utf-8")
    selection_hash = sha256_file(selection_path)
    reports = {}
    for experiment in matrix["training_experiments"]:
        for seed in matrix["seeds"]:
            identity = identities[(experiment["id"], seed)]
            run_path = tmp_path / identity["run_relative"]
            report = {
                "schema_version": 2,
                "report_fingerprint": None,
                "evaluation_protocol": FINAL_TEST_PROTOCOL,
                "split": "test",
                "source_run": str(run_path),
                "source_artifact_manifest": str(
                    run_path / "artifact_manifest.json"
                ),
                "source_artifact_manifest_fingerprint": identity["fingerprint"],
                "dataset_identifier": matrix["dataset_tag"],
                "dataset_content_fingerprint": identity["dataset_content"],
                "source_content_fingerprint": "shared-source-content",
                "model_type": "synthetic",
                "trainable_parameters": 100,
                "training_task_semantics": {
                    "future_pose_loss_enabled": True,
                    "future_pose_loss_weight": 1.0,
                    "auxiliary_pose_loss_enabled": False,
                    "auxiliary_pose_loss_weight": 0.0,
                    "pose_metrics_role": "main_learned_output",
                },
                "checkpoint": {
                    "name": "best_intention",
                    "path": str(run_path / "best_intention_model.pt"),
                    "sha256": identity["hash"],
                    "epoch": identity["epoch"],
                    "selection_split": "validation",
                    "selection_metric": "validation_intention_macro_f1",
                    "selection_value": identity["value"],
                },
                "matrix_authorization": {
                    "selection_file": str(selection_path),
                    "selection_file_sha256": selection_hash,
                    "matrix_id": matrix["matrix_id"],
                    "experiment_id": experiment["id"],
                    "seed": seed,
                    "authorized_checkpoint_sha256": identity["hash"],
                    "test_metrics_read_during_authorization": False,
                },
                "test_metrics": sample_metrics(
                    terminal=experiment["family"] == "secondary_endpose",
                    offset=(seed - 42) * 0.5,
                ),
                "test_used_for_model_or_checkpoint_selection": False,
            }
            path = final_dir / f"{experiment['id']}_seed{seed}.json"
            write_final_report(path, report)
            reports[(experiment["id"], seed)] = path
    return {
        "matrix": matrix,
        "matrix_path": matrix_path,
        "selection_path": selection_path,
        "final_dir": final_dir,
        "identities": identities,
        "reports": reports,
        "project_root": tmp_path,
    }


def add_grouped_report(fixture: dict, experiment_id: str, seed: int) -> Path:
    root = fixture["project_root"] / "postprocess"
    report_dir = root / f"{experiment_id}_seed{seed}"
    report_dir.mkdir(parents=True)
    predictions = report_dir / "test_predictions.csv"
    predictions.write_text(
        "participant,sequence_id,endpoint_timestamp_ns,sample_key,target_intention_id,"
        "predicted_intention_id,target_receiving_hand,predicted_receiving_hand,"
        "pose_valid,fair_common,oracle_position_error_cm,"
        "oracle_orientation_error_deg,persistence_position_error_cm,"
        "persistence_orientation_error_deg,constant_velocity_position_error_cm,"
        "constant_velocity_orientation_error_deg\n"
        "P1,S1,1,a,2,2,left,left,true,true,5,15,7,17,6,16\n"
        "P1,S1,2,b,2,2,left,left,true,true,6,16,8,18,7,17\n"
        "P1,S1,3,c,2,2,left,left,true,true,7,17,9,19,8,18\n"
        "P1,S1,4,d,2,2,left,left,true,true,6,16,8,18,7,17\n"
        "P1,S1,5,e,2,2,left,left,true,false,10,20,10,20,,\n",
        encoding="utf-8",
    )
    predictions_hash = sha256_file(predictions)
    normalized, _ = prepare_prediction_frame(pd.read_csv(predictions))
    fair_common = summarize_windows(
        normalized, discover_pose_methods(normalized)
    )["pose_fair_common"]
    identity = fixture["identities"][(experiment_id, seed)]
    sidecar = report_dir / "test_predictions.json"
    final_report_path = fixture["reports"][(experiment_id, seed)]
    final_report = json.loads(final_report_path.read_text(encoding="utf-8"))
    sidecar_payload = {
                "schema_version": 3,
                "report_fingerprint": None,
                "result_role": "primary_validation_selected_checkpoint",
                "checkpoint": "best_intention_model.pt",
                "checkpoint_sha256": identity["hash"],
                "checkpoint_epoch": identity["epoch"],
                "checkpoint_selection_split": "validation",
                "checkpoint_selection_metric": "validation_intention_macro_f1",
                "checkpoint_selection_value": identity["value"],
                "predictions_csv": str(predictions),
                "predictions_csv_sha256": predictions_hash,
                "rows": 5,
                "split": "test",
                "dataset_content_fingerprint": "dataset-content",
                "source_content_fingerprint": "shared-source-content",
                "artifact_freeze": {
                    "manifest_fingerprint": identity["fingerprint"],
                },
                "final_test_authorization": {
                    "path": str(final_report_path),
                    "sha256": sha256_file(final_report_path),
                    "report_fingerprint": final_report["report_fingerprint"],
                    "evaluation_protocol": FINAL_TEST_PROTOCOL,
                    "matrix_authorization": final_report[
                        "matrix_authorization"
                    ],
                },
                "full_split_export": True,
                "sequence_filter": [],
                "frozen_split_endpoint_fingerprint": TEST_ENDPOINT_FINGERPRINT,
                "exported_endpoint_fingerprint": TEST_ENDPOINT_FINGERPRINT,
                "frozen_split_endpoint_count": 5,
                "exported_endpoint_count": 5,
                "pose_comparison": {
                    "fair_common_sample_key_fingerprint": (
                        sample_key_fingerprint(["a", "b", "c", "d"])
                    )
                },
                "baseline_policy": {
                    "maximum_observation_age_seconds": 0.5,
                    "velocity_lookback_seconds": 0.5,
                    "minimum_velocity_fit_span_seconds": 0.1,
                    "timestamp_basis": "hand_timestamp_ns source captures",
                },
            }
    sidecar_payload["report_fingerprint"] = canonical_json_hash(sidecar_payload)
    sidecar.write_text(json.dumps(sidecar_payload), encoding="utf-8")
    sidecar_hash = sha256_file(sidecar)
    report = {
        "schema_version": "grouped_prediction_evaluation_v1",
        "predictions_csv": str(predictions),
        "predictions_csv_sha256": predictions_hash,
        "prediction_rows": 5,
        "checkpoint_binding": {
            "status": "bound_single_checkpoint",
            "result_role": "checkpoint_bound_grouped_primary",
            "source_prediction_report": str(sidecar),
            "source_prediction_report_sha256": sidecar_hash,
            "predictions_csv_sha256": predictions_hash,
            "checkpoint_sha256": identity["hash"],
            "checkpoint_selection_split": "validation",
            "checkpoint_selection_metric": "validation_intention_macro_f1",
            "dataset_content_fingerprint": "dataset-content",
            "split": "test",
        },
        "window_level": {
            "pose_fair_common": fair_common
        },
    }
    grouped = report_dir / "grouped_metrics.json"
    grouped.write_text(json.dumps(report), encoding="utf-8")
    return root


def summarize(fixture: dict, postprocess_root: Path | None = None) -> dict:
    return build_matrix_summary(
        matrix=fixture["matrix"],
        matrix_path=fixture["matrix_path"],
        selection_path=fixture["selection_path"],
        final_test_dir=fixture["final_dir"],
        postprocess_root=postprocess_root,
        project_root=fixture["project_root"],
        artifact_validator=lambda path: json.loads(
            path.read_text(encoding="utf-8")
        ),
    )


def test_historical_artifact_validator_does_not_require_training_checkout() -> None:
    manifest = Path("historical/artifact_manifest.json")
    with patch("summarize_thesis_v2_matrix.validate_artifact_freeze") as validate:
        validate.return_value = {"status": "complete"}
        assert validate_historical_artifact_freeze(manifest) == {"status": "complete"}
        validate.assert_called_once_with(manifest, require_current_git_state=False)


def test_authoritative_summary_keeps_seed_rows_and_aggregates_separate(
    tmp_path: Path,
) -> None:
    fixture = build_fixture(tmp_path)
    postprocess = add_grouped_report(fixture, "primary_model", 42)
    summary = summarize(fixture, postprocess)
    assert summary["one_row_one_checkpoint"] is True
    assert summary["seed_row_count"] == 4
    assert len({row["checkpoint_sha256"] for row in summary["seed_rows"]}) == 4
    primary_42 = next(
        row
        for row in summary["seed_rows"]
        if row["experiment_id"] == "primary_model" and row["seed"] == 42
    )
    assert primary_42["t1_fair_common_status"] == "available_checkpoint_bound"
    assert primary_42["t1_fair_learned_model_position_mean_cm"] == 6.0
    assert primary_42["t1_fair_persistence_position_mean_cm"] == 8.0
    assert primary_42["t1_fair_constant_velocity_position_mean_cm"] == 7.0
    primary_43 = next(
        row
        for row in summary["seed_rows"]
        if row["experiment_id"] == "primary_model" and row["seed"] == 43
    )
    assert primary_43["t1_fair_common_status"] == "not_available"
    assert primary_43["t1_fair_persistence_position_mean_cm"] is None
    terminal = next(
        row
        for row in summary["seed_rows"]
        if row["experiment_id"] == "terminal_model" and row["seed"] == 42
    )
    assert terminal["thesis_task_role"] == "secondary_terminal_endpose"
    assert terminal["terminal_fair_shared_samples"] == 7
    assert terminal["terminal_fair_learned_oracle_hand_samples"] == 7
    assert terminal["terminal_fair_learned_end_to_end_samples"] == 7
    assert terminal["terminal_fair_persistence_samples"] == 7
    assert terminal["terminal_main_pose_reporting_regime"] == (
        "strictly_before_aggregation"
    )
    assert terminal[
        "terminal_fair_strictly_before_aggregation_learned_end_to_end_samples"
    ] == 4
    assert terminal[
        "terminal_fair_partially_overlapping_aggregation_persistence_samples"
    ] == 3
    aggregates = summary["seed_aggregation"]["rows"]
    assert all(
        row["result_semantics"]
        == "across_seed_summary_not_an_executable_checkpoint"
        for row in aggregates
    )
    assert next(
        row for row in aggregates if row["experiment_id"] == "primary_model"
    )["metrics"]["test_pose_position_mean_cm"]["n"] == 2

    output = tmp_path / "summary"
    paths = write_outputs(summary, output)
    assert all(path.is_file() for path in paths.values())
    seed_payload = json.loads(paths["seed_json"].read_text(encoding="utf-8"))
    assert len(seed_payload["seed_rows"]) == 4
    aggregate_payload = json.loads(
        paths["aggregate_json"].read_text(encoding="utf-8")
    )
    assert aggregate_payload["matrix"]["sha256"] == summary["matrix"]["sha256"]
    assert aggregate_payload["validation_selection"]["sha256"] == summary[
        "validation_selection"
    ]["sha256"]
    assert aggregate_payload["source_seed_results_sha256"] == sha256_file(
        paths["seed_json"]
    )
    artifact_manifest = json.loads(
        paths["artifact_manifest"].read_text(encoding="utf-8")
    )
    assert artifact_manifest["outputs"]["markdown"]["sha256"] == sha256_file(
        paths["markdown"]
    )
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        write_outputs(summary, output)
    write_outputs(summary, output, overwrite=True)


def test_required_t1_postprocessing_is_complete_or_summary_fails(
    tmp_path: Path,
) -> None:
    fixture = build_fixture(tmp_path)
    fixture["matrix"]["postprocessing"] = {
        "required_t1_experiments": ["primary_model"],
        "seed_policy": "all_matrix_seeds",
        "require_grouped_report_in_authoritative_summary": True,
    }
    with pytest.raises(MatrixSummaryError, match="no postprocess root"):
        summarize(fixture)
    postprocess = add_grouped_report(fixture, "primary_model", 42)
    with pytest.raises(MatrixSummaryError, match="primary_model seed 43"):
        summarize(fixture, postprocess)
    add_grouped_report(fixture, "primary_model", 43)
    summary = summarize(fixture, postprocess)
    required = [
        row
        for row in summary["seed_rows"]
        if row["experiment_id"] == "primary_model"
    ]
    assert len(required) == 2
    assert all(
        row["t1_fair_common_status"] == "available_checkpoint_bound"
        for row in required
    )


def test_pose_loss_off_excludes_untrained_pose_head_from_main_results(
    tmp_path: Path,
) -> None:
    fixture = build_fixture(tmp_path)
    postprocess = add_grouped_report(fixture, "primary_model", 42)
    for seed in fixture["matrix"]["seeds"]:
        path = fixture["reports"][("primary_model", seed)]
        report = json.loads(path.read_text(encoding="utf-8"))
        report["training_task_semantics"].update(
            {
                "future_pose_loss_enabled": False,
                "future_pose_loss_weight": 0.0,
                "pose_metrics_role": "untrained_pose_head_diagnostic_only",
            }
        )
        write_final_report(path, report)

    summary = summarize(fixture, postprocess)
    rows = [
        row
        for row in summary["seed_rows"]
        if row["experiment_id"] == "primary_model"
    ]
    assert rows
    for row in rows:
        assert row["test_pose_position_mean_cm"] is None
        assert row["test_pose_coverage"] is None
        assert row["diagnostic_untrained_pose_position_mean_cm"] is not None
        assert row["t1_fair_common_status"] == (
            "untrained_pose_head_diagnostic_excluded"
        )
        assert row["t1_fair_learned_model_position_mean_cm"] is None


def test_final_report_metric_tampering_invalidates_report_fingerprint(
    tmp_path: Path,
) -> None:
    fixture = build_fixture(tmp_path)
    path = fixture["reports"][("primary_model", 42)]
    report = json.loads(path.read_text(encoding="utf-8"))
    report["test_metrics"]["intention"]["macro_f1"] = 0.999
    # Deliberately do not refresh report_fingerprint.
    path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(MatrixSummaryError, match="fingerprint mismatch"):
        summarize(fixture)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda report: report["checkpoint"].update({"sha256": "f" * 64}),
            "mixes authorization and checkpoint hashes",
        ),
        (
            lambda report: report.update({"source_run": "another/run"}),
            "source_run differs",
        ),
        (
            lambda report: report["matrix_authorization"].update({"seed": 99}),
            "another seed",
        ),
        (
            lambda report: report.update(
                {"source_artifact_manifest_fingerprint": "another-manifest"}
            ),
            "not exactly authorized",
        ),
    ],
)
def test_mismatched_or_mixed_final_checkpoint_report_is_rejected(
    tmp_path: Path, mutation, message: str
) -> None:
    fixture = build_fixture(tmp_path)
    path = fixture["reports"][("primary_model", 42)]
    report = json.loads(path.read_text(encoding="utf-8"))
    mutation(report)
    write_final_report(path, report)
    with pytest.raises(MatrixSummaryError, match=message):
        summarize(fixture)


def test_selection_matrix_hash_and_duplicate_report_are_rejected(
    tmp_path: Path,
) -> None:
    fixture = build_fixture(tmp_path)
    selection = json.loads(fixture["selection_path"].read_text(encoding="utf-8"))
    selection["matrix_sha256"] = "0" * 64
    fixture["selection_path"].write_text(json.dumps(selection), encoding="utf-8")
    with pytest.raises(MatrixSummaryError, match="another matrix file hash"):
        summarize(fixture)

    fixture = build_fixture(tmp_path / "duplicate")
    source = fixture["reports"][("primary_model", 42)]
    duplicate = fixture["final_dir"] / "copied_final_report.json"
    duplicate.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
    with pytest.raises(MatrixSummaryError, match="Unexpected/duplicate"):
        summarize(fixture)


def test_changed_prediction_csv_invalidates_optional_grouped_baselines(
    tmp_path: Path,
) -> None:
    fixture = build_fixture(tmp_path)
    postprocess = add_grouped_report(fixture, "primary_model", 42)
    predictions = (
        postprocess / "primary_model_seed42" / "test_predictions.csv"
    )
    predictions.write_text("sample_key\nchanged\n", encoding="utf-8")
    with pytest.raises(MatrixSummaryError, match="prediction CSV hash is stale"):
        summarize(fixture, postprocess)


def test_changed_grouped_json_metric_is_rejected_against_prediction_csv(
    tmp_path: Path,
) -> None:
    fixture = build_fixture(tmp_path)
    postprocess = add_grouped_report(fixture, "primary_model", 42)
    grouped_path = postprocess / "primary_model_seed42" / "grouped_metrics.json"
    grouped = json.loads(grouped_path.read_text(encoding="utf-8"))
    grouped["window_level"]["pose_fair_common"]["methods"]["persistence"][
        "position_mean_cm"
    ] += 1.0
    grouped_path.write_text(json.dumps(grouped), encoding="utf-8")
    with pytest.raises(MatrixSummaryError, match="differs from the hash-bound"):
        summarize(fixture, postprocess)


@pytest.mark.parametrize(
    ("cell", "message"),
    [(-1, "nonnegative integer"), (1.5, "nonnegative integer")],
)
def test_invalid_confusion_counts_are_rejected(
    tmp_path: Path, cell: float, message: str
) -> None:
    fixture = build_fixture(tmp_path)
    path = fixture["reports"][("primary_model", 42)]
    report = json.loads(path.read_text(encoding="utf-8"))
    report["test_metrics"]["intention"]["confusion_matrix"][0][0] = cell
    write_final_report(path, report)
    with pytest.raises(MatrixSummaryError, match=message):
        summarize(fixture)


@pytest.mark.parametrize("metric", ["accuracy", "macro_f1", "macro_f1_supported"])
def test_classification_summary_is_recomputed_from_confusion(
    tmp_path: Path, metric: str
) -> None:
    fixture = build_fixture(tmp_path)
    path = fixture["reports"][("primary_model", 42)]
    report = json.loads(path.read_text(encoding="utf-8"))
    report["test_metrics"]["intention"][metric] = 0.123456
    write_final_report(path, report)
    with pytest.raises(MatrixSummaryError, match=f"Stored {metric} conflicts"):
        summarize(fixture)


def test_zero_sample_empty_classification_is_normalized_to_undefined() -> None:
    values = {
        "accuracy": None,
        "macro_f1": None,
        "macro_f1_supported": None,
        "per_class_precision": [],
        "per_class_recall": [],
        "per_class_f1": [],
        "support": [],
        "samples": 0,
        "confusion_matrix": [],
        "class_names": ["fetch", "handover"],
    }
    fields = _classification_fields(
        values,
        prefix="test_assistance_type",
        expected_names=("fetch", "handover"),
    )
    assert fields["test_assistance_type_accuracy"] is None
    assert fields["test_assistance_type_macro_f1"] is None
    assert fields["test_assistance_type_fetch_f1"] is None
    assert fields["test_assistance_type_samples"] == 0


def test_dataset_fingerprint_is_required_and_constant_only_within_experiment(
    tmp_path: Path,
) -> None:
    fixture = build_fixture(tmp_path / "missing")
    path = fixture["reports"][("primary_model", 42)]
    report = json.loads(path.read_text(encoding="utf-8"))
    report["dataset_content_fingerprint"] = ""
    write_final_report(path, report)
    with pytest.raises(MatrixSummaryError, match="no dataset_content_fingerprint"):
        summarize(fixture)

    fixture = build_fixture(tmp_path / "inconsistent")
    path = fixture["reports"][("primary_model", 43)]
    report = json.loads(path.read_text(encoding="utf-8"))
    report["dataset_content_fingerprint"] = "different-primary-data"
    write_final_report(path, report)
    with pytest.raises(MatrixSummaryError, match="differs from source manifest"):
        summarize(fixture)

    fixture = build_fixture(tmp_path / "per_experiment")
    summary = summarize(fixture)
    assert summary["seed_row_count"] == 4
    assert {
        row["dataset_content_fingerprint"] for row in summary["seed_rows"]
    } == {"dataset-content", "terminal-derived-target-data"}


def test_source_content_must_match_across_all_matrix_cells(tmp_path: Path) -> None:
    fixture = build_fixture(tmp_path)
    report_path = fixture["reports"][("primary_model", 42)]
    report = json.loads(report_path.read_text(encoding="utf-8"))
    manifest_path = Path(report["source_artifact_manifest"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["dataset"]["source_content_fingerprint"] = "different-source"
    manifest["manifest_fingerprint"] = canonical_json_hash(
        {**manifest, "manifest_fingerprint": None}
    )
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    report["source_content_fingerprint"] = "different-source"
    report["source_artifact_manifest_fingerprint"] = manifest[
        "manifest_fingerprint"
    ]
    write_final_report(report_path, report)

    selection_path = fixture["selection_path"]
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    authorization = next(
        row
        for row in selection["final_test_runs"]
        if row["experiment_id"] == "primary_model" and row["seed"] == 42
    )
    authorization["artifact_manifest_fingerprint"] = manifest[
        "manifest_fingerprint"
    ]
    selection_path.write_text(json.dumps(selection), encoding="utf-8")
    selection_hash = sha256_file(selection_path)
    for path in fixture["reports"].values():
        value = json.loads(path.read_text(encoding="utf-8"))
        value["matrix_authorization"]["selection_file_sha256"] = selection_hash
        write_final_report(path, value)

    with pytest.raises(MatrixSummaryError, match="differs across matrix cells"):
        summarize(fixture)


def test_terminal_fair_denominator_must_match_pose_coverage(tmp_path: Path) -> None:
    fixture = build_fixture(tmp_path)
    path = fixture["reports"][("terminal_model", 42)]
    report = json.loads(path.read_text(encoding="utf-8"))
    report["test_metrics"]["pose_fair_common"][
        "coverage_denominator_pose_targets"
    ] = 5
    write_final_report(path, report)
    with pytest.raises(MatrixSummaryError, match="denominator differs"):
        summarize(fixture)


def test_main_pose_counts_cannot_exceed_or_disagree_with_coverage(
    tmp_path: Path,
) -> None:
    fixture = build_fixture(tmp_path / "too_many")
    path = fixture["reports"][("primary_model", 42)]
    report = json.loads(path.read_text(encoding="utf-8"))
    report["test_metrics"]["pose_coverage"]["pose_targets"] = 5
    write_final_report(path, report)
    with pytest.raises(MatrixSummaryError, match="outside the pose-target"):
        summarize(fixture)

    fixture = build_fixture(tmp_path / "reference_count")
    path = fixture["reports"][("primary_model", 42)]
    report = json.loads(path.read_text(encoding="utf-8"))
    report["test_metrics"]["pose_coverage"]["predicted_reference_valid"] = 7
    write_final_report(path, report)
    with pytest.raises(MatrixSummaryError, match="predicted_reference_valid"):
        summarize(fixture)


def test_grouped_binding_requires_exact_dataset_and_actual_row_count(
    tmp_path: Path,
) -> None:
    fixture = build_fixture(tmp_path / "dataset")
    postprocess = add_grouped_report(fixture, "primary_model", 42)
    grouped_path = postprocess / "primary_model_seed42" / "grouped_metrics.json"
    grouped = json.loads(grouped_path.read_text(encoding="utf-8"))
    grouped["checkpoint_binding"]["dataset_content_fingerprint"] = None
    grouped_path.write_text(json.dumps(grouped), encoding="utf-8")
    with pytest.raises(MatrixSummaryError, match="another dataset fingerprint"):
        summarize(fixture, postprocess)

    fixture = build_fixture(tmp_path / "rows")
    postprocess = add_grouped_report(fixture, "primary_model", 42)
    grouped_path = postprocess / "primary_model_seed42" / "grouped_metrics.json"
    grouped = json.loads(grouped_path.read_text(encoding="utf-8"))
    grouped["prediction_rows"] = 999
    grouped_path.write_text(json.dumps(grouped), encoding="utf-8")
    with pytest.raises(MatrixSummaryError, match="row count differs"):
        summarize(fixture, postprocess)


def test_grouped_primary_rejects_filtered_test_export(tmp_path: Path) -> None:
    fixture = build_fixture(tmp_path)
    postprocess = add_grouped_report(fixture, "primary_model", 42)
    grouped_path = postprocess / "primary_model_seed42" / "grouped_metrics.json"
    grouped = json.loads(grouped_path.read_text(encoding="utf-8"))
    sidecar_path = Path(
        grouped["checkpoint_binding"]["source_prediction_report"]
    )
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    sidecar["full_split_export"] = False
    sidecar["sequence_filter"] = ["S1"]
    sidecar["report_fingerprint"] = canonical_json_hash(
        {**sidecar, "report_fingerprint": None}
    )
    sidecar_path.write_text(json.dumps(sidecar), encoding="utf-8")
    grouped["checkpoint_binding"]["source_prediction_report_sha256"] = (
        sha256_file(sidecar_path)
    )
    grouped_path.write_text(json.dumps(grouped), encoding="utf-8")

    with pytest.raises(MatrixSummaryError, match="filtered test subset"):
        summarize(fixture, postprocess)
