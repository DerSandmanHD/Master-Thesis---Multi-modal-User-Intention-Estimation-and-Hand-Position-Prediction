from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
EVALUATION = ROOT / "Training" / "evaluation"
if str(EVALUATION) not in sys.path:
    sys.path.insert(0, str(EVALUATION))

from evaluate_causal_intention_baselines import (  # noqa: E402
    _apply_normalization,
    _normalization,
    _validate_frozen_bindings,
    fit_softmax_regression,
    predict_softmax,
)
from artifact_freeze import canonical_json_hash  # noqa: E402


def test_softmax_baseline_is_deterministic_and_multiclass() -> None:
    features = np.asarray(
        [
            [-3.0, -2.0],
            [-2.0, -3.0],
            [0.0, 3.0],
            [0.5, 2.5],
            [3.0, -1.0],
            [2.5, -0.5],
        ]
    )
    targets = np.asarray([0, 0, 1, 1, 2, 2])
    first = fit_softmax_regression(
        features, targets, class_count=3, l2=1e-4, max_iterations=300
    )
    second = fit_softmax_regression(
        features, targets, class_count=3, l2=1e-4, max_iterations=300
    )
    assert np.array_equal(predict_softmax(features, first), targets)
    assert np.array_equal(first.weights, second.weights)
    assert first.objective == second.objective


def test_train_only_normalization_is_reused_without_future_context() -> None:
    train = np.asarray([[1.0], [2.0], [3.0]])
    normalized, metadata = _normalization(train)
    assert np.allclose(normalized.mean(axis=0), 0.0)
    test = np.asarray([[4.0]])
    transformed = _apply_normalization(test, metadata)
    assert np.allclose(transformed, (test - train.mean(axis=0)) / train.std(axis=0))
    assert set(metadata) == {"mean", "std", "fit_split"}
    assert metadata["fit_split"] == "train"


def test_prediction_binding_accepts_legacy_null_dataset_identifier() -> None:
    class TestSplit:
        def __len__(self) -> int:
            return 7

        @staticmethod
        def endpoint_fingerprint() -> str:
            return "endpoint-fingerprint"

    class Bundle:
        provenance = {
            "dataset_content_fingerprint": "dataset-content",
            "source_content_fingerprint": "source-content",
        }
        test = TestSplit()

    descriptor = {
        "dataset_tag": "dataset_v3_causal_20260815_n214_5d136a34",
        "dataset_content_fingerprint": "dataset-content",
        "source_content_fingerprint": "source-content",
    }
    prediction_report = {
        "dataset_identifier": None,
        "dataset_content_fingerprint": "dataset-content",
        "source_content_fingerprint": "source-content",
        "frozen_split_endpoint_fingerprint": "endpoint-fingerprint",
        "frozen_split_endpoint_count": 7,
        "split": "test",
        "full_split_export": True,
        "report_fingerprint": None,
    }
    prediction_report["report_fingerprint"] = canonical_json_hash(
        prediction_report
    )
    _validate_frozen_bindings(
        bundle=Bundle(),
        descriptor=descriptor,
        prediction_report=prediction_report,
    )
