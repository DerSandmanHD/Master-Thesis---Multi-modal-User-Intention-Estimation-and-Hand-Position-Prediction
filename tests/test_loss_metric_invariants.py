from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch
from torch import nn


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "Training"))

from metrics import classification_metrics  # noqa: E402
from train_residual import residual_multitask_loss  # noqa: E402


def _classification_outputs() -> dict[str, torch.Tensor]:
    return {
        "assistance_logits": torch.tensor(
            [[2.0, -1.0], [-1.0, 2.0]], requires_grad=True
        ),
        "assistance_type_logits": torch.tensor(
            [[1.0, -1.0], [-1.0, 1.0]], requires_grad=True
        ),
        "receiving_hand_logits": torch.tensor(
            [[1.0, -1.0], [-1.0, 1.0]], requires_grad=True
        ),
    }


def _classification_batch() -> dict[str, torch.Tensor]:
    return {
        "intention": torch.tensor([0, 2]),
        "receiving_hand": torch.tensor([-1, 1]),
    }


def _training_config(**overrides: float) -> dict:
    config = {
        "assistance_loss_weight": 1.0,
        "assistance_type_loss_weight": 1.0,
        "receiving_hand_loss_weight": 1.0,
        "pose_loss_weight": 1.0,
        "orientation_loss_weight": 0.25,
        "auxiliary_pose_loss_weight": 0.0,
    }
    config.update(overrides)
    return config


def _loss(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    config: dict,
) -> tuple[torch.Tensor, dict[str, float]]:
    return residual_multitask_loss(
        outputs,
        batch,
        nn.CrossEntropyLoss(),
        nn.CrossEntropyLoss(),
        nn.CrossEntropyLoss(),
        config,
    )


def _assert_nonzero_finite_gradient(value: torch.Tensor) -> None:
    assert value.grad is not None
    assert torch.isfinite(value.grad).all()
    assert float(value.grad.abs().sum()) > 0.0


class LossMetricInvariantTests(unittest.TestCase):
    def test_classification_metrics_add_precision_and_recall(self) -> None:
        metrics = classification_metrics(
            predictions=torch.tensor([0, 0, 1, 2]),
            targets=torch.tensor([0, 1, 1, 2]),
            num_classes=4,
        )

        self.assertTrue(
            {
                "accuracy",
                "macro_f1",
                "macro_f1_supported",
                "per_class_f1",
                "support",
                "confusion_matrix",
                "samples",
            }.issubset(metrics)
        )
        self.assertEqual(
            metrics["per_class_precision"], [0.5, 1.0, 1.0, 0.0]
        )
        self.assertEqual(metrics["per_class_recall"], [1.0, 0.5, 1.0, 0.0])

    def test_pose_weight_zero_has_no_pose_dependency_or_nan_poisoning(
        self,
    ) -> None:
        outputs = _classification_outputs()
        pose_outputs = {
            "position_delta": torch.full(
                (2, 3), float("nan"), requires_grad=True
            ),
            "quaternion_delta": torch.full(
                (2, 4), float("nan"), requires_grad=True
            ),
            "pose_candidates": torch.full(
                (2, 2, 7), float("nan"), requires_grad=True
            ),
            "auxiliary_pose_candidates": torch.full(
                (2, 2, 7), float("nan"), requires_grad=True
            ),
        }
        outputs.update(pose_outputs)

        # Pose batch keys are deliberately absent: disabled losses must not
        # inspect them.
        loss, components = _loss(
            outputs,
            _classification_batch(),
            _training_config(
                pose_loss_weight=0.0, auxiliary_pose_loss_weight=0.0
            ),
        )
        self.assertTrue(bool(torch.isfinite(loss)))
        self.assertEqual(components["position"], 0.0)
        self.assertEqual(components["orientation"], 0.0)
        self.assertEqual(components["auxiliary_position"], 0.0)
        self.assertEqual(components["auxiliary_orientation"], 0.0)

        loss.backward()
        for key in (
            "assistance_logits",
            "assistance_type_logits",
            "receiving_hand_logits",
        ):
            _assert_nonzero_finite_gradient(outputs[key])
        for value in pose_outputs.values():
            self.assertIsNone(value.grad)

    def test_auxiliary_weight_zero_skips_only_dual_horizon_branch(self) -> None:
        outputs = _classification_outputs()
        pose_candidates = torch.zeros((2, 2, 7))
        pose_candidates[:, :, 6] = 1.0
        pose_candidates.requires_grad_()
        auxiliary_candidates = torch.full(
            (2, 2, 7), float("nan"), requires_grad=True
        )
        outputs.update(
            {
                "pose_candidates": pose_candidates,
                "auxiliary_pose_candidates": auxiliary_candidates,
            }
        )

        batch = _classification_batch()
        pose_target = torch.zeros((2, 7))
        pose_target[:, 6] = 1.0
        pose_target[1, 0] = 1.0
        batch.update(
            {
                "residual_pose_valid": torch.tensor([False, True]),
                "pose_target": pose_target,
            }
        )

        # Auxiliary target/validity keys are deliberately absent.
        loss, components = _loss(
            outputs,
            batch,
            _training_config(
                pose_loss_weight=1.0, auxiliary_pose_loss_weight=0.0
            ),
        )
        self.assertTrue(bool(torch.isfinite(loss)))
        self.assertGreater(components["position"], 0.0)
        self.assertEqual(components["auxiliary_position"], 0.0)

        loss.backward()
        _assert_nonzero_finite_gradient(pose_candidates)
        self.assertIsNone(auxiliary_candidates.grad)
        for key in (
            "assistance_logits",
            "assistance_type_logits",
            "receiving_hand_logits",
        ):
            _assert_nonzero_finite_gradient(outputs[key])

    def test_flat_head_uses_three_class_loss_and_keeps_handover_tasks(self) -> None:
        intention_logits = torch.tensor(
            [[3.0, 0.0, -1.0], [-1.0, 0.0, 3.0]], requires_grad=True
        )
        hand_logits = torch.tensor(
            [[1.0, -1.0], [-1.0, 1.0]], requires_grad=True
        )
        pose_candidates = torch.full(
            (2, 2, 7), float("nan"), requires_grad=True
        )
        outputs = {
            "intention_logits": intention_logits,
            "receiving_hand_logits": hand_logits,
            "pose_candidates": pose_candidates,
        }
        config = _training_config(pose_loss_weight=0.0)
        config.update(
            {
                "intention_loss_weight": 1.0,
                "resolved_flat_intention_class_weights": [1.0, 1.0, 1.0],
            }
        )
        loss, components = _loss(outputs, _classification_batch(), config)
        self.assertTrue(bool(torch.isfinite(loss)))
        self.assertGreater(components["intention"], 0.0)
        self.assertEqual(components["assistance"], 0.0)
        self.assertEqual(components["assistance_type"], 0.0)
        loss.backward()
        _assert_nonzero_finite_gradient(intention_logits)
        _assert_nonzero_finite_gradient(hand_logits)
        self.assertIsNone(pose_candidates.grad)

    def test_negative_loss_weights_are_rejected(self) -> None:
        for weight_name in (
            "assistance_loss_weight",
            "assistance_type_loss_weight",
            "receiving_hand_loss_weight",
            "pose_loss_weight",
            "orientation_loss_weight",
            "auxiliary_pose_loss_weight",
            "auxiliary_orientation_loss_weight",
            "intention_loss_weight",
        ):
            with self.subTest(weight_name=weight_name):
                config = _training_config(pose_loss_weight=0.0)
                config[weight_name] = -0.1
                with self.assertRaisesRegex(ValueError, weight_name):
                    _loss(
                        _classification_outputs(),
                        _classification_batch(),
                        config,
                    )

    def test_nonfinite_loss_weights_are_rejected(self) -> None:
        for invalid_weight in (float("nan"), float("inf")):
            with self.subTest(invalid_weight=invalid_weight):
                config = _training_config(pose_loss_weight=invalid_weight)
                with self.assertRaisesRegex(ValueError, "pose_loss_weight"):
                    _loss(
                        _classification_outputs(),
                        _classification_batch(),
                        config,
                    )


if __name__ == "__main__":
    unittest.main()
