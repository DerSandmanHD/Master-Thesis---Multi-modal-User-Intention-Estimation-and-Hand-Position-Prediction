from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import torch
from torch.utils.data import TensorDataset


ROOT = Path(__file__).resolve().parents[1]
TRAINING = ROOT / "Training"
if str(TRAINING) not in sys.path:
    sys.path.insert(0, str(TRAINING))

from training_control import (  # noqa: E402
    POSE_DIAGNOSTIC_UNAVAILABLE_REASON,
    available_validation_checkpoints,
    finite_diagnostic_improved,
    next_primary_patience,
)
from train import make_loader as make_standard_loader  # noqa: E402
from train_residual import make_loader as make_residual_loader  # noqa: E402


class TrainingControlTests(unittest.TestCase):
    def test_training_shuffle_is_seed_bound_not_model_rng_bound(self) -> None:
        dataset = TensorDataset(torch.arange(24))
        config = {
            "seed": 42,
            "batch_size": 4,
            "num_workers": 0,
            "sampling_mode": "window_uniform",
        }
        first = make_standard_loader(
            dataset, config, shuffle=True, device=torch.device("cpu")
        )
        # Simulate different architecture initializers consuming different
        # amounts of global RNG before the first epoch starts.
        torch.rand(10_000)
        second = make_standard_loader(
            dataset, config, shuffle=True, device=torch.device("cpu")
        )
        torch.rand(17)
        first_order = torch.cat([batch[0] for batch in first]).tolist()
        second_order = torch.cat([batch[0] for batch in second]).tolist()
        self.assertEqual(first_order, second_order)

        residual = make_residual_loader(
            dataset, config, shuffle=True, device=torch.device("cpu")
        )
        torch.rand(31)
        residual_order = torch.cat([batch[0] for batch in residual]).tolist()
        self.assertEqual(first_order, residual_order)

    def test_pose_only_improvement_cannot_reset_primary_patience(self) -> None:
        self.assertEqual(
            next_primary_patience(
                4,
                primary_improved=False,
                diagnostic_improved=True,
            ),
            5,
        )

    def test_primary_improvement_resets_patience(self) -> None:
        self.assertEqual(
            next_primary_patience(
                4,
                primary_improved=True,
                diagnostic_improved=False,
            ),
            0,
        )

    def test_nonfinite_pose_metric_never_creates_diagnostic_improvement(self) -> None:
        for value in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(value=value):
                self.assertFalse(finite_diagnostic_improved(value, 10.0))
        self.assertTrue(finite_diagnostic_improved(9.0, 10.0))

    def test_missing_finite_pose_checkpoint_is_optional_and_reported(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            primary = root / "best_intention_model.pt"
            primary.write_bytes(b"primary")
            pose = root / "best_pose_model.pt"
            checkpoints, status = available_validation_checkpoints(primary, pose)

            self.assertEqual(checkpoints, (("best_intention", primary),))
            self.assertFalse(status["available"])
            self.assertIsNone(status["path"])
            self.assertEqual(status["reason"], POSE_DIAGNOSTIC_UNAVAILABLE_REASON)

    def test_finite_pose_checkpoint_remains_available_as_diagnostic(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            primary = root / "best_intention_model.pt"
            pose = root / "best_pose_model.pt"
            primary.write_bytes(b"primary")
            pose.write_bytes(b"pose")
            checkpoints, status = available_validation_checkpoints(primary, pose)

            self.assertEqual(
                checkpoints,
                (("best_intention", primary), ("best_pose", pose)),
            )
            self.assertTrue(status["available"])
            self.assertEqual(status["role"], "diagnostic_only")
            self.assertIsNone(status["reason"])


if __name__ == "__main__":
    unittest.main()
