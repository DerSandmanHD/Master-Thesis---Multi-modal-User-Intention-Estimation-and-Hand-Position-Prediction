from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
TRAINING = ROOT / "Training"
if str(TRAINING) not in sys.path:
    sys.path.insert(0, str(TRAINING))

from model import (  # noqa: E402
    FUSION_MODES,
    HierarchicalGatedMultimodalTransformer,
    HierarchicalResidualPoseTransformer,
)


def modality_schema() -> dict:
    return {
        "version": "unit_test_v1",
        "active_modalities": ["gaze", "hands"],
        "groups": {
            "gaze": {
                "input_indices": [0, 1, 4],
                "availability_indices": [4],
            },
            "hands": {
                "input_indices": [2, 3, 5, 6],
                "availability_indices": [5, 6],
            },
        },
    }


def model_arguments() -> dict:
    return {
        "input_dim": 8,
        "window_size": 5,
        "d_model": 8,
        "nhead": 2,
        "num_layers": 1,
        "dim_feedforward": 16,
        "dropout": 0.0,
    }


class ModelFusionTests(unittest.TestCase):
    def test_all_fusion_modes_keep_head_dimensions_and_diagnostics(self) -> None:
        x = torch.randn(3, 5, 8)
        x[:, :, 4:7] = 1.0
        for mode in FUSION_MODES:
            with self.subTest(mode=mode):
                model = HierarchicalGatedMultimodalTransformer(
                    **model_arguments(),
                    fusion_mode=mode,
                    modality_schema=(
                        modality_schema() if mode == "modality_gated" else None
                    ),
                )
                outputs = model(x)
                self.assertEqual(outputs["assistance_logits"].shape, (3, 2))
                self.assertEqual(outputs["assistance_type_logits"].shape, (3, 2))
                self.assertEqual(outputs["pose"].shape, (3, 7))
                self.assertEqual(outputs["gate"].shape, (3, 2))
                self.assertEqual(outputs["fusion_weights"].shape, (3, 2))
                self.assertTrue(
                    torch.equal(outputs["gate"], outputs["fusion_weights"])
                )
                expected_modalities = 2 if mode == "modality_gated" else 0
                self.assertEqual(
                    outputs["modality_weights"].shape,
                    (3, expected_modalities),
                )
                self.assertEqual(
                    outputs["modality_available"].shape,
                    (3, expected_modalities),
                )
                self.assertTrue(
                    all(
                        torch.isfinite(value).all()
                        for value in outputs.values()
                        if torch.is_floating_point(value)
                    )
                )

    def test_modality_gate_masks_missing_groups_and_all_missing_windows(self) -> None:
        model = HierarchicalGatedMultimodalTransformer(
            **model_arguments(),
            fusion_mode="modality_gated",
            modality_schema=modality_schema(),
        ).eval()
        self.assertEqual(model.modality_names, ("gaze", "hands"))

        x = torch.randn(3, 5, 8)
        x[:, :, 4:7] = 0.0
        x[0, :, 4] = 1.0  # gaze only
        x[2, :, 4:7] = 1.0  # both modalities
        outputs = model(x)
        weights = outputs["modality_weights"]
        available = outputs["modality_available"]

        self.assertTrue(torch.equal(available[0], torch.tensor([True, False])))
        self.assertTrue(torch.equal(weights[0], torch.tensor([1.0, 0.0])))
        self.assertTrue(torch.equal(available[1], torch.tensor([False, False])))
        self.assertTrue(torch.equal(weights[1], torch.tensor([0.0, 0.0])))
        self.assertTrue(torch.isfinite(outputs["assistance_logits"][1]).all())
        self.assertTrue(torch.equal(available[2], torch.tensor([True, True])))
        self.assertAlmostEqual(float(weights[2].sum().detach()), 1.0, places=6)

        # Missing values and features outside the resolved semantic schema do
        # not bypass modality masking through a parallel all-feature path.
        perturbed = x.clone()
        perturbed[0, :, 2:4] = 1e6  # unavailable hand values
        perturbed[0, :, 7] = -1e6  # deliberately unassigned feature
        perturbed_outputs = model(perturbed)
        self.assertTrue(
            torch.allclose(
                outputs["assistance_logits"][0],
                perturbed_outputs["assistance_logits"][0],
                atol=0.0,
                rtol=0.0,
            )
        )

    def test_modality_gate_and_encoders_receive_gradients(self) -> None:
        torch.manual_seed(7)
        model = HierarchicalGatedMultimodalTransformer(
            **model_arguments(),
            fusion_mode="modality_gated",
            modality_schema=modality_schema(),
        )
        x = torch.randn(4, 5, 8)
        x[:, :, 4:7] = 1.0
        outputs = model(x)
        loss = outputs["assistance_logits"].square().mean()
        loss.backward()

        gate_gradient = model.modality_gate[-1].weight.grad
        self.assertIsNotNone(gate_gradient)
        self.assertTrue(torch.isfinite(gate_gradient).all())
        for encoder in model.modality_encoders:
            gradient = encoder[0].weight.grad
            self.assertIsNotNone(gradient)
            self.assertTrue(torch.isfinite(gradient).all())

    def test_default_residual_is_legacy_compatible_and_flat_changes_only_intent(self) -> None:
        args = model_arguments()
        hierarchical = HierarchicalResidualPoseTransformer(**args)
        legacy_state = hierarchical.state_dict()
        self.assertFalse(any(key.startswith("modality_") for key in legacy_state))
        self.assertIn("gate.0.weight", legacy_state)
        self.assertIn("assistance_head.weight", legacy_state)
        self.assertIn("assistance_type_head.weight", legacy_state)
        self.assertNotIn("intention_head.weight", legacy_state)
        reloaded = HierarchicalResidualPoseTransformer(**args)
        reloaded.load_state_dict(legacy_state, strict=True)

        flat = HierarchicalResidualPoseTransformer(
            **args,
            intention_head_mode="flat",
        )
        x = torch.randn(2, 5, 8)
        references = torch.zeros(2, 2, 7)
        references[:, :, 6] = 1.0
        hierarchical_outputs = hierarchical(x, references)
        flat_outputs = flat(x, references)

        self.assertEqual(hierarchical_outputs["assistance_logits"].shape, (2, 2))
        self.assertEqual(
            hierarchical_outputs["assistance_type_logits"].shape, (2, 2)
        )
        self.assertNotIn("intention_logits", hierarchical_outputs)
        self.assertEqual(flat_outputs["intention_logits"].shape, (2, 3))
        self.assertNotIn("assistance_logits", flat_outputs)
        self.assertNotIn("assistance_type_logits", flat_outputs)
        self.assertEqual(flat_outputs["receiving_hand_logits"].shape, (2, 2))
        self.assertEqual(flat_outputs["pose_candidates"].shape, (2, 2, 7))

        hierarchical_non_intent = {
            key: value.shape
            for key, value in hierarchical.state_dict().items()
            if not key.startswith(("assistance_head.", "assistance_type_head."))
        }
        flat_non_intent = {
            key: value.shape
            for key, value in flat.state_dict().items()
            if not key.startswith("intention_head.")
        }
        self.assertEqual(hierarchical_non_intent, flat_non_intent)

    def test_modality_schema_is_required_and_range_checked(self) -> None:
        with self.assertRaises(ValueError):
            HierarchicalGatedMultimodalTransformer(
                **model_arguments(), fusion_mode="modality_gated"
            )
        broken = modality_schema()
        broken["groups"]["gaze"]["availability_indices"] = [99]
        with self.assertRaises(ValueError):
            HierarchicalGatedMultimodalTransformer(
                **model_arguments(),
                fusion_mode="modality_gated",
                modality_schema=broken,
            )


if __name__ == "__main__":
    unittest.main()
