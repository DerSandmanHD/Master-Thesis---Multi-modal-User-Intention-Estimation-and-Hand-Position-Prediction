from __future__ import annotations

import sys
from pathlib import Path

import torch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Training"))

from prediction_utils import (  # noqa: E402
    assistance_predictions,
    assistance_type_predictions,
    intention_predictions,
    intention_probabilities,
)


def test_flat_intention_semantics() -> None:
    outputs = {
        "intention_logits": torch.tensor(
            [[3.0, 1.0, 0.0], [0.0, 2.0, 4.0]], dtype=torch.float32
        )
    }
    probabilities = intention_probabilities(outputs)
    assert probabilities.shape == (2, 3)
    assert torch.allclose(probabilities.sum(dim=-1), torch.ones(2))
    assert intention_predictions(outputs).tolist() == [0, 2]
    assert assistance_predictions(outputs).tolist() == [0, 1]
    assert assistance_type_predictions(outputs).tolist() == [0, 1]


def test_hierarchical_intention_semantics_preserve_node_decisions() -> None:
    outputs = {
        "assistance_logits": torch.tensor([[3.0, 1.0], [0.0, 4.0]]),
        "assistance_type_logits": torch.tensor([[0.0, 2.0], [3.0, 1.0]]),
    }
    probabilities = intention_probabilities(outputs)
    assert probabilities.shape == (2, 3)
    assert torch.allclose(probabilities.sum(dim=-1), torch.ones(2))
    assert intention_predictions(outputs).tolist() == [0, 1]
    assert assistance_predictions(outputs).tolist() == [0, 1]
    assert assistance_type_predictions(outputs).tolist() == [1, 0]
