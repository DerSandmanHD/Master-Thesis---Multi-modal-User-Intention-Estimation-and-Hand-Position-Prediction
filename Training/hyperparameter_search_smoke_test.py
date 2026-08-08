#!/usr/bin/env python3
"""Smoke-test deterministic hyperparameter generation and constraints."""

from __future__ import annotations

from generate_hyperparameter_trials import generate_parameter_sets


def main() -> int:
    first = generate_parameter_sets(24, 20260808)
    second = generate_parameter_sets(24, 20260808)
    different = generate_parameter_sets(24, 20260809)
    assert first == second
    assert first != different
    assert len(first) == len({str(sorted(item.items())) for item in first}) == 24
    for parameters in first:
        assert parameters["d_model"] % parameters["nhead"] == 0
        assert parameters["dim_feedforward"] >= parameters["d_model"]
        assert 1e-5 <= parameters["learning_rate"] <= 1e-3
        assert parameters["batch_size"] in {16, 32, 64}
    print("Hyperparameter search smoke test passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
