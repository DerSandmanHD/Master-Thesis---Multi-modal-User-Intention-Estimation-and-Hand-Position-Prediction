from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
EVALUATION = ROOT / "Training" / "evaluation"
if str(EVALUATION) not in sys.path:
    sys.path.insert(0, str(EVALUATION))

from audit_sampling_window_duration import distribution, window_durations  # noqa: E402


def test_distribution_reports_required_robust_statistics() -> None:
    values = np.asarray([0.02, 0.03, 0.04, 0.05, 0.06])
    result = distribution(values)
    assert result["count"] == 5
    assert result["median"] == 0.04
    assert result["quartile_25"] == 0.03
    assert result["quartile_75"] == 0.05
    assert np.isclose(result["iqr"], 0.02)
    assert result["minimum"] == 0.02
    assert result["maximum"] == 0.06


def test_sixty_samples_span_fifty_nine_intervals() -> None:
    class Record:
        timestamps_ns = np.arange(100, dtype=np.int64) * 40_000_000

    class Dataset:
        records = [Record()]
        window_size = 60
        indices = [(0, 59), (0, 69)]

    durations = window_durations(Dataset())
    assert np.allclose(durations, [59 * 0.04, 59 * 0.04])
