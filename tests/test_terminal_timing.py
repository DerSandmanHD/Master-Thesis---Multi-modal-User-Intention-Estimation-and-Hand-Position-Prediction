import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'Training'))
from evaluate_terminal_timing import block_plan, terminal_regime, cohort
from summarize_terminal_timing import within_recording, prepare_axes, aggregate, METHODS


def test_disjoint_windows_anchor_to_annotation_not_global_grid():
    labels = np.r_[np.zeros(13), np.full(145, 2)]
    blocks, start, remainder = block_plan(labels, 60)
    assert start == 13
    assert blocks == [(0, 13, 72), (1, 73, 132)]
    assert remainder == 25
    assert all(b-a+1 == 60 for _, a, b in blocks)


def test_no_padding_short_segments_and_no_multiple_onsets():
    assert block_plan([0, 0, 2, 2], 60) == ([], 2, 2)
    assert block_plan([0, 0], 60) == ([], None, 0)
    with pytest.raises(ValueError, match='Multiple'):
        block_plan([2, 2, 0, 2], 2)


def test_target_overlap_boundaries_exclude_observed_final_pose():
    metadata = dict(eligible=True, aggregation_capture_start_timestamp_ns=100,
                    target_capture_timestamp_ns=150)
    assert terminal_regime(99, metadata) == 'strictly_before_aggregation'
    assert terminal_regime(100, metadata) == 'partially_overlapping_aggregation'
    assert terminal_regime(149, metadata) == 'partially_overlapping_aggregation'
    assert terminal_regime(150, metadata) == 'at_or_after_target'
    assert terminal_regime(99, {'eligible': False}) == 'no_accepted_target'


def fixture_rows():
    rows = []
    for seed in (42, 43, 44):
        for scheme in ('disjoint60', 'frozen_stride10'):
            for sequence, errors in [('a', [30., 10.]), ('b', [2.])]:
                for i, error in enumerate(errors):
                    row = dict(seed=seed, scheme=scheme, sequence_id=sequence, participant='p',
                        regime='strictly_before_aggregation', fair_common=True, target_valid=True,
                        block_index=i, input_end_relative_s=1.9+i*2,
                        time_to_terminal_s=3-i*2, input_end_ns=100+i*100)
                    row.update({m+'_position_error_cm': error for m in METHODS})
                    rows.append(row)
    return pd.DataFrame(rows)


def test_first_last_is_within_recording_and_does_not_mix_cohorts():
    frame = fixture_rows()
    extra = frame.iloc[0].to_dict()
    extra.update(regime='partially_overlapping_aggregation', input_end_ns=999)
    extra.update({m+'_position_error_cm': 0. for m in METHODS})
    frame = pd.concat([frame, pd.DataFrame([extra])], ignore_index=True)
    result = within_recording(frame)
    assert set(result.sequence_id) == {'a'}  # b has only one eligible window
    assert result.delta_last_minus_first_cm.eq(-20.).all()
    assert len(result) == 3*2*4


def test_time_grouping_recomputes_denominators_and_keeps_overlap_separate():
    frame = fixture_rows()
    extra = frame.iloc[0].to_dict()
    extra.update(regime='partially_overlapping_aggregation')
    extra.update({m+'_position_error_cm': 1000. for m in METHODS})
    frame = pd.concat([frame, pd.DataFrame([extra])], ignore_index=True)
    frame = prepare_axes(frame, {'remaining_time_bins_seconds': [0, .5, 1, 2, 3],
                                'elapsed_time_bins_seconds': [0, 1, 2, 3, 4, 6]})
    _, summary = aggregate(frame)
    row = summary.loc[(summary.axis == 'block') & (summary.order == 0) &
                      (summary.regime == 'strictly_before_aggregation') &
                      (summary.method == 'learned_end_to_end')].iloc[0]
    assert row.windows == 2
    assert row.sequences == 2
    assert row.mean_cm == 16.
    assert row.seed_sd_cm == 0.
    assert len(cohort(frame, 'strictly_before_aggregation')) == 18


def test_sequence_weighting_does_not_overweight_longer_recording():
    frame = fixture_rows()
    # Put all times into one bin: recording a mean 20, recording b mean 2.
    frame.time_to_terminal_s = 2.5
    frame = prepare_axes(frame, {'remaining_time_bins_seconds': [0, .5, 1, 2, 3],
                                'elapsed_time_bins_seconds': [0, 1, 2, 3, 4, 6]})
    _, summary = aggregate(frame)
    row = summary.loc[(summary.axis == 'remaining') & (summary.order == 3) &
                      (summary.regime == 'strictly_before_aggregation') &
                      (summary.method == 'learned_end_to_end')].iloc[0]
    assert row.mean_cm == 14.
    assert row.sequence_mean_cm == 11.
