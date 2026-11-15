import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'Training'))
from terminal_residual_diagnostics import vector_measures, paired_future


def test_persistence_has_zero_correction_and_undefined_angle():
    m=vector_measures([[10,0,0]],[[0,0,0]])
    assert m.model_error_cm.iloc[0]==10
    assert not m.angle_eligible.iloc[0]
    assert np.isnan(m.direction_angle_deg.iloc[0])
    assert m.magnitude_squared_term_cm2.iloc[0]==100


def test_perfect_direction_and_length():
    m=vector_measures([[3,4,0]],[[3,4,0]])
    assert m.model_error_cm.iloc[0]==0
    assert m.direction_angle_deg.iloc[0]==pytest.approx(0)
    assert m.improves.iloc[0]


def test_opposite_direction_exposes_error_despite_correct_length():
    m=vector_measures([[10,0,0]],[[-10,0,0]])
    assert m.model_error_cm.iloc[0]==20
    assert m.magnitude_abs_error_cm.iloc[0]==0
    assert m.direction_squared_term_cm2.iloc[0]==400
    assert m.direction_angle_deg.iloc[0]==180
    assert m.opposite_direction.iloc[0]


def test_excessive_length_along_right_direction():
    m=vector_measures([[10,0,0]],[[25,0,0]])
    assert m.model_error_cm.iloc[0]==15
    assert m.direction_squared_term_cm2.iloc[0]==0
    assert m.longer_than_actual.iloc[0]
    assert m.worsens.iloc[0]


def test_zero_target_and_negligible_motion_do_not_get_artificial_angles():
    m=vector_measures([[0,0,0],[1,0,0]],[[10,0,0],[-10,0,0]])
    assert m.direction_angle_deg.isna().all()
    assert not m.angle_eligible.any()
    np.testing.assert_allclose(m.model_error_cm,[10,11])


def test_random_vectors_obey_exact_squared_error_identity():
    rng=np.random.default_rng(42)
    m=vector_measures(rng.normal(size=(100,3))*10,rng.normal(size=(100,3))*10)
    np.testing.assert_allclose(m.model_error_cm**2,
                               m.magnitude_squared_term_cm2+m.direction_squared_term_cm2,atol=1e-9)


def test_primary_filter_excludes_partial_and_unavailable_references():
    f=pd.DataFrame({'regime':['strictly_before_aggregation','partially_overlapping_aggregation',
                              'strictly_before_aggregation','at_or_after_target'],
                    'fair_common':[True,True,False,False]})
    assert paired_future(f).index.tolist()==[0]
