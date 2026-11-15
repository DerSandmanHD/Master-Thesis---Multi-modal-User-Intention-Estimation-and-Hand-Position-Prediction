import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'Training'))
from terminal_shrinkage import attenuate,score,validation_search,choose_alpha


def frame(target,model):
    values={'sample_key':['a'],'sequence_id':['video'],'participant':['p']}
    for prefix,position in [('target',target),('persistence',[0,0,0]),('persistence_predicted_hand',[0,0,0]),
                            ('learned_oracle_hand',model),('learned_end_to_end',model)]:
        values.update({f'{prefix}_{a}_m':[position[i]] for i,a in enumerate('xyz')})
    return pd.DataFrame(values)


def test_alpha_endpoints_and_interpolation():
    ref=np.array([[1.,2.,3.]])
    pred=np.array([[3.,2.,1.]])
    np.testing.assert_array_equal(attenuate(ref,pred,0),ref)
    np.testing.assert_array_equal(attenuate(ref,pred,1),pred)
    np.testing.assert_array_equal(attenuate(ref,pred,.5),[[2,2,2]])


@pytest.mark.parametrize('alpha',[-.01,1.01,float('nan')])
def test_rejects_invalid_factors(alpha):
    with pytest.raises(ValueError):attenuate(np.zeros((1,3)),np.ones((1,3)),alpha)


def test_validation_can_select_partial_attenuation():
    f=frame([1,0,0],[2,0,0])
    search=validation_search({s:f for s in [42,43,44]},[0,.5,1])
    alpha,error=choose_alpha(search)
    assert alpha==.5 and error==0


def test_validation_can_select_persistence_and_conservative_ties():
    f=frame([0,0,0],[1,0,0])
    assert choose_alpha(validation_search({s:f for s in [42,43,44]},[0,.5,1]))[0]==0
    same=frame([1,0,0],[0,0,0])
    assert choose_alpha(validation_search({s:same for s in [42,43,44]},[0,.5,1]))[0]==0


def test_score_uses_separate_reference_for_predicted_hand():
    f=frame([1,0,0],[2,0,0]);f['persistence_predicted_hand_x_m']=3
    scores=score(f,0)
    assert scores.gt_hand_error_cm.iloc[0]==100
    assert scores.predicted_hand_error_cm.iloc[0]==200


def test_prediction_does_not_depend_on_future_target():
    f=frame([1,0,0],[2,0,0])
    a=score(f,.5)
    f['target_x_m']=999
    b=score(f,.5)
    np.testing.assert_array_equal(a[['prediction_x_m','prediction_y_m','prediction_z_m']],
                                  b[['prediction_x_m','prediction_y_m','prediction_z_m']])
    assert a.predicted_hand_error_cm.iloc[0]!=b.predicted_hand_error_cm.iloc[0]
