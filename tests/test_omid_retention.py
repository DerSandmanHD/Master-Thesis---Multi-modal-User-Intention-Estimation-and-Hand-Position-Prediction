import sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'Training'))
from data import Normalizer
from evaluate_omid_retention import mask_object_features


def test_mask_preserves_nonobject_features_targets_and_original():
    names=['gaze_valid']+[f'aruco_{i}_{suffix}' for i in range(6,15)
        for suffix in ('robot_x_m','robot_y_m','robot_z_m','gaze_angle_rad','gaze_distance_m','valid')]
    n=len(names);features=np.ones((3,2*n),dtype=np.float32)
    record=SimpleNamespace(features=features.copy(),targets=np.array([1,2,3]))
    dataset=SimpleNamespace(records=[record],endpoint_fingerprint=lambda:'unchanged')
    norm=Normalizer(mean=np.full(n,2.),std=np.full(n,4.),feature_names=names)
    masked=mask_object_features(dataset,norm).records[0]
    np.testing.assert_array_equal(record.features,features)
    np.testing.assert_array_equal(masked.targets,record.targets)
    np.testing.assert_array_equal(masked.features[:,[0,n]],features[:,[0,n]])
    for i,name in enumerate(names[1:],1):
        expected=(-.5,1.) if name.endswith('_valid') else (0.,0.)
        np.testing.assert_array_equal(masked.features[:,i],np.full(3,expected[0]))
        np.testing.assert_array_equal(masked.features[:,n+i],np.full(3,expected[1]))
