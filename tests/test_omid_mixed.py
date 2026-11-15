import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch.utils.data import WeightedRandomSampler

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'Training'))
from omid_mixed_finetune import (
    domain_sampling_weights, expected_label_mass, validation_score,
    verify_bundle, verify_isolation, require_pose_cohort,
)


def test_sampler_balances_unequal_domains_and_is_reproducible():
    weights = domain_sampling_weights([1000, 10])
    assert weights[:1000].sum().item() == pytest.approx(.5)
    assert weights[1000:].sum().item() == pytest.approx(.5)
    def sample():
        return list(WeightedRandomSampler(weights, 20000, replacement=True,
                    generator=torch.Generator().manual_seed(42)))
    first = sample()
    assert first == sample()
    assert .48 < sum(i >= 1000 for i in first) / len(first) < .52


@pytest.mark.parametrize('sizes', [[0, 5], [5], [-1, 5]])
def test_reject_invalid_domains(sizes):
    with pytest.raises(ValueError):
        domain_sampling_weights(sizes)


def test_selection_gives_each_validation_domain_equal_weight():
    metrics = {'old': {'intention': {'macro_f1': .4}, 'windows': 100000},
               'new': {'intention': {'macro_f1': .8}, 'windows': 10}}
    assert validation_score(metrics) == pytest.approx(.6)
    metrics['old']['intention']['macro_f1'] = float('nan')
    with pytest.raises(ValueError):
        validation_score(metrics)


def test_loss_mass_matches_sampler_including_missing_labels():
    class Dataset:
        def __init__(self, size, counts): self.size, self.counts = size, counts
        def __len__(self): return self.size
        def receiving_hand_counts(self): return self.counts
    mass = expected_label_mass([Dataset(100, [20, 10]), Dataset(10, [0, 5])], 'receiving_hand_counts')
    np.testing.assert_allclose(mass, [.1, .3])


def bundle(prefix):
    class Windows(list):
        def endpoint_fingerprint(self): return 'fixed'
    return SimpleNamespace(feature_columns=['a'], train=Windows([1]), validation=Windows([1]), test=Windows([1]),
                           split_metadata={'sequences': {k: [prefix + k] for k in ('train', 'validation', 'test')}})


def test_isolation_detects_cross_domain_test_leakage():
    domains = {'old': bundle('old'), 'new': bundle('new')}
    verify_isolation(domains)
    domains['new'].split_metadata['sequences']['train'] = ['oldtest']
    with pytest.raises(ValueError, match='Overlapping'):
        verify_isolation(domains)


def test_prior_cohort_check_rejects_window_drift():
    data = bundle('old')
    meta = {'feature_columns': ['a'], 'split': {
        'sequences': data.split_metadata['sequences'],
        'window_eligibility': {'endpoint_fingerprints': {k: 'fixed' for k in ('train', 'validation', 'test')}}}}
    verify_bundle(data, meta)
    meta['split']['window_eligibility']['endpoint_fingerprints']['test'] = 'different'
    with pytest.raises(ValueError, match='test windows'):
        verify_bundle(data, meta)


def test_pose_comparison_rejects_different_cohort():
    prior = {'pose_fair_common': {'sample_key_fingerprint': 'abc', 'shared_samples': 2,
                                 'coverage_denominator_pose_targets': 3}}
    require_pose_cohort(prior, prior)
    changed = {'pose_fair_common': {**prior['pose_fair_common'], 'shared_samples': 1}}
    with pytest.raises(ValueError, match='cohort'):
        require_pose_cohort(changed, prior)
