import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'reproducibility'))
from report_metrics import distribution, quantile
from replay_checkpoint import relocate
from artifacts import load_lock, selected


def test_spatial_distribution_uses_euclidean_errors_and_strict_hits():
    d = distribution([0, 5, 10, 20])
    assert d['mean_cm'] == 8.75
    assert d['median_cm'] == 7.5
    assert d['p90_cm'] == pytest.approx(17)
    assert d['hit_at_5_cm'] == .25
    assert d['hit_at_20_cm'] == .75
    assert quantile([4], .9) == 4
    with pytest.raises(ValueError):
        distribution([float('nan')])


def test_relocation_changes_only_historical_root_and_preserves_input():
    original = {'path': '/old/repo/Data/file', 'other': '/old/repository/file',
                'nested': ['/old/repo/config.json', 42]}
    result = relocate(original, '/old/repo', Path('/new'))
    assert result['path'] == '/new/Data/file'
    assert result['other'] == original['other']
    assert result['nested'] == ['/new/config.json', 42]
    assert original['path'].startswith('/old/')


def test_lock_contains_complete_main_lopo_and_processed_datasets():
    lock = load_lock()
    main = [x for x in selected(lock, ['main']) if x['path'].startswith('Training/runs/')
            and x['path'].endswith('/config.json')]
    lopo = [x for x in selected(lock, ['lopo']) if x['path'].endswith('/config.json')]
    masters = [x for x in lock['files'] if x['path'].startswith('Data_collection/master_datasets/')
               and x['path'].endswith('_master.csv')]
    new = [x for x in lock['files'] if '/object_marker_free_v1/master_datasets/' in x['path']
           and x['path'].endswith('_master.csv')]
    assert (len(main), len(lopo), len(masters), len(new)) == (48, 25, 214, 50)
