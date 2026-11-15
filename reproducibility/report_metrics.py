#!/usr/bin/env python3
"""Recompute the primary paired t+1 statistics from verified prediction CSVs."""
import argparse
import csv
import json
import math
from pathlib import Path
import statistics

from artifacts import load_lock, verify


def quantile(values, q):
    ordered = sorted(values)
    index = (len(ordered) - 1) * q
    low = math.floor(index)
    high = math.ceil(index)
    return ordered[low] + (ordered[high] - ordered[low]) * (index - low)


def distribution(values):
    if not values or not all(math.isfinite(v) and v >= 0 for v in values):
        raise ValueError('Invalid position errors')
    return {'mean_cm': statistics.mean(values),
            'rmse_cm': math.sqrt(statistics.mean(v * v for v in values)),
            'median_cm': quantile(values, .5), 'p90_cm': quantile(values, .9),
            **{f'hit_at_{t}_cm': sum(v < t for v in values) / len(values) for t in (5, 10, 15, 20)}}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    a = p.parse_args()
    lock = load_lock()
    by_path = {x['path']: x for x in lock['files']}
    base = f'Training/reports/{lock["dataset_tag"]}/thesis_final_v2_corrected_alignment/postprocess'
    results = {}
    for seed in (42, 43, 44):
        path = f'{base}/residual_current_gate_seed{seed}/test_predictions.csv'
        if verify(a.root, [by_path[path]])[0][1] != 'verified':
            raise ValueError('Prediction export failed hash verification')
        with (a.root / path).open() as handle:
            rows = [r for r in csv.DictReader(handle) if r['fair_common'].lower() == 'true']
        if len(rows) != 213:
            raise ValueError('Primary three-method cohort must contain 213 windows')
        methods = {'learned': 'oracle_position_error_cm', 'persistence': 'persistence_position_error_cm',
                   'constant_velocity': 'constant_velocity_position_error_cm'}
        results[seed] = {name: distribution([float(r[col]) for r in rows]) for name, col in methods.items()}
    summary = {method: {stat: {'mean': statistics.mean(results[s][method][stat] for s in results),
                              'sample_sd': statistics.stdev(results[s][method][stat] for s in results)}
                        for stat in results[42][method]} for method in results[42]}
    print(json.dumps({'cohort': '213 shared t+1 windows; annotated receiving hand; no intention gate',
                      'seed_sd_is_not_participant_uncertainty': True,
                      'per_seed': results, 'across_seeds': summary}, indent=2))


if __name__ == '__main__':
    main()
