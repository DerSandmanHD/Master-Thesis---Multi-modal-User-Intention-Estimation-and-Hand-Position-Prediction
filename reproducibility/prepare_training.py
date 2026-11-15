#!/usr/bin/env python3
"""Create new configurations and a command plan; never start training."""
import argparse
import json
from pathlib import Path
import shlex

from artifacts import load_lock, verify
from replay_checkpoint import relocate


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument('--family', choices=['main', 'lopo'], default='main')
    p.add_argument('--output-dir', type=Path, required=True, help='New, empty reproduction directory')
    a = p.parse_args()
    root = a.root.resolve()
    lock = load_lock()
    candidates = [x for x in lock['files'] if a.family in x['groups']
                  and x['path'].startswith('Training/runs/') and x['path'].endswith('/config.json')]
    expected = 48 if a.family == 'main' else 25
    if len(candidates) != expected:
        raise ValueError('Unexpected frozen configuration count')
    if any(s != 'verified' for _, s in verify(root, candidates)):
        raise ValueError('Frozen configuration verification failed')
    output = a.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    for item in candidates:
        config = relocate(json.loads((root / item['path']).read_text()),
                          lock['historical_cluster_root'], root)
        name = Path(item['path']).parent.name
        config_path = output / (name + '.json')
        config_path.write_text(json.dumps(config, indent=2) + '\n')
        entry = 'train_residual.py' if config['model_type'] in {
            'hierarchical_residual_pose_transformer_v2',
            'hierarchical_dual_horizon_residual_pose_transformer_v2'} else 'train.py'
        # Terminal model identifier is defined by the historical configuration.
        if 'residual' in config['model_type']:
            entry = 'train_residual.py'
        command = ['python', str(root / 'Training' / entry), '--config', str(config_path),
                   '--run-dir', str(output / 'runs' / name), '--seed', str(config['training']['seed']),
                   '--dataset-tag', lock['dataset_tag'], '--experiment-tag', output.name,
                   '--skip-test-evaluation']
        rows.append({'source_config': item['path'], 'source_sha256': item['sha256'],
                     'command': shlex.join(command)})
    (output / 'training_plan.json').write_text(json.dumps(rows, indent=2) + '\n')
    print(json.dumps({'configurations': len(rows), 'plan': str(output / 'training_plan.json'),
                      'training_started': False}))


if __name__ == '__main__':
    main()
