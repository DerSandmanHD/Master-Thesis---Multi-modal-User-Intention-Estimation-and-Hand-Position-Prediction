#!/usr/bin/env python3
"""Re-evaluate a frozen main/LOPO checkpoint from processed data at a new root.

No training or raw recording access. Stored manifests remain unchanged.
"""
import argparse
import json
import os
from pathlib import Path
import sys

from artifacts import load_lock, selected, verify


def relocate(value, old_root, new_root):
    if isinstance(value, dict):
        return {k: relocate(v, old_root, new_root) for k, v in value.items()}
    if isinstance(value, list):
        return [relocate(v, old_root, new_root) for v in value]
    if isinstance(value, str) and value.startswith(old_root + '/'):
        return str(new_root / value[len(old_root) + 1:])
    return value


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument('--run', required=True, help='Repository-relative frozen main or LOPO run directory')
    p.add_argument('--output', type=Path, required=True, help='New report; existing files are refused')
    p.add_argument('--device', choices=['cpu', 'cuda', 'mps', 'auto'], default='cpu')
    a = p.parse_args()
    root = a.root.resolve()
    os.chdir(root)
    relative = Path(a.run)
    if relative.is_absolute() or '..' in relative.parts:
        raise ValueError('Run must be repository-relative')
    if a.output.exists():
        raise ValueError('Output exists; choose a new report path')
    lock = load_lock()
    prefix = relative.as_posix() + '/'
    run_items = [x for x in lock['files'] if x['path'].startswith(prefix)]
    if not run_items or not any(set(x['groups']) & {'main', 'lopo'} for x in run_items):
        raise ValueError('Run is not an indexed main/LOPO experiment')
    config_path = root / relative / 'config.json'
    # Verify before reading executable configuration or deserializing weights.
    required = selected(lock, ['data']) + run_items
    if any(status != 'verified' for _, status in verify(root, required)):
        raise ValueError('Required processed data or run artifacts failed verification')
    config = json.loads(config_path.read_text())
    if config['data'].get('visual_embeddings', {}).get('enabled', False):
        if any(status != 'verified' for _, status in verify(root, selected(lock, ['visual']))):
            raise ValueError('Visual artifacts failed verification')
    sys.path.insert(0, str(root / 'Training'))
    from evaluate_frozen_run import choose_device, prepare_frozen_bundle, evaluate
    from artifact_freeze import sha256_file
    metadata = json.loads((root / relative / 'data_metadata.json').read_text())
    manifest = json.loads((root / relative / 'artifact_manifest.json').read_text())
    config = relocate(config, lock['historical_cluster_root'], root)
    bundle = prepare_frozen_bundle(config, metadata, master_dir=root / 'Data_collection/master_datasets')
    if bundle.normalizer.to_dict() != metadata['normalizer']:
        raise ValueError('Normalizer differs from frozen training statistics')
    expected = metadata['split']['window_eligibility']['endpoint_fingerprints']
    for split in ('train', 'validation', 'test'):
        if getattr(bundle, split).endpoint_fingerprint() != expected[split]:
            raise ValueError('Window cohort differs: ' + split)
    identity = manifest['output_artifacts']['checkpoints']['best_intention']
    checkpoint = root / relative / identity['path']
    if sha256_file(checkpoint) != identity['sha256']:
        raise ValueError('Selected checkpoint mismatch')
    model_type, metrics, parameters = evaluate(config=config, bundle=bundle,
                                              checkpoint_path=checkpoint, device=choose_device(a.device))
    report = {'scope': 'processed-data checkpoint replay; not a new model selection',
              'source_run': a.run, 'checkpoint_sha256': identity['sha256'],
              'model_type': model_type, 'trainable_parameters': parameters, 'test_metrics': metrics}
    with a.output.open('x') as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write('\n')


if __name__ == '__main__':
    main()
