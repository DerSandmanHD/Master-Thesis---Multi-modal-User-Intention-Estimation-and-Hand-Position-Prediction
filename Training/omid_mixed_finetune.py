"""Exploratory old/new rehearsal fine-tuning with frozen, disjoint test sets."""
import argparse
import copy
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import ConcatDataset, DataLoader, WeightedRandomSampler

from artifact_freeze import sha256_file
from data import checkpoint_provenance, prepare_data, save_data_metadata
from evaluate_omid_retention import mask_object_features
from train_residual import (
    build_residual_model, checkpoint_payload, class_weights, make_loader,
    run_epoch, set_seed,
)


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def domain_sampling_weights(sizes):
    """Equal domain probability; uniform windows within each domain."""
    if len(sizes) != 2 or min(sizes) <= 0:
        raise ValueError('Two nonempty training domains required')
    return torch.cat([torch.full((n,), 0.5 / n, dtype=torch.double) for n in sizes])


def expected_label_mass(datasets, method):
    # Counts per all eligible windows, including unlabeled-hand windows in the
    # denominator. This matches the sampler before run_epoch filters labels.
    return sum(np.asarray(getattr(d, method)(), dtype=float) / len(d) * 0.5
               for d in datasets).tolist()


def validation_score(metrics):
    values = [float(metrics[d]['intention']['macro_f1']) for d in ('old', 'new')]
    if not all(math.isfinite(v) for v in values):
        raise ValueError('Nonfinite validation score')
    return sum(values) / 2


def verify_bundle(bundle, previous_meta):
    """Reject drift of ANY split, not just held-out windows."""
    if bundle.feature_columns != previous_meta['feature_columns']:
        raise ValueError('Feature schema changed')
    for split in ('train', 'validation', 'test'):
        if set(bundle.split_metadata['sequences'][split]) != set(previous_meta['split']['sequences'][split]):
            raise ValueError(f'{split} sequence membership changed')
        if getattr(bundle, split).endpoint_fingerprint() != previous_meta['split']['window_eligibility']['endpoint_fingerprints'][split]:
            raise ValueError(f'{split} windows changed')


def verify_isolation(bundles):
    seen = set()
    for domain, bundle in bundles.items():
        for split in ('train', 'validation', 'test'):
            ids = bundle.split_metadata['sequences'][split]
            if len(set(ids)) != len(ids) or seen.intersection(ids):
                raise ValueError(f'Overlapping sequences: {domain}/{split}')
            if not ids or not len(getattr(bundle, split)):
                raise ValueError(f'Empty split: {domain}/{split}')
            seen.update(ids)


def require_pose_cohort(metrics, prior):
    keys = ('sample_key_fingerprint', 'shared_samples', 'coverage_denominator_pose_targets')
    for key in keys:
        if metrics['pose_fair_common'][key] != prior['pose_fair_common'][key]:
            raise ValueError(f'Test pose cohort changed: {key}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--seed', type=int, choices=(42, 43, 44), required=True)
    parser.add_argument('--preflight-only', action='store_true')
    args = parser.parse_args()
    root = args.data_root.resolve()
    protocol_path = root / 'mixed_protocol_v1.json'
    protocol = json.loads(protocol_path.read_text())
    previous = root / 'adaptation_runs_v1' / f'seed{args.seed}'
    source_info = json.loads((previous / 'source.json').read_text())
    source_path = Path(source_info['checkpoint'])
    source = source_path.parent
    if sha256_file(source_path) != source_info['sha256']:
        raise ValueError('Original checkpoint changed')
    if sha256_file(source / 'data_metadata.json') != source_info['normalizer_sha256']:
        raise ValueError('Original normalizer metadata changed')
    if sha256_file(root / 'adaptation_protocol.json') != source_info['protocol_sha256']:
        raise ValueError('Original new-recording split protocol changed')
    base = json.loads((source / 'config.json').read_text())
    source_meta = json.loads((source / 'data_metadata.json').read_text())
    new_meta = json.loads((previous / 'finetune_data_metadata.json').read_text())
    adaptation = json.loads((root / 'adaptation_protocol.json').read_text())
    if protocol['training'] != adaptation['training']['finetune']:
        raise ValueError('Training settings must match prior fine-tuning')
    state = torch.load(source_path, map_location='cpu', weights_only=True)
    configs = {'old': copy.deepcopy(base['data']), 'new': copy.deepcopy(base['data'])}
    master = Path(configs['old']['master_dir'])
    configs['old']['master_dir'] = str(master if master.is_absolute() else args.repo / master)
    for key in ('manifest_filter', 'dataset_contract', 'validation_participants',
                'test_participants', 'train_participants', 'visual_embeddings'):
        configs['new'].pop(key, None)
    configs['new'].update(master_dir=str(root / 'master_datasets'), sequence_split=adaptation['sequence_split'])
    for config in configs.values():
        config['frozen_normalizer'] = source_meta['normalizer']
    bundles = {domain: prepare_data(config, args.seed) for domain, config in configs.items()}
    verify_bundle(bundles['old'], source_meta)
    verify_bundle(bundles['new'], new_meta)
    verify_isolation(bundles)
    for bundle in bundles.values():
        if bundle.normalizer.to_dict() != source_meta['normalizer']:
            raise ValueError('Normalizer must remain frozen')
        if len(bundle.normalizer.output_feature_names) != state['input_dim']:
            raise ValueError('Model input dimension changed')
    if bundles['old'].split_metadata['pose_target'] != bundles['new'].split_metadata['pose_target']:
        raise ValueError('Incompatible pose targets')
    suffix = 'mixed_preflight_v1' if args.preflight_only else 'mixed_runs_v1'
    out = root / suffix / f'seed{args.seed}'
    out.mkdir(parents=True, exist_ok=False)
    write(out / 'protocol.json', protocol)
    write(out / 'source.json', {**source_info, 'mixed_protocol_sha256': sha256_file(protocol_path)})
    support = {}
    for domain, bundle in bundles.items():
        save_data_metadata(bundle, out / f'{domain}_data_metadata.json')
        support[domain] = {
            split: dict(sequences=len(bundle.split_metadata['sequences'][split]),
                        windows=len(getattr(bundle, split)),
                        fingerprint=getattr(bundle, split).endpoint_fingerprint())
            for split in ('train', 'validation', 'test')
        }
    write(out / 'support.json', support)
    set_seed(args.seed)
    device = torch.device('cpu' if args.preflight_only else 'cuda')
    if not args.preflight_only and not torch.cuda.is_available():
        raise RuntimeError('GPU required for training')
    model = build_residual_model(state['model_type'], input_dim=state['input_dim'],
                                 window_size=state['window_size'], model_config=state['model_config']).to(device)
    model.load_state_dict(state['model_state_dict'], strict=True)
    if args.preflight_only:
        write(out / 'preflight.json', {'passed': True, 'source_sha256': source_info['sha256']})
        print(json.dumps(support), flush=True)
        return
    settings = protocol['training']
    config = copy.deepcopy(base)
    for key in ('hyperparameter_search_v3', 'run_context', 'evaluation'):
        config.pop(key, None)
    config.update(run_name=f'omid_mixed_finetune_seed{args.seed}', data=configs['old'],
                  mixed_data=configs, mixed_protocol=protocol)
    config['training'].update(seed=args.seed, num_workers=0, epochs=settings['epochs'],
                              learning_rate=settings['learning_rate'],
                              early_stopping_patience=settings['patience'])
    write(out / 'config.json', config)
    train_sets = [bundles[d].train for d in ('old', 'new')]
    sizes = [len(d) for d in train_sets]
    sampler = WeightedRandomSampler(domain_sampling_weights(sizes), sum(sizes), replacement=True,
                                    generator=torch.Generator().manual_seed(args.seed))
    train = DataLoader(ConcatDataset(train_sets), batch_size=config['training']['batch_size'],
                       sampler=sampler, num_workers=0, pin_memory=True,
                       generator=torch.Generator().manual_seed(args.seed))
    counts = expected_label_mass(train_sets, 'intention_counts')
    hands = expected_label_mass(train_sets, 'receiving_hand_counts')
    criteria = (nn.CrossEntropyLoss(weight=class_weights([counts[0], sum(counts[1:])], device)),
                nn.CrossEntropyLoss(weight=class_weights(counts[1:3], device)),
                nn.CrossEntropyLoss(weight=class_weights(hands, device)))
    write(out / 'sampling.json', dict(domain_probability={'old': 0.5, 'new': 0.5},
          replacement=True, draws_per_epoch=sum(sizes), windows=dict(zip(('old', 'new'), sizes)),
          expected_intention_mass=counts, expected_hand_mass=hands,
          loss_class_weights=[c.weight.cpu().tolist() for c in criteria]))
    validation = {d: make_loader(b.validation, config['training'], shuffle=False, device=device)
                  for d, b in bundles.items()}
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings['learning_rate'],
                                  weight_decay=config['training']['weight_decay'])
    history = []
    best, stale = -math.inf, 0
    selected = out / 'best_intention_model.pt'
    for epoch in range(1, settings['epochs'] + 1):
        tm = run_epoch(model, train, device, *criteria, config['training'], optimizer)
        vm = {d: run_epoch(model, loader, device, *criteria, config['training'])
              for d, loader in validation.items()}
        score = validation_score(vm)
        history.append(dict(epoch=epoch, train=tm, validation=vm, selection_score=score))
        write(out / 'history.json', history)
        print(f'seed={args.seed} epoch={epoch} old_val={vm["old"]["intention"]["macro_f1"]:.6f} '
              f'new_val={vm["new"]["intention"]["macro_f1"]:.6f} mean={score:.6f}', flush=True)
        if score > best:
            best, stale = score, 0
            payload = checkpoint_payload(model, config, bundles['old'],
                         sum(p.numel() for p in model.parameters()), epoch,
                         'equal_domain_mean_validation_intention_macro_f1', score)
            payload['dataset_provenance'] = {
                'protocol': 'mixed_old_new_rehearsal_v1',
                'domains': {d: checkpoint_provenance(b) for d, b in bundles.items()},
            }
            payload['mixed_protocol'] = protocol
            payload['source_sha256'] = source_info['sha256']
            torch.save(payload, selected)
        else:
            stale += 1
        if stale >= settings['patience']:
            break
    # Selection and its checksum are fixed before any test inference.
    selected_hash = sha256_file(selected)
    write(out / 'selection.json', dict(path=str(selected), sha256=selected_hash,
          score=best, criterion='mean of old/new validation intention macro-F1'))
    selected_state = torch.load(selected, map_location=device, weights_only=True)
    model.load_state_dict(selected_state['model_state_dict'], strict=True)
    prior_old = json.loads((root / 'retention_v1' / f'seed{args.seed}' / 'test_results.json').read_text())
    prior_new = json.loads((previous / 'test_results.json').read_text())
    test_sets = {'original': bundles['old'].test,
                 'objects_masked': mask_object_features(bundles['old'].test, bundles['old'].normalizer),
                 'new_marker_free': bundles['new'].test}
    results = {}
    # Unweighted loss is diagnostic only; reported F1/pose metrics do not use it.
    eval_criteria = (nn.CrossEntropyLoss(), nn.CrossEntropyLoss(), nn.CrossEntropyLoss())
    for condition, dataset in test_sets.items():
        loader = make_loader(dataset, config['training'], shuffle=False, device=device)
        metrics = run_epoch(model, loader, device, *eval_criteria, config['training'])
        for variant in ('unchanged', 'finetune', 'scratch'):
            prior = prior_new[variant] if condition == 'new_marker_free' else prior_old[variant][condition]
            require_pose_cohort(metrics, prior)
        results[condition] = metrics
        print('test', condition, metrics['intention']['macro_f1'], flush=True)
    if sha256_file(selected) != selected_hash or sha256_file(source_path) != source_info['sha256']:
        raise ValueError('Evaluation changed checkpoint')
    write(out / 'test_results.json', results)
    write(out / 'audit.json', dict(seed=args.seed, selected_epoch=selected_state['epoch'],
          validation_selection_value=best, test_used_for_checkpoint_selection=False,
          exploratory_after_observed_tests=True, all_prior_test_pose_cohorts_matched=True,
          support=support, source_sha256=source_info['sha256'], checkpoint_sha256=selected_hash))
    (out / 'COMPLETE').touch()


if __name__ == '__main__':
    main()
