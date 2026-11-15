"""Frozen terminal model: position error for inputs at different times."""
import argparse
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

import artifact_freeze
from artifact_freeze import canonical_json_hash, sha256_file, validate_artifact_freeze
from data import WindowDataset, prepare_data
from endpose_v2 import DUAL_HORIZON_MODEL_TYPE
from metrics import sample_key_fingerprint
from select_matrix_checkpoints import validate_embedded_final_test_authorization
from train_residual import build_residual_model, set_seed


METHODS = ('learned_end_to_end', 'learned_oracle_hand',
           'persistence_predicted_hand', 'persistence')
REGIMES = ('strictly_before_aggregation', 'partially_overlapping_aggregation')


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')


def block_plan(intentions, length):
    """Anchor exact disjoint blocks at the first annotated handover row."""
    handover = np.flatnonzero(np.asarray(intentions) == 2)
    if not len(handover):
        return [], None, 0
    start, end = int(handover[0]), int(handover[-1])
    if not np.all(np.asarray(intentions)[start:end + 1] == 2):
        raise ValueError('Multiple handover segments require explicit event IDs')
    blocks = [(k, s, s + length - 1)
              for k, s in enumerate(range(start, end - length + 2, length))]
    return blocks, start, (end - start + 1) % length


def terminal_regime(endpoint_ns, metadata):
    if not metadata or not metadata.get('eligible'):
        return 'no_accepted_target'
    if endpoint_ns < metadata['aggregation_capture_start_timestamp_ns']:
        return REGIMES[0]
    if endpoint_ns < metadata['target_capture_timestamp_ns']:
        return REGIMES[1]
    return 'at_or_after_target'


def cohort(frame, regime):
    return frame.loc[(frame['regime'] == regime) & frame['fair_common']]


def verify_reproduction(frame, prior):
    checks = {}
    for regime in REGIMES:
        selected = cohort(frame, regime)
        expected = prior['pose_by_terminal_target_regime'][regime]
        if len(selected) != expected['shared_samples']:
            raise ValueError(f'Reproduced cohort count changed: {regime}')
        valid = frame.loc[(frame.regime == regime) & frame.target_valid]
        if len(valid) != expected['coverage_denominator_pose_targets']:
            raise ValueError(f'Reproduced target denominator changed: {regime}')
        checks[regime] = {'samples': len(selected), 'methods': {}}
        for method in ('learned_end_to_end', 'learned_oracle_hand', 'persistence'):
            value = float(selected[f'{method}_position_error_cm'].mean())
            reference = expected['methods'][method]['position_mae_cm']
            if not np.isclose(value, reference, atol=1e-4, rtol=1e-5):
                raise ValueError(f'Historical position error differs: {regime}/{method}: {value} vs {reference}')
            checks[regime]['methods'][method] = dict(reproduced_cm=value, historical_cm=reference)
    keys = frame.loc[frame.fair_common, 'sample_key'].tolist()
    if sample_key_fingerprint(keys) != prior['pose_fair_common']['sample_key_fingerprint']:
        raise ValueError('Historical paired sample identities changed')
    return checks


def infer(model, dataset, indices, scheme, seed, device, block_ids=None):
    loader = DataLoader(Subset(dataset, indices), batch_size=64, shuffle=False,
                        num_workers=0, pin_memory=device.type == 'cuda')
    rows = []
    cursor = 0
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            output = model(batch['features'].to(device), batch['hand_reference_pose'].to(device))
            predictions = output['pose_candidates'].cpu().numpy()
            predicted_hands = output['receiving_hand_logits'].argmax(-1).cpu().numpy()
            for i in range(len(predictions)):
                idx = indices[cursor + i]
                record_idx, end = dataset.indices[idx]
                record = dataset.records[record_idx]
                start = end - dataset.window_size + 1
                _, handover_start, _ = block_plan(record.intentions, dataset.window_size)
                metadata = record.pose_target_metadata
                end_ns = int(record.timestamps_ns[end])
                regime = terminal_regime(end_ns, metadata)
                gt_hand = int(batch['receiving_hand'][i])
                predicted_hand = int(predicted_hands[i])
                target_valid = bool(batch['pose_valid'][i]) and gt_hand in (0, 1)
                references = batch['hand_reference_pose'][i].numpy()
                refs_valid = batch['hand_reference_valid'][i].numpy().astype(bool)
                paired = target_valid and bool(refs_valid.all())
                target = np.asarray(metadata['pose'], dtype=np.float32) if metadata.get('eligible') else None
                if target_valid and not np.allclose(target, batch['pose_target'][i].numpy(), atol=1e-6):
                    raise ValueError('Per-window target differs from fixed sequence terminal target')
                if paired != (regime in REGIMES and gt_hand in (0, 1) and bool(refs_valid.all())):
                    raise ValueError('Temporal regime and target validity disagree')
                row = dict(seed=seed, scheme=scheme, sequence_id=record.sequence_id,
                    participant=record.participant, sample_key=f'{record.sequence_id}|{end_ns}',
                    input_start_row=start, input_end_row=end, input_samples=dataset.window_size,
                    input_start_ns=int(record.timestamps_ns[start]), input_end_ns=end_ns,
                    input_start_relative_row=start-handover_start, input_end_relative_row=end-handover_start,
                    input_start_relative_s=(int(record.timestamps_ns[start])-int(record.timestamps_ns[handover_start]))/1e9,
                    input_end_relative_s=(end_ns-int(record.timestamps_ns[handover_start]))/1e9,
                    input_span_s=(end_ns-int(record.timestamps_ns[start]))/1e9,
                    block_index=block_ids[idx] if block_ids is not None else -1,
                    regime=regime, target_valid=target_valid, fair_common=paired,
                    ground_truth_hand=gt_hand, predicted_hand=predicted_hand,
                    left_reference_valid=bool(refs_valid[0]), right_reference_valid=bool(refs_valid[1]),
                    target_capture_ns=metadata.get('target_capture_timestamp_ns'),
                    aggregation_start_ns=metadata.get('aggregation_capture_start_timestamp_ns'))
                target_ns = metadata.get('target_capture_timestamp_ns')
                row['time_to_terminal_s'] = (target_ns-end_ns)/1e9 if target_ns is not None else None
                for axis, name in enumerate('xyz'):
                    row[f'target_{name}_m'] = float(target[axis]) if target is not None else None
                method_predictions = {
                    'learned_end_to_end': predictions[i, predicted_hand],
                    'learned_oracle_hand': predictions[i, max(0, gt_hand)],
                    'persistence_predicted_hand': references[predicted_hand],
                    'persistence': references[max(0, gt_hand)],
                }
                for method, pose in method_predictions.items():
                    row[f'{method}_position_error_cm'] = float(np.linalg.norm(pose[:3]-target[:3])*100) if paired else None
                    for axis, name in enumerate('xyz'):
                        row[f'{method}_{name}_m'] = float(pose[axis])
                rows.append(row)
            cursor += len(predictions)
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--final-test-report', type=Path, required=True)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--verify-source-only', action='store_true')
    a = parser.parse_args()
    repo = a.repo.resolve()
    # Historical manifests contain paths relative to the original repository,
    # not to this isolated reporting snapshot. Retain every hash check while
    # resolving those original source artifacts against their actual root.
    artifact_freeze.PROJECT_ROOT = repo
    prior = json.loads(a.final_test_report.read_text())
    protocol = json.loads(a.protocol.read_text())
    run = Path(prior['source_run'])
    freeze = validate_artifact_freeze(run/'artifact_manifest.json', require_current_git_state=False)
    if prior['report_fingerprint'] != canonical_json_hash({**prior, 'report_fingerprint': None}):
        raise ValueError('Historical report checksum changed')
    if prior['source_artifact_manifest_fingerprint'] != freeze['manifest_fingerprint']:
        raise ValueError('Historical report belongs to a different source freeze')
    validate_embedded_final_test_authorization(prior, authorization_base=a.final_test_report.parent, project_root=repo)
    config = json.loads((run/'config.json').read_text())
    metadata = json.loads((run/'data_metadata.json').read_text())
    seed = int(config['training']['seed'])
    if seed not in protocol['seeds'] or config['model_type'] != DUAL_HORIZON_MODEL_TYPE:
        raise ValueError('Wrong seed/model family')
    if config['data']['pose_target']['target_definition_version'] != protocol['terminal_target_version']:
        raise ValueError('Wrong terminal target definition')
    if config['data']['window_size'] != protocol['input_samples']:
        raise ValueError('Protocol must retain the trained input length')
    checkpoint_path = run/'best_intention_model.pt'
    checkpoint_hash = sha256_file(checkpoint_path)
    if checkpoint_hash != prior['checkpoint']['sha256']:
        raise ValueError('Frozen checkpoint changed')
    if a.verify_source_only:
        print('Verified frozen source', seed, checkpoint_hash, flush=True)
        return
    config['data']['frozen_normalizer'] = metadata['normalizer']
    bundle = prepare_data(config['data'], seed)
    for key in ('dataset_content_fingerprint', 'source_content_fingerprint'):
        if bundle.provenance[key] != metadata['provenance'][key]:
            raise ValueError(f'Dataset changed: {key}')
    if bundle.feature_columns != metadata['feature_columns'] or bundle.normalizer.to_dict() != metadata['normalizer']:
        raise ValueError('Features/normalizer changed')
    for split in ('train', 'validation', 'test'):
        if bundle.split_metadata['sequences'][split] != metadata['split']['sequences'][split]:
            raise ValueError(f'Split changed: {split}')
        if getattr(bundle, split).endpoint_fingerprint() != metadata['split']['window_eligibility']['endpoint_fingerprints'][split]:
            raise ValueError(f'Frozen windows changed: {split}')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required for this evaluation job')
    device = torch.device('cuda')
    set_seed(seed)
    state = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    model = build_residual_model(state['model_type'], input_dim=state['input_dim'],
             window_size=state['window_size'], model_config=state['model_config']).to(device)
    model.load_state_dict(state['model_state_dict'], strict=True)
    a.output.mkdir(parents=True, exist_ok=False)
    write_json(a.output/'protocol.json', protocol)
    print('Frozen source and data verified', seed, flush=True)
    original = bundle.test
    selected = [i for i, (r, e) in enumerate(original.indices) if original.records[r].intentions[e] == 2]
    regular = infer(model, original, selected, 'frozen_stride10', seed, device)
    reproduction = verify_reproduction(regular, prior['test_metrics'])
    regular.to_csv(a.output/'frozen_window_predictions.csv', index=False)
    print('Historical terminal errors reproduced', seed, flush=True)
    # Use the same normalized records, targets and eligibility checks, at exact
    # handover-relative endpoints. Auxiliary training targets are unnecessary
    # for inference from the already frozen primary terminal output head.
    dc = config['data']
    dense = WindowDataset(copy.deepcopy(original.records), window_size=dc['window_size'], stride=1,
        pose_intent_ids=dc['pose_intent_ids'], minimum_observed_fraction=dc['minimum_observed_fraction'],
        max_timestamp_gap_seconds=dc['max_timestamp_gap_seconds'], include_hand_references=True,
        max_hand_reference_age_seconds=dc['max_hand_reference_age_seconds'])
    lookup = {pair: i for i, pair in enumerate(dense.indices)}
    indices, block_ids, availability, target_audit = [], {}, [], []
    for r, record in enumerate(original.records):
        blocks, start, remainder = block_plan(record.intentions, dc['window_size'])
        target_audit.append(dict(sequence_id=record.sequence_id, participant=record.participant,
             trailing_incomplete_samples=remainder, **record.pose_target_metadata))
        for block, begin, end in blocks:
            idx = lookup.get((r, end))
            availability.append(dict(sequence_id=record.sequence_id, block_index=block,
                input_start_relative_row=begin-start, input_end_relative_row=end-start,
                input_eligible=idx is not None,
                regime=terminal_regime(int(record.timestamps_ns[end]), record.pose_target_metadata)))
            if idx is not None:
                indices.append(idx)
                block_ids[idx] = block
    disjoint = infer(model, dense, indices, 'disjoint60', seed, device, block_ids)
    disjoint.to_csv(a.output/'disjoint_window_predictions.csv', index=False)
    pd.DataFrame(availability).to_csv(a.output/'block_availability.csv', index=False)
    write_json(a.output/'terminal_target_audit.json', target_audit)
    if sha256_file(checkpoint_path) != checkpoint_hash:
        raise ValueError('Inference changed the source checkpoint')
    checks = dict(seed=seed, source_run=str(run), checkpoint_sha256=checkpoint_hash,
        source_report_sha256=sha256_file(a.final_test_report),
        protocol_sha256=sha256_file(a.protocol), source_artifact_manifest_fingerprint=freeze['manifest_fingerprint'],
        test_sequences=bundle.split_metadata['sequences']['test'],
        test_window_fingerprint=original.endpoint_fingerprint(), training_performed=False,
        checkpoint_selection_changed=False, historical_reproduction=reproduction,
        output_sha256={p.name: sha256_file(p) for p in a.output.iterdir() if p.is_file()})
    write_json(a.output/'audit.json', checks)
    (a.output/'COMPLETE').touch()
    print('COMPLETE', seed, 'original handover windows', len(regular), 'disjoint blocks', len(disjoint), flush=True)


if __name__ == '__main__':
    main()
