"""Supplementary paired best-intention/best-pose terminal test diagnostics."""
import argparse
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import artifact_freeze
from artifact_freeze import canonical_json_hash, sha256_file, validate_artifact_freeze
from data import prepare_data
from endpose_v2 import DUAL_HORIZON_MODEL_TYPE
from evaluate_terminal_timing import infer, verify_reproduction, write_json
from select_matrix_checkpoints import validate_embedded_final_test_authorization
from terminal_residual_diagnostics import assert_same_cases, paired_future
from train_residual import build_residual_model, set_seed


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo', type=Path, required=True)
    p.add_argument('--output-root', type=Path, required=True)
    p.add_argument('--seed', type=int, choices=(42,43,44), required=True)
    a = p.parse_args()
    repo = a.repo.resolve()
    artifact_freeze.PROJECT_ROOT = repo
    base = repo/'Training/reports/dataset_v3_causal_20260815_n214_5d136a34'
    source_report = base/'thesis_final_v2_corrected_alignment/final_test'/f'terminal_endpose_learned_seed{a.seed}.json'
    prior = json.loads(source_report.read_text())
    if prior['report_fingerprint'] != canonical_json_hash({**prior, 'report_fingerprint': None}):
        raise ValueError('Historical official report changed')
    run = Path(prior['source_run'])
    freeze = validate_artifact_freeze(run/'artifact_manifest.json', require_current_git_state=False)
    if freeze['manifest_fingerprint'] != prior['source_artifact_manifest_fingerprint']:
        raise ValueError('Historical source freeze changed')
    # Preserve official main-checkpoint authorization. The second checkpoint is
    # evaluated only under this explicitly supplementary post-hoc protocol.
    validate_embedded_final_test_authorization(prior, authorization_base=source_report.parent, project_root=repo)
    protocol_path = a.output_root/'code/Training/configs/evaluation/terminal_residual_diagnosis_v1.json'
    protocol = json.loads(protocol_path.read_text())
    config = json.loads((run/'config.json').read_text())
    metadata = json.loads((run/'data_metadata.json').read_text())
    metrics = json.loads((run/'metrics.json').read_text())
    if config['model_type'] != DUAL_HORIZON_MODEL_TYPE or config['training']['seed'] != a.seed:
        raise ValueError('Wrong model/seed')
    if config['data']['pose_target']['target_definition_version'] != protocol['target_version']:
        raise ValueError('Wrong terminal target')
    config['data']['frozen_normalizer'] = metadata['normalizer']
    bundle = prepare_data(config['data'], a.seed)
    if bundle.normalizer.to_dict() != metadata['normalizer'] or bundle.feature_columns != metadata['feature_columns']:
        raise ValueError('Normalizer/schema drift')
    for key in ('dataset_content_fingerprint', 'source_content_fingerprint'):
        if bundle.provenance[key] != metadata['provenance'][key]:
            raise ValueError('Dataset drift: '+key)
    for split in ('train','validation','test'):
        if bundle.split_metadata['sequences'][split] != metadata['split']['sequences'][split]:
            raise ValueError('Split drift: '+split)
        if getattr(bundle,split).endpoint_fingerprint() != metadata['split']['window_eligibility']['endpoint_fingerprints'][split]:
            raise ValueError('Window drift: '+split)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required')
    device = torch.device('cuda')
    set_seed(a.seed)
    out = a.output_root/f'seed{a.seed}'
    out.mkdir(parents=True, exist_ok=False)
    write_json(out/'protocol.json', protocol)
    indices = [i for i,(r,e) in enumerate(bundle.test.indices) if bundle.test.records[r].intentions[e] == 2]
    frames, identities, validation, hashes = {}, {}, {}, {}
    frozen_hashes = {v['sha256'] for v in freeze['output_artifacts']['checkpoints'].values()}
    for name in protocol['checkpoints']:
        checkpoint = run/f'{name}_model.pt'
        digest = sha256_file(checkpoint)
        info = metrics['checkpoints'][name]
        if digest != info['sha256'] or digest not in frozen_hashes:
            raise ValueError('Checkpoint differs from preexisting freeze')
        if not info['selection_metric'].startswith('validation_'):
            raise ValueError('Checkpoint was not validation-selected')
        if name == 'best_intention' and digest != prior['checkpoint']['sha256']:
            raise ValueError('Main checkpoint differs from official test')
        state = torch.load(checkpoint, map_location='cpu', weights_only=True)
        model = build_residual_model(state['model_type'], input_dim=state['input_dim'],
                 window_size=state['window_size'], model_config=state['model_config']).to(device)
        model.load_state_dict(state['model_state_dict'], strict=True)
        frame = infer(model, bundle.test, indices, 'frozen_stride10', a.seed, device)
        if name == 'best_intention':
            reproduction = verify_reproduction(frame, prior['test_metrics'])
            old_folder = base/'terminal_input_timing_v1'/f'seed{a.seed}'
            old_audit = json.loads((old_folder/'audit.json').read_text())
            old_path = old_folder/'frozen_window_predictions.csv'
            if sha256_file(old_path) != old_audit['output_sha256'][old_path.name]:
                raise ValueError('Previous timing export changed')
            old = pd.read_csv(old_path)
            assert_same_cases(frame, old)
            for method in ('learned_end_to_end','learned_oracle_hand','persistence','persistence_predicted_hand'):
                col = method+'_position_error_cm'
                if not np.allclose(frame[col], old[col], equal_nan=True, atol=1e-4):
                    raise ValueError('Previous predictions differ')
        else:
            assert_same_cases(frame, frames['best_intention'])
        frame['checkpoint'] = name
        frame.to_csv(out/f'{name}_predictions.csv', index=False)
        frames[name] = frame
        identities[name] = dict(path=str(checkpoint), sha256=digest, **{k:info[k] for k in ('epoch','selection_metric','selection_value')})
        hashes[name] = digest
        v = metrics['validation_by_checkpoint'][name]
        validation[name] = dict(intention_f1=v['intention']['macro_f1'],
            pure_future=v['pose_by_terminal_target_regime']['strictly_before_aggregation'])
        print(a.seed,name,'pure future paired',len(paired_future(frame)),flush=True)
    for name in protocol['checkpoints']:
        if sha256_file(run/f'{name}_model.pt') != hashes[name]:
            raise ValueError('Inference modified checkpoint')
    if len(paired_future(frames['best_intention'])) != 201:
        raise ValueError('Primary historical cohort not reproduced')
    write_json(out/'validation_results.json', validation)
    write_json(out/'checkpoint_identities.json', identities)
    write_json(out/'audit.json', dict(seed=a.seed, training_performed=False, supplementary=True,
        checkpoint_selection_changed=False, protocol_sha256=sha256_file(protocol_path),
        test_window_fingerprint=bundle.test.endpoint_fingerprint(),
        source_report_sha256=sha256_file(source_report), artifact_fingerprint=freeze['manifest_fingerprint'],
        main_reproduction=reproduction, same_cases=True,
        output_sha256={p.name:sha256_file(p) for p in out.iterdir() if p.is_file()}))
    (out/'COMPLETE').touch()


if __name__ == '__main__':
    main()
