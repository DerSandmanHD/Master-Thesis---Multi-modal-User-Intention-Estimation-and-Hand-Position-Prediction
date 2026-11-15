"""Export only validation predictions for frozen terminal checkpoints."""
import argparse
import json
from pathlib import Path

import torch

import artifact_freeze
from artifact_freeze import sha256_file, validate_artifact_freeze
from data import prepare_data
from endpose_v2 import DUAL_HORIZON_MODEL_TYPE
from evaluate_terminal_timing import infer, verify_reproduction, write_json
from terminal_residual_diagnostics import assert_same_cases, paired_future
from train_residual import build_residual_model, set_seed


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo',type=Path,required=True)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--seed',type=int,choices=(42,43,44),required=True)
    a=parser.parse_args()
    artifact_freeze.PROJECT_ROOT=a.repo.resolve()
    protocol_path=a.root/'code/Training/configs/evaluation/terminal_shrinkage_v1.json'
    protocol=json.loads(protocol_path.read_text())
    run=a.repo/'Training/runs/dataset_v3_causal_20260815_n214_5d136a34/thesis_final_v2_validation/terminal_endpose_learned'/f'terminal_endpose_learned_seed{a.seed}'
    freeze=validate_artifact_freeze(run/'artifact_manifest.json',require_current_git_state=False)
    config=json.loads((run/'config.json').read_text())
    metadata=json.loads((run/'data_metadata.json').read_text())
    metrics=json.loads((run/'metrics.json').read_text())
    if metrics.get('test_evaluation_skipped') is not True or 'test' in metrics or 'test_by_checkpoint' in metrics:
        raise ValueError('Source must contain validation-only metrics')
    if config['model_type']!=DUAL_HORIZON_MODEL_TYPE or config['training']['seed']!=a.seed:
        raise ValueError('Wrong source model/seed')
    if config['data']['pose_target']['target_definition_version']!=protocol['target_version']:
        raise ValueError('Wrong target definition')
    config['data']['frozen_normalizer']=metadata['normalizer']
    bundle=prepare_data(config['data'],a.seed)
    for key in ('dataset_content_fingerprint','source_content_fingerprint'):
        if bundle.provenance[key]!=metadata['provenance'][key]:raise ValueError('Data drift')
    if bundle.normalizer.to_dict()!=metadata['normalizer'] or bundle.feature_columns!=metadata['feature_columns']:
        raise ValueError('Normalizer/schema drift')
    for split in ('train','validation','test'):
        if bundle.split_metadata['sequences'][split]!=metadata['split']['sequences'][split]:raise ValueError('Split drift')
        if getattr(bundle,split).endpoint_fingerprint()!=metadata['split']['window_eligibility']['endpoint_fingerprints'][split]:
            raise ValueError('Window drift')
    if not torch.cuda.is_available():raise RuntimeError('GPU required for inference')
    device=torch.device('cuda');set_seed(a.seed)
    out=a.root/f'seed{a.seed}';out.mkdir(parents=True,exist_ok=False)
    write_json(out/'protocol.json',protocol)
    dataset=bundle.validation
    indices=[i for i,(r,e) in enumerate(dataset.indices) if dataset.records[r].intentions[e]==2]
    frames={};identities={};reproduction={}
    frozen={v['sha256'] for v in freeze['output_artifacts']['checkpoints'].values()}
    for family in protocol['checkpoint_families']:
        path=run/f'{family}_model.pt';digest=sha256_file(path)
        identity=metrics['checkpoints'][family]
        if digest!=identity['sha256'] or digest not in frozen:raise ValueError('Checkpoint drift')
        if not identity['selection_metric'].startswith('validation_'):raise ValueError('Invalid checkpoint selection')
        state=torch.load(path,map_location='cpu',weights_only=True)
        model=build_residual_model(state['model_type'],input_dim=state['input_dim'],window_size=state['window_size'],
                                  model_config=state['model_config']).to(device)
        model.load_state_dict(state['model_state_dict'],strict=True)
        frame=infer(model,dataset,indices,'frozen_stride10',a.seed,device)
        reproduction[family]=verify_reproduction(frame,metrics['validation_by_checkpoint'][family])
        if frames:assert_same_cases(frame,frames[protocol['checkpoint_families'][0]])
        frames[family]=frame
        frame.to_csv(out/f'{family}_validation_predictions.csv',index=False)
        identities[family]=dict(checkpoint_sha256=digest,epoch=identity['epoch'],selection_metric=identity['selection_metric'])
        if sha256_file(path)!=digest:raise ValueError('Inference changed source checkpoint')
        print(a.seed,family,'pure future validation windows',len(paired_future(frame)),flush=True)
    write_json(out/'checkpoint_identities.json',identities)
    write_json(out/'audit.json',dict(seed=a.seed,split='validation',test_predictions_loaded=False,retraining=False,
        protocol_sha256=sha256_file(protocol_path),artifact_fingerprint=freeze['manifest_fingerprint'],
        validation_sequences=bundle.split_metadata['sequences']['validation'],
        test_sequences=bundle.split_metadata['sequences']['test'],
        validation_fingerprint=dataset.endpoint_fingerprint(),reproduction=reproduction,
        output_sha256={p.name:sha256_file(p) for p in out.iterdir() if p.is_file()}))
    (out/'COMPLETE').touch()


if __name__=='__main__':main()
