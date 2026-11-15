"""Frozen old-test retention and object-input masking comparison; no fitting."""
import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from data import prepare_data, save_data_metadata
from artifact_freeze import sha256_file
from train_residual import build_residual_model, make_loader, run_epoch, set_seed


def mask_object_features(dataset, normalizer):
    """Match marker-free builder semantics without changing eligibility/targets."""
    result=copy.deepcopy(dataset)
    names=normalizer.feature_names; n=len(names)
    indices=[i for i,name in enumerate(names) if name.startswith('aruco_')]
    if len(indices)!=54:raise ValueError(f'Unexpected object schema: {len(indices)}')
    for record in result.records:
        for i in indices:
            if names[i].endswith('_valid'):
                record.features[:,i]=(0.0-normalizer.mean[i])/normalizer.std[i]
                record.features[:,n+i]=1.0
            else:
                record.features[:,i]=0.0
                record.features[:,n+i]=0.0
    if result.endpoint_fingerprint()!=dataset.endpoint_fingerprint():
        raise ValueError('Masking changed test cohort')
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root',type=Path,required=True)
    p.add_argument('--repo',type=Path,required=True)
    p.add_argument('--seed',type=int,required=True)
    a=p.parse_args();set_seed(a.seed)
    runs=a.data_root/'adaptation_runs_v1'/f'seed{a.seed}'
    source_info=json.loads((runs/'source.json').read_text())
    source=Path(source_info['checkpoint']).parent
    source_meta=json.loads((source/'data_metadata.json').read_text())
    base=json.loads((source/'config.json').read_text())
    out=a.data_root/'retention_v1'/f'seed{a.seed}';out.mkdir(parents=True,exist_ok=False)
    def save(name,value):(out/name).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    device=torch.device('cuda')
    if not torch.cuda.is_available():raise RuntimeError('GPU required')
    results={};identities={};fingerprints=set();pose_fingerprints=set()
    for variant in ('unchanged','finetune','scratch'):
        ckpath=source/'best_intention_model.pt' if variant=='unchanged' else runs/variant/'best_intention_model.pt'
        expected=source_info['sha256'] if variant=='unchanged' else json.loads((runs/'selection.json').read_text())[variant]['sha256']
        if sha256_file(ckpath)!=expected:raise ValueError('Checkpoint hash changed')
        meta=source_meta if variant=='unchanged' else json.loads((runs/f'{variant}_data_metadata.json').read_text())
        config=copy.deepcopy(base)
        master=Path(config['data']['master_dir'])
        if not master.is_absolute():master=a.repo/master
        config['data']['master_dir']=str(master)
        config['data']['frozen_normalizer']=meta['normalizer']
        config['training'].update(num_workers=0,seed=a.seed)
        bundle=prepare_data(config['data'],a.seed)
        if bundle.feature_columns!=meta['feature_columns']:raise ValueError('Feature mismatch')
        if bundle.split_metadata['sequences']['test']!=source_meta['split']['sequences']['test']:
            raise ValueError('Historical test split changed')
        fingerprint=bundle.test.endpoint_fingerprint();fingerprints.add(fingerprint)
        old=source_meta['split']['window_eligibility']['endpoint_fingerprints']['test']
        if fingerprint!=old:raise ValueError('Historical test windows changed')
        save_data_metadata(bundle,out/f'{variant}_data_metadata.json')
        state=torch.load(ckpath,map_location=device,weights_only=True)
        model=build_residual_model(state['model_type'],input_dim=state['input_dim'],window_size=state['window_size'],model_config=state['model_config']).to(device)
        model.load_state_dict(state['model_state_dict'],strict=True)
        criteria=(nn.CrossEntropyLoss(),nn.CrossEntropyLoss(),nn.CrossEntropyLoss())
        results[variant]={}
        for condition in ('original','objects_masked'):
            dataset=bundle.test if condition=='original' else mask_object_features(bundle.test,bundle.normalizer)
            loader=make_loader(dataset,config['training'],shuffle=False,device=device)
            metrics=run_epoch(model,loader,device,*criteria,config['training'])
            results[variant][condition]=metrics
            pose_fingerprints.add(metrics['pose_fair_common']['sample_key_fingerprint'])
            print(variant,condition,metrics['intention']['macro_f1'],flush=True)
        identities[variant]=dict(checkpoint=str(ckpath),sha256=expected,normalizer=meta['normalizer'])
        if sha256_file(ckpath)!=expected:raise ValueError('Evaluation modified checkpoint')
    if len(fingerprints)!=1 or len(pose_fingerprints)!=1:raise ValueError('Unpaired evaluation')
    save('identities.json',identities)
    save('test_results.json',results)
    save('audit.json',dict(seed=a.seed,training_performed=False,window_fingerprint=next(iter(fingerprints)),
        pose_fingerprint=next(iter(pose_fingerprints)),test_sequences=bundle.split_metadata['sequences']['test'],
        test_participants=bundle.split_metadata['participants']['test'],windows=len(bundle.test),
        object_masking='geometry/gaze missing; validity zero observed; targets and windows unchanged',
        historical_test_previously_observed=True))
    (out/'COMPLETE').touch()


if __name__=='__main__':main()
