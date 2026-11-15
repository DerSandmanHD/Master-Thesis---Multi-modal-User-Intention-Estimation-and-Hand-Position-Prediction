"""Frozen same-person adaptation comparison; test evaluated only after selection."""
import argparse
import copy
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn

from data import prepare_data, save_data_metadata
from artifact_freeze import sha256_file
from train_residual import build_residual_model, make_loader, run_epoch, class_weights, set_seed, checkpoint_payload


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root', type=Path, required=True)
    p.add_argument('--source-run', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--preflight-only', action='store_true')
    args=p.parse_args(); root=args.data_root.resolve(); source=args.source_run.resolve()
    protocol=json.loads((root/'adaptation_protocol.json').read_text())
    base=json.loads((source/'config.json').read_text())
    source_meta=json.loads((source/'data_metadata.json').read_text())
    source_path=source/'best_intention_model.pt'
    checkpoint=torch.load(source_path,map_location='cpu',weights_only=True)
    config=copy.deepcopy(base)
    for k in ('manifest_filter','dataset_contract','validation_participants','test_participants','train_participants','visual_embeddings'):
        config['data'].pop(k,None)
    config['data'].update(master_dir=str(root/'master_datasets'), sequence_split=protocol['sequence_split'])
    config['training'].update(seed=args.seed,num_workers=0)
    device=torch.device('cpu' if args.preflight_only else 'cuda')
    if not args.preflight_only and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
    args.output.mkdir(parents=True,exist_ok=False)
    write(args.output/'protocol.json',protocol)
    write(args.output/'source.json',dict(checkpoint=str(source_path),sha256=sha256_file(source_path),
          normalizer_sha256=sha256_file(source/'data_metadata.json'),protocol_sha256=sha256_file(root/'adaptation_protocol.json')))
    bundles={}
    for mode in ('scratch','finetune'):
        data_config=copy.deepcopy(config['data'])
        if mode=='finetune':data_config['frozen_normalizer']=source_meta['normalizer']
        bundle=prepare_data(data_config,args.seed)
        if bundle.feature_columns != source_meta['feature_columns']:raise ValueError('Source feature schema mismatch')
        if len(bundle.normalizer.output_feature_names)!=checkpoint['input_dim']:raise ValueError('Source input dimension mismatch')
        support={}
        for name in ('train','validation','test'):
            dataset=getattr(bundle,name)
            support[name]=dict(windows=len(dataset),intentions=dataset.intention_counts(),hands=dataset.receiving_hand_counts())
            if not len(dataset) or min(dataset.intention_counts())<=0 or min(dataset.receiving_hand_counts()[:2])<=0:
                raise ValueError(f'Missing label support: {name}: {support[name]}')
        write(args.output/f'{mode}_support.json',support)
        save_data_metadata(bundle,args.output/f'{mode}_data_metadata.json')
        bundles[mode]=bundle
    for name in ('train','validation','test'):
        if getattr(bundles['scratch'],name).endpoint_fingerprint()!=getattr(bundles['finetune'],name).endpoint_fingerprint():
            raise ValueError('Comparison window mismatch')
    if args.preflight_only:
        model=build_residual_model(checkpoint['model_type'],input_dim=checkpoint['input_dim'],window_size=checkpoint['window_size'],model_config=checkpoint['model_config'])
        model.load_state_dict(checkpoint['model_state_dict'],strict=True)
        write(args.output/'preflight.json',{'passed':True})
        return
    criteria=None
    selected={}
    for mode in ('scratch','finetune'):
        set_seed(args.seed)
        bundle=bundles[mode]; cfg=copy.deepcopy(config); cfg['data']['frozen_normalizer']=source_meta['normalizer'] if mode=='finetune' else None
        settings=protocol['training'][mode]; cfg['training'].update(learning_rate=settings['learning_rate'],epochs=settings['epochs'])
        out=args.output/mode;out.mkdir()
        write(out/'config.json',cfg)
        model=build_residual_model(checkpoint['model_type'],input_dim=checkpoint['input_dim'],window_size=checkpoint['window_size'],model_config=checkpoint['model_config']).to(device)
        if mode=='finetune':model.load_state_dict(checkpoint['model_state_dict'],strict=True)
        counts=bundle.train.intention_counts()
        criteria=(nn.CrossEntropyLoss(weight=class_weights([counts[0],sum(counts[1:])],device)),
                  nn.CrossEntropyLoss(weight=class_weights(counts[1:3],device)),
                  nn.CrossEntropyLoss(weight=class_weights(bundle.train.receiving_hand_counts(),device)))
        train=make_loader(bundle.train,cfg['training'],shuffle=True,device=device)
        validation=make_loader(bundle.validation,cfg['training'],shuffle=False,device=device)
        optimizer=torch.optim.AdamW(model.parameters(),lr=settings['learning_rate'],weight_decay=cfg['training']['weight_decay'])
        best=-math.inf; stale=0;history=[]
        for epoch in range(1,settings['epochs']+1):
            tm=run_epoch(model,train,device,*criteria,cfg['training'],optimizer)
            vm=run_epoch(model,validation,device,*criteria,cfg['training'])
            score=float(vm['intention']['macro_f1'])
            if not math.isfinite(score):raise ValueError('Nonfinite validation score')
            history.append(dict(epoch=epoch,train=tm,validation=vm))
            write(out/'history.json',history)
            print(mode,args.seed,epoch,'val_f1',score,flush=True)
            if score>best:
                best=score;stale=0
                payload=checkpoint_payload(model,cfg,bundle,sum(p.numel() for p in model.parameters()),epoch,'validation_intention_macro_f1',score)
                torch.save(payload,out/'best_intention_model.pt')
            else:stale+=1
            if stale>=settings['patience']:break
        selected[mode]=out/'best_intention_model.pt'
    # All choices fixed before reading any test results.
    write(args.output/'selection.json',{k:dict(path=str(v),sha256=sha256_file(v)) for k,v in selected.items()})
    results={}
    for mode,path in {'unchanged':source_path,**selected}.items():
        bundle=bundles['scratch' if mode=='scratch' else 'finetune']
        state=torch.load(path,map_location=device,weights_only=True)
        model=build_residual_model(state['model_type'],input_dim=state['input_dim'],window_size=state['window_size'],model_config=state['model_config']).to(device)
        model.load_state_dict(state['model_state_dict'],strict=True)
        loader=make_loader(bundle.test,config['training'],shuffle=False,device=device)
        results[mode]=run_epoch(model,loader,device,*criteria,config['training'])
    write(args.output/'test_results.json',results)
    (args.output/'COMPLETE').touch()


if __name__=='__main__': main()
