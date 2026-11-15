"""Paired frozen retention summary, including the prior new-recording test."""
import argparse
import csv
import json
import statistics
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('--data-root',type=Path,required=True)
    root=p.parse_args().data_root;out=root/'retention_v1';rows=[];audits=[]
    for seed in (42,43,44):
        folder=out/f'seed{seed}'
        if not (folder/'COMPLETE').exists():raise ValueError(f'Incomplete {seed}')
        audit=json.loads((folder/'audit.json').read_text());audits.append(audit)
        old=json.loads((folder/'test_results.json').read_text())
        new=json.loads((root/'adaptation_runs_v1'/f'seed{seed}'/'test_results.json').read_text())
        for variant in ('unchanged','finetune','scratch'):
            for condition,m in {**old[variant],'new_marker_free':new[variant]}.items():
                fair=m['pose_fair_common'];pose=fair['methods']['learned_end_to_end']
                rows.append(dict(seed=seed,variant=variant,condition=condition,
                    intention_macro_f1=m['intention']['macro_f1'],hand_macro_f1=m['receiving_hand']['macro_f1'],
                    position_mean_cm=pose['position_mae_cm'],shared_pose_samples=fair['shared_samples'],
                    pose_fingerprint=fair['sample_key_fingerprint']))
    if len({a['window_fingerprint'] for a in audits})!=1:raise ValueError('Old-test windows differ between seeds')
    if len({a['pose_fingerprint'] for a in audits})!=1:raise ValueError('Old-test pose cohorts differ')
    with (out/'comparison.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
    summary={};lines=['# Frozen old/new test comparison','',
        f"Historical test: {len(audits[0]['test_sequences'])} recordings, participants {', '.join(audits[0]['test_participants'])}, {audits[0]['windows']} windows.",
        'No training, checkpoint selection or normalization fitting on these test recordings.',
        'Pose: predicted hand on identical valid GT targets with both wrist references; no intention gate.',
        'Mean ± sample standard deviation over seeds 42/43/44; not a confidence interval over people.','']
    metrics=('intention_macro_f1','hand_macro_f1','position_mean_cm')
    for condition in ('original','objects_masked','new_marker_free'):
        summary[condition]={};lines += [f'## {condition}','','| Model | Intention F1 | Hand F1 | Position (cm) |','|---|---:|---:|---:|']
        for variant in ('unchanged','finetune','scratch'):
            group=[r for r in rows if r['condition']==condition and r['variant']==variant]
            values={key:dict(mean=statistics.mean(r[key] for r in group),std=statistics.stdev(r[key] for r in group)) for key in metrics}
            summary[condition][variant]=values
            lines.append('| '+variant+' | '+' | '.join(f"{v['mean']:.4f} ± {v['std']:.4f}" for v in values.values())+' |')
        lines.append('')
    lines+=['Old test was previously observed in the thesis; this is a post-adaptation retention check, not an untouched final holdout.',
            'Synthetic input masking keeps old MPS tracking and does not remove physical markers from the image.',
            'New videos use native VIO/handtracking and one participant. Cross-dataset differences cannot isolate marker removal.']
    (out/'COMPARISON.md').write_text('\n'.join(lines)+'\n')
    (out/'comparison.json').write_text(json.dumps(summary,indent=2)+'\n')
    print('\n'.join(lines))


if __name__=='__main__':main()
