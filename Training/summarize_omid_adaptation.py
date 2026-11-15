"""Summarize predeclared model variants on the same held-out recordings."""
import argparse
import csv
import json
import statistics
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root',type=Path,required=True)
    a=p.parse_args();root=a.data_root/'adaptation_runs_v1'
    rows=[]
    for seed in (42,43,44):
        folder=root/f'seed{seed}'
        if not (folder/'COMPLETE').exists():raise RuntimeError(f'Run incomplete: {folder}')
        results=json.loads((folder/'test_results.json').read_text())
        fingerprints={m['pose_fair_common']['sample_key_fingerprint'] for m in results.values()}
        if len(fingerprints)!=1:raise ValueError('Unpaired pose cohorts')
        for variant,m in results.items():
            pose=m['pose_fair_common']['methods']['learned_end_to_end']
            rows.append(dict(seed=seed,variant=variant,intention_macro_f1=m['intention']['macro_f1'],
                hand_macro_f1=m['receiving_hand']['macro_f1'],
                position_mean_cm=pose['position_mae_cm'],
                shared_pose_samples=m['pose_fair_common']['shared_samples'],
                sample_key_fingerprint=m['pose_fair_common']['sample_key_fingerprint']))
    if len({r['sample_key_fingerprint'] for r in rows})!=1:raise ValueError('Seed test cohorts differ')
    with (root/'comparison.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
    aggregate={}
    for variant in ('unchanged','finetune','scratch'):
        selected=[r for r in rows if r['variant']==variant]
        aggregate[variant]={}
        for metric in ('intention_macro_f1','hand_macro_f1','position_mean_cm'):
            values=[r[metric] for r in selected]
            if any(v is None for v in values):raise ValueError('Missing primary metric')
            aggregate[variant][metric]={'mean':statistics.mean(values),'seed_std':statistics.stdev(values)}
    (root/'comparison.json').write_text(json.dumps(aggregate,indent=2)+'\n')
    lines=['# Omid marker-free adaptation','',
           '30 training / 10 validation / 10 test recordings, one participant. Seeds 42/43/44.',
           'Checkpoint selection: validation intention macro-F1. No test-based tuning.',
           'Pose metric: predicted receiving hand on the identical valid-target cohort with both reference wrists available; excludes intention gating.',
           'Means and standard deviations below describe variation across model seeds, not unseen-person uncertainty.',
           '','| Variant | Intention macro-F1 | Hand macro-F1 | Position mean (cm) |','|---|---:|---:|---:|']
    for name,metrics in aggregate.items():
        lines.append('| '+name+' | '+' | '.join(f"{v['mean']:.4f} ± {v['seed_std']:.4f}" for v in metrics.values())+' |')
    lines+=['','The new recordings use native handtracking/VIO instead of MPS. Differences cannot be attributed solely to missing object markers.',
            'Full metrics, oracle/persistence diagnostics, sample counts and validation histories are preserved in each seed directory.']
    (root/'COMPARISON.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(aggregate,indent=2))


if __name__=='__main__':main()
