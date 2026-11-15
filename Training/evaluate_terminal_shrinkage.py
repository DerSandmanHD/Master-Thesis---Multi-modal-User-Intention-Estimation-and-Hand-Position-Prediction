"""Freeze alpha on validation, then score only three predeclared test variants."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from terminal_residual_diagnostics import assert_same_cases, paired_future
from terminal_shrinkage import score,validation_search,choose_alpha


FAMILIES={'best_intention':'Intention-Checkpoint','best_pose':'Pose-Checkpoint (Diagnose)'}


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path,data):path.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')


def checked_outputs(folder,audit):
    for filename,digest in audit['output_sha256'].items():
        if sha(folder/filename)!=digest:raise ValueError('Artifact checksum changed: '+filename)


def select_using_validation(root,protocol):
    """This function has no argument/path for a test prediction file."""
    frames={family:{} for family in FAMILIES}; identities={}; artifacts={};audits={}
    for seed in protocol['seeds']:
        folder=root/f'seed{seed}'
        if not (folder/'COMPLETE').exists():raise ValueError('Validation export incomplete')
        audit=json.loads((folder/'audit.json').read_text())
        if audit['split']!='validation' or audit['test_predictions_loaded'] or audit['retraining']:
            raise ValueError('Validation-only export requirement violated')
        if audit['protocol_sha256']!=sha(root/'code/Training/configs/evaluation/terminal_shrinkage_v1.json'):
            raise ValueError('Protocol mismatch')
        checked_outputs(folder,audit)
        identities[str(seed)]=json.loads((folder/'checkpoint_identities.json').read_text())
        audits[seed]=audit
        for family in FAMILIES:
            path=folder/f'{family}_validation_predictions.csv'
            frame=pd.read_csv(path)
            if not frame.seed.eq(seed).all():raise ValueError('Wrong validation seed')
            frames[family][seed]=paired_future(frame)
            if len(frames[family][seed])!=256:raise ValueError('Unexpected validation cohort')
            if seed>42:assert_same_cases(frames[family][42],frames[family][seed])
            artifacts[str(path.relative_to(root))]=sha(path)
        assert_same_cases(frames['best_intention'][seed],frames['best_pose'][seed])
    results=[];selected={}
    for family in FAMILIES:
        grid=validation_search(frames[family],protocol['alpha_grid'])
        alpha,error=choose_alpha(grid)
        selected[family]=dict(alpha=alpha,validation_predicted_hand_cm=error)
        grid['checkpoint']=family;results.append(grid)
    grid=pd.concat(results,ignore_index=True)
    grid.to_csv(root/'validation_grid.csv',index=False)
    selection=dict(protocol_sha256=sha(root/'code/Training/configs/evaluation/terminal_shrinkage_v1.json'),
        criterion=protocol['selection'],selected=selected,source_checkpoints=identities,
        validation_artifact_sha256=artifacts,test_predictions_used=False,
        primary_family='best_intention')
    path=root/'selection.json'
    if path.exists():
        old=json.loads(path.read_text());old.pop('frozen_at_utc')
        if old!=selection:raise ValueError('Cannot alter an existing frozen selection')
    else:
        selection['frozen_at_utc']=datetime.now(timezone.utc).isoformat()
        with path.open('x') as f:json.dump(selection,f,indent=2,allow_nan=False);f.write('\n')
    return grid,selected,identities,audits,frames,sha(path)


def evaluate_frozen_test(repo,root,protocol,selected,identities,audits,validation_frames):
    source=repo/'Training/reports/dataset_v3_causal_20260815_n214_5d136a34'/protocol['test_source']
    rows=[];per_window=[];sources={}
    # This is the first point where any test prediction file is opened.
    for seed in protocol['seeds']:
        folder=source/f'seed{seed}'
        audit=json.loads((folder/'audit.json').read_text())
        if not (folder/'COMPLETE').exists() or not audit['same_cases']:
            raise ValueError('Invalid existing test export')
        checked_outputs(folder,audit)
        test_identities=json.loads((folder/'checkpoint_identities.json').read_text())
        all_frames={}
        for family in FAMILIES:
            if test_identities[family]['sha256']!=identities[str(seed)][family]['checkpoint_sha256']:
                raise ValueError('Validation and test checkpoints differ')
            path=folder/f'{family}_predictions.csv'
            frame=paired_future(pd.read_csv(path))
            if not frame.seed.eq(seed).all() or len(frame)!=201:raise ValueError('Wrong test seed/cohort')
            if family=='best_pose':assert_same_cases(frame,all_frames['best_intention'])
            if seed>42:assert_same_cases(frame,reference[family])
            if set(frame.sample_key)&set(validation_frames[family][seed].sample_key):
                raise ValueError('Validation/test sample leakage')
            if set(audits[seed]['validation_sequences'])&set(frame.sequence_id):
                raise ValueError('Validation/test recording leakage')
            all_frames[family]=frame
            sources[f'seed{seed}/{family}']=dict(path=str(path),sha256=sha(path))
            for variant,alpha in [('persistence',0.),('selected',selected[family]['alpha']),('original',1.)]:
                scores=score(frame,alpha)
                if variant in ('persistence','original'):
                    gtcol='persistence_position_error_cm' if alpha==0 else 'learned_oracle_hand_position_error_cm'
                    predcol='persistence_predicted_hand_position_error_cm' if alpha==0 else 'learned_end_to_end_position_error_cm'
                    if not np.allclose(scores.gt_hand_error_cm,frame[gtcol],atol=1e-4):raise ValueError('GT endpoint reproduction failed')
                    if not np.allclose(scores.predicted_hand_error_cm,frame[predcol],atol=1e-4):raise ValueError('Predicted-hand endpoint reproduction failed')
                seq=scores.groupby('sequence_id')[['predicted_hand_error_cm','gt_hand_error_cm']].mean()
                rows.append(dict(checkpoint=family,seed=seed,variant=variant,alpha=alpha,
                    windows=len(frame),videos=frame.sequence_id.nunique(),
                    predicted_hand_cm=scores.predicted_hand_error_cm.mean(),gt_hand_cm=scores.gt_hand_error_cm.mean(),
                    sequence_predicted_hand_cm=seq.predicted_hand_error_cm.mean(),sequence_gt_hand_cm=seq.gt_hand_error_cm.mean()))
                scores['checkpoint']=family;scores['seed']=seed;scores['variant']=variant;scores['alpha']=alpha
                per_window.append(scores)
        if seed==42:reference=all_frames
    return pd.DataFrame(rows),pd.concat(per_window,ignore_index=True),sources


def summarize(by_seed):
    return by_seed.groupby(['checkpoint','variant'],sort=False).agg(alpha=('alpha','first'),windows=('windows','first'),
        videos=('videos','first'),predicted_hand_cm=('predicted_hand_cm','mean'),predicted_hand_sd=('predicted_hand_cm','std'),
        gt_hand_cm=('gt_hand_cm','mean'),gt_hand_sd=('gt_hand_cm','std'),
        sequence_predicted_hand_cm=('sequence_predicted_hand_cm','mean'),sequence_gt_hand_cm=('sequence_gt_hand_cm','mean')).reset_index()


def plot(grid,summary,selected,folder):
    fig,axs=plt.subplots(1,3,figsize=(15,4.8))
    for family,color in [('best_intention','#2563a6'),('best_pose','#d97718')]:
        g=grid.loc[grid.checkpoint.eq(family)].groupby('alpha').predicted_hand_cm.agg(['mean','std'])
        axs[0].plot(g.index,g['mean'],color=color,label=FAMILIES[family])
        axs[0].fill_between(g.index,g['mean']-g['std'],g['mean']+g['std'],color=color,alpha=.12)
        alpha=selected[family]['alpha'];axs[0].scatter([alpha],[g.loc[alpha,'mean']],color=color,s=45)
    axs[0].set(xlabel='Alpha · retained residual fraction',ylabel='Validation error (cm)',title='Validation-only alpha selection')
    axs[0].legend(fontsize=8)
    for ax,family in zip(axs[1:],FAMILIES):
        g=summary.loc[summary.checkpoint.eq(family)].set_index('variant').loc[['persistence','selected','original']]
        x=np.arange(3)
        ax.bar(x-.17,g.predicted_hand_cm,.34,yerr=g.predicted_hand_sd,label='Predicted hand',color='#2563a6',capsize=3)
        ax.bar(x+.17,g.gt_hand_cm,.34,yerr=g.gt_hand_sd,label='GT hand (diagnostic)',color='#d97718',capsize=3)
        ax.set_xticks(x,[f'Persistence\nα=0',f'Validation selection\nα={selected[family]["alpha"]:g}',f'Original\nα=1'])
        ax.set(title=FAMILIES[family]+' · fixed test',ylabel='Test error (cm)')
        ax.legend(fontsize=8)
    for ax in axs:ax.grid(axis='y',alpha=.2);ax.set_ylim(bottom=0)
    fig.suptitle('Frozen terminal residual attenuation · 3 seeds · same 201 test windows',fontsize=13)
    fig.text(.5,.005,'Bands/error bars: seed SD, not participant confidence intervals. Test alpha was fixed before test exports were opened.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.04,1,.94))
    for ext in ('png','pdf'):fig.savefig(folder/f'01_validation_and_fixed_test.{ext}',dpi=180,bbox_inches='tight')
    plt.close(fig)


def report(root,grid,summary,selected):
    lines=['# Abschwächung der terminalen Residualkorrektur','',
        'Ergänzender nachträglicher Versuch mit vorhandenen Endpose-Checkpoints. Kein neues Training und keine Änderung der ursprünglichen Thesis-Modelle. Beide zuvor festgelegten Checkpoint-Familien werden getrennt behandelt; die primäre Familie bleibt best_intention.',
        'Positionsvorhersage = letzte Handreferenz + α × (Modellvorhersage − letzte Handreferenz). α=0 ist Persistence; α=1 ist das bisherige Modell. Empfangshand, Intention und Orientierung bleiben unverändert.',
        'Ein gemeinsamer Faktor pro Checkpoint-Familie für Seeds 42/43/44. Auswahl auf 256 identischen reinen Zukunfts-Validierungsfenstern: niedrigster mittlerer Positionsfehler mit vorhergesagter Hand. Festes Raster 0 bis 1 in Schritten von 0,05, bei Gleichstand kleineres α. Auswahl mit Prüfsummen vor dem Lesen der Testexports gespeichert.','',
        '## Auswahl ausschließlich auf Validierung','',
        '| Checkpoint | Gewähltes α | Validation: α=0 | Validation: gewähltes α | Validation: α=1 |',
        '|---|---:|---:|---:|---:|']
    for family in FAMILIES:
        g=grid.loc[grid.checkpoint.eq(family)].groupby('alpha').predicted_hand_cm.mean()
        alpha=selected[family]['alpha']
        lines.append(f'| {FAMILIES[family]} | {alpha:g} | {g.loc[0]:.2f} cm | {g.loc[alpha]:.2f} cm | {g.loc[1]:.2f} cm |')
    lines+=['','## Test mit unverändertem Faktor','',
        '201 identische reine Zukunftsfenster aus 20 alten Testvideos; gültige Referenzen beider Hände. Jede Methode nutzt dieselbe vorhergesagte Hand innerhalb ihrer Checkpoint-Familie. Mittelwert ± Stichproben-SD über drei Seeds.','',
        '| Checkpoint | Variante | α | Vorhergesagte Hand ↓ | GT-Hand, diagnostisch ↓ |',
        '|---|---|---:|---:|---:|']
    labels={'persistence':'Persistence','selected':'Auf Validation gewählter Faktor','original':'Bisheriges Modell'}
    for family in FAMILIES:
        for variant in ('persistence','selected','original'):
            r=summary.loc[summary.checkpoint.eq(family)&summary.variant.eq(variant)].iloc[0]
            lines.append(f'| {FAMILIES[family]} | {labels[variant]} | {r.alpha:g} | {r.predicted_hand_cm:.2f} ± {r.predicted_hand_sd:.2f} cm | {r.gt_hand_cm:.2f} ± {r.gt_hand_sd:.2f} cm |')
    lines.append('')
    for family in FAMILIES:
        r=summary.loc[summary.checkpoint.eq(family)&summary.variant.eq('selected')].iloc[0]
        b=summary.loc[summary.checkpoint.eq(family)&summary.variant.eq('persistence')].iloc[0]
        o=summary.loc[summary.checkpoint.eq(family)&summary.variant.eq('original')].iloc[0]
        lines+=['',f'{FAMILIES[family]}: gewählter Faktor minus Persistence = {r.predicted_hand_cm-b.predicted_hand_cm:+.2f} cm; '
                f'gewählter Faktor minus ursprüngliches Modell = {r.predicted_hand_cm-o.predicted_hand_cm:+.2f} cm. '
                f'Bei gleichem Gewicht pro Video: gewählter Faktor {r.sequence_predicted_hand_cm:.2f} cm, Persistence {b.sequence_predicted_hand_cm:.2f} cm.','']
    lines+=['## Einordnung','',
        'Die Auswahl verbessert oder erhält den gewählten Validation-Fehler definitionsgemäß. Erst der anschließend eingefrorene Testvergleich beschreibt, ob sich diese Auswahl auf die alten Testpersonen überträgt. Testwerte wurden nicht zur Änderung von α oder zur Auswahl einer Checkpoint-Familie verwendet.',
        'Falls α=0 gewählt wird, verwendet diese Variante ausschließlich Persistence für die Position. Falls α>0 gewählt wird, bleibt ein entsprechender Anteil der neuronalen Korrektur erhalten. Keine Garantie eines Vorteils gegenüber α=0.',
        'GT-Hand und vorhergesagte Hand sind unterschiedliche Informationsbedingungen. Die GT-Hand-Persistence liegt bei 18,25 cm; mit vorhergesagter Hand hängt die Baseline auch von der Handentscheidung des jeweiligen Checkpoints ab.',
        'Wahre Zielposition, Restabstand und verbleibende Zeit werden ausschließlich zur Evaluation verwendet. Die Abschwächung selbst benötigt nur aktuelle Handreferenz, Modellvorhersage und den global eingefrorenen Faktor.',
        'Überlappende Fenster sind abhängig; Seed-SD ist kein Konfidenzintervall über Personen. Unterschiedliche Videolängen werden zusätzlich durch gleich pro Video gewichtete Mittelwerte berücksichtigt.',
        'Der Versuch wurde durch bereits bekannte Testergebnisse motiviert und ist eine explorative Folgeuntersuchung, keine unangetastete finale Holdout-Evaluation. Es wird kein verbessertes Modell nachträglich als ursprüngliches Thesis-Hauptergebnis eingesetzt.',
        'Nur Positionskorrektur; keine Aussage über geänderte Orientierung, Intention oder Empfangshand. Markerfreie t+1-Fine-Tuning-Modelle gehören nicht zu diesem Endpose-Experiment.','',
        '## Dateien','',
        '- `selection.json`: vor Testauswertung gespeicherter Faktor mit Validation-/Checkpoint-Prüfsummen.',
        '- `validation_grid.csv`: alle vorab festgelegten Faktoren auf Validation; kein vollständiges Test-Raster.',
        '- `test_summary.csv`, `test_by_seed.csv`: nur α=0, α=1 und gewählter Faktor auf Test.',
        '- `test_predictions.csv`: resultierende Positionen und Fehler pro Fenster.',
        '- `seed*/`: ausschließlich neue Validation-Inferenz und deren Reproduktionsaudit.',
        '- `figures/01_validation_and_fixed_test.{png,pdf}`: Auswahlkurve und eingefrorener Testvergleich.','']
    (root/'REPORT.md').write_text('\n'.join(lines))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo',type=Path,required=True);p.add_argument('--root',type=Path,required=True)
    a=p.parse_args();root=a.root
    protocol=json.loads((root/'code/Training/configs/evaluation/terminal_shrinkage_v1.json').read_text())
    grid,selected,identities,audits,frames,selection_hash=select_using_validation(root,protocol)
    test_read_started=datetime.now(timezone.utc).isoformat()
    by_seed,predictions,sources=evaluate_frozen_test(a.repo,root,protocol,selected,identities,audits,frames)
    if sha(root/'selection.json')!=selection_hash:raise ValueError('Selection changed during test evaluation')
    summary=summarize(by_seed)
    by_seed.to_csv(root/'test_by_seed.csv',index=False);summary.to_csv(root/'test_summary.csv',index=False)
    predictions.to_csv(root/'test_predictions.csv',index=False)
    write(root/'protocol.json',protocol)
    write(root/'audit.json',dict(selection_sha256=selection_hash,test_read_started_utc=test_read_started,
        validation_only_selection=True,test_reselection=False,retraining=False,source_test_exports=sources))
    folder=root/'figures';folder.mkdir(exist_ok=True)
    plot(grid,summary,selected,folder);report(root,grid,summary,selected)
    outputs={str(p.relative_to(root)):sha(p) for p in sorted(root.rglob('*')) if p.is_file()
             and p.name not in ('COMPLETE','output_manifest.json') and 'code' not in p.relative_to(root).parts}
    write(root/'output_manifest.json',outputs);(root/'COMPLETE').touch()
    print((root/'REPORT.md').read_text())


if __name__=='__main__':main()
