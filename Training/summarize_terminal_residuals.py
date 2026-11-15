"""Report frozen supplementary terminal checkpoint and residual diagnostics."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from terminal_residual_diagnostics import (add_vectors, assert_same_cases, grouped_summaries,
                                           paired_future, scalar_summary)


NAMES = {'best_intention':'Intention-Checkpoint','best_pose':'Pose-Checkpoint'}


def load(root):
    frames, validation = [], []
    for seed in (42,43,44):
        folder = root/f'seed{seed}'
        if not (folder/'COMPLETE').exists():
            raise ValueError('Incomplete seed '+str(seed))
        audit = json.loads((folder/'audit.json').read_text())
        if audit['training_performed'] or audit['checkpoint_selection_changed'] or not audit['same_cases']:
            raise ValueError('Frozen supplementary protocol violated')
        for filename, digest in audit['output_sha256'].items():
            if hashlib.sha256((folder/filename).read_bytes()).hexdigest() != digest:
                raise ValueError('Output checksum mismatch')
        per_seed = {}
        val = json.loads((folder/'validation_results.json').read_text())
        for name in NAMES:
            frame = pd.read_csv(folder/f'{name}_predictions.csv')
            per_seed[name] = frame
            frames.append(paired_future(frame))
            g = val[name]['pure_future']
            validation.append(dict(seed=seed,checkpoint=name,windows=g['shared_samples'],
                model_gt_cm=g['methods']['learned_oracle_hand']['position_mae_cm'],
                model_predicted_hand_cm=g['methods']['learned_end_to_end']['position_mae_cm'],
                persistence_gt_cm=g['methods']['persistence']['position_mae_cm'],
                intention_f1=val[name]['intention_f1']))
        assert_same_cases(per_seed['best_intention'],per_seed['best_pose'])
        if seed > 42:
            assert_same_cases(per_seed['best_intention'], reference)
        else:
            reference = per_seed['best_intention']
    return pd.concat(frames,ignore_index=True), pd.DataFrame(validation)


def aggregate(by_seed):
    keys = ['checkpoint','axis','bin','order']
    keep = ['windows','sequences']
    numbers = [c for c in by_seed.columns if c not in keys+keep+['seed']]
    spec = {c:(c,'first') for c in keep}
    for c in numbers:
        spec[c+'_mean'] = (c,'mean')
        spec[c+'_seed_sd'] = (c,'std')
    return by_seed.groupby(keys,sort=False).agg(**spec).reset_index()


def figures(vectors, summary, folder):
    plt.rcParams.update({'font.size':10, 'axes.spines.top':False,'axes.spines.right':False})
    fig, axs = plt.subplots(1,3,figsize=(15,4.8))
    for name,color in (('best_intention','#2563a6'),('best_pose','#d97718')):
        g = summary.loc[(summary.checkpoint==name)&(summary.axis=='displacement')].sort_values('order')
        for ax,col in zip(axs,('model_gt_cm','error_difference_cm','predicted_displacement_cm')):
            ax.plot(g.order,g[col+'_mean'],marker='o',color=color,label=NAMES[name])
        if name=='best_intention':
            axs[0].plot(g.order,g.persistence_gt_cm_mean,'--',color='#527564',label='Persistence (GT hand)')
            axs[2].plot(g.order,g.actual_displacement_cm_mean,'--',color='#527564',label='Actual displacement')
    labels=[f'{r.bin}\nn={r.windows}' for r in g.itertuples()]
    for ax in axs: ax.set_xticks(g.order,labels,rotation=25)
    axs[0].set_title('Position error · same GT hand')
    axs[1].set_title('Model error minus persistence')
    axs[1].axhline(0,color='gray',lw=1)
    axs[2].set_title('Predicted versus actual displacement')
    for ax in axs:
        ax.set_xlabel('Actual displacement group (cm)')
        ax.set_ylabel('cm')
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    fig.suptitle('Terminal residual diagnosis · pure future · same 201 windows',fontsize=13)
    fig.tight_layout()
    for ext in ('png','pdf'): fig.savefig(folder/f'01_displacement_groups.{ext}',dpi=180,bbox_inches='tight')
    plt.close(fig)
    fig,axs = plt.subplots(2,2,figsize=(12,9))
    for row,(name,color) in enumerate((('best_intention','#2563a6'),('best_pose','#d97718'))):
        g = vectors.loc[vectors.checkpoint==name].groupby('sample_key').agg(
            actual=('actual_displacement_cm','first'), predicted=('predicted_displacement_cm','mean'),
            delta=('error_difference_cm','mean'), angle=('direction_angle_deg','mean'))
        limit=max(g.actual.max(),g.predicted.max())*1.04
        axs[row,0].scatter(g.actual,g.predicted,color=color,s=18,alpha=.65)
        axs[row,0].plot([0,limit],[0,limit],'--',color='gray')
        axs[row,0].set(xlabel='Actual displacement (cm)',ylabel='Predicted displacement (cm)',title=NAMES[name]+' · seed-mean per window')
        valid=g.angle.dropna()
        axs[row,1].hist(valid,bins=np.linspace(0,180,19),color=color,alpha=.8)
        axs[row,1].set(xlabel='Direction error (degrees)',ylabel='Windows',title='Both displacement lengths ≥ 2 cm; mean of available seed angles')
    fig.text(.5,.01,'One point per window, averaged over seeds. Overlapping windows are dependent. '
             'Angle histograms omit small vectors; eligible seeds can differ per window.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.035,1,1))
    for ext in ('png','pdf'): fig.savefig(folder/f'02_lengths_and_directions.{ext}',dpi=180,bbox_inches='tight')
    plt.close(fig)


def value(r,name,unit=' cm'):
    x,s=r[name+'_mean'],r[name+'_seed_sd']
    return '–' if pd.isna(x) else f'{x:.2f} ± {s:.2f}{unit}'


def report(root, summary, validation, sequences, hand):
    overall=summary.loc[summary.axis=='overall']
    lines=['# Ergänzende Diagnose der terminalen Handpositionsvorhersage','',
        'Diese nachträgliche Analyse betrifft ausschließlich das korrigierte Endpose-Modell auf den alten Testvideos. Kein neues Training, keine neue Checkpoint-Auswahl; die ursprünglichen Thesis-Hauptergebnisse bleiben unverändert.',
        'Drei Seeds 42/43/44. Die beiden bereits auf Validation ausgewählten Checkpoints werden auf denselben 201 rein zukünftigen Testfenstern aus 20 Videos ausgewertet. Der vollständige Testsplit umfasst 23 Videos. Fenster mit teilweise beobachtetem Ziel werden für die Hauptdiagnose ausgeschlossen.',
        'Für die Residualanalyse wird immer die tatsächliche Empfangshand als Referenz verwendet. Der Abstand vom aktuellen Handgelenk zur festen terminalen Zielpose ist die tatsächliche Verschiebung. Diese geschätzte stabile Handpose ist keine unabhängige physikalische Kontaktmessung.','',
        '## Validierung und Test getrennt','',
        'Bereits gespeicherte Validierungswerte: reines Zukunftsziel, gepaarte Referenzen, vorhergesagte Hand. Die Auswahl von best_pose verwendete die breitere ursprüngliche Validation-End-to-End-Metrik einschließlich gültiger teilweise überlappender Ziele.','',
        '| Checkpoint | Validation: Modell | Validation: Persistence (GT-Hand) | Validation: Intention F1 |',
        '|---|---:|---:|---:|']
    for name in NAMES:
        g=validation.loc[validation.checkpoint==name]
        lines.append(f'| {NAMES[name]} | {g.model_predicted_hand_cm.mean():.2f} cm | {g.persistence_gt_cm.mean():.2f} cm | {g.intention_f1.mean():.4f} |')
    lines+=['','Neue Testauswertung: identische reine Zukunftsfenster, beide Referenzhände gültig. Hauptvergleich verwendet die korrekte Hand für beide Verfahren.','',
        '| Checkpoint | Modell: GT-Hand | Persistence: GT-Hand | Modell: vorhergesagte Hand | Persistence: dieselbe vorhergesagte Hand |',
        '|---|---:|---:|---:|---:|']
    for name in NAMES:
        r=overall.loc[overall.checkpoint==name].iloc[0]
        lines.append('| '+NAMES[name]+' | '+' | '.join(value(r,m) for m in ('model_gt_cm','persistence_gt_cm','model_predicted_hand_cm','persistence_predicted_hand_cm'))+' |')
    lines+=['','## Gelernte Verschiebungen','',
        '| Checkpoint | Tatsächliche Verschiebung | Gelernte Verschiebung | Anteil Fenster mit Verbesserung gegenüber Persistence | Richtungsfehler* |',
        '|---|---:|---:|---:|---:|']
    for name in NAMES:
        r=overall.loc[overall.checkpoint==name].iloc[0]
        lines.append(f'| {NAMES[name]} | {value(r,"actual_displacement_cm")} | {value(r,"predicted_displacement_cm")} | {100*r.improvement_fraction_mean:.1f}% | {value(r,"mean_angle_deg","°")} |')
    lines+=['','*Winkel werden nur ausgewertet, wenn tatsächliche und vorhergesagte Verschiebung jeweils mindestens 2 cm betragen. Dies ist eine vorab festgelegte deskriptive Grenze, keine validierte Sensorrauschgrenze. Null- oder Kleinstbewegungen erhalten keinen künstlichen Richtungswinkel.','']
    for name in NAMES:
        counts=sequences.loc[sequences.checkpoint==name].groupby('seed').angle_windows.sum()
        lines.append(NAMES[name]+': auswertbare Richtungswinkel je Seed = '+', '.join(f'{seed}: {int(n)}' for seed,n in counts.items())+' von jeweils 201 Fenstern.')
    for name in NAMES:
        r=overall.loc[overall.checkpoint==name].iloc[0]
        lines += [f'### {NAMES[name]} nach tatsächlicher Bewegungsgröße','',
            '| Tatsächliche Verschiebung | Fenster / Videos | Gelernte Verschiebung | Modell minus Persistence | Anteil verbesserter Fenster |',
            '|---|---:|---:|---:|---:|']
        g=summary.loc[(summary.checkpoint==name)&(summary.axis=='displacement')].sort_values('order')
        for z in g.itertuples():
            gain='–' if pd.isna(z.improvement_fraction_mean) else f'{100*z.improvement_fraction_mean:.1f}%'
            lines.append(f'| {z.bin} cm | {z.windows} / {z.sequences} | {z.predicted_displacement_cm_mean:.2f} cm | {z.error_difference_cm_mean:+.2f} cm | {gain} |')
        sg=sequences.loc[sequences.checkpoint==name].groupby('sequence_id').error_difference_cm.mean()
        lines += ['',f'Gleichgewichtung der Videos: Modell minus Persistence = {sg.mean():+.2f} cm. '
                  f'Bei {int((sg<0).sum())}/{len(sg)} Videos ist der mittlere GT-Hand-Fehler im Seed-Mittel niedriger.',
                  f'Der geometrische Richtungsterm trägt im Mittel {100*r.direction_squared_share_mean:.1f}% zur Summe der quadratischen Residualfehler bei. '
                  f'Bei {100*r.opposite_direction_fraction_mean:.1f}% der auswertbaren Richtungswinkel zeigt die Korrektur um mehr als 90° am tatsächlichen Zielvektor vorbei.',
                  f'Diagnostisches Gedankenexperiment: Bei perfekter, aus der Ground Truth bekannter Richtung und unverändert geschätzter Länge blieben {r.perfect_direction_same_magnitude_error_cm_mean:.2f} cm Fehler. '
                  'Dies ist eine nicht einsetzbare Oracle-Rechnung, keine zusätzliche Modellvariante.','']
    lines+=['## Grenzen der Interpretation','',
        '- Überlappende Fenster sind abhängig. Prozentangaben über Fenster sind keine unabhängigen Erfolgsversuche. Zusätzlich werden gleiche Video-Gewichte und Fallzahlen ausgewiesen.',
        '- Mittelwert ± Stichproben-SD über drei Seeds ist kein Konfidenzintervall über Personen. Die 20 Videos stammen von drei bekannten Testpersonen.',
        '- Der Richtungs- und Größenterm zerlegt den quadratischen Fehler geometrisch. Das beweist keine kausale Trainingsursache; Richtung und Länge wirken zusammen.',
        '- Unterschiede zwischen GT-Hand und vorhergesagter Hand werden separat ausgewiesen. Ein schlechterer End-to-End-Wert kann auch durch Handwahl entstehen.',
        '- Das Ziel stammt aus Aria-Handtracking und einer stabilen Endpose-Aggregation; Kalibrier- und Trackingfehler können die Diagnose beeinflussen.',
        '- Beide Checkpoints waren bereits vor dieser Testanalyse auf Validation ausgewählt. Die Tests waren zuvor bekannt. Dieses Ergebnis darf nicht zum nachträglichen Ersetzen des primären Thesis-Checkpoints verwendet werden.',
        '- Die jüngsten markerfreien Fine-Tuning-Läufe sagen t+1 voraus und gehören nicht zu dieser terminalen Diagnose.','',
        '## Artefakte','',
        '`vectors.csv`: jede tatsächliche und vorhergesagte Verschiebung sowie Fehler, Winkel und geometrische Zerlegung.',
        '`summary.csv`, `summary_by_seed.csv`: Gruppen nach Verschiebung und verbleibender Zeit.',
        '`per_sequence.csv`: Video-Mittelwerte; `hand_selection.csv`: Unterschiede durch Handwahl.',
        '`validation_comparison.csv`: unveränderte gespeicherte Validation-Werte; `seed*/`: neue Inferenzexports mit Checkpoint-Identitäten und Prüfsummen.',
        '`figures/`: PNG/PDF-Diagramme. `protocol.json`: vor Testinferenz festgelegte Auswertungsregeln.','']
    (root/'REPORT.md').write_text('\n'.join(lines))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    root=p.parse_args().root
    protocol=json.loads((root/'seed42/protocol.json').read_text())
    raw,validation=load(root)
    if not raw.groupby(['seed','checkpoint']).size().eq(201).all():
        raise ValueError('Unexpected primary cohort size')
    vectors=add_vectors(raw,protocol['angle_min_cm'])
    vectors,by_seed=grouped_summaries(vectors,protocol)
    summary=aggregate(by_seed)
    seq=[]
    for (checkpoint,seed,sequence),g in vectors.groupby(['checkpoint','seed','sequence_id']):
        seq.append(dict(checkpoint=checkpoint,seed=seed,sequence_id=sequence,participant=g.participant.iloc[0],**scalar_summary(g)))
    sequences=pd.DataFrame(seq)
    hand=vectors.groupby(['checkpoint','seed','wrong_hand']).agg(windows=('sample_key','size'),
        gt_error_cm=('model_error_cm','mean'), predicted_hand_error_cm=('learned_end_to_end_position_error_cm','mean'),
        hand_selection_difference_cm=('hand_selection_error_difference_cm','mean')).reset_index()
    for filename,frame in [('vectors.csv',vectors),('summary_by_seed.csv',by_seed),('summary.csv',summary),
                           ('per_sequence.csv',sequences),('hand_selection.csv',hand),('validation_comparison.csv',validation)]:
        frame.to_csv(root/filename,index=False)
    (root/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    folder=root/'figures';folder.mkdir(exist_ok=True)
    figures(vectors,summary,folder)
    report(root,summary,validation,sequences,hand)
    manifest={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(root.rglob('*'))
        if p.is_file() and p.name not in ('COMPLETE','output_manifest.json') and 'code' not in p.relative_to(root).parts}
    (root/'output_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    (root/'COMPLETE').touch()
    print((root/'REPORT.md').read_text())


if __name__=='__main__':main()
