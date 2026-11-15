"""Add mixed fine-tuning to the frozen three-condition comparison."""
import argparse
import csv
import json
import statistics
from pathlib import Path


CONDITIONS = {
    'original': 'Alte Videos – Original',
    'objects_masked': 'Alte Videos – Objekte maskiert',
    'new_marker_free': 'Neue markerfreie Videos',
}
VARIANTS = {'unchanged': 'Unchanged', 'finetune': 'Fine-Tune',
            'scratch': 'Scratch', 'mixed_finetune': 'Mixed Fine-Tune'}
METRICS = ('intention_macro_f1', 'hand_macro_f1', 'position_mean_cm')


def collect(root):
    rows, audits = [], []
    for seed in (42, 43, 44):
        folder = root / 'mixed_runs_v1' / f'seed{seed}'
        for required in (folder, root / 'retention_v1' / f'seed{seed}',
                         root / 'adaptation_runs_v1' / f'seed{seed}'):
            if not (required / 'COMPLETE').exists():
                raise ValueError(f'Incomplete: {required}')
        audit = json.loads((folder / 'audit.json').read_text())
        if not audit['all_prior_test_pose_cohorts_matched'] or audit['test_used_for_checkpoint_selection']:
            raise ValueError('Invalid evaluation audit')
        audits.append(audit)
        old = json.loads((root / 'retention_v1' / f'seed{seed}' / 'test_results.json').read_text())
        new = json.loads((root / 'adaptation_runs_v1' / f'seed{seed}' / 'test_results.json').read_text())
        mixed = json.loads((folder / 'test_results.json').read_text())
        for variant in VARIANTS:
            results = mixed if variant == 'mixed_finetune' else {**old[variant], 'new_marker_free': new[variant]}
            for condition, metrics in results.items():
                fair = metrics['pose_fair_common']
                rows.append(dict(seed=seed, variant=variant, condition=condition,
                    intention_macro_f1=metrics['intention']['macro_f1'],
                    hand_macro_f1=metrics['receiving_hand']['macro_f1'],
                    position_mean_cm=fair['methods']['learned_end_to_end']['position_mae_cm'],
                    shared_pose_samples=fair['shared_samples'],
                    pose_fingerprint=fair['sample_key_fingerprint']))
    for condition in CONDITIONS:
        group = [r for r in rows if r['condition'] == condition]
        if len(group) != 12 or len({(r['pose_fingerprint'], r['shared_pose_samples']) for r in group}) != 1:
            raise ValueError(f'Unpaired or incomplete comparison: {condition}')
    return rows, audits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True)
    root = parser.parse_args().data_root
    out = root / 'mixed_runs_v1'
    rows, audits = collect(root)
    with (out / 'comparison.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    summary = {}
    for condition in CONDITIONS:
        summary[condition] = {}
        for variant in VARIANTS:
            group = [r for r in rows if r['variant'] == variant and r['condition'] == condition]
            summary[condition][variant] = {
                metric: dict(mean=statistics.mean(r[metric] for r in group),
                             std=statistics.stdev(r[metric] for r in group))
                for metric in METRICS
            }
    lines = ['# Vergleich mit gemischtem Fine-Tuning', '',
        'Mittelwerte über Seeds 42, 43, 44. Position: mittlerer euklidischer Fehler in cm, vorhergesagte Empfangshand, identische gültige Zielkohorte mit beiden Handreferenzen, ohne Intention-Gating.', '',
        '| Testdatensatz | Modell | Intention F1 ↑ | Hand F1 ↑ | Position ↓ |',
        '|---|---|---:|---:|---:|']
    for condition, label in CONDITIONS.items():
        for variant, name in VARIANTS.items():
            values = summary[condition][variant]
            cells = []
            for metric in METRICS:
                value = values[metric]['mean']
                best = (min if metric == 'position_mean_cm' else max)(summary[condition][v][metric]['mean'] for v in VARIANTS)
                cell = (f'{value:.2f} cm' if metric == 'position_mean_cm' else f'{value:.4f}').replace('.', ',')
                cells.append(f'**{cell}**' if value == best else cell)
            lines.append('| ' + (label if variant == 'unchanged' else '') + ' | ' + name + ' | ' + ' | '.join(cells) + ' |')
    lines += ['', '## Seed-Streuung', '', '| Testdatensatz | Modell | Intention F1 ± SD | Hand F1 ± SD | Position cm ± SD |', '|---|---|---:|---:|---:|']
    for condition, label in CONDITIONS.items():
        for variant, name in VARIANTS.items():
            values = summary[condition][variant]
            cells = [f"{values[m]['mean']:.4f} ± {values[m]['std']:.4f}" for m in METRICS]
            lines.append('| ' + label + ' | ' + name + ' | ' + ' | '.join(cells) + ' |')
    lines += ['', 'Mixed Fine-Tune beginnt beim ursprünglichen Modell und trainiert auf den 170 alten und 30 neuen Trainingsvideos. Beide Datensätze haben je 50% Sampling-Wahrscheinlichkeit; Fenster innerhalb eines Datensatzes werden gleich gewichtet. Normalisierung bleibt eingefroren.',
        'Ein Checkpoint pro Seed, gewählt allein nach dem Mittel der beiden Validierungs-Intention-F1-Werte. Alte und neue Tests bleiben separat und verwenden dieselben Fenster/Zielkohorten wie zuvor.',
        'Explorative Folgeuntersuchung nach Sichtung früherer Testergebnisse. Die Tests sind keine erstmals verwendeten finalen Holdouts. Trainingsumfang und Validierungsauswahl unterscheiden sich vom bisherigen Fine-Tuning; eine Verbesserung wäre kein isolierter Nachweis für den Mischeffekt.',
        'Neue Videos: eine Person, natives Tracking statt altem MPS. Künstliche Objektmaskierung entfernt Eingangsmerkmale, keine Marker aus Bildpixeln. Seed-SD ist keine Unsicherheit über Personen.', '',
        'Gewählte Epochen: ' + ', '.join(f"Seed {a['seed']}: {a['selected_epoch']}" for a in audits) + '.']
    (out / 'COMPARISON.md').write_text('\n'.join(lines) + '\n')
    (out / 'comparison.json').write_text(json.dumps(summary, indent=2) + '\n')
    (out / 'COMPLETE').touch()
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
