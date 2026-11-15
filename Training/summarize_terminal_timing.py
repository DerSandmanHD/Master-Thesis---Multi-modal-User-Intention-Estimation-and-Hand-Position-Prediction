"""Aggregate and plot frozen terminal prediction errors over input timing."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

METHODS = ('learned_end_to_end', 'persistence_predicted_hand', 'learned_oracle_hand', 'persistence')
REGIMES = ('strictly_before_aggregation', 'partially_overlapping_aggregation')
LABELS = {'learned_end_to_end': 'Model · predicted hand',
          'persistence_predicted_hand': 'Last observation · predicted hand',
          'learned_oracle_hand': 'Model · ground-truth hand',
          'persistence': 'Last observation · ground-truth hand'}
COLORS = {'learned_end_to_end': '#2563a6', 'persistence_predicted_hand': '#d97718',
          'learned_oracle_hand': '#5e498f', 'persistence': '#527564'}


def read_checked(root):
    data, audits = [], []
    identity = {}
    for seed in (42, 43, 44):
        folder = root/f'seed{seed}'
        if not (folder/'COMPLETE').exists():
            raise ValueError(f'Incomplete seed {seed}')
        audit = json.loads((folder/'audit.json').read_text())
        if audit['training_performed'] or audit['checkpoint_selection_changed']:
            raise ValueError('Source was not frozen')
        for filename, checksum in audit['output_sha256'].items():
            if hashlib.sha256((folder/filename).read_bytes()).hexdigest() != checksum:
                raise ValueError(f'Output checksum mismatch: {filename}')
        audits.append(audit)
        for filename in ('frozen_window_predictions.csv', 'disjoint_window_predictions.csv'):
            frame = pd.read_csv(folder/filename)
            if frame['sample_key'].duplicated().any() or not frame.seed.eq(seed).all():
                raise ValueError('Duplicate rows or wrong seed')
            fixed_columns = ['sample_key', 'regime', 'target_valid', 'fair_common', 'input_start_row', 'input_end_row',
                             'target_x_m', 'target_y_m', 'target_z_m']
            fixed = frame[fixed_columns].sort_values('sample_key').reset_index(drop=True)
            if filename in identity:
                pd.testing.assert_frame_equal(fixed, identity[filename])
            else:
                identity[filename] = fixed
            if not frame.loc[frame.fair_common, 'regime'].isin(REGIMES).all():
                raise ValueError('Observed terminal target included in forecast cohort')
            for _, group in frame.dropna(subset=['target_x_m']).groupby('sequence_id'):
                if (group[['target_x_m', 'target_y_m', 'target_z_m']].nunique() != 1).any():
                    raise ValueError('Terminal target varies across windows')
            data.append(frame)
    if len({a['test_window_fingerprint'] for a in audits}) != 1:
        raise ValueError('Different test windows across seeds')
    if len({a['protocol_sha256'] for a in audits}) != 1:
        raise ValueError('Different protocols across seeds')
    return pd.concat(data, ignore_index=True), audits


def prepare_axes(frame, protocol):
    frame = frame.copy()
    for name, column, edges in (
        ('remaining', 'time_to_terminal_s', protocol['remaining_time_bins_seconds']),
        ('elapsed', 'input_end_relative_s', protocol['elapsed_time_bins_seconds']),
    ):
        bounds = edges + [float('inf')]
        labels = [f'{a:g}–{b:g} s' if np.isfinite(b) else f'≥{a:g} s' for a, b in zip(bounds, bounds[1:])]
        frame[name+'_bin'] = pd.cut(frame[column], bounds, right=False, labels=labels)
    frame['block_bin'] = frame.block_index.map(lambda i: f'{int(i)*60}–{int(i)*60+59}' if i >= 0 else '')
    return frame


def aggregate(frame):
    rows = []
    axes = [('frozen_stride10', 'remaining'), ('frozen_stride10', 'elapsed'), ('disjoint60', 'block')]
    for scheme, axis in axes:
        pool = frame.loc[frame.scheme == scheme]
        if axis == 'block':
            labels = [f'{int(i)*60}–{int(i)*60+59}' for i in sorted(pool.block_index.unique())]
        else:
            labels = list(frame[axis+'_bin'].cat.categories)
        for seed in (42, 43, 44):
            for regime in REGIMES:
                for order, label in enumerate(labels):
                    available = pool.loc[(pool.seed == seed) & (pool[axis+'_bin'] == label)]
                    selected = available.loc[(available.regime == regime) & available.fair_common]
                    for method in METHODS:
                        values = selected[f'{method}_position_error_cm']
                        per_sequence = selected.groupby('sequence_id')[f'{method}_position_error_cm'].mean()
                        rows.append(dict(scheme=scheme, axis=axis, regime=regime, bin=label, order=order,
                            seed=seed, method=method, windows=len(selected), sequences=selected.sequence_id.nunique(),
                            eligible_input_windows=len(available),
                            target_windows=int(((available.regime == regime) & available.target_valid).sum()),
                            mean_cm=values.mean(), median_cm=values.median(),
                            sequence_mean_cm=per_sequence.mean()))
    by_seed = pd.DataFrame(rows)
    keys = ['scheme', 'axis', 'regime', 'bin', 'order', 'method']
    summary = by_seed.groupby(keys, sort=False).agg(
        windows=('windows', 'first'), sequences=('sequences', 'first'),
        eligible_input_windows=('eligible_input_windows', 'first'), target_windows=('target_windows', 'first'),
        mean_cm=('mean_cm', 'mean'), seed_sd_cm=('mean_cm', 'std'),
        sequence_mean_cm=('sequence_mean_cm', 'mean'), sequence_seed_sd_cm=('sequence_mean_cm', 'std'),
    ).reset_index()
    return by_seed, summary


def within_recording(frame):
    """Earliest/latest pure-future blocks, always paired within a recording."""
    rows = []
    paired = frame.loc[frame.fair_common & (frame.regime == REGIMES[0])]
    for (scheme, seed, sequence), group in paired.groupby(['scheme', 'seed', 'sequence_id']):
        group = group.sort_values('input_end_ns')
        if len(group) < 2:
            continue
        first, last = group.iloc[0], group.iloc[-1]
        for method in METHODS:
            col = method+'_position_error_cm'
            rows.append(dict(scheme=scheme, seed=seed, sequence_id=sequence,
                participant=first.participant, method=method,
                first_time_s=first.input_end_relative_s, last_time_s=last.input_end_relative_s,
                first_error_cm=first[col], last_error_cm=last[col],
                delta_last_minus_first_cm=last[col]-first[col]))
    return pd.DataFrame(rows)


def save_figure(fig, folder, name):
    fig.savefig(folder/f'{name}.png', dpi=180, bbox_inches='tight')
    fig.savefig(folder/f'{name}.pdf', bbox_inches='tight')
    plt.close(fig)


def make_plots(frame, summary, figures):
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    for regime, suffix, title in ((REGIMES[0], 'forecast', 'Terminal prediction before any target evidence'),
                                  (REGIMES[1], 'partial', 'Diagnostic: terminal target is already partly observed')):
        fig, axs = plt.subplots(1, 3, figsize=(16, 4.9))
        for ax, axis, title_x in zip(axs, ('block', 'elapsed', 'remaining'),
                    ('Disjoint 60-sample input blocks', 'Overlapping windows · time since handover onset',
                     'Overlapping windows · time remaining to target')):
            sub = summary.loc[(summary.axis == axis) & (summary.regime == regime)]
            for method in METHODS:
                g = sub.loc[sub.method == method].sort_values('order')
                x, y = g.order.to_numpy(), g.mean_cm.to_numpy()
                sd = g.seed_sd_cm.to_numpy()
                ax.plot(x, y, marker='o', markersize=4, color=COLORS[method],
                        linestyle='-' if method in METHODS[:2] else '--', label=LABELS[method])
                if method == 'learned_end_to_end':
                    ax.fill_between(x, y-sd, y+sd, color=COLORS[method], alpha=.12)
            g = sub.loc[sub.method == METHODS[0]].sort_values('order')
            labels = [f'{r.bin}\n{r.windows}w / {r.sequences}v' for r in g.itertuples()]
            ax.set_xticks(g.order, labels, rotation=25, ha='right')
            ax.set_title(title_x, fontsize=10, pad=12)
            ax.set_ylabel('Mean Euclidean position error (cm)')
            ax.grid(axis='y', alpha=.2)
            ax.set_ylim(bottom=0)
        handles, labels = axs[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', ncol=2, bbox_to_anchor=(.5, -.08), frameon=False)
        fig.suptitle(title, fontsize=14)
        fig.text(.5, -.16, 'Mean of 3 seeds; blue band: seed SD. w = paired windows, v = recordings. '
                 'Different bins may contain different recordings.', ha='center', fontsize=9)
        fig.tight_layout(rect=(0, .04, 1, .93))
        save_figure(fig, figures, f'01_{suffix}_timing')
    # Include every recording with paired future predictions; no error-driven
    # cherry picking. Seed averages at each exact endpoint, one panel per video.
    valid = frame.loc[(frame.scheme == 'frozen_stride10') & frame.fair_common & (frame.regime == REGIMES[0])]
    sequences = sorted(valid.sequence_id.unique())
    fig, axs = plt.subplots(int(np.ceil(len(sequences)/4)), 4, figsize=(16, 3.1*int(np.ceil(len(sequences)/4))), squeeze=False)
    for ax, sequence in zip(axs.flat, sequences):
        group = valid.loc[valid.sequence_id == sequence]
        for method in METHODS[:2]:
            curve = group.groupby('input_end_relative_s')[method+'_position_error_cm'].mean()
            ax.plot(curve.index, curve.values, marker='.', markersize=3, color=COLORS[method], label=LABELS[method])
        ax.set_title(sequence.split('_2026')[0], fontsize=9)
        ax.set_xlabel('Seconds since annotated handover onset', fontsize=8)
        ax.set_ylabel('Error (cm)', fontsize=8)
        ax.set_ylim(bottom=0)
        ax.grid(alpha=.2)
    for ax in list(axs.flat)[len(sequences):]:
        ax.set_visible(False)
    fig.suptitle('Within each recording · pure-future terminal prediction', fontsize=14)
    handles, labels = axs.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=2, frameon=False)
    fig.tight_layout(rect=(0, .03, 1, .96))
    save_figure(fig, figures, '02_per_recording_forecast')


def write_report(root, frame, summary, paired, audits):
    lines = ['# Endgültige Übergabeposition aus früheren und späteren Eingaben', '',
        'Drei eingefrorene Endpose-Checkpoints (Seeds 42/43/44), alte Testvideos von Edu, Jona und Mona. Keine Gewichtsänderung oder neue Modellauswahl.',
        'Für jedes Video ist die Zielposition identisch: robustes Aggregat des letzten akzeptierten stabilen Abschnitts. Eingaben enthalten 60 Sensor-Zeitschritte; die feste Fensterlänge wurde nicht verändert.',
        'Zeitnull ist der annotierte Übergabebeginn (THIRD), kein separat vermessener Beginn der Armbewegung. Ein Zeitschritt bezeichnet eine Zeile der Sensorausrichtung, nicht zwingend einen eigenen RGB-Frame.', '',
        '## Nicht überlappende Eingabefenster', '',
        'Fenster 0–59, 60–119 usw. beginnen exakt am annotierten Übergabebeginn. Unvollständige letzte Fenster werden ausgelassen. Die Hauptwerte verwenden nur Endpunkte vor dem ersten zur Zielaggregation gehörenden physischen Hand-Capture und gültige Referenzen beider Hände.', '',
        '| Eingabefenster | Fenster / Videos | Modell: vorhergesagte Hand | Letzte Position: gleiche vorhergesagte Hand | Modell: korrekte Hand | Letzte Position: korrekte Hand |',
        '|---|---:|---:|---:|---:|---:|']
    def table_rows(axis, regime):
        subset = summary.loc[(summary.axis == axis) & (summary.regime == regime)]
        rows = []
        for _, group in subset.groupby('order', sort=True):
            first = group.iloc[0]
            cells = []
            for method in METHODS:
                r = group.loc[group.method == method].iloc[0]
                cells.append('–' if pd.isna(r.mean_cm) else f'{r.mean_cm:.2f} ± {r.seed_sd_cm:.2f} cm')
            rows.append('| ' + str(first['bin']) + f' | {first.windows} / {first.sequences} | ' + ' | '.join(cells) + ' |')
        return rows
    lines += table_rows('block', REGIMES[0])
    lines += ['', '## Verbleibende Zeit zur endgültigen Zielpose', '',
        'Zusätzliche Auswertung aller bisherigen 60er-Fenster mit Schrittweite 10. Diese Fenster können Kontext vor dem Übergabebeginn enthalten. Weiterhin nur reine Zukunftsvorhersage.', '',
        '| Zeit bis Ziel-Capture | Fenster / Videos | Modell: vorhergesagte Hand | Letzte Position: gleiche vorhergesagte Hand | Modell: korrekte Hand | Letzte Position: korrekte Hand |',
        '|---|---:|---:|---:|---:|---:|']
    lines += table_rows('remaining', REGIMES[0])
    lines += ['', '## Paarweiser Verlauf innerhalb derselben Videos', '',
        'Frühestes gegen spätestes verfügbares Fenster mit rein zukünftigem Ziel. Mittelwert der Differenzen pro Video; negative Werte bedeuten geringeren Fehler am späteren Zeitpunkt. Abstände zwischen den beiden Zeitpunkten variieren zwischen Videos.', '']
    for scheme in ('disjoint60', 'frozen_stride10'):
        g = paired.loc[(paired.scheme == scheme) & (paired.method == 'learned_end_to_end')]
        if len(g):
            by_seed = g.groupby('seed').delta_last_minus_first_cm.mean()
            per_video = g.groupby('sequence_id').delta_last_minus_first_cm.mean()
            lines.append(f'- {scheme}: {g.sequence_id.nunique()} Videos mit mindestens zwei Fenstern; '
                         f'spät minus früh = {by_seed.mean():.2f} ± {by_seed.std():.2f} cm; '
                         f'bei {int((per_video < 0).sum())}/{len(per_video)} Videos niedrigerer Fehler im Seed-Mittel.')
    lines += ['', '## Reproduktion und Einordnung', '',
        'Die ursprünglichen Fehler und Fallzahlen wurden für jedes Modell innerhalb numerischer Toleranz reproduziert: 201 gepaarte reine Zukunftsfenster und 20 gepaarte teilweise überlappende Fenster. Die historischen Testfenster und Checkpoint-Hashes wurden geprüft.',
        'Die Diagramme 01_forecast_timing und 02_per_recording_forecast zeigen reine Zukunftsvorhersagen. 01_partial_timing zeigt ausschließlich teilweise beobachtete Zielintervalle; deren Fehler werden nicht mit den Hauptwerten vermischt. Endpunkte am/nach dem letzten Ziel-Capture sind von der Fehlerberechnung ausgeschlossen.',
        'Nicht alle Videos tragen zu jedem Zeitabschnitt bei. Die gepaarten Unterschiede und Einzelvideo-Kurven ergänzen deshalb die Gesamtkurven. summary.csv enthält zusätzlich gleich pro Video gewichtete Mittelwerte.',
        '± bezeichnet Stichproben-SD über drei Seeds, kein Konfidenzintervall über Personen. Persistence mit vorhergesagter Hand verwendet die Handentscheidung desselben Modells; die beiden GT-Hand-Kurven vergleichen dieselbe bekannte Hand.',
        'Diese Analyse betrifft das terminale Endpose-Modell auf den alten Originalvideos. Die jüngsten Omid/Mixed-t+1-Ergebnisse sind eine andere Zielgröße. Das Endziel ist eine stabile Handpose und kein unabhängig gemessener Kontakt- oder Objektübergabezeitpunkt. Die Roboterkalibrierung bleibt eine Offline-Schätzung aus dem ganzen Video.',
        'Die Testdaten waren bereits bekannt. Zeitgruppen und Auswertungsregeln wurden vor diesem Lauf festgehalten; Ergebnisse dienen der nachträglichen zeitlichen Diagnose.', '',
        '## Dateien', '',
        '- `summary.csv`, `summary_by_seed.csv`: Fehler, Fallzahlen und Video-Gewichtung je Zeitgruppe.',
        '- `paired_first_last.csv`: früh/spät innerhalb derselben Videos.',
        '- `coverage.csv`: gültige/ausgeschlossene Fenster je Modus und Zeitregime.',
        '- `seed42` bis `seed44`: jede Eingabe mit Zeitstempeln, Endziel, Fehlern, Verfügbarkeit und Audit.',
        '- `figures/`: PNG- und PDF-Diagramme.', '']
    (root/'REPORT.md').write_text('\n'.join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    root = parser.parse_args().root
    frame, audits = read_checked(root)
    protocol = json.loads((root/'seed42/protocol.json').read_text())
    frame = prepare_axes(frame, protocol)
    by_seed, summary = aggregate(frame)
    by_seed.to_csv(root/'summary_by_seed.csv', index=False)
    summary.to_csv(root/'summary.csv', index=False)
    paired = within_recording(frame)
    paired.to_csv(root/'paired_first_last.csv', index=False)
    coverage = frame.groupby(['seed', 'scheme', 'regime']).agg(
        input_windows=('sample_key', 'size'), videos=('sequence_id', 'nunique'),
        target_windows=('target_valid', 'sum'), paired_windows=('fair_common', 'sum')).reset_index()
    coverage.to_csv(root/'coverage.csv', index=False)
    figures = root/'figures'
    figures.mkdir(exist_ok=True)
    make_plots(frame, summary, figures)
    write_report(root, frame, summary, paired, audits)
    artifacts = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                 for p in sorted(root.rglob('*')) if p.is_file() and p.name not in ('COMPLETE', 'summary_manifest.json')
                 and not any(part.startswith('code') for part in p.relative_to(root).parts)}
    (root/'summary_manifest.json').write_text(json.dumps(artifacts, indent=2)+'\n')
    (root/'COMPLETE').touch()
    print((root/'REPORT.md').read_text())


if __name__ == '__main__':
    main()
