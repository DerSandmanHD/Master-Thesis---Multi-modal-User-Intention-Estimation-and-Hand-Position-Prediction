# Neue VRS- und MPS-Aufnahmen verarbeiten

Diese Anleitung beschreibt den normalen Ablauf, nachdem neue Aria-Gen2-VRS-Dateien
und die zugehörigen MPS-Ergebnisse für Handtracking und SLAM vorliegen.

Die Kurzfassung ist:

```text
Dateinamen und MPS-Struktur prüfen
    |
    +--> aria_vrs_to_mp4.sbatch
    +--> aria_audio_all.sbatch
    `--> aria_detect_tags_all.sbatch
              |
              v
      manuelle Timestamp- und Semantik-Review
              |
              v
      apply_manual_reviews.py + dataset_qa.py
              |
              v
      aria_build_master_dataset.sbatch
```

Die ersten drei Jobs sind voneinander unabhängig. Sie können parallel laufen.
`aria_build_master_dataset.sbatch` darf dagegen erst nach Review, Annotation und
abschließender QA gestartet werden.

## 1. Vor dem ersten Job: Namen vereinheitlichen

Der vollständige `sequence_id` muss für VRS-Datei, MPS-Ordner und alle späteren
Artefakte identisch sein. Das erwartete Schema ist beispielsweise:

```text
Data_collection/Data_vrs/David_1_20260715_120000.vrs
Data_collection/Data_vrs/mps_David_1_20260715_120000_vrs/
├── hand_tracking/hand_tracking_results.csv
└── slam/closed_loop_trajectory.csv
```

Groß-/Kleinschreibung ist hier wichtig. `David` und `david` werden von der
aktuellen Dataset- und Trainingspipeline als zwei verschiedene Teilnehmer
behandelt. Das könnte den personbasierten Train-/Validation-/Test-Split
verfälschen. Deshalb alle David-Aufnahmen auf `David_...` normalisieren, bevor
MP4, Audio, Marker-CSVs oder Master-Datasets erzeugt werden.

Falls die kleingeschriebenen Dateien bereits auf dem Linux-Cluster liegen, kann
folgender Code verwendet werden. Er benennt sowohl die VRS-Datei als auch den
passenden MPS-Ordner um und bricht bei Namenskollisionen ab:

```bash
VRS_DIR="Data_collection/Data_vrs"
shopt -s nullglob

for old_vrs in "$VRS_DIR"/david_*.vrs; do
  old_id="$(basename "$old_vrs" .vrs)"
  new_id="David_${old_id#david_}"
  old_mps="$VRS_DIR/mps_${old_id}_vrs"
  new_mps="$VRS_DIR/mps_${new_id}_vrs"

  if [[ -e "$VRS_DIR/$new_id.vrs" || -e "$new_mps" ]]; then
    echo "ERROR: Ziel existiert bereits: $new_id" >&2
    exit 1
  fi

  mv -- "$old_vrs" "$VRS_DIR/$new_id.vrs"
  if [[ -d "$old_mps" ]]; then
    mv -- "$old_mps" "$new_mps"
  fi
done
```

Nicht erst nach der Verarbeitung umbenennen. Andernfalls müssten auch MP4,
WAV, Marker-CSV, Annotationen und eventuell Master-Datasets konsistent
umbenannt werden.

## 2. Upload und MPS-Vollständigkeit prüfen

Alle VRS-Dateien müssen direkt in `Data_collection/Data_vrs/` liegen. Die
MPS-Ordner liegen ebenfalls direkt darin.

Anzahl der VRS-Dateien und MPS-Ordner prüfen:

```bash
find Data_collection/Data_vrs -maxdepth 1 -type f -name '*.vrs' | wc -l
find Data_collection/Data_vrs -maxdepth 1 -type d -name 'mps_*_vrs' | wc -l
```

Für jede VRS-Datei Handtracking und Closed-Loop-SLAM prüfen:

```bash
missing=0

while IFS= read -r vrs; do
  sequence_id="$(basename "$vrs" .vrs)"
  mps_dir="Data_collection/Data_vrs/mps_${sequence_id}_vrs"

  for required in \
    "$mps_dir/hand_tracking/hand_tracking_results.csv" \
    "$mps_dir/slam/closed_loop_trajectory.csv"
  do
    if [[ ! -s "$required" ]]; then
      echo "MISSING: $required"
      missing=$((missing + 1))
    fi
  done
done < <(find Data_collection/Data_vrs -maxdepth 1 -type f -name '*.vrs' | sort)

echo "Fehlende MPS-Dateien: $missing"
```

Für die aktuell neu aufgenommenen Personen kann zusätzlich kontrolliert werden:

```bash
find Data_collection/Data_vrs -maxdepth 1 -type f -name '*.vrs' \
  | grep -Ei '/(David|Florian|Angelina|Yuzhi)_' \
  | sort
```

Hier sollten für den aktuellen Upload insgesamt 39 neue VRS-Dateien erscheinen.

## 3. Automatische Cluster-Jobs

Aus dem Repository-Stammverzeichnis ausführen.

### Empfohlene Reihenfolge, wenn die Jobs einzeln gestartet werden

```bash
sbatch singularity/aria_vrs_to_mp4.sbatch
sbatch singularity/aria_audio_all.sbatch
sbatch singularity/aria_detect_tags_all.sbatch
```

Dabei gilt:

1. `aria_vrs_to_mp4.sbatch` erzeugt fehlende MP4-Dateien für die visuelle Review.
2. `aria_audio_all.sbatch` extrahiert Audio, erkennt `START`, `SECOND`, `DONE`
   und `THIRD` und erzeugt die Debug-WAVs. Dieser Job verarbeitet derzeit alle
   VRS-Dateien erneut und sichert die vorherigen Timestamp-Dateien.
3. `aria_detect_tags_all.sbatch` erzeugt die AprilTag-/ArUco-Posen. Bereits
   vorhandene gültige Marker-CSVs werden standardmäßig übersprungen.

Die drei Jobs haben untereinander keine fachliche Abhängigkeit und dürfen zur
Zeitersparnis auch gleichzeitig eingereicht werden. Wenn sie technisch strikt
nacheinander laufen sollen, können SLURM-Abhängigkeiten verwendet werden:

```bash
mp4_job=$(sbatch --parsable singularity/aria_vrs_to_mp4.sbatch)
audio_job=$(sbatch --parsable --dependency=afterok:"$mp4_job" singularity/aria_audio_all.sbatch)
tags_job=$(sbatch --parsable --dependency=afterok:"$audio_job" singularity/aria_detect_tags_all.sbatch)

echo "MP4 job:   $mp4_job"
echo "Audio job: $audio_job"
echo "Tags job:  $tags_job"
```

`aria_smoke.sbatch`, `aria_code_check.sbatch` und die Trainings-SBatch-Dateien
sind für das normale Hinzufügen neuer Aufnahmen nicht erforderlich.

### Jobstatus und Logs prüfen

```bash
squeue -u "$USER"
sacct -j JOB_ID --format=JobID,JobName,State,ExitCode,Elapsed
```

Die zugehörigen Dateien `aria_vrs_to_mp4.<JOB_ID>.out`,
`aria_audio_all.<JOB_ID>.out` und `aria_tags_all.<JOB_ID>.out` müssen ohne
Fehler enden. Insbesondere im Tag-Log sollte abschließend keine fehlende oder
ungültige Marker-CSV gemeldet werden.

## 4. Dataset-QA nach den automatischen Jobs

Zunächst mit den automatisch erkannten Timestamps ausführen:

```bash
singularity exec ~/singularity/aria_master.simg \
  python3 Code/dataset_qa.py \
  --data-root Data_collection \
  --timestamps Data_collection/Data_vrs/timestamps_summary.json
```

Erzeugt beziehungsweise aktualisiert:

```text
Data_collection/dataset_manifest.csv
Data_collection/dataset_qa_report.json
```

Nur die vier relevanten Personen anzeigen:

```bash
awk -F, '
  NR == 1 || tolower($2) ~ /^(david|florian|angelina|yuzhi)$/
' Data_collection/dataset_manifest.csv
```

Typische `next_action`-Werte sind:

| `next_action` | Bedeutung |
|---|---|
| `download_or_process_mps` | Handtracking oder SLAM fehlt |
| `fix_timestamps` | mindestens ein Sprachkommando fehlt oder ist ungültig |
| `review_or_exclude_sequence` | Handover-Handtracking ist unbrauchbar oder zu schwach |
| `run_aruco_extraction` | Marker-CSV fehlt oder passt zeitlich nicht |
| `convert_mp4` | MP4 fehlt |
| `extract_wav` | Review-WAV fehlt |
| `annotate_sequence` | Zielobjekt und/oder Empfangshand fehlen |
| `build_master_dataset` | alle Eingaben sind vorhanden; Master-Dataset fehlt |

`next_action` zeigt immer nur das aktuell wichtigste Problem. Deshalb die QA
nach jedem behobenen Blocker erneut ausführen.

Fehlende Backup-VRS auf dem Cluster sind nur eine Warnung, wenn der lokale
Backup-Ordner dort absichtlich nicht vorhanden ist. Entscheidend ist, dass die
VRS-Datei unter `Data_collection/Data_vrs/` existiert.

## 5. Timestamps und semantische Labels manuell prüfen

Dieser Schritt ist nicht durch ein SBatch-Skript abgedeckt. Das Review-Werkzeug
öffnet ein Videofenster und spielt Audio ab; es sollte daher lokal auf einem
Rechner mit GUI ausgeführt werden. Dafür müssen mindestens folgende aktuellen
Artefakte lokal vorliegen:

```text
Data_collection/Data_mp4/
Data_collection/Data_vrs/debug_audio/
Data_collection/dataset_manifest.csv
Data_collection/Aruco_CSV/                 # empfohlen
Data_collection/manual_timestamp_review.csv
```

Zuerst Sequenzen mit Timestamp-Problemen prüfen:

```bash
python3 Code/review_timestamps_video.py \
  --data-root Data_collection \
  --only-next-action fix_timestamps
```

Danach Sequenzen mit noch fehlender Semantik prüfen:

```bash
python3 Code/review_timestamps_video.py \
  --data-root Data_collection \
  --only-next-action annotate_sequence
```

Bei einer Sequenz müssen festgelegt beziehungsweise bestätigt werden:

- `decision`: `accept_auto`, `manual_fix`, `exclude` oder `uncertain`
- alle vier Zeiten: `START`, `SECOND`, `DONE`, `THIRD`
- `target_object_id`: Objektmarker 6 bis 14
- `receiving_hand`: für das aktuelle Training `left` oder `right`
- `annotation_confidence`: möglichst `certain`

Bei Timestamp-Problemen können Zielobjekt und Hand direkt im gleichen
Review-Durchlauf annotiert werden. Die Änderungen landen in:

```text
Data_collection/manual_timestamp_review.csv
```

Wichtig: `target_object_id` und `receiving_hand` sind Pflichtfelder für den
standardmäßigen Master-Batch-Job. Fehlen sie, wird die Sequenz übersprungen.

## 6. Reviews anwenden und erneut QA ausführen

Die aktualisierte `manual_timestamp_review.csv` muss zusammen mit den aktuellen
automatischen Timestamp-Dateien auf dem System liegen, auf dem der Import läuft.

Reviews validiert anwenden:

```bash
singularity exec ~/singularity/aria_master.simg \
  python3 Code/apply_manual_reviews.py \
  --data-root Data_collection
```

Das erzeugt unter anderem:

```text
Data_collection/Data_vrs/timestamps_summary.reviewed.json
Data_collection/Data_vrs/timestamps_manual_overrides.json
Data_collection/manual_timestamp_review_report.json
```

Wenn `Applied: ..., rejected: 0` nicht erreicht wird, zuerst die im Report
genannten fehlerhaften Zeiten korrigieren. Nicht mit abgelehnten Reviews
weiterbauen.

Danach unbedingt die QA mit der reviewed-Datei wiederholen:

```bash
singularity exec ~/singularity/aria_master.simg \
  python3 Code/dataset_qa.py \
  --data-root Data_collection \
  --timestamps Data_collection/Data_vrs/timestamps_summary.reviewed.json
```

Für jede verwendete neue Sequenz sollten nun mindestens diese Bedingungen
erfüllt sein:

- VRS vorhanden
- Handtracking und SLAM vorhanden
- Marker-CSV vorhanden und zeitlich passend
- alle vier Timestamps vorhanden und geordnet
- Review nicht `uncertain`
- Zielobjekt-ID vorhanden
- Empfangshand `left` oder `right`
- kein `missing_handover_hand_tracking`

Warnungen wie `target_object_not_detected`, `receiving_hand_tracking_low` oder
sehr kurze Phasen müssen fachlich geprüft und dürfen nicht nur ignoriert werden.

## 7. Master-Datasets bauen

Zuerst einen Dry-Run starten:

```bash
sbatch --export=ALL,DRY_RUN=1 \
  singularity/aria_build_master_dataset.sbatch
```

Im Job-Log und in
`Data_collection/master_datasets/master_batch_report.json` prüfen, ob die
erwarteten neuen Sequenzen als `would_build` ausgewählt werden. Falls eine neue
Sequenz unter `skipped` steht, stehen dort auch die Gründe, zum Beispiel
`missing_target_object_id`, `missing_receiving_hand` oder
`incomplete_timestamps`.

Anschließend den echten resumierbaren Build starten:

```bash
sbatch singularity/aria_build_master_dataset.sbatch
```

Der Standard ist `OVERWRITE=0`. Bereits vollständig vorhandene Master-Datasets
werden übersprungen und nur fehlende neue Datasets gebaut. Für das normale
Hinzufügen der 39 Aufnahmen daher **nicht** `OVERWRITE=1` setzen. Das wäre nur
nötig, wenn bereits gebaute Master-Datasets aufgrund geänderter Eingaben
absichtlich neu erzeugt werden sollen.

Erwartete Ausgaben pro Sequenz:

```text
Data_collection/master_datasets/<sequence_id>_master.csv
Data_collection/master_datasets/<sequence_id>_master_report.json
```

Der SBatch-Job führt nach dem Build automatisch noch einmal Dataset-QA aus.

## 8. Abschließende Kontrolle

Master-Batch-Bericht prüfen:

```bash
sed -n '1,240p' Data_collection/master_datasets/master_batch_report.json
```

Wichtig sind:

- `errors` muss `0` sein,
- die erwarteten neuen Sequenzen müssen unter `records` als `built` oder bei
  einem späteren Wiederholungslauf als `already_exists` erscheinen,
- unbeabsichtigt übersprungene neue Sequenzen müssen vor dem Training geklärt
  werden.

Anzahl und Namen der neuen Master-Datasets prüfen:

```bash
find Data_collection/master_datasets -maxdepth 1 -type f -name '*_master.csv' \
  | grep -Ei '/(David|Florian|Angelina|Yuzhi)_' \
  | sort
```

Erst danach bei Bedarf die Trainingsjobs starten:

```bash
sbatch Training/hierarchical_baseline.sbatch
sbatch Training/hierarchical_residual_v2.sbatch
```

Die Trainingsjobs gehören nicht mehr zur Aufnahmeverarbeitung selbst.

