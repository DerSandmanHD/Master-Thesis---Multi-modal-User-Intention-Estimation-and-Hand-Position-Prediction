# Nachträgliche Dataset-Review-Korrekturen

**Stand:** 29. Juli 2026

Die visuelle Videoprüfung bestätigte zwei fehlerhaft gesetzte
`DONE`-Zeitpunkte. Die Aufnahmen selbst besitzen ausreichend lange
Transitionen; der allgemeine Mindestwert von 0,5 Sekunden wird nicht
abgesenkt.

| Sequenz | alter DONE-Wert | neuer DONE-Wert | THIRD | neue Transition |
|---|---:|---:|---:|---:|
| `Suthan_3_20260604_165324` | 31,164 s | 30,763 s | 31,597 s | 0,834 s |
| `Vanessa_6_20260624_133209` | 32,045 s | 31,830 s | 32,497 s | 0,667 s |

Die kanonische lokale Annotationstabelle ist
`Data_collection/manual_timestamp_review.csv`. Sie wird ab diesem Stand als
einzige Datei unter `Data_collection/` in Git versioniert. Der SHA-256-Wert
direkt nach den beiden Korrekturen lautet:

```text
884f76fa847d4003ec58d017e26e1bf649954caa1f23ac4f58c41b8cad660c45
```

Der Hash bezieht sich auf die durch `.gitattributes` festgelegte
Git-kanonische LF-Version der CSV.

Vor dem nächsten Dataset-Build sind auf dem Cluster folgende Schritte
erforderlich:

1. Eine dort eventuell abweichende `manual_timestamp_review.csv` sichern und
   gegen die versionierte Datei vergleichen.
2. Die manuellen Reviews mit `Code/apply_manual_reviews.py` validiert in eine
   neue `timestamps_summary.reviewed.json` übernehmen.
3. Dataset-QA mit
   `--min-handover-hand-valid-ratio 0.70` erneut ausführen.
4. Die Master-CSVs von `Suthan_3` und `Vanessa_6` neu bauen.
5. Manifest, erwartete 169 geeignete Mastersequenzen und Split erneut
   validieren, bevor ein Training beginnt.

Die sechs Aufnahmen mit vollständig fehlendem Zielobjektmarker sowie
`Jola_5` bleiben bewusst ausgeschlossen. Es wurden keine Rohdaten,
Master-CSVs oder bestehenden Checkpoints verändert.
