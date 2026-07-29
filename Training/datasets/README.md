# Dataset versions

Jeder Trainingsstand erhält vor dem ersten langen Lauf einen unveränderlichen
Dataset-Tag:

```text
dataset_v<version>_<YYYYMMDD>_n<sequences>_<fingerprint8>
```

Beispiel:

```text
dataset_v2_20260815_n180_ab12cd34
```

Der Fingerprint muss aus dem nach QA ausgewählten Datasetbestand stammen. Ein
Descriptor dokumentiert mindestens Manifest, Inhaltsfingerprint, Split,
Builderversion und Snapshotpfad. Der große Snapshot selbst bleibt außerhalb
von Git; nur Descriptor, Manifest-Snapshot in den Runs und Hashes werden
versioniert.

`dataset_version_template.json` wird für neue Stände kopiert und vollständig
ausgefüllt. Ein einmal für finale Trainingsläufe verwendeter Dataset-Tag wird
nicht nachträglich auf andere CSV-Inhalte umgebogen.

Der vorhandene Descriptor
`dataset_v1_20260729_n156_seq457a80f1.json` registriert rückwirkend den
Datasetstand der zwölf `final_clean_v1`-Läufe. Da diese historischen Runs noch
keinen Inhaltsfingerprint speicherten, wird ausdrücklich nur der belegte
Manifest- und Sequenzfingerprint angegeben.

Vor einem neuen Benchmark:

1. Master-Datasets und QA abschließen.
2. Einen unveränderlichen Snapshot validieren.
3. Den Template-Descriptor kopieren und mit den Werten dieses Snapshots
   ausfüllen.
4. Den Descriptor in `Training/run_registry.json` registrieren.
5. Genau diesen `dataset_tag` an die Jobs unter `Training/jobs/` übergeben.
