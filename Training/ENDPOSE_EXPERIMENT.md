# Reproducible terminal end-pose experiment

This experiment is separate from the existing Residual-v2 experiment. The
existing model predicts the receiving-hand pose at **t+1 second**. The new
`residual_v2_endpose` model predicts one **robust terminal receiving-hand pose**
for every eligible handover sequence. Existing configs, checkpoints, metrics,
and run directories are neither modified nor overwritten.

## Frozen data and split

- Dataset tag: `dataset_v2_20260802_n214_5d136a34`
- Selected sequences: 214
- Sequence fingerprint:
  `5d136a34b915f4e6a81fda70d34c959be48b4be79f0f7922decfdaae65ad12cd`
- Source-content fingerprint:
  `d97c840904d01f687feba713cc98c72b226f388a6d3636c4ce9546194938444a`
- Validation participants: Atilla, Ermal, Vanessa
- Test participants: Edu, Jona, Mona
- Seeds: 42, 43, 44

The loader verifies these contracts before audit, dry-run, training, or
evaluation. The feature set, 60-frame windows, stride 10, participant split,
Residual-v2 architecture, and validation-selected hyperparameters are inherited
from the final n214 Residual-v2 configuration.

## Terminal target definition

For the annotated receiving hand, the target builder searches backwards from
the last `handover` row (the phase after `THIRD`) for the latest stable
0.5-second tracking interval. Position is the coordinate-wise median. Quaternion
orientation is the principal-eigenvector average, which is invariant to the
equivalent signs `q` and `-q`.

The following limits are fixed in
`Training/configs/models/residual_transformer_endpose_v1.json` before model
training:

- at least 8 valid hand samples;
- at least 70% valid samples in the interval;
- at least 0.35 seconds between first and last valid sample;
- 90th-percentile positional deviation no greater than 5 cm;
- 90th-percentile angular deviation no greater than 25 degrees;
- stable interval ending no more than 1 second before the sequence's handover
  end.

If no candidate passes, the sequence is marked as uncertain. It remains in the
intent and receiving-hand tasks, but every pose target for that sequence is
invalid, so it contributes neither pose loss nor pose metrics. Every valid
handover window of an accepted sequence receives exactly the same terminal
target. `time_to_sequence_end_seconds` is measured from the window endpoint to
the final handover timestamp.

## Execution order

From the repository root on the cluster:

```bash
DATASET_TAG=dataset_v2_20260802_n214_5d136a34
EXPERIMENT_TAG=residual_v2_endpose_v1

AUDIT_JOB=$(sbatch --parsable \
  --export=ALL,DATASET_TAG="$DATASET_TAG",EXPERIMENT_TAG="$EXPERIMENT_TAG" \
  Training/jobs/audit_endpose_v1.sbatch)
```

Inspect the audit and dry-run first. Training is allowed only when both report
success:

```bash
python3 -c 'import json; p="Training/reports/dataset_v2_20260802_n214_5d136a34/residual_v2_endpose_v1/audit"; print(json.load(open(p+"/endpose_target_audit.json"))["status"], json.load(open(p+"/dry_run.json"))["status"])'
```

Then submit the three independent training seeds and a dependent finalizer:

```bash
TRAIN_JOB=$(sbatch --parsable \
  --export=ALL,DATASET_TAG="$DATASET_TAG",EXPERIMENT_TAG="$EXPERIMENT_TAG" \
  Training/jobs/train_endpose_v1.sbatch)

sbatch --dependency=afterok:"$TRAIN_JOB" \
  --export=ALL,DATASET_TAG="$DATASET_TAG",EXPERIMENT_TAG="$EXPERIMENT_TAG" \
  Training/jobs/finalize_endpose_v1.sbatch
```

Checkpoint selection is validation-only:

- `best_intention_model.pt`: maximum validation intent macro-F1;
- `best_pose_model.pt`: minimum validation terminal-pose position error.

The test split is evaluated only after those checkpoints have been selected.
No test result influences architecture, hyperparameters, quality thresholds, or
checkpoint choice.

## Comparison and outputs

The final comparison contains two complementary views:

1. the original native t+1 metric, retained only as context because it has a
   different target;
2. a fair common-target comparison in which the existing t+1 checkpoints and
   the new end-pose checkpoints are both evaluated against the same robust
   terminal targets.

The report directory is
`Training/reports/dataset_v2_20260802_n214_5d136a34/residual_v2_endpose_v1/`.
It contains:

- per-sequence audit CSV and JSON;
- dry-run JSON;
- per-seed and aggregated model CSV files;
- error-versus-time-remaining CSV files;
- CPU/CUDA latency JSON and CSV files;
- PNG and PDF figures;
- `comparison.json` and a concise generated `README.md`.

Training runs are written only below
`Training/runs/dataset_v2_20260802_n214_5d136a34/residual_v2_endpose_v1/`.
