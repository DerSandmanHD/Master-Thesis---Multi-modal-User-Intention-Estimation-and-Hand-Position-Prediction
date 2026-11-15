# Reproduction from processed research data

This guide reproduces the reported experiments **starting from processed data**.
The primary inputs are the frozen master CSVs (aligned tracking/gaze/object/motion
features and targets), CLIP embeddings, training-only PCA, split/normalizer
metadata and selected checkpoints. Original RGB/video/audio recordings are not
required for the workflows below. Re-extracting tracking or embeddings from raw
recordings is a separate upstream experiment outside this release's scope.

The classification target is the current annotated interaction state. Only the
wrist-pose target is in the future. Spatial features use offline SLAM and an
offline recording-wide marker anchor. The fixed test had been inspected during
development; replay does not turn these exploratory comparisons into confirmation.

## 1. What Git contains and what must be obtained separately

`reproducibility/processed_artifacts.lock.json` inventories 3,386 files on the
research cluster, with byte size, SHA-256 and resource groups. It was assembled
on 2026-11-15 from the experiment-bound manifests and the available cluster
artifacts. `reference=frozen_sha256` means the file was checked against an
existing recorded hash; `cluster_snapshot_2026-11-15` means a new inventory hash,
not a retrospective claim of a historical freeze. Group membership overlaps.

The index is small enough for Git. The data, weights, prediction exports and
historical run directories remain external. **A clone alone is not a complete
data release.** Access requires permission from the repository maintainer and
the existing HPC account/VPN, or an equivalent separately supplied package.
No public download URL is currently provided. The index contains recording
identifiers used by the existing experiment artifacts; keep the data package
under the same access controls as the research dataset. This checkout and its
Git history are not an anonymized public release; see [DATA_ACCESS.md](DATA_ACCESS.md).

| Group | Contents | Needed for |
|---|---|---|
| `results` | Saved reports, predictions, grouped statistics and protocols | Metric recomputation |
| `data` | 214 selected original masters/reports and dataset selection inputs | Main/LOPO evaluation and training |
| `main` | 48 frozen main run folders, selected and diagnostic checkpoints, configs/normalizers | Main checkpoint replay |
| `lopo` | 25 seed-42 fold runs and diagnostic checkpoints | LOPO replay |
| `visual` | 214 CLIP cache files, manifest and PCA NPZ/JSON | Visual variants; no encoder needed to consume embeddings |
| `adaptation` | 50 new masters/reports, trial_000 sources, fine-tuning/scratch/mixed runs and protocols | Adaptation |
| `snapshots` | Archived supplementary source trees | Historical supplementary reruns |
| `tracking` | Intermediate gaze/hand/VIO/SLAM/marker data and reviewed annotations | Optional provenance/upstream processed-data inspection |

Intermediate tracking files are redundant for training when the exact master
CSVs have been supplied; they are retained as a separate optional group. The
container, OpenCLIP dependency directory and encoder weights are described under
`external_software`; they are not silently included in repository-relative transfers.
Encoder weights are needed only to extract new embeddings, not to train on caches.

## 2. Obtain and verify the external package

From the repository root, using Python 3.10 or later (standard library only):

```bash
python3 reproducibility/artifacts.py summary
python3 reproducibility/artifacts.py files --groups results > /tmp/thesis-results.files
```

Obtain access instructions separately from the maintainer. Configure your own
SSH alias with the authorized host, account and port; none is a public download
endpoint. Set the remote repository path and transfer **only the selected index
entries**. Replace the placeholders below; a trailing slash is required:

```bash
THESIS_REMOTE='your-authorized-ssh-alias'
THESIS_REMOTE_REPO='/absolute/path/to/the/research-repository'
rsync -an --ignore-existing -e ssh --files-from=/tmp/thesis-results.files \
  "$THESIS_REMOTE:$THESIS_REMOTE_REPO/" ./
# Review the dry run, then use -a instead of -an to perform this transfer.
python3 reproducibility/artifacts.py verify --groups results
```

For checkpoint replay, generate another file list with
`--groups data main lopo visual adaptation snapshots` and transfer it the same
way. This is a multi-GB package; inspect `summary --groups ...` first.
`--ignore-existing` protects existing artifacts. If verification detects a bad
existing file, use a clean destination or resolve it explicitly; do not silently
replace frozen results. The verifier separates missing files, permission/I/O
errors, size differences and SHA-256 differences and returns a nonzero exit code.
It writes nothing and never needs the original recordings.

The inventory preserves the cluster's repository-relative layout. There are
231 masters in the cluster's original master directory, but only 214 are selected.
Use the indexed selection manifest, never train on an unfiltered directory glob.
The original split is 170/21/23 recordings from 19/3/3 distinct participants,
with 15,189/1,978/2,199 retained train/validation/test windows. The new dataset
uses 30/10/10 recordings and 2,753/896/915 windows from one participant.

## 3. Software environment

Use the historical cluster image `~/singularity/aria_master.simg` for the main
runs: Python 3.10.12, torch 2.4.1+cu124, torchvision 0.19.1+cu124, CUDA runtime
12.4, NumPy 2.2.6, pandas 2.3.3, scipy 1.15.3, scikit-learn 1.5.2,
projectaria-tools 2.1.2 and projectaria-mps 1.2.1. The run-level
`artifact_manifest.json/runtime` is the authority for each experiment.
`environment_exact.yml` describes a later local environment, not these runs.

The image is present, but its historical binary digest was not recorded.
Its embedded definition adds Ultralytics 8.4.53 and ultralytics-thop 2.0.19
compared with `singularity/aria.recipe`. Core version pins agree. A fresh build
from that recipe is not claimed to be the identical historical image.
For preserving the current environment, record both the image digest and the
embedded definition when packaging it; neither proves a missing historical digest.

CLIP extraction used OpenCLIP 2.26.1 from
`~/.cache/aria_python/open_clip_2_26_1/` and `~/.cache/clip/ViT-B-32.pt`
(SHA-256 `40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af`).
PCA: randomized sklearn solver, random state 20260808, 512 to 32 dimensions,
29,470 cached frames from exactly the 170 training recordings. Random visual
control seed: 20260815. Preserve the existing projection rather than refit it.
The new tracking conversion used `Data_collection/object_marker_free_v1/runtime_aria230/`
with projectaria-tools 2.3.0 and the container's base packages. Processed masters
can be consumed without rerunning that conversion or MPS.

## 4. Recompute reported metrics without model inference

```bash
python3 reproducibility/report_metrics.py > /tmp/thesis-t1-metrics.json
```

This verifies each primary prediction CSV and recomputes the mean, Euclidean
RMSE, linearly interpolated median/P90 and strict Hit@5/10/15/20 cm. It uses
exactly the shared 213-window, ground-truth-hand cohort and reports each seed
plus mean and sample SD. Expected mean errors: learned 14.903 cm, persistence
14.804 cm, constant velocity 33.490 cm (rounding only).

For grouped uncertainty, use the existing prediction sidecar to preserve
checkpoint binding, and write to new output paths:

```bash
D=dataset_v3_causal_20260815_n214_5d136a34
P=Training/reports/$D/thesis_final_v2_corrected_alignment/postprocess/residual_current_gate_seed42
python Training/evaluation/evaluate_grouped_predictions.py \
  --predictions "$P/test_predictions.csv" --prediction-report "$P/test_predictions.json" \
  --bootstrap-iterations 10000 --bootstrap-seed 42 --sequence-bootstrap \
  --report-out /tmp/thesis-grouped-seed42.json --table-out /tmp/thesis-grouped-seed42.csv
```

Do not mix populations: architecture/ablation pose errors use 181 windows with
both references; kinematic three-way comparison uses 213; the participant-balanced
learned/persistence analysis uses 214 and a different weighting; cascade uses
217 target-valid handover windows. `pose_fair_common` in a run report is the
181 cohort, while CSV `fair_common` is the 213 cohort. Terminal evaluation uses
201 strictly pure-future paired windows, not the fixed-horizon target.
Seed SD is optimization variability, not participant uncertainty. LOPO hand
macro-F1 has fixed-class and supported-class variants; the maximum of the
participant-balanced fixed-class statistic is 0.64 for this hand distribution.

## 5. Re-evaluate existing main and LOPO checkpoints

Historical scripts contain absolute cluster paths and strict freeze validators.
Do not edit their frozen configs/manifests to relocate them, and do not run
`evaluate_frozen_run.py` on a later checkout expecting its historical Git-state
check to pass automatically. Use the processed-data replay entry point:

```bash
D=dataset_v3_causal_20260815_n214_5d136a34
python reproducibility/replay_checkpoint.py \
  --run "Training/runs/$D/thesis_final_v2_validation/residual_current_gate/residual_current_gate_seed42" \
  --device cpu --output /tmp/thesis-replay-seed42.json
```

The replay verifies indexed inputs/weights before deserialization, relocates
paths in memory, rebuilds datasets through the existing loader and terminal
wrapper, checks dataset/schema/split identity, normalization and window
fingerprints, and calls the existing evaluation implementation. It refuses an
existing output file. It neither trains nor reselects checkpoints. Use the same
entry point for each of the 48 main run folders and 25 LOPO fold folders.
The 213-window kinematic comparison is recomputable from its existing exports
(section 4); the replay's run metrics retain the original 181-window comparison.

The replay is an additional portability tool, not a replacement of historical
reports. It has not yet been validated by re-evaluating every checkpoint on a
fresh machine. CPU/GPU and library differences may produce numerical differences;
compare regenerated values, cohorts and checkpoint hashes to the frozen reports.

## 6. Re-train from the processed data

```bash
python3 reproducibility/artifacts.py verify --groups data main visual
python3 reproducibility/prepare_training.py --family main --output-dir /tmp/thesis-main-retrain
python3 reproducibility/prepare_training.py --family lopo --output-dir /tmp/thesis-lopo-retrain
```

These commands only create a new plan and relocated configurations; inspect the
JSON plans and submit the listed training commands through your scheduler.
They never launch jobs. Main has 48 runs (15 fixed-horizon plus one terminal
configuration, each with seeds 42/43/44); LOPO has 25 seed-42 folds, each using
the next participant alphabetically for validation. All training commands skip
test evaluation. New runs use separate directories. Subsequent selection must
remain validation-only (`Training/select_matrix_checkpoints.py` and the frozen
matrix protocol); create a separate matching matrix/selection for a new run tree,
not an authorization pretending that new checkpoints are the historical ones.

The primary LR is 3e-4. The terminal LR 3.81605632501e-4 was inherited from
historical v2 random-search trial_022; it was not retuned for the corrected
terminal target. The terminal architecture, normalization of pose loss, sampling,
auxiliary target and training budget differ from the primary task. Keep the
resolved terminal config. The source of adaptation is the later grouped-search
trial_000 family, not the primary matrix checkpoint family.

## 7. Supplementary studies

Obtain `adaptation`, `snapshots`, `data` and the relevant main checkpoints.
The historical entry points and protocols are now included in Git as well as
archived in the external package:

| Study | Entry points | Frozen input/output locations |
|---|---|---|
| New-only and scratch | `Training/omid_adaptation.py` | `Data_collection/object_marker_free_v1/adaptation_protocol.json`, `adaptation_runs_v1/seed{42,43,44}/` |
| Old-domain retention | `Training/evaluate_omid_retention.py` | `retention_v1/seed{42,43,44}/` |
| Mixed fine-tuning | `Training/omid_mixed_finetune.py`, `summarize_omid_mixed.py` | `mixed_runs_v1/seed<seed>/protocol.json`, `mixed_runs_v1/seed{42,43,44}/` |
| Terminal timing | `Training/evaluate_terminal_timing.py`, `summarize_terminal_timing.py` | `Training/reports/<D>/terminal_input_timing_v1/` |
| Terminal checkpoints/residuals | `Training/evaluate_terminal_checkpoints.py`, `summarize_terminal_residuals.py` | `Training/reports/<D>/terminal_residual_diagnosis_v1/` |
| Validation attenuation | `Training/export_terminal_validation.py`, `evaluate_terminal_shrinkage.py` | `Training/reports/<D>/terminal_residual_shrinkage_v1/` |

For exact historical execution, use each study's `code/` snapshot, its protocol,
job command and stored source checkpoint identities. These historical scripts
retain absolute paths and some write outputs beside inputs: run them in an
isolated copy at compatible paths, not over the original archive. Their
individual CLI options are documented by `--help` and the `Training/OMID_*.md`
and `Training/TERMINAL_*.md` files. Adaptation metadata is stored at seed level,
above the `finetune/` and `scratch/` subfolders. Fine-tuning keeps the source
normalizer; scratch fits on new training data; mixed uses domain-balanced sampling
and equally weighted old/new validation. The data-loader additions preserve
these explicit recording splits and frozen source normalizers.

All follow-ups are exploratory. New recordings are from one person in all three
partitions; marker removal also changes tracking provenance. Timing cohorts have
16/3/1/1 recordings for successive blocks; the closest remaining-time bin has
five windows. Do not interpret these as controlled population-wide improvements.


### Generate the supplementary thesis tables

`Training/build_supplementary_thesis.py` regenerates six LaTeX tables and the
terminal residual-length figure from seven saved result files, without training
or checkpoint inference. Obtain the `results` and `adaptation` resources first.
It verifies each input's size and SHA-256 against the release lock and writes a
manifest of input/output hashes. Python plus matplotlib is required.

```bash
python Training/build_supplementary_thesis.py \
  --root . --output-dir /tmp/thesis-supplementary-regenerated
```

The output directory must not already exist. Original thesis files and frozen
results are never overwritten. Compare the generated numeric table bodies with
the submission; captions and layout may have been edited independently. The
plot's PDF bytes may vary with renderer metadata and library versions.

### Historical terminology and latency

[Training/EXPERIMENT_PROVENANCE_NOTES.md](Training/EXPERIMENT_PROVENANCE_NOTES.md)
explains the separate `trial_000` adaptation sources, the terminal learning-rate
origin, the development pilot and `robot_anchor_interpolated` semantics. The
saved latency reports concern a historical v2 model, separate CLIP benchmarks
and exploratory live sessions. They do not measure the active v3 system from
sensor capture through physical robot execution.

## 8. Validation and limits

The cluster audit found matching hashes for 214 original and 50 new masters,
48 main and 25 LOPO primary checkpoints, 12 adaptation source/trained checkpoints,
214 CLIP cache files, PCA and encoder weights. A standard-library inventory
check can be run on any supplied package. Unit tests cover corrupt/missing inputs,
unsafe paths, cohort diagnostics, attenuation and adaptation sampling/masking.
A complete fresh-machine run of all experiments has not been performed for this
release. No new training or evaluation jobs were launched while preparing it.
Original raw recordings and external MPS reprocessing are deliberately outside
this processed-data reproduction contract. This does not claim bitwise identical
GPU retraining or independent raw-to-MPS regeneration.


Validation of the supplementary-table entry point on 2026-11-15 regenerated all
six tables and the residual plot from seven SHA-256-verified saved inputs.
Numeric tokens in all six table bodies matched the supplied thesis version 13.
Existing-output protection and same-size input-corruption rejection were checked;
seven reproduction-tool tests passed. This was report/figure regeneration,
not a new model evaluation, training run or fresh-machine reproduction.
