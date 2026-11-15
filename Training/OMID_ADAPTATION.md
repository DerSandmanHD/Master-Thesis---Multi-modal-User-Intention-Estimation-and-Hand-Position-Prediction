# Omid adaptation, frozen 2026-11-02

Dataset: `Data_collection/object_marker_free_v1`. The user confirmed the event
proposals and receiving hands. Only robot Tag 0 is present, with the existing
0.10 m size assumption. Table/object markers are absent.

START uses the first RGB frame's device timestamp. Confirmed SECOND/DONE/THIRD
retain their original device timestamps. Audio-relative and video-relative
seconds are distinct fields in `timestamps_confirmed_sensor.json`.

Native handtracking and high-frequency VIO replace MPS. Device-frame linear
velocity is computed by rotating odometry velocity with the inverse device
rotation. Invalid native poses are masked; multiple odometry sessions are
rejected. The existing causal observation joins and separate t+1-second targets
remain. Object features are absent, with the original 92-feature schema intact.
Robot anchoring remains an offline whole-recording estimate. This experiment
does not isolate marker removal from the change in tracking provenance.

## Recording split

- Training: 1–30; 15 left / 15 right; includes the development pilot.
- Validation: 31–36 and 42–45; 5 left / 5 right (33 is left).
- Test: 37–41 and 46–50; 5 left / 5 right.

Participant remains Omid throughout. These are disjoint recordings, not new
participants or verified independent sessions. All clips were visually checked
before splitting. No model test metrics informed splitting or training settings.

## Models and selection

Source: selected trial_000 original-split runs from residual_v3_hp_search_v1,
seeds 42/43/44. Each source checkpoint and normalizer is hash-recorded.

1. Unchanged source model with original normalization.
2. Fine-tune all weights: original normalization, LR 3e-5, at most 40 epochs,
   early-stopping patience 10.
3. Random initialization of the same architecture: only the new 30 training
   videos, normalization fitted only on them, LR 3e-4, at most 80 epochs,
   patience 15.

Other settings and losses come from the source config. Validation intention
macro-F1 selects one checkpoint for every task. Both models are selected before
the test is evaluated. Preflight verifies feature schemas, split completeness,
class/hand support and equal window fingerprints.

Compare intention macro-F1, hand macro-F1, and mean Euclidean position error
in cm. Primary pose comparison uses the identical valid-target cohort where
both reference wrists exist, predicted receiving hands, and no intention gate.
Full oracle/persistence/gated diagnostics and sample counts remain available.
Three-seed standard deviation describes seed variation, not person uncertainty.

## Jobs and artifacts

Preparation/preflight: 2388274. GPU array: 2388287. Comparison: 2388289.
Each stage depends on successful completion of the preceding stage.

Isolated source snapshot: `training_code_v1/`. Outputs: `native_training_v1/`,
`master_datasets/`, `adaptation_preflight_v1/`, and `adaptation_runs_v1/seed42`
through `seed44`. Final comparison is written as COMPARISON.md, comparison.csv
and comparison.json under adaptation_runs_v1. Old datasets and source runs are
not rewritten.

Local checks: 19 isolation/native-conversion/marker-free tests passed. A real
pilot master and source checkpoint passed strict loading and forward/backward
training with finite serialized metrics.
