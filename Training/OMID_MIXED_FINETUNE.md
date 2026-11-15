# Mixed old/new fine-tuning, 2026-11-03

The earlier table compares unchanged, new-only fine-tuning, and new-only random
initialization. This experiment adds `mixed_finetune`: original selected main
checkpoint (trial_000, matching seed), with all weights fine-tuned jointly on
the original 170 training recordings and the existing 30 new Omid training
recordings. It is a single model, with the existing architecture.

Both original splits are preserved: old validation 21/test 23 recordings; new
validation 10/test 10. Sequence membership and eligible-window fingerprints
must match previous metadata for every split. The original data contract is
enforced. New and old sequence IDs must be disjoint. All normalizers remain
the original source training normalizer; no test statistics are fitted.

Sampling assigns probability 0.5 to each dataset and uniform probability to
windows within that dataset. Sampling uses replacement and an epoch contains
the sum of both training window counts. This balances domains in expectation,
not exactly in every batch. Class weights use inverse expected class mass
under this sampler, including the fraction of hand-labeled windows; task loss
weights retain the source settings.

Three seeds 42/43/44; LR 3e-5; at most 40 epochs; patience 10; batch 32; other
optimizer/model settings unchanged. Strict improvement of the equally weighted
mean of old-original and new-marker-free validation intention macro-F1 selects
one checkpoint per seed for all tasks. Test inference begins only after that
checkpoint and checksum have been recorded. Settings are frozen in
`Data_collection/object_marker_free_v1/mixed_protocol_v1.json` before launch.

Evaluate that checkpoint under the same three existing conditions: original
old inputs, old object features masked, and new marker-free inputs. Reuse the
original mask implementation. Assert identical pose cohorts against all three
earlier variants. Position is mean Euclidean error in cm with predicted hand,
on valid GT targets with both wrist references and without intention gating.
The combined report preserves earlier results and adds one row per condition.

This is an exploratory follow-up after inspecting earlier test results, not
an evaluation on untouched final holdouts. No new test results guide checkpoint
selection or hyperparameter search. Training exposure and validation criterion
differ from new-only fine-tuning, so this does not isolate mixing as the only
cause. One-person/native-tracking and whole-recording robot-anchor limitations
remain. Seed variation is not uncertainty over people.

Code snapshot: `mixed_code_v1/`; preflight: `mixed_preflight_v1/seed42/`;
outputs: `mixed_runs_v1/seed42` through `seed44`. Automated dependent summary
writes `mixed_runs_v1/COMPARISON.md`, `comparison.csv`, and `comparison.json`.
Job IDs are recorded in `mixed_jobs.json` under the dataset root after submission.

Submitted jobs: preflight **2389008**, GPU array **2389009** (tasks 0–2),
dependent summary **2389010**. Local validation: 16 tests passed, covering
domain balancing, reproducible sampling, loss-class mass, validation selection,
split leakage, prior window identity, pose-cohort identity, marker masking,
and native tracking conversion. The remote code snapshot has SHA-256 checksums.

All jobs completed successfully (GPU elapsed 3:28, 3:36, 4:26; stderr empty).
Selected epochs: 3, 2, 6; early stopping after epochs 13, 12, 16. Full results
and checkpoints were downloaded locally and verified against remote checksums.
All nine previous comparison rows are unchanged. All prior paired test cohorts
match. Local audits verified that selected checkpoints maximize the recorded
equal-domain validation score, and checkpoint SHA-256 matches selection.json.

Mean mixed-model scores (seeds 42/43/44):

| Condition | Intention macro-F1 | Hand macro-F1 | Position cm |
|---|---:|---:|---:|
| Old original | 0.8341 | 0.9621 | 16.67 |
| Old object features masked | 0.8261 | 0.9655 | 16.53 |
| New marker-free | 0.9585 | 0.9509 | 10.28 |

See `Data_collection/object_marker_free_v1/mixed_runs_v1/COMPARISON.md` for
all four variants, seed variation and interpretation limits. Relative to
new-only fine-tuning, mixed training improves mean old-domain metrics and
new-domain position, with slightly lower new-domain intention/hand F1.
The unchanged model retains lower old-domain position error. These descriptive
three-seed differences are not a significance claim.
