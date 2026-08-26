# Current thesis status

This file is the short entry point for the currently valid thesis state. It
does not replace the technical protocol or machine-readable records.

## Active scope

- **Primary objective:** multimodal intention recognition and receiving-wrist
  pose prediction at **t+1 s**.
- **Secondary experiment:** terminal receiving-hand endpose; it is separate
  from the primary t+1 target.
- **Active dataset:** `dataset_v3_causal_20260815_n214_5d136a34`.
- **Master alignment:** `causal_backward_device_time_v1`.
- **CLIP RGB timing:** `vrs_rgb_device_time_v2`.

## Execution status

According to the run registry, the active v3 protocol has completed its core
training, authorized evaluation, participant-wise LOPO analysis, and required
postprocessing. Reporting is also complete: SLURM job 2246329 produced the
authoritative 48-row matrix summary, and job 2246134 produced three synchronized
Device-Time qualitative overlays. The authoritative report fingerprint is
`ec078d5ed0d1eda3c2b009b92b3575da57f45cd2a3bbaa2ceca1154544184b9c`.

Verification on 2026-08-26 completed 174 unit/invariant tests, all 22
scientific smoke checks, a fresh 214/214 causal-master preflight, local
recomputation of every summary/qualitative output hash, and read-only
revalidation of all 48 training artifact freezes.

The project author confirmed on 2026-08-26 that `Test` is a pseudonym for a
real participant. Its six selected training sequences remain in place, so the
frozen study comprises **214 sequences from 25 participants**. No dataset
change or rerun is required. The dated
[identity-provenance resolution](Training/reports/dataset_v3_causal_20260815_n214_5d136a34/IDENTITY_PROVENANCE_RESOLUTION_20260826.md)
supersedes the pre-confirmation warning in the immutable final summary without
altering that result artifact.

Historical `dataset_v2_20260802_n214_5d136a34` artifacts remain historical.
They must not be interpreted or reported as results of the active causal v3
protocol.

## Authoritative outputs

- [Active-v3 final summary](Training/reports/dataset_v3_causal_20260815_n214_5d136a34/thesis_final_v2_corrected_alignment/final_summary/FINAL_MATRIX_SUMMARY.md)
- [Summary artifact manifest](Training/reports/dataset_v3_causal_20260815_n214_5d136a34/thesis_final_v2_corrected_alignment/final_summary/summary_artifact_manifest.json)
- [Transfer-ready thesis results](Thesis/experiment_results_n214.md)
- [Qualitative artifact manifest](Training/reports/dataset_v3_causal_20260815_n214_5d136a34/thesis_final_v2_corrected_alignment/qualitative/qualitative_artifact_manifest.json)
- [Updated LOPO summary](Training/reports/dataset_v3_causal_20260815_n214_5d136a34/thesis_v2_group_cv_seed42/summary_v2/group_cv_summary.json)
- [Identity-provenance resolution](Training/reports/dataset_v3_causal_20260815_n214_5d136a34/IDENTITY_PROVENANCE_RESOLUTION_20260826.md)

## Sources of truth

- [Checklist execution status](THESIS_CHECKLIST_STATUS.md) — item-by-item completion and remaining external/Git work.
- [Final verification record](Training/reports/dataset_v3_causal_20260815_n214_5d136a34/FINAL_VERIFICATION_20260826.md) — executed tests, dataset QA and artifact-hash checks.
- [Run registry](Training/run_registry.json) — current execution and blocker status.
- [v3 dataset descriptor](Training/datasets/dataset_v3_causal_20260815_n214_5d136a34.json) — frozen dataset contract and alignment metadata.
- [Implementation status](Training/IMPLEMENTATION_STATUS_P0_P5.json) — requirement-level implementation and runtime evidence.
- [Final thesis protocol](Training/THESIS_FINAL_PROTOCOL_V2.md) — scientific rules, targets, and reporting policy.
- [Experiment matrix](Training/configs/experiment_matrix_v2.json) — active variants, seeds, and evaluation plan.
