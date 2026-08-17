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
postprocessing. The authoritative matrix summary and qualitative rendering are
still blocked and are therefore not reportable as final thesis results. Consult
the registry for the exact blockers and recovery state.

Historical `dataset_v2_20260802_n214_5d136a34` artifacts remain historical.
They must not be interpreted or reported as results of the active causal v3
protocol.

## Sources of truth

- [Run registry](Training/run_registry.json) — current execution and blocker status.
- [v3 dataset descriptor](Training/datasets/dataset_v3_causal_20260815_n214_5d136a34.json) — frozen dataset contract and alignment metadata.
- [Implementation status](Training/IMPLEMENTATION_STATUS_P0_P5.json) — requirement-level implementation and runtime evidence.
- [Final thesis protocol](Training/THESIS_FINAL_PROTOCOL_V2.md) — scientific rules, targets, and reporting policy.
- [Experiment matrix](Training/configs/experiment_matrix_v2.json) — active variants, seeds, and evaluation plan.
