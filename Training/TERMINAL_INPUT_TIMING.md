# Input timing and terminal receiving-hand position

Requested on 2026-11-07: evaluate the same final handover position from earlier
and later input windows, as suggested by the supervisor.

Use the corrected v3 terminal model, seeds 42/43/44, and the existing old test
split (Edu, Jona, Mona). The checkpoints remain validation-selected
`best_intention`. Preserve source checkpoint, artifact-manifest, normalizer,
feature schema, target-definition and split hashes. No training or selection.
The authoritative source reports are `terminal_endpose_learned_seed*.json`
under the corrected v3 matrix's `final_test` directory.

The existing model expects **60 aligned sensor samples**. Primary diagnostic:
non-overlapping windows 0–59, 60–119, etc., anchored at the first annotated
handover row. The 10-sample window in the supervisor's message was an example;
input length is held constant at the trained value. Trailing incomplete blocks
are omitted without padding. This time origin is the THIRD/handover annotation,
not an independently measured physical arm-motion onset or necessarily RGB
frame 0.

Complementary diagnostic: all original 60-sample/stride-10 handover windows,
which can include pre-handover context. Reconstruct the original 201 pure-future
and 20 partial-evidence paired windows and reproduce their historical errors
before reporting new block results. Dense stride-1 construction on the same
normalized **test** records supplies exact handover-aligned non-overlap windows;
eligibility, reference ages and target extraction remain identical.

Each accepted recording has one fixed terminal position from the latest
accepted stable segment, based on unique physical hand captures. Primary
results include only endpoints strictly before the first contributing capture.
Endpoints inside aggregation are separate partial-evidence diagnostics;
endpoints at/after the final contributing capture are excluded. Both hand
references must be available, giving model-independent paired cohorts.

Compare learned and last-observation predictions with the same predicted hand.
Also retain learned/persistence GT-hand diagnostics, matching the thesis.
All errors are conditional on annotated handover, with no intention gate.
Report per-window Euclidean error in cm; seed means/SD; windows and recording
counts; equal-recording means; and earliest/latest errors paired within each
recording. Plot all eligible recordings, rather than selecting examples by
their error trend. The temporal bins are frozen in
`Training/configs/evaluation/terminal_timing_v1.json`.

Limitations: old tests were previously observed; aggregate bins can have
different contributing recordings; seed SD is not person uncertainty; stable
terminal hand pose is not an independently annotated physical contact event;
whole-recording robot calibration remains offline. The recent Omid and mixed
fine-tuning experiments predict t+1 and are not terminal adaptations.

Scripts: `evaluate_terminal_timing.py`, `summarize_terminal_timing.py`.
Slurm jobs: `jobs/terminal_timing.sbatch`, `jobs/terminal_timing_summary.sbatch`.
Outputs: `Training/reports/dataset_v3_causal_20260815_n214_5d136a34/terminal_input_timing_v1/`.
`jobs.json` records execution IDs and failed attempts, while source snapshots
and checksums preserve the evaluated code. The first attempt stopped before
data/inference because historical relative source-config paths were resolved
against the reporting snapshot; explicit original-repository path resolution
corrects this while retaining all hash checks.

Local validation: 13 timing/target invariant tests passed. After the path fix,
the six directly relevant timing tests passed again.

## Completed result

Evaluation array **2391695**, summary **2391697** completed successfully.
Per-seed inference took 38–41 seconds; summary/plots took 16 seconds. Historical
pure-future and partial-evidence position errors were reproduced for all three
seeds. The downloaded 30 result files passed SHA-256 verification; PNG figures
were visually inspected.

The exact disjoint protocol creates 39 complete blocks per seed. Of these,
21 have a purely future accepted target and both reference hands, 5 are paired
partial-evidence blocks, and the rest are unavailable or excluded. Pure-future
block means are 17.81 cm (samples 0–59, 16 videos), 6.92 cm (60–119, 3 videos),
15.94 cm (120–179, 1 video), and 14.22 cm (180–239, 1 video). The changing
cohort prevents interpreting that aggregate sequence as a single movement.
Only three recordings have at least two paired pure-future disjoint blocks;
all three show lower error in the last block, averaging a 25.27 cm reduction.

The original stride-10 protocol retains 201 paired pure-future windows across
20 recordings. At least two windows exist in all 20. The latest versus earliest
eligible input reduces learned predicted-hand error in 17/20 recordings; equal
recording means are 32.54 cm early and 13.49 cm late (mean reduction 19.05 cm).
Matched predicted-hand persistence also improves: 27.35 cm early and 8.15 cm
late. Thus later input is usually more informative, but this diagnostic does
not establish a general learned-model advantage over last observation.

See `REPORT.md`, `summary.csv`, `paired_first_last.csv` and all three PNG/PDF
figures under the report root. Interpretation remains descriptive and post hoc;
physical motion onset and exact contact time were not newly annotated.
