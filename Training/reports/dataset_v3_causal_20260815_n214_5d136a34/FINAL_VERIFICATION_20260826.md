# Final verification — active causal v3

Date: 2026-08-26

Scope: read-only verification of the active thesis dataset, source changes and
existing frozen artifacts. No SLURM training job was submitted, no checkpoint
was changed, and no existing result file was overwritten. A subsequent manual
identity confirmation was recorded additively without modifying the frozen
summary.

## Bound state

- Branch: `improvement_pipeline`
- Base commit before the uncommitted finalization changes:
  `87112f5febfc07cdd5c80c167ca35d7c667513b5`
- Active dataset: `dataset_v3_causal_20260815_n214_5d136a34`
- Dataset content fingerprint:
  `339c313d56c7be8564a3035cd8334bad103e15659315c33135e96b95e820245a`
- Source content fingerprint:
  `b2cfd10991c5638277795aadcabfd322d3e9eb41dffb70696b63dc657f5f22cb`
- Matrix SHA-256:
  `20ebefa13be76635b2f78139a6a8506dcf1eae6fedd8fdc1c7bdcf4d53e7e6d2`
- Authoritative report fingerprint:
  `ec078d5ed0d1eda3c2b009b92b3575da57f45cd2a3bbaa2ceca1154544184b9c`
- Qualitative manifest fingerprint:
  `3d1673d12bc5d189de72905d0746164aa18a46292b9798fca0cb63e3b8cac650`

The final reporting checkout is identified by the Git commit containing this
record and its merge into `main`. A Git tag is intentionally still pending and
is not invented here.

## Verification results

| Check | Result |
|---|---|
| Linux unit/scientific invariant tests | **PASS — 174 passed**, 16 NumPy deprecation warnings |
| Repository scientific smoke scripts | **PASS — 22/22** |
| Active causal masters | **PASS — 214/214**, expected alignment and sequence fingerprint |
| Active validation-run artifact freezes | **PASS — 48/48** |
| Summary output hashes | **PASS — 5/5** |
| Authoritative combined report fingerprint | **PASS** |
| Qualitative output hashes | **PASS — 9/9** |
| JSON syntax and unique requirement IDs | **PASS** |
| Registry smoke test | **PASS** |
| Markdown relative links | **PASS** |
| `git diff --check` | **PASS** |
| Participant-pseudonym provenance | **RESOLVED — `Test` confirmed** |

The full pytest suite was executed in the TCML Linux Aria container with a
temporary, isolated pytest 8.4.1 layer:

```bash
singularity exec "$HOME/singularity/aria_master.simg" \
  env PYTHONPATH=/tmp/thesis_pytest_8_4_1 \
  python3 -m pytest -q tests
```

Result:

```text
174 passed, 16 warnings in 98.03s
```

The repository's `Training/run_scientific_tests.py` repeated all 174 tests and
then executed its smoke list. It exposed one stale assertion in
`Training/run_registry_smoke_test.py` that still expected the active v3 dataset
to be unmaterialized. The assertion was corrected to validate the actual
materialized descriptor; that check and the two remaining smoke scripts then
passed. Across the run and its verified continuation, all 22 listed smoke
scripts passed.

The actual dataset was rechecked with:

```bash
python3 Training/verify_causal_masters.py \
  --master-dir Data_collection/master_datasets \
  --manifest Data_collection/dataset_manifest.csv \
  --expected-alignment-version causal_backward_device_time_v1 \
  --expected-sequence-fingerprint \
    5d136a34b915f4e6a81fda70d34c959be48b4be79f0f7922decfdaae65ad12cd
```

Result:

```text
Causal masters verified: 214 sequences,
source=b2cfd10991c5638277795aadcabfd322d3e9eb41dffb70696b63dc657f5f22cb
```

All 48 manifests below the active validation root were passed through
`validate_artifact_freeze(..., require_complete=True,
require_current_git_state=False)`. The latter flag permits the later
reporting-only checkout while continuing to verify the training identity and
all run-local hashes.

Result:

```text
Artifact freezes valid: 48
```

The macOS `aria_conda` environment cannot run the whole suite because importing
its native `cv2` module aborts the process (`exit 134`). The same exact test
sources pass in the intended Linux container, so this is recorded as a local
binary-environment limitation rather than a repository test failure.

## Current control-file hashes

| File | SHA-256 |
|---|---|
| `THESIS_CURRENT_STATUS.md` | `74b21232fd026f2416d1acf739632d7efd8eaf1cd6b50acd4a3497f30255b893` |
| `THESIS_CHECKLIST_STATUS.md` | `51d56465eebc18e282fbf6cd9b974b41c1bd13048eab672ba873fcccf54939c7` |
| `Training/run_registry.json` | `f933c7afd401868788f1fa4cb6f75b421ce14ca69e5a640f39f927e146aa6ea6` |
| `Training/IMPLEMENTATION_STATUS_P0_P5.json` | `1962c29af311d8766587200054517687980514b4ccaf60cea77fadaccef9af67` |
| `Training/THESIS_FINAL_PROTOCOL_V2.md` | `56feaf53fed9bd5c003e28c92513cacf0797c44a610dafc915824ea8f7ff60ed` |
| `Training/configs/experiment_matrix_v2.json` | `20ebefa13be76635b2f78139a6a8506dcf1eae6fedd8fdc1c7bdcf4d53e7e6d2` |
| `Training/reports/dataset_v3_causal_20260815_n214_5d136a34/IDENTITY_PROVENANCE_RESOLUTION_20260826.md` | `1bfd2757b9c43f43e63e9acd1104315b54e738f5e5053e631d85469331b88724` |

These hashes describe the control files included in the authorized finalization
commit. Git records the checkout identity; the frozen dataset, matrix and
result fingerprints remain unchanged by the commit and merge.

## Identity resolution after verification

The project author directly confirmed on 2026-08-26 that `Test` is a real
participant pseudonym. The six selected training sequences remain in the
active dataset, so the defensible final wording is **214 sequences from 25
participants**. No dataset rebuild, retraining, or result regeneration is
required. The additive resolution is recorded in
[`IDENTITY_PROVENANCE_RESOLUTION_20260826.md`](IDENTITY_PROVENANCE_RESOLUTION_20260826.md).
