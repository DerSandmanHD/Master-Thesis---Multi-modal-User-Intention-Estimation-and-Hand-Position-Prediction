# Identity provenance resolution — participant pseudonym `Test`

Date: 2026-08-26

Active dataset: `dataset_v3_causal_20260815_n214_5d136a34`

## Resolution

The project author directly confirmed during the finalization review that
`Test` is a pseudonym for a real participant, not a label for technical trial
recordings. This resolves the manual identity-provenance flag raised by the
active split audit.

The following six already selected training sequences belong to that
participant:

- `Test_2_20260624_122206`
- `Test_3_20260624_122314`
- `Test_4_20260624_122416`
- `Test_5_20260624_122509`
- `Test_6_20260624_122607`
- `Test_7_20260624_122709`

## Consequences for the frozen study

- The active dataset remains unchanged at **214 sequences from 25
  participants**.
- `Test` remains one participant-disjoint group in the training split.
- No dataset rebuild, split change, retraining, checkpoint selection, or
  evaluation rerun is required.
- Other recordings carrying the same pseudonym that were already excluded for
  independent QA or manual-exclusion reasons remain excluded. This resolution
  does not override their sequence-level status.

The repository evidence for the six selected sequences is the existing
`split_confounding_v2/split_audit.json`; the human-identity interpretation is
provided by the direct project-author confirmation above.

## Frozen-report compatibility

The hash-bound authoritative summary was generated before this confirmation
and contains an identity-provenance warning. That result artifact is
intentionally left untouched. This dated resolution supersedes only that
warning for thesis reporting; it does not modify any metric, cohort, dataset,
or artifact fingerprint.
