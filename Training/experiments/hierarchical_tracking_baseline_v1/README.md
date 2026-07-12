# Hierarchical Tracking Baseline v1

This directory stores immutable metadata for the accepted baseline result. It
does not duplicate the Python source tree.

## Source

- Source commit used on the cluster: `04c3156`
- Training config snapshot: `config_snapshot.json`
- Result summary: `result_summary.json`
- Full documentation:
  `Thesis/status_testing_hierarchical_tracking_baseline_v1.md`
- Cluster run directory:
  `Training/runs/hierarchical_baseline_20260712_101448`

Restore the exact source code with:

```bash
git checkout 04c3156
```

Do not copy the complete `Training/` source into this experiment directory.
Source duplication creates divergent implementations and ambiguous imports.
Use Git commits and annotated tags for code, and keep only configs, metrics and
small metadata artifacts per experiment.

The cluster files `config.json`, `data_metadata.json` and `metrics.json` should
be copied into this directory for the permanent experiment archive. Do not add
the large `best_model.pt` checkpoint to Git; archive it separately with the run
identifier and commit hash.

