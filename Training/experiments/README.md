# Experiment archive

This directory stores small, reproducible snapshots of completed or otherwise
important training experiments. It is not a second source-code tree.

## Convention

Create one directory per named experiment, for example:

```text
Training/experiments/hierarchical_tracking_baseline_v1/
  README.md
  config_snapshot.json
  result_summary.json
  config.json          # optional: copied from the actual run directory
  data_metadata.json   # optional: copied from the actual run directory
  metrics.json         # optional: copied from the actual run directory
```

Each experiment must record:

- the exact Git commit used for training;
- the immutable configuration used for training;
- the participant split and dataset counts;
- the run directory or job identifier;
- validation and test results;
- known limitations and exclusions.

## Source-code versioning

Do not copy `train.py`, `model.py`, `dataset.py`, and their dependencies into
every experiment directory. Duplicate source trees drift, obscure bug fixes,
and make it unclear which version is authoritative. Keep the active source in
`Training/` and recover an experiment's code through its Git commit or tag.

Use a Git tag for an accepted milestone. Use a branch only when an experiment
requires a maintained implementation that must continue to diverge from the
main training pipeline.

Large checkpoints such as `best_model.pt` should not be committed to normal
Git history. Store them on the cluster, in an artifact store, or with Git LFS,
and document their location and checksum in the experiment README.
