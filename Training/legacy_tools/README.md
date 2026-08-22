# Historical standalone tools

These scripts are retained only to reproduce earlier exploratory v1/v2 work.
They are not part of the active causal v3 thesis protocol and have no current
call sites in the repository.

- `create_clip_latency_fixture.py`: creates a single RGB fixture for an older
  CLIP latency benchmark.
- `generate_endpose_v2_trials.py`: generates the completed terminal-endpose
  v2 hyperparameter-search trials.
- `select_final_model.py`: applies the earlier visual-model selection flow;
  the active protocol uses `select_matrix_checkpoints.py` instead.

Do not use these scripts for new results.  The active entry points and their
roles are listed in `../SCRIPT_GUIDE.md`.
