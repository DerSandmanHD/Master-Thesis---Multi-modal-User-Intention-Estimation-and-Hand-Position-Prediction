# Causal intention baselines

Retrospective descriptive baselines on the frozen v3 split. Fitting uses train windows only; no test metric selects a feature, parameter, or hyperparameter.

| Method | Test accuracy | Test macro-F1 | Continue F1 | Fetch F1 | Handover F1 |
|---|---:|---:|---:|---:|---:|
| majority_class | 0.7058 | 0.2758 | 0.8275 | 0.0000 | 0.0000 |
| elapsed_time_since_start_logistic | 0.7294 | 0.4660 | 0.8742 | 0.0000 | 0.5239 |
| last_sensor_frame_logistic | 0.8558 | 0.7884 | 0.9156 | 0.7355 | 0.7141 |
