# Empirical sampling and observation-window duration

Dataset: `dataset_v3_causal_20260815_n214_5d136a34` (214 sequences).

| Quantity | Median | IQR | 5th–95th percentile | Min–max |
|---|---:|---:|---:|---:|
| Positive Δt within configured gap limit (s) | 0.033333 | 0.000001 | 0.033333–0.033334 | 0.033333–0.033334 |
| Actual valid 60-sample window duration (s) | 1.966667 | 0.000001 | 1.966666–1.966667 | 1.966666–1.966668 |

A 60-sample window spans 59 timestamp intervals. The reported window distribution is measured directly for every accepted train/validation/test window.
