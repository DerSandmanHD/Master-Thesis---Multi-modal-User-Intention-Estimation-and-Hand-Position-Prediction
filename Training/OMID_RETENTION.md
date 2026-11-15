# Frozen historical-test retention evaluation

Authorized 2026-11-02 after the initial Omid adaptation comparison.

Evaluate unchanged, fine-tuned and scratch models, seeds 42/43/44, on exactly
the historical source-model test split. Assert original dataset content
contract, exact sequence lists and test endpoint fingerprints. Do not retrain,
select another checkpoint, or adjust preprocessing using test performance.
Use each model's saved normalizer, and verify checkpoint hashes before/after.

Two conditions: original inputs, and synthetic removal of the 54 object-marker
features. For the latter, geometric/gaze features become missing (normalized
value zero, observed mask zero); object-validity features are observed zero,
as in the new marker-free masters. Do not mask robot/hand features or targets.
Keep original windows even when object features are removed, to ensure pairing.

Report intention/hand macro-F1 and position errors. The primary position metric
uses predicted receiving hands on the fixed valid-target cohort with both
reference wrists, without an intention gate. Preserve complete oracle,
persistence and gated diagnostics in per-seed JSON. Check identical cohort
fingerprints across models, conditions and seeds.

The historical test has been observed before; this is a post-adaptation
retention check. Synthetic masking retains old MPS tracking and visible physical
markers. It is not equivalent to new native-tracking videos. Seed standard
deviation is not a participant confidence interval.

GPU array: 2388397. Outputs: `Data_collection/object_marker_free_v1/retention_v1`.
The automatic summary also includes the previously evaluated new-video test.
