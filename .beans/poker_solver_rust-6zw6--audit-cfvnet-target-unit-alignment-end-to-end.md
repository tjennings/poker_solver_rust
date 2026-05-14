---
# poker_solver_rust-6zw6
title: Audit CFVNet target-unit alignment end to end
status: in-progress
type: bug
priority: high
created_at: 2026-05-14T04:46:09Z
updated_at: 2026-05-14T04:46:09Z
parent: poker_solver_rust-lnpl
---

Trace a small set of boundary records through Rust datagen, Python loading, model target scaling, ONNX output interpretation, Rust evaluator conversion, and compare-solve oracle comparison. Prove the same value unit is used at every boundary before more large training runs.

## Notes

Kickoff audit finding: local Python training still encodes targets as `cfv * pot / (pot + effective_stack)` in `crates/cfvnet/python/cfvnet/data.py`, while Rust's current `BoundaryInferenceMode::Direct` expects solver-native BCFV output. The existing docs already name this as the `direct_normalized_legacy` contract for current Python-exported checkpoints.

Re-ran the v5 spot with the contract-compatible `direct_normalized_legacy` runtime mode:

- Previous `model.onnx`: OOP mean_abs `0.262599`, rmse `0.277203`, corr `0.7813`, mag_ratio `0.813`; IP mean_abs `0.545222`, rmse `0.575535`, corr `0.8924`, mag_ratio `0.995`.
- Latest epoch 391 `best.onnx`: OOP mean_abs `0.266676`, rmse `0.281418`, corr `0.7697`, mag_ratio `0.837`; IP mean_abs `0.547883`, rmse `0.578119`, corr `0.8992`, mag_ratio `1.003`.

This is much better than the incorrect `direct` runs, but still fails max_mean_abs `0.25`, especially IP. Additional training still did not improve this spot: latest `best.onnx` is slightly worse than previous `model.onnx` on mean_abs under the correct legacy-normalized contract.
