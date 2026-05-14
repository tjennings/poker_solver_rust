---
# poker_solver_rust-ekg1
title: Compare previous and latest CFVNet v5 exports
status: completed
type: task
priority: normal
created_at: 2026-05-14T04:37:35Z
updated_at: 2026-05-14T04:38:33Z
---

Run the original compare-solve diagnostic on the previous v5 `model.onnx` and compare it with the latest `best.onnx` run.

## Summary of Changes

Re-ran the original compare-solve diagnostic on `local_data/models/turn_boundary_cfvnet_v5/model.onnx`, the previous export from before the latest `best.pt` export.

Previous `model.onnx` aggregate:
- OOP mean_abs `0.383081`, rmse `0.392524`, corr `0.7813`, mag_ratio `1.627`
- IP mean_abs `0.832239`, rmse `0.851982`, corr `0.8924`, mag_ratio `1.990`

Latest `best.onnx` aggregate from the epoch 391 `best.pt` export:
- OOP mean_abs `0.391610`, rmse `0.401388`, corr `0.7697`, mag_ratio `1.673`
- IP mean_abs `0.837361`, rmse `0.855863`, corr `0.8992`, mag_ratio `2.007`

Conclusion: additional training did not improve the compare-solve gate. Mean_abs worsened by `0.008529` OOP and `0.005122` IP, while IP correlation improved slightly from `0.8924` to `0.8992`. Both exports still fail the max_mean_abs `0.25` gate.
