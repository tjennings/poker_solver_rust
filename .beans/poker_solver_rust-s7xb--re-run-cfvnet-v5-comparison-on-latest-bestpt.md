---
# poker_solver_rust-s7xb
title: Re-run CFVNet v5 comparison on latest best.pt
status: completed
type: task
priority: normal
created_at: 2026-05-14T04:30:23Z
updated_at: 2026-05-14T04:31:32Z
---

Export the latest v5 `best.pt` artifact if needed and re-run the original compare-solve diagnostic against it.

## Summary of Changes

Exported latest v5 `best.pt` to `local_data/models/turn_boundary_cfvnet_v5/best.onnx` and re-ran the original compare-solve diagnostic against that export. The checkpoint reports epoch 391 with val_huber `1.3860028411727399e-05`.

Result: boundary CFV gate still fails. Aggregate metrics were OOP mean_abs `0.391610`, rmse `0.401388`, corr `0.7697`, mag_ratio `1.673`; IP mean_abs `0.837361`, rmse `0.855863`, corr `0.8992`, mag_ratio `2.007`. Exact-subtree raw control remained clean at OOP mean_abs `0.013713` corr `0.9965` and IP mean_abs `0.026847` corr `0.9946`.

Conclusion: latest `best.pt` does not fix the comparison failure. Runtime magnitude is no longer exploding; remaining failure is model/target quality or target-unit alignment.
