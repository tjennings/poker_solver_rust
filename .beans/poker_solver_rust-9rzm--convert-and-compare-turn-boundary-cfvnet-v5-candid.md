---
# poker_solver_rust-9rzm
title: Convert and compare turn_boundary_cfvnet_v5 candidate
status: completed
type: task
priority: normal
created_at: 2026-05-14T04:00:43Z
updated_at: 2026-05-14T04:04:35Z
parent: poker_solver_rust-8e9f
---

Convert local_data/models/turn_boundary_cfvnet_v5 best candidate to ONNX and run compare-solve boundary CFV gate on the known diagnostic turn spot.\n\n- [x] Identify export command and output path\n- [x] Convert best.pt to ONNX\n- [x] Run compare-solve diagnostic gate\n- [x] Report metrics and pass/fail result

## Summary of Changes

Converted local_data/models/turn_boundary_cfvnet_v5/best.pt to local_data/models/turn_boundary_cfvnet_v5/model.onnx plus model.onnx.data using the existing Python BoundaryNet exporter. The checkpoint was epoch 225 with val_huber=1.5054589221108472e-05 and ONNXRuntime verification passed.

Ran compare-solve on the diagnostic turn spot with the v5 ONNX artifact. Canonical direct mode failed the boundary gate: aggregate OOP mean_abs=147.780289, IP mean_abs=154.519104, OOP corr=0.8743, IP corr=0.9292. direct_normalized_legacy also failed: aggregate OOP mean_abs=73.951195, IP mean_abs=77.369743, OOP corr=0.8743, IP corr=0.9292. exact_subtree raw control stayed healthy at OOP mean_abs=0.013713 corr=0.9965 and IP mean_abs=0.026847 corr=0.9946.

Conclusion: v5 best.pt converts successfully, but its output magnitude is incompatible with the current boundary evaluator contracts on this spot. It should not be promoted for policy decisions.
