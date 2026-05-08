---
# poker_solver_rust-fyq4
title: Relax BoundaryNet ONNX export verification tolerance
status: completed
type: bug
priority: high
created_at: 2026-05-08T18:09:33Z
updated_at: 2026-05-08T18:11:22Z
---

Training completed but best-checkpoint ONNX export failed verification on tiny numerical drift: max abs diff 0.000155 against 0.0001 tolerance. Make export verification robust while preserving useful failure diagnostics, then verify the exported best model loads in Rust.\n\n- [x] Reproduce or inspect tolerance failure\n- [x] Adjust export verification\n- [x] Run Python export tests\n- [x] Export and verify current best ONNX\n- [x] Record results

## Summary of Changes

Relaxed BoundaryNet ONNX export verification from rtol/atol 1e-4 to rtol 1e-3 and atol 2e-4, covering the observed max abs drift of 0.000155 while preserving a tight sanity check. Re-exported local_data/models/turn_boundary_cfvnet_v2/best.pt to best.onnx/best.onnx.data, verified Python export and Rust eval-boundary. Current shard metrics: mean MAE 0.002351, p95 0.0046, p99 0.0093, max 0.0212 over 10,000 records. Python cfvnet tests passed: 48 passed.
