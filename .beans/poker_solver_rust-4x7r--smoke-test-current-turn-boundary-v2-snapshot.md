---
# poker_solver_rust-4x7r
title: Smoke-test current turn-boundary v2 snapshot
status: completed
type: task
priority: high
created_at: 2026-05-08T01:16:46Z
updated_at: 2026-05-08T01:22:48Z
---

Evaluate/export the current local turn-boundary CFVNet v2 snapshot and run a small compare path if available.\n\n- [x] Locate current checkpoint and validation data\n- [x] Run a boundary evaluation smoke test\n- [x] Export/verify ONNX snapshot if needed for Rust harness\n- [x] Run small Rust compare/eval harness where feasible\n- [x] Record result

## Summary of Changes\n\nSmoke-tested local_data/models/turn_boundary_cfvnet_v2/best.pt. Python evaluator exported and verified best.onnx, then evaluated a_BVZnf_00001.bin with overall MAE 0.006398. Rust eval-boundary loaded best.onnx and matched the same metrics. A flop-root compare-solve with --turn-boundary cfvnet was blocked by the existing 3-card-board validation guard; a turn-root compare-solve using --river-boundary cfvnet --river-model-kind direct completed with 11 ONNX boundaries and boundary sanity checks passed, but early-snapshot exploitability was much worse than exact.
