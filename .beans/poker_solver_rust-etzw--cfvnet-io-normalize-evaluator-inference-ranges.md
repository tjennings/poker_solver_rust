---
# poker_solver_rust-etzw
title: 'CFVNet IO: normalize evaluator inference ranges'
status: completed
type: task
priority: high
created_at: 2026-05-14T01:10:39Z
updated_at: 2026-05-14T01:26:03Z
parent: poker_solver_rust-8e9f
blocked_by:
    - poker_solver_rust-kai4
---

Normalize runtime evaluator inputs to match training.\n\n- [x] Use canonical helper when mapping game-order reaches to 1326 ranges\n- [x] Zero board-blocked combos before model inference\n- [x] Renormalize ranges after river blocker adjustment in river-enumerated mode\n- [x] Apply to Burn and ONNX evaluator paths\n\nPrimary file: crates/cfvnet/src/eval/boundary_evaluator.rs

## Summary of Changes

Wired the canonical range sanitization helper into BoundaryNet inference. Burn and ONNX paths now sanitize and normalize mapped 1326 ranges against the public board before inference. River-enumerated mode renormalizes ranges after each candidate river blocker is zeroed. Zero-mass ranges remain deterministic all-zero inputs. Added focused tests for per-river normalization and zero-mass river batches. Verified with cargo test -p cfvnet boundary_evaluator.
