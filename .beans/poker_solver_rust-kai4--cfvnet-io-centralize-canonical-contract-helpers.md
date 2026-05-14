---
# poker_solver_rust-kai4
title: 'CFVNet IO: centralize canonical contract helpers'
status: completed
type: task
priority: high
created_at: 2026-05-14T01:10:31Z
updated_at: 2026-05-14T01:21:30Z
parent: poker_solver_rust-8e9f
---

Add explicit helper APIs for the BoundaryNet IO contract.\n\n- [x] Add range finite/non-negative/blocker validation helpers\n- [x] Add normalize-after-blockers helper for 1326 canonical ranges\n- [x] Add model-output conversion helpers for chip EV and legacy half-pot bcfv\n- [x] Add small unit tests for conversion math\n\nPrimary file: crates/cfvnet/src/eval/boundary_evaluator.rs

## Summary of Changes

Implemented BoundaryNet IO contract helpers in `crates/cfvnet/src/eval/boundary_evaluator.rs`: canonical 1326 range sanitization/normalization after blockers, normalized-to-chip EV conversion, legacy half-pot BCFV conversion, and focused unit tests. Targeted `cargo test -p cfvnet boundary_evaluator` passes.
