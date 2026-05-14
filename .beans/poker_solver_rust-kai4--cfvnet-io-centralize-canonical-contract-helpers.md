---
# poker_solver_rust-kai4
title: 'CFVNet IO: centralize canonical contract helpers'
status: in-progress
type: task
priority: high
created_at: 2026-05-14T01:10:31Z
updated_at: 2026-05-14T01:18:29Z
parent: poker_solver_rust-8e9f
---

Add explicit helper APIs for the BoundaryNet IO contract.\n\n- [ ] Add range finite/non-negative/blocker validation helpers\n- [ ] Add normalize-after-blockers helper for 1326 canonical ranges\n- [ ] Add model-output conversion helpers for chip EV and legacy half-pot bcfv\n- [ ] Add small unit tests for conversion math\n\nPrimary file: crates/cfvnet/src/eval/boundary_evaluator.rs
