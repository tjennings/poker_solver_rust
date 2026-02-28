---
# poker_solver_rust-kfnw
title: Add LCFR iteration weighting to exhaustive postflop solver
status: completed
type: bug
priority: normal
created_at: 2026-02-28T18:01:42Z
updated_at: 2026-02-28T18:39:52Z
---

The exhaustive postflop solver uses plain vanilla CFR with no iteration weighting. The MCCFR backend uses LCFR (linear weighting by iteration number), and the preflop solver has full DCFR support. Add LCFR weighting to the exhaustive solver to match the MCCFR backend and improve convergence.

## Plan

## Wave 1 — Foundation
- [x] Create `cfr/dcfr.rs` with `DcfrParams`, weights, discounting, flooring methods
- [x] Update `cfr/mod.rs` to export new module
- [x] Move `CfrVariant` from `preflop/config.rs` to `cfr/dcfr.rs`
- [x] Update `preflop/mod.rs` re-exports

## Wave 2 — Consumers (parallel)
- [x] Refactor `preflop/solver.rs` to use `DcfrParams`
- [x] Add LCFR/DCFR to `postflop_exhaustive.rs` + `cfr_variant` to `PostflopModelConfig`

## Wave 3 — Integration
- [x] Merge worktrees, full test suite + clippy
- [x] Idiomatic Rust review
- [x] Architecture review


## Summary of Changes

Extracted shared DCFR module (`cfr/dcfr.rs`) with `CfrVariant` enum and `DcfrParams` struct. Moved `CfrVariant` from `preflop/config.rs` to `cfr/dcfr.rs` with backward-compatible re-exports. Refactored `PreflopSolver` to use `DcfrParams` (5 inline fields → 1 struct). Added LCFR/DCFR iteration weighting and discounting to the exhaustive postflop solver via `DcfrParams`. Added `cfr_variant` field to `PostflopModelConfig` (default: `linear`). 16 new unit tests for `DcfrParams`.
