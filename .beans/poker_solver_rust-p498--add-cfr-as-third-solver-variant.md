---
# poker_solver_rust-p498
title: Add CFR+ as third solver variant
status: completed
type: feature
priority: normal
created_at: 2026-02-25T23:21:10Z
updated_at: 2026-02-25T23:29:39Z
---

Add CFR+ (Tammelin 2014) as a third preflop solver variant alongside Vanilla and DCFR. CFR+ floors negative regrets to zero each iteration and uses linear strategy weighting.

## Tasks
- [x] Add CfrPlus variant to CfrVariant enum in config.rs
- [x] Add deserialization test for cfrplus
- [x] Update solver.rs: linear strategy weighting for CfrPlus
- [x] Update solver.rs: floor regrets after accumulation
- [x] Update solver.rs: skip DCFR discounting for CfrPlus
- [x] Update solver.rs: avg_positive_regret uses cumulative approach
- [x] Add unit tests (floor regrets, valid strategy, no discounting)
- [x] Add integration test (avg_positive_regret convergence)
- [x] Update sample config comment
- [x] Verify all tests pass and clippy clean


## Summary of Changes

Added CFR+ (Tammelin 2014) as a third preflop solver variant:
- `CfrVariant::CfrPlus` enum variant with serde `cfrplus`
- Regrets floored to zero after each iteration via `floor_regrets()`
- Uniform regret weighting (weight=1) + linear strategy weighting (weight=T)
- DCFR discounting skipped for CFR+
- `avg_positive_regret()` uses cumulative approach (like Vanilla)
- 4 unit tests + 1 integration convergence test
- Updated sample config docs
