---
# poker_solver_rust-xso3
title: Collapse low-SPR river action explosion
status: completed
type: bug
priority: high
created_at: 2026-05-08T01:21:06Z
updated_at: 2026-05-08T01:24:36Z
---

Lazy sparse insert attribution shows sustained new infosets concentrated on river, SPR bucket 0, history length 16-31, mostly 2-action nodes. Investigate and collapse low-SPR river betting actions that are creating leak-like state growth.

Tasks:
- [x] Inspect lazy MP action generation around low-SPR river states.
- [x] Add a conservative collapse/guard for river SPR bucket 0 action generation.
- [x] Add focused lazy action tests covering low-SPR river behavior.
- [x] Update training docs if the lazy_sparse backend behavior changes.
- [x] Run focused lazy tests.

## Summary of Changes

Lazy MP action generation now suppresses new lead/raise/all-in aggression on river SPR-0 states while preserving check, fold, call, and all-in-call resolution. Added focused lazy action tests for unopened river SPR-0, facing a bet with chips behind, and facing an all-in call. Updated training docs to document the lazy_sparse low-SPR river collapse.

Focused tests passed:
- cargo test -q -p poker-solver-core lazy_river_spr_zero
- cargo test -q -p poker-solver-core lazy_
- cargo test -q -p poker-solver-trainer lazy_sparse
