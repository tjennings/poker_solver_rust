---
# poker_solver_rust-o6oc
title: Add Direct CFVNet legacy normalized conversion
status: completed
type: bug
priority: high
created_at: 2026-05-08T20:20:35Z
updated_at: 2026-05-08T20:31:39Z
---

Current Direct CFVNet checkpoint outputs normalized chip_ev/(pot+stack) values. Add the temporary conversion back to solver-native bcfv at inference so the current local model can be used while preserving clear documentation of the compatibility tradeoff.

- [x] Verify baseline full test suite is clean and under one minute
- [x] Patch Direct evaluator conversion for current normalized checkpoint
- [x] Update tests and docs for temporary Direct compatibility
- [x] Run focused and full verification
- [x] Commit code and bean

## Summary of Changes

Added `direct_normalized_legacy` compatibility mode. It converts current Direct checkpoint outputs from chip_ev/(pot+stack) to bcfv using `2*y*(pot+stack)/pot - 1`; UI Direct CFVNet now sends this mode; `direct` remains native bcfv for future checkpoints.
