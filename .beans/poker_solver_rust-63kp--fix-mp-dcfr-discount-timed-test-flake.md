---
# poker_solver_rust-63kp
title: Fix MP DCFR discount timed test flake
status: completed
type: bug
priority: high
created_at: 2026-05-19T14:16:57Z
updated_at: 2026-05-19T14:28:25Z
---

The post-change full-suite verification failed because blueprint_mp::trainer::tests::dcfr_discount_reduces_strategy_sums exceeded its 1s timed-test limit at 1.193s. Investigate and repair the test/runtime so the full suite passes and remains under the one-minute target.\n\n- [x] Commit this blocking test bean\n- [x] Dispatch Rust worker to repair the timed test flake\n- [x] Integrate the fix\n- [x] Re-run the full test suite under one minute

## Summary of Changes\n\nStabilized MP DCFR discounting for small toy storages by using a serial discount path up to 4096 slots while preserving the existing Rayon path for larger storages. Added focused threshold coverage at 4096 and 4097 slots. Verified targeted DCFR tests and a final full cargo test pass in 54.47s.
