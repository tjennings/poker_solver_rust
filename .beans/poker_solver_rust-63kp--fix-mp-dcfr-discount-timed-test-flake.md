---
# poker_solver_rust-63kp
title: Fix MP DCFR discount timed test flake
status: in-progress
type: bug
priority: high
created_at: 2026-05-19T14:16:57Z
updated_at: 2026-05-19T14:17:46Z
---

The post-change full-suite verification failed because blueprint_mp::trainer::tests::dcfr_discount_reduces_strategy_sums exceeded its 1s timed-test limit at 1.193s. Investigate and repair the test/runtime so the full suite passes and remains under the one-minute target.\n\n- [x] Commit this blocking test bean\n- [x] Dispatch Rust worker to repair the timed test flake\n- [ ] Integrate the fix\n- [ ] Re-run the full test suite under one minute
