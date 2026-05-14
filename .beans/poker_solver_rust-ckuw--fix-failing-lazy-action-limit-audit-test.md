---
# poker_solver_rust-ckuw
title: Fix failing lazy action-limit audit test
status: completed
type: bug
priority: critical
created_at: 2026-05-14T14:40:57Z
updated_at: 2026-05-14T14:46:18Z
---

Pre-code cargo test on branch codex/negative-action-purge-config failed before negative-action purge implementation.

Failing test:
- blueprint_mp::lazy_mccfr::tests::lazy_action_limit_audit_allows_one_all_in_aggression_past_raise_rows
- Location: crates/core/src/blueprint_mp/lazy_mccfr.rs around line 1585/1586
- Observed: assertion expected nonzero action-limit audit counts, but got zero.

Acceptance criteria:
- [x] Identify whether the test expectation or action-limit audit behavior is stale.
- [x] Fix the failing test or implementation without changing unrelated solver semantics.
- [x] Re-run the focused failing test.
- [x] Re-run the full suite gate or document remaining speed issue.

## Summary of Changes

Added a test-only mutex guard around lazy action-limit audit counter reset, action generation, and snapshot reads in the two audit tests. This preserves production solver/action semantics while preventing the two resettable global-counter tests from racing each other under parallel test execution.

Verification passed:
- cargo fmt --check
- cargo test -p poker-solver-core blueprint_mp::lazy_mccfr::tests::lazy_action_limit_audit -- --nocapture
- cargo test -p poker-solver-core --lib blueprint_mp::lazy_mccfr::tests::lazy_action_limit_audit -- --nocapture
- cargo test
