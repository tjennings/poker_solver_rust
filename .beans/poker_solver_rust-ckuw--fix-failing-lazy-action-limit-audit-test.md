---
# poker_solver_rust-ckuw
title: Fix failing lazy action-limit audit test
status: in-progress
type: bug
priority: critical
created_at: 2026-05-14T14:40:57Z
updated_at: 2026-05-14T14:41:12Z
---

Pre-code cargo test on branch codex/negative-action-purge-config failed before negative-action purge implementation.

Failing test:
- blueprint_mp::lazy_mccfr::tests::lazy_action_limit_audit_allows_one_all_in_aggression_past_raise_rows
- Location: crates/core/src/blueprint_mp/lazy_mccfr.rs around line 1585/1586
- Observed: assertion expected nonzero action-limit audit counts, but got zero.

Acceptance criteria:
- [ ] Identify whether the test expectation or action-limit audit behavior is stale.
- [ ] Fix the failing test or implementation without changing unrelated solver semantics.
- [ ] Re-run the focused failing test.
- [ ] Re-run the full suite gate or document remaining speed issue.
