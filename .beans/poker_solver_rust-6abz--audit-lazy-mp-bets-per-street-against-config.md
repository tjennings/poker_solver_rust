---
# poker_solver_rust-6abz
title: Audit lazy MP bets per street against config
status: completed
type: bug
priority: high
created_at: 2026-05-08T01:42:37Z
updated_at: 2026-05-08T01:49:27Z
---

Lazy sparse training still inserts many low-SPR river infosets. Audit whether observed per-street raise counts exceed configured raise rows, including all-in aggression paths that may bypass sized raise limits.

Tasks:
- [x] Add lazy action-limit telemetry for per-street max raise count and over-config decision/action counts.
- [x] Report the audit fields in lazy_sparse no-TUI heartbeat.
- [x] Add focused tests showing all-in aggression can be observed past sized raise rows or is correctly counted.
- [x] Update training docs with the new action-limit audit fields.
- [x] Run focused lazy/trainer tests.

## Summary of Changes

- Added lazy action-limit telemetry for max per-street raise counts, over-cap decisions/actions, and aggressive all-ins.
- Reported the audit in no-TUI lazy_sparse heartbeats as action_limit[max=..., over_dec=..., over_aggr=..., allin_aggr=...].
- Treated configured raise rows plus one all-in aggression as allowed, and suppressed further aggressive all-ins after that cap.
- Documented the heartbeat field and covered the cap behavior with focused lazy tests.

Focused checks passed:
- cargo test -q -p poker-solver-core lazy_
- cargo test -q -p poker-solver-trainer lazy_sparse
- git diff --check -- crates/core/src/blueprint_mp/lazy_mccfr.rs crates/trainer/src/main.rs docs/training.md
