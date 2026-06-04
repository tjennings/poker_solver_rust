---
# poker_solver_rust-0lrl
title: Review Phase 2 trainer/TUI baseline integration
status: in-progress
type: task
priority: high
created_at: 2026-06-04T04:04:01Z
updated_at: 2026-06-04T04:04:01Z
parent: poker_solver_rust-l6r9
---

Independent review of implementation commit `3f7ee811 Integrate blueprint baseline validation` for Phase 2 slice `poker_solver_rust-nobl`.

Review focus:
- Baseline validation config defaults disabled and enabled config parses as intended.
- Trainer loads baseline and computes reports on cadence against `active_storage()` without dense projection.
- `BaselineGamePreconditions` are filled from actual original `GameConfig` values, not fabricated pinned constants.
- Wrong config/precondition cases are rejected and covered by tests.
- No-TUI output and TUI metrics/rendering include aggregate TV, root/first-response/worst spot TV, coverage, skipped/invalid/unsupported counts, and top 5 worst spots with diagnostic data.
- Sample 20bb validation config is stack/action equivalent to `cash_hu_20bb_cev.json` and keeps pruning/eviction disabled.
- Docs accurately describe config, command, metrics, limitations, and strategy-frequency-only validation.
- Default test suite stays under one minute.

Reviewer should report blockers with file/line references and recommend whether Phase 2 can be closed or needs an integration fix.
