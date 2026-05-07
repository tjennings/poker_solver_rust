---
# poker_solver_rust-ruv3
title: Reduce MP TUI telemetry stalls and add no-TUI heartbeat
status: completed
type: bug
priority: high
created_at: 2026-05-07T04:42:25Z
updated_at: 2026-05-07T04:46:15Z
---

Large 6-max MP action abstractions show long pauses under the TUI while --no-tui keeps CPU fully utilized. Add backpressure so expensive telemetry scans do not overlap, and add periodic no-TUI progress output so throughput can be compared without the dashboard.\n\n- [x] Add MP no-TUI heartbeat/progress output\n- [x] Prevent overlapping MP TUI telemetry scans\n- [x] Verify focused trainer tests

## Summary of Changes

- Added a no-TUI MP heartbeat that prints iterations, interval/average iters per second, elapsed time, sampled regret stats, and prune percentage every 10 seconds.
- Added backpressure to the MP TUI telemetry bridge so expensive full-storage telemetry scans cannot overlap on large action abstractions.
- Kept no-TUI regret reporting sampled to avoid turning progress output into another full-memory scan.

## Verification

- cargo test -q -p poker-solver-trainer sample_mp_regret_summary_reports_scaled_stats
- cargo test -q -p poker-solver-trainer mp_snapshot_save_creates_strategy_and_metadata
- cargo test -q -p poker-solver-trainer train_blueprint_mp_no_tui_cli_parses
