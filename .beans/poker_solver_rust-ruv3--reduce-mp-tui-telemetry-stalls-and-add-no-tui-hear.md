---
# poker_solver_rust-ruv3
title: Reduce MP TUI telemetry stalls and add no-TUI heartbeat
status: in-progress
type: bug
priority: high
created_at: 2026-05-07T04:42:25Z
updated_at: 2026-05-07T04:42:25Z
---

Large 6-max MP action abstractions show long pauses under the TUI while --no-tui keeps CPU fully utilized. Add backpressure so expensive telemetry scans do not overlap, and add periodic no-TUI progress output so throughput can be compared without the dashboard.\n\n- [ ] Add MP no-TUI heartbeat/progress output\n- [ ] Prevent overlapping MP TUI telemetry scans\n- [ ] Verify focused trainer tests
