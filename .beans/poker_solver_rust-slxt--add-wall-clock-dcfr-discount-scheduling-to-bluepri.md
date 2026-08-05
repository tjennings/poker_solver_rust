---
# poker_solver_rust-slxt
title: Add wall-clock DCFR discount scheduling to blueprint_mp
status: in-progress
type: feature
priority: high
created_at: 2026-08-05T14:24:48Z
updated_at: 2026-08-05T14:24:48Z
---

Implement the approved time-based discount interval design for blueprint_mp.

- [ ] Verify algorithm and architecture decisions
- [ ] Establish clean baseline with full test suite under one minute
- [ ] Implement config, scheduler, checkpoint behavior, telemetry, and docs
- [ ] Add deterministic fake-clock and compatibility tests
- [ ] Complete independent code review and repairs
- [ ] Run the entire test suite under one minute
- [ ] Integrate an atomic implementation commit

Reference: poker_solver_rust-ft2v development handoff and poker_solver_rust-qh5t literature research.
