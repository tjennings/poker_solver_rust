---
# poker_solver_rust-lr0y
title: Restore complete test suite runtime below one minute
status: in-progress
type: bug
priority: high
created_at: 2026-08-04T12:59:50Z
updated_at: 2026-08-04T12:59:50Z
---

The required baseline cargo test run exceeded one minute and was terminated at 61.20 seconds while still compiling. Determine whether this is a cold-cache artifact; if the warmed complete suite still exceeds one minute, identify and fix the bottleneck before resuming feature work.

- [ ] Run a warmed complete cargo test suite and measure wall time
- [ ] Diagnose any remaining runtime above one minute
- [ ] Implement and review a fix if required
- [ ] Confirm the complete suite passes in under one minute
- [ ] Summarize the outcome
