---
# poker_solver_rust-9aym
title: Instrument lazy sparse MP compute loop timings
status: in-progress
type: task
priority: high
created_at: 2026-05-07T17:05:30Z
updated_at: 2026-05-07T17:05:30Z
---

Add low-overhead timing telemetry for lazy_sparse MP training so CPU utilization pauses can be correlated with deal sampling, bucket assignment, traversal, discounting, and console stats collection.\n\n- [ ] Add timing counters for lazy compute-loop components\n- [ ] Print timing fields in lazy no-TUI heartbeat\n- [ ] Include console stats timing so reporting pauses can be confirmed or ruled out\n- [ ] Add focused regression coverage\n- [ ] Run focused verification
