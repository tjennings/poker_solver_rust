---
# poker_solver_rust-q0zk
title: Add lazy sparse storage activity telemetry
status: in-progress
type: task
priority: high
created_at: 2026-05-08T00:41:24Z
updated_at: 2026-05-08T00:41:24Z
---

Training throughput collapses after sparse storage reaches hundreds of millions of entries even with cheap stats. Add lightweight telemetry to separate allocation pressure from repeated lookup pressure and hit/miss behavior.\n\nTasks:\n- [ ] Add sparse storage activity counters for read probes/hits, write probes/hits, and inserts.\n- [ ] Report per-heartbeat activity deltas/rates in no-TUI lazy_sparse output.\n- [ ] Add focused tests for counter behavior and heartbeat formatting if appropriate.\n- [ ] Update training docs with the new telemetry fields.\n- [ ] Run focused sparse/lazy tests.
