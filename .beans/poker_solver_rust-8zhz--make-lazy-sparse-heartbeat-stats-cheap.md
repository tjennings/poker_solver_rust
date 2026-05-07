---
# poker_solver_rust-8zhz
title: Make lazy sparse heartbeat stats cheap
status: in-progress
type: bug
priority: high
created_at: 2026-05-07T18:58:27Z
updated_at: 2026-05-07T18:58:27Z
---

Lazy sparse MP no-TUI heartbeat currently calls SparseMpStorage::stats(), which walks every sparse node to count slots. At hundreds of millions of entries this telemetry blocks training for 60-130s and causes a major IPS collapse.\n\nTasks:\n- [ ] Add exact live sparse storage counters for entries, slots, and per-shard entry counts.\n- [ ] Change heartbeat-facing stats/entry_count to read counters instead of scanning all nodes.\n- [ ] Preserve snapshot/load behavior and stats correctness in tests.\n- [ ] Update training docs to clarify cheap live sparse telemetry.\n- [ ] Run focused sparse/lazy tests.
