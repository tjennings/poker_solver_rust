---
# poker_solver_rust-tkmj
title: Add lazy sparse MP tail traversal telemetry
status: in-progress
type: task
priority: high
created_at: 2026-05-07T17:11:08Z
updated_at: 2026-05-07T17:11:08Z
---

Add telemetry/logging for lazy_sparse MP long-tail compute pauses before changing scheduling: track max per-deal Rayon job time, max single traverser time, slow event counts, and identifying context in the no-TUI heartbeat.\n\n- [ ] Add tail timing counters and context\n- [ ] Print tail fields in lazy no-TUI heartbeat\n- [ ] Add focused regression coverage\n- [ ] Run focused verification
