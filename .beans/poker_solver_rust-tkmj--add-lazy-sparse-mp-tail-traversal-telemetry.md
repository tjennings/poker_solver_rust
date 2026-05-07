---
# poker_solver_rust-tkmj
title: Add lazy sparse MP tail traversal telemetry
status: completed
type: task
priority: high
created_at: 2026-05-07T17:11:08Z
updated_at: 2026-05-07T17:14:34Z
---

Add telemetry/logging for lazy_sparse MP long-tail compute pauses before changing scheduling: track max per-deal Rayon job time, max single traverser time, slow event counts, and identifying context in the no-TUI heartbeat.\n\n- [x] Add tail timing counters and context\n- [x] Print tail fields in lazy no-TUI heartbeat\n- [x] Add focused regression coverage\n- [x] Run focused verification

## Summary of Changes

Added lazy_sparse MP long-tail telemetry for the maximum per-deal Rayon job, maximum single traverser, slow job count, slow traverser count, traversal count, and identifying iteration/seat context. The lazy no-TUI heartbeat now prints a tail[...] section next to aggregate timings, letting us confirm whether one long traversal is holding the batch barrier before changing scheduling. Updated training docs to describe the tail fields.

Verification: cargo test -q -p poker-solver-core lazy_timing_snapshot_tracks_compute_components; cargo test -q -p poker-solver-core lazy_; cargo test -q -p poker-solver-trainer lazy_sparse
