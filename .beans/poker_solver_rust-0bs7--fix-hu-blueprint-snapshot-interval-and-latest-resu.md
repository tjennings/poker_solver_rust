---
# poker_solver_rust-0bs7
title: Fix HU blueprint snapshot interval and latest-resume behavior
status: in-progress
type: bug
priority: high
created_at: 2026-06-09T13:03:45Z
updated_at: 2026-06-09T13:03:45Z
---

The HU blueprint trainer appears not to save snapshots on the configured snapshot interval and/or not resume from the latest available snapshot when snapshots.resume is enabled. Investigate snapshot scheduling, warmup handling, output directory layout, latest snapshot discovery, resume semantics, and logging/TUI visibility. Acceptance: a small regression test proves periodic snapshots are emitted when configured, latest snapshot selection is deterministic, resume loads the newest valid snapshot, and docs/config examples make the behavior clear.
