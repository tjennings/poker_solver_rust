---
# poker_solver_rust-0bs7
title: Fix HU blueprint snapshot interval and latest-resume behavior
status: completed
type: bug
priority: high
created_at: 2026-06-09T13:03:45Z
updated_at: 2026-06-09T13:25:51Z
---

The HU blueprint trainer appears not to save snapshots on the configured snapshot interval and/or not resume from the latest available snapshot when snapshots.resume is enabled. Investigate snapshot scheduling, warmup handling, output directory layout, latest snapshot discovery, resume semantics, and logging/TUI visibility. Acceptance: a small regression test proves periodic snapshots are emitted when configured, latest snapshot selection is deterministic, resume loads the newest valid snapshot, and docs/config examples make the behavior clear.

## Summary of Changes

Investigated the configured HU baseline validation output directory and found existing snapshots on disk at `local_data/blueprints/hu_20bb_baseline_validation`: `snapshot_0000` through `snapshot_0011`. The latest numbered snapshot has metadata `iteration=2961501400` and `elapsed_minutes=140`. The edited sample config was not resuming because `snapshots.resume` was still `false`.

Fixes delivered:

- Updated the HU sample config to keep the long-run/snapshot cadence changes and set `snapshots.resume: true`, so it resumes the existing output directory instead of starting over.
- Changed HU resume selection to scan valid numbered snapshots and `final/`, requiring `regrets.bin` plus readable `metadata.json` with both `iteration` and `elapsed_minutes`. Metadata-missing checkpoints are skipped instead of resuming at iteration 0.
- Changed checkpoint ordering to pick the newest valid checkpoint by metadata iteration, then elapsed minutes, then final checkpoint status, then numbered snapshot index. This prevents stale `final/` from masking a newer numbered snapshot while still allowing an equal-or-newer `final/` to win.
- Restored `last_snapshot_time` from resumed metadata so timed snapshots wait one configured interval after the loaded checkpoint instead of firing immediately after resume.
- Kept `snapshot_count` aligned so resuming from `snapshot_NNNN` writes `snapshot_NNNN+1`, and resuming from `final/` writes after the highest valid numbered snapshot.
- Seeded HU TUI metrics with `trainer.iterations` after the TUI metrics atomic replaces the trainer atomic, so the dashboard does not show zero after resume.
- Updated training docs to describe latest valid checkpoint ordering and metadata requirements.
- Added focused resume tests for stale `final/`, final tie/newer ordering, selected numbered snapshot counts, restored snapshot time, and metadata-missing snapshots.

Verification:

- Preflight full suite passed; initial warm gate before implementation was `real 43.03`.
- `cargo test -p poker-solver-core resume --quiet` passed: 9 tests.
- `cargo test -p poker-solver-trainer --no-run --quiet` passed.
- `git diff --check` passed.
- Full suite passed after changes. The first post-change run was compile-warmup-heavy at `real 110.61`; immediate warm confirmation passed under the gate at `real 44.28`.

Review:

- Snapshot/resume research identified stale `final/`, resume cadence, and HU TUI iteration seeding issues.
- Implementation review found no blockers; two edge semantics were tightened before close: final tie ordering and metadata-missing checkpoints.
