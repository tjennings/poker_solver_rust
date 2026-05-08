---
# poker_solver_rust-1nu0
title: Add lazy sparse MP TUI support
status: completed
type: task
priority: high
created_at: 2026-05-08T02:04:22Z
updated_at: 2026-05-08T02:11:25Z
parent: poker_solver_rust-5kvv
---

Restore train-blueprint-mp TUI support when training.backend is lazy_sparse.

Tasks:
- [x] Remove the lazy_sparse --no-tui hard block and add a lazy TUI run path.
- [x] Bridge lazy sparse iterations, quit, snapshot trigger, and telemetry into MP TUI metrics.
- [x] Provide useful live strategy/regret updates from sparse storage without dense tree scans.
- [x] Update docs for lazy_sparse TUI behavior and limitations.
- [x] Run focused trainer/core tests.

## Summary of Changes

- Added a lazy_sparse MP TUI run path instead of requiring --no-tui.
- Bridged lazy sparse iterations, quit, snapshot hotkey, sampled regret telemetry, prune telemetry, and sampled strategy-delta movement into the MP TUI.
- Added bounded sparse telemetry sampling so TUI updates do not full-scan dense storage.
- Added lazy sparse snapshot saving for TUI hotkey snapshots via sparse_entries.bin plus metadata.json.
- Documented lazy_sparse TUI behavior and the current eager-only limitation for configured scenario hand grids.

Focused checks passed:
- cargo test -q -p poker-solver-core sparse_storage
- cargo test -q -p poker-solver-trainer lazy_sparse
- cargo test -q -p poker-solver-trainer mp_tui
- cargo test -q -p poker-solver-trainer sparse_mp_telemetry_pushes_regret_and_strategy_delta
- cargo test -q -p poker-solver-trainer lazy_mp_snapshot_save_creates_sparse_entries_and_metadata
- cargo test -q -p poker-solver-core telemetry_sample_reports_sparse_regret_and_strategy_movement_signal
- git diff --check -- crates/core/src/blueprint_mp/sparse_storage.rs crates/trainer/src/main.rs docs/training.md
