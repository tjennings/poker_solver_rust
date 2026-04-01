---
# poker_solver_rust-d3zl
title: Eliminate per-sample model cloning in turn datagen
status: completed
type: task
priority: normal
created_at: 2026-04-01T13:30:05Z
updated_at: 2026-04-01T13:39:21Z
---

## Summary of Changes

Separated boundary evaluation from parallel solving in the NdArray turn datagen path. The neural network model is now loaded once and used sequentially for boundary evaluation (Phase 1), then games are solved in parallel without any model dependency (Phase 2). This eliminates the wasteful per-sample `model.clone_inner()` call.

### Changes:
- Extracted boundary evaluation logic into `evaluate_game_boundaries()` function
- Refactored `solve_turn_situation()` into a thin wrapper (test-only) calling build + evaluate + solve
- Updated NdArray main loop to two-phase pipeline: sequential boundary eval, parallel solve
- Removed `SyncModel` wrapper (no longer needed since model is not shared across threads)
- Removed `#[allow(dead_code)]` from `build_turn_game` and `solve_and_extract` (now used in production)
- Added 2 new tests: `evaluate_game_boundaries_sets_cfvs` and `two_phase_matches_monolithic`

All 113 cfvnet tests pass, 0 warnings.
