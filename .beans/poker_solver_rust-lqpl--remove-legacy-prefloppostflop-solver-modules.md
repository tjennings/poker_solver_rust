---
# poker_solver_rust-lqpl
title: Remove legacy preflop/postflop solver modules
status: completed
type: epic
priority: normal
created_at: 2026-03-11T15:42:00Z
updated_at: 2026-03-11T16:21:30Z
---

Remove the old LCFR preflop solver (preflop/) and old subgame-based postflop solver (blueprint/) along with all CLI commands, tests, benchmarks, configs, and documentation. Keep blueprint_v2/, abstraction/, Tauri UX, cfvnet, and simulation (agents + v2 blueprints) functional.

Plan: docs/plans/2026-03-11-remove-legacy-solvers.md

## Work Streams (parallel worktrees)

- [x] Stream 1: Delete tests + benchmarks (Task 1)
- [x] Stream 2: Delete sample configs (Task 2)
- [x] Stream 3: Trainer cleanup - delete support files + remove old CLI commands (Tasks 3-4)
- [x] Stream 4: Core cleanup - delete preflop/ + blueprint/, refactor simulation.rs + info_key.rs (Tasks 5-8)
- [x] Stream 5: Tauri cleanup - refactor exploration.rs + simulation.rs + lib.rs (Tasks 9-10)
- [x] Stream 6: Documentation updates (Tasks 13-14)

## Post-merge

- [x] Clean up Cargo dependencies (Task 11)
- [x] Run full test suite + clippy (Task 12)
- [x] Final verification (Task 15)

## Summary of Changes

Removed ~22,000 lines of legacy code:
- Deleted crates/core/src/preflop/ (15 files, LCFR preflop solver)
- Deleted crates/core/src/blueprint/ (12 files, old subgame postflop solver)
- Removed 12 CLI commands from trainer (solve-preflop, solve-postflop, diag-buckets, etc.)
- Deleted 6 trainer support modules (tui, diagnostics, lhe_viz, etc.)
- Deleted 8 test/bench files, 14 sample configs
- Removed 8 dead Cargo dependencies (sled, lru, zstd, tracing, dashmap, etc.)
- Refactored simulation.rs (removed BlueprintAgent, RealTimeSolvingAgent)
- Refactored info_key.rs (inlined AbstractionMode enum)
- Refactored Tauri exploration.rs (removed Bundle/PreflopSolve/SubgameSolve sources)
- Updated docs (training, architecture, explorer, CLAUDE.md, MEMORY.md)
- Moved cbv.rs and cbv_compute.rs to blueprint_v2/
- Added poker::parse_card utility

768 tests pass, 0 failures, clippy clean (no new warnings).
