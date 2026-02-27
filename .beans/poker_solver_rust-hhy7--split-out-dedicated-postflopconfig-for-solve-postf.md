---
# poker_solver_rust-hhy7
title: Split out dedicated PostflopConfig for solve-postflop
status: completed
type: task
priority: normal
created_at: 2026-02-27T18:19:25Z
updated_at: 2026-02-27T18:23:27Z
---

solve-postflop currently deserializes PreflopTrainingConfig even though it only needs postflop_model. Add a thin PostflopSolveConfig wrapper and simplify minimal_postflop.yaml.

- [x] Add PostflopSolveConfig struct in trainer main.rs
- [x] Update run_solve_postflop to use PostflopSolveConfig
- [x] Simplify minimal_postflop.yaml to remove unused preflop boilerplate
- [x] Verify: cargo test, cargo clippy, config parsing

## Summary of Changes

Added `PostflopSolveConfig` wrapper struct in trainer main.rs so `solve-postflop` deserializes only the `postflop_model` field instead of the full `PreflopTrainingConfig`. Simplified `minimal_postflop.yaml` to remove unused game structure boilerplate (positions, blinds, antes, stacks). All 34 trainer tests pass, no new clippy warnings, and the config parses and solves correctly end-to-end.
