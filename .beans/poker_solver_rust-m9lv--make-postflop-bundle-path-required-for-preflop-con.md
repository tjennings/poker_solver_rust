---
# poker_solver_rust-m9lv
title: Make postflop bundle path required for preflop config
status: completed
type: task
priority: normal
created_at: 2026-02-27T18:27:53Z
updated_at: 2026-02-27T18:39:41Z
---

Remove inline postflop_model from PreflopConfig. Make postflop_model_path required in PreflopTrainingConfig. Update diag-buckets and trace-hand to use PostflopSolveConfig. Update minimal_preflop.yaml and docs/training.md.

- [x] Remove postflop_model from PreflopConfig (core)
- [x] Make postflop_model_path required in PreflopTrainingConfig (trainer)
- [x] Remove --postflop-model CLI preset flag from solve-preflop
- [x] Update run_solve_preflop: always load bundle, never build inline
- [x] Update diag-buckets and trace-hand to use PostflopSolveConfig
- [x] Update minimal_preflop.yaml with postflop_model_path
- [x] Update docs/training.md
- [x] Verify: cargo test, cargo clippy

## Summary of Changes

Removed `postflop_model` field from `PreflopConfig` in core. Made `postflop_model_path` required (non-optional) in `PreflopTrainingConfig`. Removed `--postflop-model` CLI preset flag and `--stack-depth`/`--players` flags from solve-preflop (config file is now required). Updated `run_solve_preflop` to always load from a pre-built bundle. Updated `diag-buckets` and `trace-hand` commands to use `PostflopSolveConfig`. Updated `minimal_preflop.yaml` to include `postflop_model_path`. Restructured `docs/training.md` with separate Postflop Config section.
