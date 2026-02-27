---
# poker_solver_rust-0kdp
title: Multi-SPR postflop model selection
status: completed
type: feature
priority: normal
created_at: 2026-02-27T17:51:58Z
updated_at: 2026-02-27T20:48:23Z
---

Build a PostflopAbstraction per configured SPR and have the preflop solver
select the closest SPR model at each showdown terminal.

## Tasks

- [x] Task 1: Add select_closest_spr helper with tests
- [x] Task 2: Change PostflopState to hold multiple abstractions
- [x] Task 3: Update PostflopAbstraction::build to accept explicit SPR
- [x] Task 4: Update PostflopBundle for multi-SPR persistence
- [x] Task 5: Update trainer to build and attach multiple SPR models
- [x] Task 6: Update exploitability module for multi-SPR
- [x] Task 7: Update explorer bundle loading for multi-SPR
- [x] Task 8: Full integration test
- [x] Task 9: Update documentation

## Summary of Changes

All multi-SPR postflop work completed and merged to main via `worktree-multi-spr-postflop` branch. Key commits: select_closest_spr helper, PostflopState holds Vec<PostflopAbstraction>, multi-SPR bundle persistence with legacy compat, trainer builds/persists multi-SPR models, explorer loads multi-SPR bundles, docs updated. Also removed deep-cfr crate and made preflop solving require a pre-built postflop bundle.
