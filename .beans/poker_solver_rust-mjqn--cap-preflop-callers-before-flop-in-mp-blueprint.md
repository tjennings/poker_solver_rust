---
# poker_solver_rust-mjqn
title: Cap preflop callers before flop in MP blueprint
status: completed
type: feature
priority: normal
created_at: 2026-05-14T00:39:57Z
updated_at: 2026-05-14T00:56:22Z
---

Add configurable maximum number of players allowed to continue from preflop to the flop during train-blueprint-mp. Enforce by removing call/check-call after the first call unless the actor is closing action, so non-closing overcalls can be pruned while action-closing calls remain legal.

- [x] Research current MP preflop action generation and config path
- [x] Design configurable cap semantics and defaults
- [x] Implement cap without changing postflop action generation
- [x] Add focused tests for call removal and action-closing exception
- [x] Update training docs/config docs for user-facing config changes
- [x] Run full test suite under 1 minute and trainer/sample sanity checks

## Summary of Changes

- Added optional action_abstraction.max_flop_players with uncapped default.
- Applied the cap in both eager and lazy MP preflop action generation.
- Preserved closing calls up to the cap and removed non-closing calls that would occupy the last allowed flop-player slot.
- Set the 6-max sample blueprint config to max_flop_players: 3.
- Updated architecture and training docs.

## Verification

- cargo fmt
- cargo test -p poker-solver-core preflop_flop_player_cap -- --nocapture
- cargo test -p poker-solver-core blueprint_mp::config -- --nocapture
- cargo test -p poker-solver-trainer mp_ -- --nocapture
- /usr/bin/time -p cargo test: passed on warmed rerun in 49.75s real time
