---
# poker_solver_rust-qj9e
title: Precompute per-player buffer ranges for discount functions
status: todo
type: task
priority: normal
created_at: 2026-02-28T23:58:14Z
updated_at: 2026-02-28T23:58:14Z
---

## Problem
`discount_regrets` and `discount_strategy_sums` in `preflop/solver.rs:387-424` iterate all nodes and skip non-matching positions. Called twice per iteration with LCFR. The position check in the inner loop is unnecessary overhead.

## Fix
Store `player_node_ranges: [Range<usize>; 2]` (or equivalent start/end indices into the flat buffer) per player, precomputed at construction time. Then discounting becomes a single slice operation per player.

## Verification
- `cargo test -p poker-solver-core`
- `cargo clippy`

## TODO
- [ ] Add per-player buffer range fields to PreflopSolver
- [ ] Compute ranges during tree construction / solver init
- [ ] Refactor discount_regrets to use precomputed ranges
- [ ] Refactor discount_strategy_sums to use precomputed ranges
- [ ] Run tests and clippy
