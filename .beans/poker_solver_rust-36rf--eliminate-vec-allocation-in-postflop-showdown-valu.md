---
# poker_solver_rust-36rf
title: Eliminate Vec allocation in postflop_showdown_value hot path
status: completed
type: task
priority: high
created_at: 2026-02-28T23:58:04Z
updated_at: 2026-03-01T00:13:31Z
---

## Problem
`postflop_showdown_value` at `preflop/solver.rs:745` allocates a `Vec<f64>` of SPR values on every call by collecting from `pf_state.abstractions`. This is called per hand-pair × per terminal × 2 positions in the hot path.

## Fix
Change `select_closest_spr` to accept `&[PostflopAbstraction]` directly instead of `&[f64]`. Iterate the abstraction slice inline — zero allocations, same result.

## Verification
- `cargo test -p poker-solver-core`
- `cargo clippy`

## TODO
- [ ] Refactor select_closest_spr to accept &[PostflopAbstraction] or iterate inline
- [ ] Remove the Vec<f64> collection at the call site
- [ ] Run tests and clippy

## Summary of Changes\nChanged `select_closest_spr` to accept an iterator over SPR values instead of `&[f64]`, removing the per-call `Vec<f64>` allocation in `postflop_showdown_value`.
