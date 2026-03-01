---
# poker_solver_rust-18ff
title: Return &[f64] from PreflopStrategy::get_probs instead of Vec<f64>
status: completed
type: task
priority: normal
created_at: 2026-02-28T23:58:17Z
updated_at: 2026-03-01T00:13:37Z
---

## Problem
`PreflopStrategy::get_probs` at `preflop/solver.rs:56` clones a `Vec<f64>` on every call. In exploitability computation (`exploitability.rs:140-142`), this is called for every opponent hand × every action × every tree depth.

## Fix
Return `&[f64]` instead of `Vec<f64>`. Change return type and use `.map(Vec::as_slice).unwrap_or(&[])`. Update all call sites in exploitability.rs accordingly.

## Verification
- `cargo test -p poker-solver-core`
- `cargo clippy`

## TODO
- [ ] Change get_probs return type to &[f64]
- [ ] Update exploitability.rs call sites
- [ ] Run tests and clippy

## Summary of Changes\nChanged `get_probs` and `get_root_probs` return type from `Vec<f64>` to `&[f64]`, eliminating Vec cloning on every call during exploitability computation.
