---
# poker_solver_rust-s4t5
title: Reuse buffers in parallel_traverse for preflop solver
status: todo
type: task
priority: high
created_at: 2026-02-28T23:58:01Z
updated_at: 2026-02-28T23:58:01Z
---

## Problem
`parallel_traverse` in `cfr/parallel.rs:33-50` allocates 2×buf_size×N_threads buffers per iteration via `.fold(...).collect()`. The preflop solver calls the allocating form at `solver.rs:447` instead of the existing `parallel_traverse_into`.

## Fix
Switch the call site in `preflop/solver.rs` to use `parallel_traverse_into` with reusable buffers stored on the solver struct. Add `strategy_delta_buf: Vec<f64>` to `PreflopSolver`, initialize in `::new`, and pass both buffers to `parallel_traverse_into`.

## Verification
- `cargo test -p poker-solver-core`
- `cargo clippy`

## TODO
- [ ] Add reusable delta buffers to PreflopSolver struct
- [ ] Switch call site from parallel_traverse to parallel_traverse_into
- [ ] Update last_instantaneous_regret to clone_from delta buffer
- [ ] Run tests and clippy
