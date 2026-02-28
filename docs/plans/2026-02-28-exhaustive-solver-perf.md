# Exhaustive Solver Performance: Reuse Delta Buffers

## Problem

`parallel_traverse` in `crates/core/src/cfr/parallel.rs` returns freshly allocated `(Vec<f64>, Vec<f64>)` each call. In `exhaustive_solve_one_flop`, this runs 500 iterations, allocating 2 × `buf_size` f64 Vecs per iteration that are immediately consumed and dropped.

## Fix

### 1. Add `parallel_traverse_into` (parallel.rs)

New function that accepts pre-allocated `&mut Vec<f64>` output buffers:
- Zero the output buffers at the start
- Run the same fold+collect pattern internally (rayon thread-local buffers are unavoidable)
- Merge thread partitions into the provided buffers instead of fresh allocations
- Keep the existing `parallel_traverse` as-is for backward compat

### 2. Update `exhaustive_solve_one_flop` (postflop_exhaustive.rs)

- Pre-allocate `dr` and `ds` buffers before the iteration loop
- Call `parallel_traverse_into` instead of `parallel_traverse`
- Buffers reused across all 500 iterations

### 3. Minor: Fix string allocations in progress callback (postflop_exhaustive.rs)

- Clone `flop_name` once before the loop
- Use `&'static str` for metric label constant

## Files Changed

| File | Change |
|-|-|
| `crates/core/src/cfr/parallel.rs` | Add `parallel_traverse_into` |
| `crates/core/src/preflop/postflop_exhaustive.rs` | Use new function + fix string allocs |

## Agent Team & Execution Order

1. **rust-developer** (single agent, worktree): implement both changes + run tests
2. **rust-perf-reviewer** (review): verify the changes
3. **idiomatic-rust-enforcer** (review): verify idiomatic patterns
