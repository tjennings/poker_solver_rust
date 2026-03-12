---
# poker_solver_rust-rs01
title: Optimize range-solver DCFR loop — eliminate alloc hotspots
status: in-progress
type: task
created_at: 2026-03-12T06:30:00Z
updated_at: 2026-03-12T06:30:00Z
---

## Problem

cfvnet datagen games/s unchanged after memory optimization. Profiling shows the range-solver DCFR inner loop is compute-bound with billions of unnecessary heap allocations.

## Fixes

### P0: Exploitability check allocates inside every 10-iteration block
**Files:** `crates/range-solver/src/utility.rs:257-274, 291-308`

`compute_mes_ev` and `compute_current_ev` allocate `Vec::with_capacity(num_private_hands)` per player per call. At 1000 iters/spot, fires 100× per spot. For 1M spots = ~200M heap allocs.

**Fix:** Pre-allocate scratch buffers once and pass as `&mut [f32]` output slices.

### P1: Two result Vec allocs per player per DCFR iteration
**Files:** `crates/range-solver/src/solver.rs:137, 189`

Inside the `for t in 0..max_num_iterations` loop, each iteration allocates two `Vec::with_capacity(num_private_hands)` per player. 1000 iters × 2 players = 2000 allocs/spot × 1M = 2B allocation events.

**Fix:** Pre-allocate outside the loop, `clear()` + reuse each iteration.

### P2: BetSizeOptions::clone() twice per spot
**File:** `crates/cfvnet/src/datagen/solver.rs:68`

`config.bet_sizes.clone()` called twice to populate `river_bet_sizes: [BetSizeOptions; 2]`. If contains Vec = 2M heap clones.

**Fix:** Use `Arc<BetSizeOptions>` or borrow.

### P3: finalize allocates per tree node
**File:** `crates/range-solver/src/utility.rs:336`

`MutexLike::new(Vec::with_capacity(...))` at every non-terminal node during `finalize`. 20-50 nodes × 1M spots = 20-50M allocs.

**Fix:** Reuse via TLS scratch buffers.

### P4: div_slice zero-check blocks vectorization
**File:** `crates/range-solver/src/sliceop.rs:32`

`is_zero()` uses `to_bits() == 0` which prevents LLVM auto-vectorization. Replace with `== 0.0`.

## Verification

- [ ] `cargo test -p range-solver` passes
- [ ] `cargo test -p cfvnet` passes
- [ ] Measure games/s before and after
