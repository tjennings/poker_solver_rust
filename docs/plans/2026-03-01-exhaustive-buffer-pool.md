# Exhaustive CFR Buffer Pool Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Eliminate per-iteration buffer allocation churn and redundant snapshot cloning in the exhaustive postflop CFR solver, reducing memory allocation by ~7 GB/flop/iteration at SPR=20.

**Architecture:** Add a `parallel_traverse_pooled` function that accepts pre-allocated thread buffers instead of allocating fresh ones per iteration via rayon `fold`. Eliminate the `snapshot`, `dr`, and `ds` buffers in `exhaustive_solve_one_flop` — pass `&regret_sum` directly as snapshot (safe because it's not mutated during traversal), and merge directly from pool into `regret_sum`/`strategy_sum`.

**Tech Stack:** Rust, rayon (parallel iteration)

---

### Task 1: Add `parallel_traverse_pooled` to `cfr/parallel.rs`

**Files:**
- Modify: `crates/core/src/cfr/parallel.rs`

**Step 1: Write the failing test**

Add to the existing `#[cfg(test)] mod tests` block in `crates/core/src/cfr/parallel.rs`:

```rust
#[timed_test]
fn parallel_traverse_pooled_matches_sequential() {
    let pairs: Vec<(u16, u16)> = (0..10_u16)
        .flat_map(|h1| (0..10_u16).map(move |h2| (h1, h2)))
        .collect();

    let mut exp_dr = vec![0.0_f64; 2];
    let mut exp_ds = vec![0.0_f64; 2];
    for &(h1, h2) in &pairs {
        MockCfr.traverse_pair(&mut exp_dr, &mut exp_ds, h1, h2);
    }

    let mut pool: Vec<(Vec<f64>, Vec<f64>)> = (0..4)
        .map(|_| (vec![0.0f64; 2], vec![0.0f64; 2]))
        .collect();
    parallel_traverse_pooled(&MockCfr, &pairs, &mut pool);

    let mut dr = vec![0.0f64; 2];
    let mut ds = vec![0.0f64; 2];
    for (pdr, pds) in &pool {
        add_into(&mut dr, pdr);
        add_into(&mut ds, pds);
    }

    assert!((dr[0] - exp_dr[0]).abs() < 1e-9, "dr mismatch: {} vs {}", dr[0], exp_dr[0]);
    assert!((ds[0] - exp_ds[0]).abs() < 1e-9, "ds mismatch: {} vs {}", ds[0], exp_ds[0]);
}

#[timed_test]
fn parallel_traverse_pooled_empty_pairs() {
    let mut pool: Vec<(Vec<f64>, Vec<f64>)> = (0..2)
        .map(|_| (vec![1.0f64; 2], vec![1.0f64; 2]))
        .collect();
    parallel_traverse_pooled(&MockCfr, &[], &mut pool);
    // Pool should be zeroed even with no pairs
    for (dr, ds) in &pool {
        assert!(dr[0].abs() < 1e-15);
        assert!(ds[0].abs() < 1e-15);
    }
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test -p poker-solver-core --lib parallel_traverse_pooled -- --nocapture`
Expected: FAIL — `parallel_traverse_pooled` does not exist yet.

**Step 3: Implement `parallel_traverse_pooled`**

Add this function to `crates/core/src/cfr/parallel.rs`, after `parallel_traverse_into`:

```rust
/// Traverse all hand pairs in parallel using a pre-allocated buffer pool.
///
/// Each pool entry is zeroed, then accumulates deltas for a chunk of pairs.
/// The caller merges pool entries into final regret/strategy buffers.
///
/// This avoids the O(threads × buf_size) allocation that `parallel_traverse_into`
/// performs every call via rayon `fold`. The pool is allocated once and reused
/// across iterations.
pub fn parallel_traverse_pooled<T: ParallelCfr>(
    ctx: &T,
    pairs: &[(u16, u16)],
    pool: &mut [(Vec<f64>, Vec<f64>)],
) {
    if pool.is_empty() || pairs.is_empty() {
        for (dr, ds) in pool.iter_mut() {
            dr.fill(0.0);
            ds.fill(0.0);
        }
        return;
    }

    let n = pool.len();
    let chunk_size = (pairs.len() + n - 1) / n;

    pool.par_iter_mut()
        .enumerate()
        .for_each(|(i, (dr, ds))| {
            dr.fill(0.0);
            ds.fill(0.0);
            let start = i * chunk_size;
            let end = (start + chunk_size).min(pairs.len());
            for &(h1, h2) in &pairs[start..end] {
                ctx.traverse_pair(dr, ds, h1, h2);
            }
        });
}
```

**Step 4: Run tests to verify they pass**

Run: `cargo test -p poker-solver-core --lib parallel_traverse_pooled -- --nocapture`
Expected: PASS (both new tests).

Also run: `cargo test -p poker-solver-core --lib cfr::parallel -- --nocapture`
Expected: PASS (all existing tests still pass).

**Step 5: Commit**

```bash
git add crates/core/src/cfr/parallel.rs
git commit -m "feat(cfr): add parallel_traverse_pooled with pre-allocated buffer pool

Eliminates per-iteration buffer allocation in rayon fold. The caller
pre-allocates a fixed pool of (dr, ds) buffer pairs which are zeroed
and reused each call. Pairs are split evenly across pool entries."
```

---

### Task 2: Update exhaustive solver to use pooled traversal + eliminate snapshot

**Files:**
- Modify: `crates/core/src/preflop/postflop_exhaustive.rs` (function `exhaustive_solve_one_flop`, lines ~593-717)

**Step 1: Replace buffer allocation and iteration loop**

In `exhaustive_solve_one_flop`, change the import line and the function body. The key changes:

1. Import `parallel_traverse_pooled` instead of `parallel_traverse_into`
2. Remove `snapshot`, `dr`, `ds` buffer allocations
3. Pre-allocate pool outside the iteration loop
4. Use `&regret_sum` directly as snapshot (scoped borrow via block)
5. Merge directly from pool into `regret_sum`/`strategy_sum`

Change the import at line 8:
```rust
use crate::cfr::parallel::{add_into, parallel_traverse_pooled, ParallelCfr};
```

Replace lines 605-660 (buffer allocation through traversal call) with:

```rust
    let buf_size = layout.total_size;
    let mut regret_sum = vec![0.0f64; buf_size];
    let mut strategy_sum = vec![0.0f64; buf_size];
    let mut current_exploitability = f64::INFINITY;
    let mut iterations_used = 0;
    let n = NUM_CANONICAL_HANDS;

    // Pre-build valid hand pairs (filter NaN equity pairs once).
    let pairs: Vec<(u16, u16)> = (0..n as u16)
        .flat_map(|h1| (0..n as u16).map(move |h2| (h1, h2)))
        .filter(|&(h1, h2)| !equity_table[h1 as usize * n + h2 as usize].is_nan())
        .collect();

    // Tell the TUI how many traversals this flop will contribute.
    if let Some(c) = counters {
        c.total_expected_traversals
            .fetch_add((pairs.len() * num_iterations) as u64, Ordering::Relaxed);
    }

    // Pre-allocate thread buffer pool: reused across all iterations.
    // Each entry is a (regret_delta, strategy_delta) pair for one rayon partition.
    let n_partitions = rayon::current_num_threads().max(1);
    let mut pool: Vec<(Vec<f64>, Vec<f64>)> = (0..n_partitions)
        .map(|_| (vec![0.0f64; buf_size], vec![0.0f64; buf_size]))
        .collect();

    // Per-flop pruning accumulators: track action-slot pruning for this flop
    // by snapshotting the global counters before/after each iteration.
    let mut flop_total_action_slots: u64 = 0;
    let mut flop_pruned_action_slots: u64 = 0;

    let flop_name_owned = flop_name.to_string();

    for iter in 0..num_iterations {
        // Snapshot global counters before traversal so we can attribute
        // the delta to this flop.
        let prev_ta = counters.map_or(0, |c| c.total_action_slots.load(Ordering::Relaxed));
        let prev_pa = counters.map_or(0, |c| c.pruned_action_slots.load(Ordering::Relaxed));

        let prune_active = config.prune_warmup > 0 && iter >= config.prune_warmup;

        // Scoped borrow: ctx borrows &regret_sum immutably during traversal.
        // No snapshot clone needed — regret_sum is not mutated until after
        // traversal completes (discounting + merge happen below).
        {
            let ctx = PostflopCfrCtx {
                tree,
                layout,
                equity_table,
                snapshot: &regret_sum,
                iteration: iter as u64,
                dcfr,
                prune_active,
                prune_explore_pct: config.prune_explore_pct,
                counters,
            };

            parallel_traverse_pooled(&ctx, &pairs, &mut pool);
        } // ctx dropped — &regret_sum borrow released
```

Replace lines 662-674 (pruning stats + discounting + merge) with:

```rust
        // Accumulate per-flop pruning stats from global counter deltas.
        let cur_ta = counters.map_or(0, |c| c.total_action_slots.load(Ordering::Relaxed));
        let cur_pa = counters.map_or(0, |c| c.pruned_action_slots.load(Ordering::Relaxed));
        flop_total_action_slots += cur_ta.saturating_sub(prev_ta);
        flop_pruned_action_slots += cur_pa.saturating_sub(prev_pa);

        // Apply DCFR discounting before merging deltas.
        if dcfr.should_discount(iter as u64) {
            dcfr.discount_regrets(&mut regret_sum, iter as u64);
            dcfr.discount_strategy_sums(&mut strategy_sum, iter as u64);
        }

        // Merge pool partition deltas directly into accumulators.
        for (pdr, pds) in pool.iter() {
            add_into(&mut regret_sum, pdr);
            add_into(&mut strategy_sum, pds);
        }
```

Lines 675 onward (`should_floor_regrets`, `regret_floor`, `compute_exploitability`, progress callback, convergence check) remain **unchanged**.

**Step 2: Run the full test suite**

Run: `cargo test -p poker-solver-core --lib -- --nocapture`
Expected: PASS — all existing tests pass with identical numerical results.

Key tests to watch:
- `exhaustive_cfr_converges_simple_tree` — verifies exploitability decreases
- `exhaustive_exploitability_decreases` — regression test
- `uniform_strategy_is_exploitable` — baseline sanity check

**Step 3: Commit**

```bash
git add crates/core/src/preflop/postflop_exhaustive.rs
git commit -m "perf(exhaustive): eliminate snapshot clone and use pooled traversal

- Remove snapshot buffer: pass &regret_sum directly as frozen snapshot
  (safe because regret_sum is not mutated during traversal)
- Remove dr/ds intermediate buffers: merge directly from pool
- Pre-allocate thread buffer pool once per flop, reuse across iterations
- Saves 3 × buf_size per flop + eliminates N_threads × 2 × buf_size
  allocation churn per iteration
- At SPR=20 with 3-bet/3-raise: eliminates ~1.1 GB persistent +
  ~5.8 GB allocation churn per flop per iteration"
```

---

### Task 3: Verify no regressions across full test suite

**Step 1: Run all core tests**

Run: `cargo test -p poker-solver-core --lib`
Expected: All tests pass. Known timer failures (pre-existing): `blueprint/subgame_cfr`, `preflop/bundle`.

**Step 2: Run clippy**

Run: `cargo clippy -p poker-solver-core`
Expected: No new warnings (pedantic enabled in core crate).

**Step 3: Commit any fixups**

If clippy or tests reveal issues, fix and commit.

---

## Agent Team & Execution Order

| Task | Agent | Notes |
|-|-|-|
| 1-2 | `rust-developer` | Sequential — Task 2 depends on Task 1 |
| 3 | `rust-developer` | Verification |
| Review | `idiomatic-rust-enforcer` + `rust-perf-reviewer` | Parallel, after Task 3 |

Tasks 1-2 are sequential (Task 2 imports from Task 1). Review agents run in parallel after implementation.

## Memory Impact Summary

At SPR=20 with 3-bet/3-raise (buf_size = 47.5M f64s = 362 MB):

| Buffer | Before | After |
|-|-|-|
| `regret_sum` | 362 MB | 362 MB |
| `strategy_sum` | 362 MB | 362 MB |
| `snapshot` | 362 MB | **eliminated** |
| `dr` | 362 MB | **eliminated** |
| `ds` | 362 MB | **eliminated** |
| Rayon partitions (8 threads) | 5.8 GB **per iteration** | 5.8 GB **once** |
| **Total persistent** | **1.8 GB** | **724 MB** |
| **Per-iteration alloc churn** | **5.8 GB** | **0** |
