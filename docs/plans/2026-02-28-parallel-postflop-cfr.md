# Parallel Within-Flop Postflop CFR — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Parallelize the exhaustive postflop CFR inner loop using the same snapshot + delta-merge pattern as the preflop solver, extracted into a shared `ParallelCfr` trait.

**Architecture:** New `cfr::parallel` module provides `ParallelCfr` trait + `parallel_traverse()`. Both preflop `Ctx` and new postflop `PostflopCfrCtx` implement the trait. The postflop traversal function gains a separate `snapshot` parameter for reads vs `dr`/`ds` for writes. `compute_exploitability` also gets parallelized.

**Tech Stack:** Rust, rayon (already a dependency)

**Design doc:** `docs/plans/2026-02-28-parallel-postflop-cfr-design.md`

---

## Agent Team & Execution Order

| Task | Agent | Dependencies |
|-|-|-|
| 1: Create `cfr::parallel` module | `rust-developer` | None |
| 2: Port preflop solver to shared trait | `rust-developer` | Task 1 |
| 3: Refactor postflop traversal signature | `rust-developer` | Task 1 |
| 4: Parallelize postflop iteration loop | `rust-developer` | Task 3 |
| 5: Parallelize `compute_exploitability` | `rust-developer` | Task 3 |
| Review round 1 | `software-architect` + `idiomatic-rust-enforcer` | Tasks 1-5 |
| Review round 2 | `rust-perf-reviewer` + `code-reviewer` | Review 1 |

Tasks 2 and 3 can run in parallel (independent files). Tasks 4 and 5 can run in parallel after task 3.

---

### Task 1: Create `cfr::parallel` module

**Files:**
- Create: `crates/core/src/cfr/parallel.rs`
- Modify: `crates/core/src/cfr/mod.rs`

**Step 1: Write the failing test**

In `crates/core/src/cfr/parallel.rs`, add the module with tests:

```rust
//! Shared parallel CFR iteration driver.
//!
//! Provides [`ParallelCfr`] trait and [`parallel_traverse`] function for
//! snapshot + delta-merge parallelism over hand pairs. Used by both
//! preflop and postflop solvers.

use rayon::prelude::*;

/// Context for one CFR iteration. Implementors provide the buffer size
/// and a traversal function that reads strategy from a frozen snapshot
/// and writes regret/strategy deltas to thread-local buffers.
pub trait ParallelCfr: Sync {
    /// Size of the regret/strategy flat buffers.
    fn buffer_size(&self) -> usize;

    /// Traverse one hand pair (both hero positions), accumulating
    /// regret deltas into `dr` and strategy deltas into `ds`.
    fn traverse_pair(&self, dr: &mut [f64], ds: &mut [f64], h1: u16, h2: u16);
}

/// Parallel fold+reduce over hand pairs. Returns merged `(regret_delta, strategy_delta)`.
///
/// Each rayon worker thread gets its own zero-initialized delta buffers.
/// After all pairs are traversed, deltas are merged via element-wise addition.
/// Results are mathematically identical to sequential traversal.
pub fn parallel_traverse<T: ParallelCfr>(
    ctx: &T,
    pairs: &[(u16, u16)],
) -> (Vec<f64>, Vec<f64>) {
    let buf_size = ctx.buffer_size();
    pairs
        .par_iter()
        .fold(
            || (vec![0.0f64; buf_size], vec![0.0f64; buf_size]),
            |(mut dr, mut ds), &(h1, h2)| {
                ctx.traverse_pair(&mut dr, &mut ds, h1, h2);
                (dr, ds)
            },
        )
        .reduce(
            || (vec![0.0; buf_size], vec![0.0; buf_size]),
            |(mut ar, mut a_s), (br, bs)| {
                add_into(&mut ar, &br);
                add_into(&mut a_s, &bs);
                (ar, a_s)
            },
        )
}

/// Element-wise `dst[i] += src[i]`.
#[inline]
pub fn add_into(dst: &mut [f64], src: &[f64]) {
    for (d, s) in dst.iter_mut().zip(src) {
        *d += s;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use test_macros::timed_test;

    /// Trivial implementor: each pair (h1, h2) adds h1+h2 to dr[0] and h1*h2 to ds[0].
    struct MockCfr;

    impl ParallelCfr for MockCfr {
        fn buffer_size(&self) -> usize { 2 }
        fn traverse_pair(&self, dr: &mut [f64], ds: &mut [f64], h1: u16, h2: u16) {
            dr[0] += f64::from(h1) + f64::from(h2);
            ds[0] += f64::from(h1) * f64::from(h2);
        }
    }

    #[timed_test]
    fn parallel_traverse_matches_sequential() {
        let pairs: Vec<(u16, u16)> = (0..10u16)
            .flat_map(|h1| (0..10u16).map(move |h2| (h1, h2)))
            .collect();

        // Sequential reference
        let mut exp_dr = vec![0.0f64; 2];
        let mut exp_ds = vec![0.0f64; 2];
        for &(h1, h2) in &pairs {
            MockCfr.traverse_pair(&mut exp_dr, &mut exp_ds, h1, h2);
        }

        // Parallel
        let (dr, ds) = parallel_traverse(&MockCfr, &pairs);

        assert!((dr[0] - exp_dr[0]).abs() < 1e-9, "dr mismatch: {} vs {}", dr[0], exp_dr[0]);
        assert!((ds[0] - exp_ds[0]).abs() < 1e-9, "ds mismatch: {} vs {}", ds[0], exp_ds[0]);
    }

    #[timed_test]
    fn parallel_traverse_empty_pairs() {
        let (dr, ds) = parallel_traverse(&MockCfr, &[]);
        assert_eq!(dr.len(), 2);
        assert_eq!(ds.len(), 2);
        assert!(dr[0].abs() < 1e-15);
        assert!(ds[0].abs() < 1e-15);
    }

    #[timed_test]
    fn add_into_works() {
        let mut dst = vec![1.0, 2.0, 3.0];
        let src = vec![10.0, 20.0, 30.0];
        add_into(&mut dst, &src);
        assert_eq!(dst, vec![11.0, 22.0, 33.0]);
    }
}
```

**Step 2: Register the module**

In `crates/core/src/cfr/mod.rs`, add:

```rust
pub mod parallel;

pub use dcfr::{CfrVariant, DcfrParams};
```

**Step 3: Run tests**

Run: `cargo test -p poker-solver-core cfr::parallel`
Expected: All 3 tests pass.

**Step 4: Commit**

```
feat: add shared ParallelCfr trait and parallel_traverse driver
```

---

### Task 2: Port preflop solver to shared trait

**Files:**
- Modify: `crates/core/src/preflop/solver.rs`

**Depends on:** Task 1

**Step 1: Import the shared module**

Add to imports in `solver.rs`:
```rust
use crate::cfr::parallel::{ParallelCfr, parallel_traverse as shared_parallel_traverse, add_into};
```

**Step 2: Implement `ParallelCfr` for `Ctx`**

Add this impl block after the `Ctx` struct definition (after line ~137):

```rust
impl ParallelCfr for Ctx<'_> {
    fn buffer_size(&self) -> usize {
        self.layout.total_size
    }

    fn traverse_pair(&self, dr: &mut [f64], ds: &mut [f64], h1: u16, h2: u16) {
        let w = self.equity.weight(h1 as usize, h2 as usize);
        for hero_pos in 0..2u8 {
            let (hh, oh) = if hero_pos == 0 { (h1, h2) } else { (h2, h1) };
            cfr_traverse(self, dr, ds, 0, hh, oh, hero_pos, 1.0, w);
        }
    }
}
```

**Step 3: Replace the standalone `parallel_traverse` function**

In `train_one_iteration()` (line ~433), change:
```rust
let (mr, ms) = parallel_traverse(&ctx, &self.pairs);
```
to:
```rust
let (mr, ms) = shared_parallel_traverse(&ctx, &self.pairs);
```

**Step 4: Remove the old standalone functions**

Delete the old `parallel_traverse()` function (lines 462-489) and `add_into()` function (lines 491-497). The `add_into` call on lines 436-437 should use the shared version.

**Step 5: Run all preflop tests**

Run: `cargo test -p poker-solver-core preflop`
Expected: All preflop tests pass (63 unit + 3 integration). Behavior is identical.

**Step 6: Commit**

```
refactor: port preflop solver to shared ParallelCfr trait
```

---

### Task 3: Refactor postflop traversal signature (snapshot split)

**Files:**
- Modify: `crates/core/src/preflop/postflop_exhaustive.rs`

**Depends on:** Task 1

**Step 1: Change `exhaustive_cfr_traverse` signature**

Split the `regret_sum: &mut [f64]` parameter into `snapshot: &[f64]` (read-only) and `dr: &mut [f64]` (write):

```rust
fn exhaustive_cfr_traverse(
    tree: &PostflopTree,
    layout: &PostflopLayout,
    equity_table: &[f64],
    snapshot: &[f64],          // frozen regrets for strategy computation
    dr: &mut [f64],            // regret deltas
    ds: &mut [f64],            // strategy deltas
    node_idx: u32,
    hero_hand: u16,
    opp_hand: u16,
    hero_pos: u8,
    reach_hero: f64,
    reach_opp: f64,
    iteration: u64,
    dcfr: &DcfrParams,
) -> f64 {
```

**Step 2: Update the function body**

Line 157 — regret matching reads from snapshot:
```rust
regret_matching_into(snapshot, start, &mut strategy[..num_actions]);
```

Lines 162-167 — recursive calls pass `snapshot, dr, ds`:
```rust
action_values[i] = exhaustive_cfr_traverse(
    tree, layout, equity_table, snapshot, dr, ds,
    child, hero_hand, opp_hand, hero_pos,
    reach_hero * strategy[i], reach_opp,
    iteration, dcfr,
);
```

Lines 177-181 — writes go to `dr` and `ds`:
```rust
dr[start + i] += regret_weight * reach_opp * (val - node_value);
// ...
ds[start + i] += strategy_weight * reach_hero * s;
```

Lines 189-194 — opponent recursive calls:
```rust
exhaustive_cfr_traverse(
    tree, layout, equity_table, snapshot, dr, ds,
    child, hero_hand, opp_hand, hero_pos,
    reach_hero, reach_opp * strategy[i],
    iteration, dcfr,
)
```

**Step 3: Update `exhaustive_solve_one_flop` to pass snapshot**

Temporarily, pass `&regret_sum` as both snapshot and the dr target (same as current behavior, just explicit). Add a snapshot clone:

```rust
for iter in 0..num_iterations {
    let snapshot = regret_sum.clone();
    for hero_hand in 0..n as u16 {
        for opp_hand in 0..n as u16 {
            let eq = equity_table[hero_hand as usize * n + opp_hand as usize];
            if eq.is_nan() { continue; }
            for hero_pos in 0..2u8 {
                exhaustive_cfr_traverse(
                    tree, layout, equity_table,
                    &snapshot,             // read from snapshot
                    &mut regret_sum,       // write deltas to regret_sum directly (for now)
                    &mut strategy_sum,
                    0, hero_hand, opp_hand, hero_pos,
                    1.0, 1.0, iter as u64, dcfr,
                );
            }
        }
    }
    // DCFR discounting unchanged...
```

**Step 4: Update test callsites**

Tests `exhaustive_cfr_fold_terminal_payoff` and `exhaustive_cfr_showdown_terminal_payoff` call `exhaustive_cfr_traverse` directly. Update them to pass `&regret_sum` as snapshot and `&mut regret_sum` as dr:

```rust
// Before:
let ev = exhaustive_cfr_traverse(
    &tree, &layout, &equity_table,
    &mut regret_sum, &mut strategy_sum,
    node_idx, hero_hand, opp_hand, 0, 1.0, 1.0, 0, &dcfr,
);

// After — use a snapshot clone since we can't borrow regret_sum mutably and immutably:
let snapshot = regret_sum.clone();
let ev = exhaustive_cfr_traverse(
    &tree, &layout, &equity_table,
    &snapshot, &mut regret_sum, &mut strategy_sum,
    node_idx, hero_hand, opp_hand, 0, 1.0, 1.0, 0, &dcfr,
);
```

**Step 5: Run tests**

Run: `cargo test -p poker-solver-core postflop_exhaustive`
Expected: All postflop exhaustive tests pass. Behavior unchanged (snapshot is a clone of regret_sum at start of iteration).

**Step 6: Commit**

```
refactor: split postflop traversal into snapshot read + delta write
```

---

### Task 4: Parallelize postflop iteration loop

**Files:**
- Modify: `crates/core/src/preflop/postflop_exhaustive.rs`

**Depends on:** Task 3

**Step 1: Add imports**

```rust
use crate::cfr::parallel::{ParallelCfr, parallel_traverse, add_into};
```

**Step 2: Create `PostflopCfrCtx` struct and implement trait**

Add above `exhaustive_solve_one_flop`:

```rust
/// Immutable context for one postflop CFR iteration.
/// Strategy is read from `snapshot`; deltas written to thread-local buffers.
struct PostflopCfrCtx<'a> {
    tree: &'a PostflopTree,
    layout: &'a PostflopLayout,
    equity_table: &'a [f64],
    snapshot: &'a [f64],
    iteration: u64,
    dcfr: &'a DcfrParams,
}

impl ParallelCfr for PostflopCfrCtx<'_> {
    fn buffer_size(&self) -> usize {
        self.layout.total_size
    }

    fn traverse_pair(&self, dr: &mut [f64], ds: &mut [f64], h1: u16, h2: u16) {
        let n = NUM_CANONICAL_HANDS;
        let eq = self.equity_table[h1 as usize * n + h2 as usize];
        if eq.is_nan() {
            return;
        }
        for hero_pos in 0..2u8 {
            exhaustive_cfr_traverse(
                self.tree, self.layout, self.equity_table,
                self.snapshot, dr, ds,
                0, h1, h2, hero_pos, 1.0, 1.0,
                self.iteration, self.dcfr,
            );
        }
    }
}
```

**Step 3: Rewrite `exhaustive_solve_one_flop` iteration loop**

Replace the triple-nested serial loop with the parallel pattern:

```rust
fn exhaustive_solve_one_flop(
    tree: &PostflopTree,
    layout: &PostflopLayout,
    equity_table: &[f64],
    num_iterations: usize,
    convergence_threshold: f64,
    flop_name: &str,
    dcfr: &DcfrParams,
    on_progress: &(impl Fn(BuildPhase) + Sync),
) -> FlopSolveResult {
    let buf_size = layout.total_size;
    let mut regret_sum = vec![0.0f64; buf_size];
    let mut strategy_sum = vec![0.0f64; buf_size];
    let mut current_exploitability = f64::INFINITY;
    let mut iterations_used = 0;
    let n = NUM_CANONICAL_HANDS;

    // Pre-build valid hand pairs (filter NaN equity pairs once).
    let pairs: Vec<(u16, u16)> = (0..n as u16)
        .flat_map(|h1| (0..n as u16).map(move |h2| (h1, h2)))
        .filter(|&(h1, h2)| {
            !equity_table[h1 as usize * n + h2 as usize].is_nan()
        })
        .collect();

    let mut snapshot = vec![0.0f64; buf_size];

    for iter in 0..num_iterations {
        snapshot.clone_from(&regret_sum);

        let ctx = PostflopCfrCtx {
            tree,
            layout,
            equity_table,
            snapshot: &snapshot,
            iteration: iter as u64,
            dcfr,
        };

        let (dr, ds) = parallel_traverse(&ctx, &pairs);

        // Apply DCFR discounting before merging deltas.
        if dcfr.should_discount(iter as u64) {
            dcfr.discount_regrets(&mut regret_sum, iter as u64);
            dcfr.discount_strategy_sums(&mut strategy_sum, iter as u64);
        }
        add_into(&mut regret_sum, &dr);
        add_into(&mut strategy_sum, &ds);
        if dcfr.should_floor_regrets() {
            dcfr.floor_regrets(&mut regret_sum);
        }

        iterations_used = iter + 1;
        if iter >= 1 {
            current_exploitability =
                compute_exploitability(tree, layout, &strategy_sum, equity_table);
        }

        on_progress(BuildPhase::FlopProgress {
            flop_name: flop_name.to_string(),
            stage: FlopStage::Solving {
                iteration: iterations_used,
                max_iterations: num_iterations,
                delta: current_exploitability,
                metric_label: "mBB/h".to_string(),
            },
        });

        if iter >= 1 && current_exploitability < convergence_threshold {
            break;
        }
    }

    FlopSolveResult {
        strategy_sum,
        delta: current_exploitability,
        iterations_used,
    }
}
```

**Step 4: Add a test that verifies parallel matches sequential**

Add a new test that runs the same solve with 1 thread and N threads, comparing strategy sums:

```rust
#[timed_test(10)]
fn parallel_solve_matches_sequential_result() {
    let config = PostflopModelConfig {
        bet_sizes: vec![1.0],
        max_raises_per_street: 0,
        postflop_solve_iterations: 5,
        ..PostflopModelConfig::exhaustive_fast()
    };
    let tree = PostflopTree::build_with_spr(&config, 3.5).unwrap();
    let node_streets = annotate_streets(&tree);
    let n = NUM_CANONICAL_HANDS;
    let layout = PostflopLayout::build(&tree, &node_streets, n, n, n);
    let equity_table = synthetic_equity_table();
    let dcfr = DcfrParams::linear();

    // Solve with default thread pool (parallel)
    let result_par = exhaustive_solve_one_flop(
        &tree, &layout, &equity_table, 5, 0.0, "par", &dcfr, &|_| {},
    );

    // Solve with 1 thread (sequential)
    let pool = rayon::ThreadPoolBuilder::new().num_threads(1).build().unwrap();
    let result_seq = pool.install(|| {
        exhaustive_solve_one_flop(
            &tree, &layout, &equity_table, 5, 0.0, "seq", &dcfr, &|_| {},
        )
    });

    assert_eq!(result_par.iterations_used, result_seq.iterations_used);
    // Strategy sums should be very close (f64 addition order may differ slightly)
    for (p, s) in result_par.strategy_sum.iter().zip(result_seq.strategy_sum.iter()) {
        assert!(
            (p - s).abs() < 1e-6,
            "strategy_sum mismatch: parallel={p}, sequential={s}"
        );
    }
}
```

**Step 5: Run tests**

Run: `cargo test -p poker-solver-core postflop_exhaustive`
Expected: All tests pass, including the new parallel-vs-sequential test.

**Step 6: Commit**

```
feat: parallelize postflop CFR inner loop via shared ParallelCfr trait
```

---

### Task 5: Parallelize `compute_exploitability`

**Files:**
- Modify: `crates/core/src/preflop/postflop_exhaustive.rs`

**Depends on:** Task 3

**Step 1: Rewrite `compute_exploitability` with rayon**

```rust
fn compute_exploitability(
    tree: &PostflopTree,
    layout: &PostflopLayout,
    strategy_sum: &[f64],
    equity_table: &[f64],
) -> f64 {
    let n = NUM_CANONICAL_HANDS;
    let mut br_values = [0.0f64; 2];

    for br_player in 0..2u8 {
        let (total, count) = (0..n as u16)
            .into_par_iter()
            .flat_map_iter(|hero_hand| {
                (0..n as u16)
                    .filter(move |&opp_hand| {
                        !equity_table[hero_hand as usize * n + opp_hand as usize].is_nan()
                    })
                    .map(move |opp_hand| (hero_hand, opp_hand))
            })
            .map(|(hero_hand, opp_hand)| {
                best_response_ev(
                    tree, layout, strategy_sum, equity_table,
                    0, hero_hand, opp_hand, br_player,
                )
            })
            .fold(
                || (0.0f64, 0u64),
                |(t, c), v| (t + v, c + 1),
            )
            .reduce(
                || (0.0, 0),
                |(t1, c1), (t2, c2)| (t1 + t2, c1 + c2),
            );

        br_values[br_player as usize] = if count > 0 {
            total / count as f64
        } else {
            0.0
        };
    }

    let pot_fraction = (br_values[0] + br_values[1]) / 2.0;
    pot_fraction * INITIAL_POT_BB * 1000.0
}
```

**Step 2: Run tests**

Run: `cargo test -p poker-solver-core postflop_exhaustive`
Expected: All exploitability tests still pass (same results, just computed in parallel).

**Step 3: Commit**

```
feat: parallelize postflop exploitability computation
```

---

### Task 6: Final validation

**Step 1: Run full test suite**

Run: `cargo test`
Expected: All tests pass. No regressions in preflop or postflop.

**Step 2: Run clippy**

Run: `cargo clippy`
Expected: Clean (no new warnings).

**Step 3: Run a real postflop solve to verify speedup**

Run: `cargo run -p poker-solver-trainer --release -- postflop-diag --flop 2c3h4d --spr 10.0`
(or whatever the equivalent diagnostic command is — check `docs/training.md`)

Expected: Noticeably faster than before. mBB/h exploitability should be similar to pre-refactor values.

**Step 4: Commit any final fixes**

---

## Summary

| Task | Files | What |
|-|-|-|
| 1 | `cfr/parallel.rs`, `cfr/mod.rs` | New shared `ParallelCfr` trait + `parallel_traverse` + `add_into` |
| 2 | `preflop/solver.rs` | Impl trait for `Ctx`, remove old standalone functions |
| 3 | `preflop/postflop_exhaustive.rs` | Split traversal into snapshot read + delta write |
| 4 | `preflop/postflop_exhaustive.rs` | `PostflopCfrCtx` impl, rewrite iteration loop with `parallel_traverse` |
| 5 | `preflop/postflop_exhaustive.rs` | Parallelize `compute_exploitability` with rayon |
| 6 | — | Full validation: tests, clippy, real solve benchmark |
