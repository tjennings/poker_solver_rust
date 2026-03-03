# Thread-Local Buffer Pooling Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Eliminate per-flop buffer allocation churn by reusing thread-local buffers across flops in the exhaustive postflop solver.

**Architecture:** Add a `FlopBuffers` struct holding the 4 large vectors (regret_sum, strategy_sum, delta pair). Create a `rayon::ThreadLocal<RefCell<FlopBuffers>>` in `build_exhaustive` so each rayon worker thread lazily allocates one buffer set and reuses it across all flops. Change `exhaustive_solve_one_flop` to accept `&mut FlopBuffers` instead of allocating internally.

**Tech Stack:** Rust, rayon (`ThreadLocal`), `std::cell::RefCell`

---

### Task 1: Add `FlopBuffers` struct and update `exhaustive_solve_one_flop` signature

**Files:**
- Modify: `crates/core/src/preflop/postflop_exhaustive.rs`

**Step 1: Add the `FlopBuffers` struct after the `SolverCounters` struct (after line 43)**

```rust
/// Reusable per-thread buffers for exhaustive CFR solving.
///
/// Allocated once per rayon worker thread and reused across flops to avoid
/// repeated allocation of large vectors (e.g. 277 MB per flop at SPR=6).
pub(crate) struct FlopBuffers {
    pub regret_sum: Vec<f64>,
    pub strategy_sum: Vec<f64>,
    pub delta: (Vec<f64>, Vec<f64>),
}

impl FlopBuffers {
    /// Allocate zeroed buffers of the given size.
    fn new(size: usize) -> Self {
        Self {
            regret_sum: vec![0.0; size],
            strategy_sum: vec![0.0; size],
            delta: (vec![0.0; size], vec![0.0; size]),
        }
    }

    /// Zero all buffers for reuse with the next flop.
    fn reset(&mut self) {
        self.regret_sum.fill(0.0);
        self.strategy_sum.fill(0.0);
        // delta buffers are zeroed per-iteration by parallel_traverse_pooled,
        // but zero them here too for clean state between flops.
        self.delta.0.fill(0.0);
        self.delta.1.fill(0.0);
    }
}
```

**Step 2: Change `exhaustive_solve_one_flop` to accept `&mut FlopBuffers`**

Replace the function signature (lines 612-623) and the internal allocation (lines 624-647):

Old signature:
```rust
fn exhaustive_solve_one_flop(
    tree: &PostflopTree,
    layout: &PostflopLayout,
    equity_table: &[f64],
    num_iterations: usize,
    convergence_threshold: f64,
    flop_name: &str,
    dcfr: &DcfrParams,
    config: &PostflopModelConfig,
    on_progress: &(impl Fn(BuildPhase) + Sync),
    counters: Option<&SolverCounters>,
) -> FlopSolveResult {
    let buf_size = layout.total_size;
    let mut regret_sum = vec![0.0f64; buf_size];
    let mut strategy_sum = vec![0.0f64; buf_size];
    ...
    let mut buf = (vec![0.0f64; buf_size], vec![0.0f64; buf_size]);
```

New signature:
```rust
fn exhaustive_solve_one_flop(
    tree: &PostflopTree,
    layout: &PostflopLayout,
    equity_table: &[f64],
    num_iterations: usize,
    convergence_threshold: f64,
    flop_name: &str,
    dcfr: &DcfrParams,
    config: &PostflopModelConfig,
    on_progress: &(impl Fn(BuildPhase) + Sync),
    counters: Option<&SolverCounters>,
    bufs: &mut FlopBuffers,
) -> FlopSolveResult {
```

Remove the 3 `vec!` allocation lines (625, 626, 647). Replace all references:
- `regret_sum` → `bufs.regret_sum`
- `strategy_sum` → `bufs.strategy_sum`
- `buf` → `bufs.delta`

In the return value, clone `strategy_sum` so `FlopSolveResult` still owns it (preserves MCCFR compatibility and test access):
```rust
FlopSolveResult {
    strategy_sum: bufs.strategy_sum.clone(),
    delta: current_exploitability,
    iterations_used,
}
```

**Step 3: Run tests to verify compilation**

Run: `cargo test -p poker-solver-core exhaustive_solve --no-run 2>&1 | tail -5`
Expected: Compilation errors in tests (they still use old signature) — that's expected, fixed in Task 2.

**Step 4: Commit**

```bash
git add crates/core/src/preflop/postflop_exhaustive.rs
git commit -m "refactor: add FlopBuffers struct and update exhaustive_solve_one_flop signature"
```

---

### Task 2: Update all callers — `build_exhaustive` and tests

**Files:**
- Modify: `crates/core/src/preflop/postflop_exhaustive.rs`

**Step 1: Add imports at top of file (line 6, after `use rayon::prelude::*`)**

```rust
use rayon::iter::ThreadLocal;
use std::cell::RefCell;
```

Note: `rayon::iter::ThreadLocal` is re-exported from `rayon`. If not available at that path, use `rayon::ThreadLocal` directly. Check rayon docs — the type is at `rayon::iter::ThreadLocal` in rayon 1.10.

**Step 2: Update `build_exhaustive` (lines 914-941)**

Replace the `par_iter` closure to use thread-local buffers:

```rust
    let buf_size = layout.total_size;
    let tl_bufs: ThreadLocal<RefCell<FlopBuffers>> = ThreadLocal::new();

    let results: Vec<Vec<f64>> = (0..num_flops)
        .into_par_iter()
        .map(|flop_idx| {
            let flop = flops[flop_idx];
            let flop_name = format!("{}{}{}", flop[0], flop[1], flop[2]);

            let equity_table = if let Some(tables) = pre_equity_tables {
                tables[flop_idx].clone()
            } else {
                let combo_map = build_combo_map(&flop);
                compute_equity_table(&combo_map, flop)
            };

            let cell = tl_bufs.get_or(|| RefCell::new(FlopBuffers::new(buf_size)));
            let mut bufs = cell.borrow_mut();
            bufs.reset();

            let result = exhaustive_solve_one_flop(
                tree,
                layout,
                &equity_table,
                num_iterations,
                config.cfr_convergence_threshold,
                &flop_name,
                &dcfr,
                config,
                on_progress,
                counters,
                &mut bufs,
            );

            let values =
                exhaustive_extract_values(tree, layout, &result.strategy_sum, &equity_table);

            on_progress(BuildPhase::FlopProgress {
                flop_name: flop_name.clone(),
                stage: FlopStage::Done,
            });
            let done = completed.fetch_add(1, Ordering::Relaxed) + 1;
            on_progress(BuildPhase::MccfrFlopsCompleted {
                completed: done,
                total: num_flops,
            });

            values
        })
        .collect();
```

**Step 3: Update all test call sites**

Each test that calls `exhaustive_solve_one_flop` needs to create a `FlopBuffers` and pass it. There are 6 call sites in tests (lines ~1154, ~1188, ~1385, ~1413, ~1423, ~1491). For each, add before the call:

```rust
let mut bufs = FlopBuffers::new(layout.total_size);
```

And append `&mut bufs` as the last argument to `exhaustive_solve_one_flop(...)`.

For tests that previously accessed `result.strategy_sum`, they still work since the result clones it.

**Step 4: Run all tests**

Run: `cargo test -p poker-solver-core 2>&1 | tail -10`
Expected: All tests pass (ignored tests stay ignored).

**Step 5: Commit**

```bash
git add crates/core/src/preflop/postflop_exhaustive.rs
git commit -m "feat: thread-local buffer pooling in exhaustive flop solver

Buffers (regret_sum, strategy_sum, delta pair) are now allocated once per
rayon worker thread and reused across flops via ThreadLocal<RefCell<FlopBuffers>>.
Eliminates 1,755 × 277 MB allocation churn for SPR=6 trees."
```

---

### Task 3: Eliminate the strategy_sum clone in the hot path

**Files:**
- Modify: `crates/core/src/preflop/postflop_exhaustive.rs`

The clone in `FlopSolveResult { strategy_sum: bufs.strategy_sum.clone(), ... }` copies 69 MB per flop. Since `build_exhaustive` calls `exhaustive_extract_values` immediately after and never uses `strategy_sum` again, we can extract values before dropping the borrow.

**Step 1: Change `exhaustive_solve_one_flop` return type**

Replace the return type and return statement. Instead of returning `FlopSolveResult`, return just the metadata:

```rust
) -> (f64, usize) {  // (exploitability, iterations_used)
    ...
    (current_exploitability, iterations_used)
}
```

**Step 2: Update `build_exhaustive` to extract values from pooled buffer**

```rust
            let (delta, iterations_used) = exhaustive_solve_one_flop(
                tree, layout, &equity_table, num_iterations,
                config.cfr_convergence_threshold, &flop_name, &dcfr,
                config, on_progress, counters, &mut bufs,
            );

            // Extract values while strategy_sum is still in the pooled buffer.
            let values =
                exhaustive_extract_values(tree, layout, &bufs.strategy_sum, &equity_table);
```

**Step 3: Update tests**

Tests that used `result.strategy_sum` now read from `bufs.strategy_sum` instead:

```rust
// Before:
let result = exhaustive_solve_one_flop(...);
assert!(result.iterations_used > 0);
let has_nonzero = result.strategy_sum.iter().any(|&v| v.abs() > 1e-15);

// After:
let (delta, iterations_used) = exhaustive_solve_one_flop(..., &mut bufs);
assert!(iterations_used > 0);
let has_nonzero = bufs.strategy_sum.iter().any(|&v| v.abs() > 1e-15);
```

For each test call site:
- `result.iterations_used` → `iterations_used`
- `result.strategy_sum` → `bufs.strategy_sum`
- `result.delta` → `delta`

**Step 4: Check if `FlopSolveResult` is still needed**

`FlopSolveResult` is still used by the MCCFR path (`postflop_mccfr.rs`). Leave it in `postflop_abstraction.rs`. Remove the import from `postflop_exhaustive.rs` line 11 if no longer referenced.

**Step 5: Run all tests**

Run: `cargo test -p poker-solver-core 2>&1 | tail -10`
Expected: All tests pass.

Run: `cargo clippy -p poker-solver-core 2>&1 | tail -10`
Expected: No new warnings (unused import, etc).

**Step 6: Commit**

```bash
git add crates/core/src/preflop/postflop_exhaustive.rs
git commit -m "perf: eliminate 69 MB strategy_sum clone per flop

Extract values directly from pooled buffer instead of cloning strategy_sum
into FlopSolveResult. The exhaustive solver now returns (delta, iterations_used)
tuple instead."
```

---

### Task 4: Verify with a real solve

**Step 1: Run the actual solver with full config**

Run: `cargo run -p poker-solver-trainer --release -- solve-postflop -c sample_configurations/full.yaml -o ./local_data/full_postflop_test 2>&1 | head -30`

Verify:
- No panics or crashes
- Flops start appearing in TUI without the long initial pause
- SPR=6 flops progress through iterations normally
- Memory usage is bounded (check with `htop` or similar — should be ~N_threads × 277 MB, not growing)

**Step 2: Compare output quality**

If you have a previous run's output, compare hand-average values to confirm numerical equivalence. The solver is deterministic for the exhaustive backend, so values should match exactly.

---

## Agent Team & Execution Order

1. **rust-developer** (single agent, sequential tasks): Tasks 1-3 are tightly coupled edits to one file. One agent handles all three in sequence.
2. **User verification**: Task 4 requires running the full solver interactively.
3. **code-reviewer**: After Task 3, review the final diff for correctness.
