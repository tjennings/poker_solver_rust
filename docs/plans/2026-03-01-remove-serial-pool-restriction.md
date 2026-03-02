# Remove Serial Pool Restriction in Exhaustive CFR

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Enable full CPU utilization during exhaustive postflop CFR by removing the 1-thread serial pool restriction.

**Architecture:** Remove the `use_inner_parallelism` parameter and serial pool from `exhaustive_solve_one_flop`. Rayon's work-stealing naturally handles nested `par_iter` — the outer flop-level and inner hand-pair-level parallelism share the global thread pool.

**Tech Stack:** Rust, rayon

---

## Agent Team & Execution Order

Single `rust-developer` agent — this is a 3-file surgical change.

---

### Task 1: Remove serial pool from `exhaustive_solve_one_flop`

**Files:**
- Modify: `crates/core/src/preflop/postflop_exhaustive.rs`

**Step 1: Remove `use_inner_parallelism` parameter from function signature**

Change line 544-554 from:
```rust
fn exhaustive_solve_one_flop(
    tree: &PostflopTree,
    layout: &PostflopLayout,
    equity_table: &[f64],
    num_iterations: usize,
    convergence_threshold: f64,
    flop_name: &str,
    dcfr: &DcfrParams,
    use_inner_parallelism: bool,
    config: &PostflopModelConfig,
    on_progress: &(impl Fn(BuildPhase) + Sync),
) -> FlopSolveResult {
```
to:
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
) -> FlopSolveResult {
```

**Step 2: Remove serial pool creation and usage**

Delete lines 576-588 (the serial pool creation):
```rust
    // When inner parallelism is disabled, run traverse and exploitability
    // on a 1-thread pool to avoid rayon nesting contention.
    let serial_pool = if !use_inner_parallelism {
        Some(
            rayon::ThreadPoolBuilder::new()
                .num_threads(1)
                .build()
                .expect("failed to build single-thread rayon pool"),
        )
    } else {
        None
    };
```

Replace lines 609-613 (traverse call):
```rust
        if let Some(pool) = &serial_pool {
            pool.install(|| parallel_traverse_into(&ctx, &pairs, &mut dr, &mut ds));
        } else {
            parallel_traverse_into(&ctx, &pairs, &mut dr, &mut ds);
        }
```
with:
```rust
        parallel_traverse_into(&ctx, &pairs, &mut dr, &mut ds);
```

Replace lines 638-642 (exploitability call):
```rust
            current_exploitability = if let Some(pool) = &serial_pool {
                pool.install(|| compute_exploitability(tree, layout, &strategy_sum, equity_table))
            } else {
                compute_exploitability(tree, layout, &strategy_sum, equity_table)
            };
```
with:
```rust
            current_exploitability = compute_exploitability(tree, layout, &strategy_sum, equity_table);
```

**Step 3: Update call site in `build_exhaustive`**

In `build_exhaustive` (line 822), remove the `num_flops == 1` argument:
```rust
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
            );
```

**Step 4: Update doc comment on `exhaustive_solve_one_flop`**

Replace lines 537-542:
```rust
/// Solve a single flop using exhaustive CFR with configurable iteration weighting.
///
/// When `use_inner_parallelism` is false, inner `parallel_traverse_into` and
/// `compute_exploitability` run on a single-thread pool. This avoids
/// 3-level rayon nesting when the outer `build_exhaustive` already
/// parallelises over multiple flops.
```
with:
```rust
/// Solve a single flop using exhaustive CFR with configurable iteration weighting.
///
/// Inner `parallel_traverse_into` and `compute_exploitability` use rayon's
/// global thread pool. When called from `build_exhaustive` (which parallelises
/// over flops), rayon's work-stealing distributes hand-pair work across all
/// available cores, dynamically rebalancing as flops converge at different rates.
```

**Step 5: Run tests**

Run: `cargo test -p poker-solver-core -- postflop_exhaustive`
Expected: All existing tests pass (no behavioral change).

**Step 6: Commit**

```bash
git add crates/core/src/preflop/postflop_exhaustive.rs
git commit -m "perf: remove serial pool restriction in exhaustive CFR

Allow rayon work-stealing to handle nested parallelism naturally.
With few flops (e.g. 2), the old 1-thread serial pool limited CPU
utilization to num_flops cores. Now all cores participate in
hand-pair traversal, dynamically rebalancing as flops converge."
```

---

### Task 2: Fix misleading progress label

**Files:**
- Modify: `crates/trainer/src/main.rs`

**Step 1: Fix progress bar label**

Change line 480:
```rust
                        phase_bar.set_message(format!("SPR={spr} MCCFR Solving"));
```
to:
```rust
                        phase_bar.set_message(format!("SPR={spr} Solving"));
```

**Step 2: Commit**

```bash
git add crates/trainer/src/main.rs
git commit -m "fix: correct misleading 'MCCFR Solving' progress label for exhaustive mode"
```

---

### Task 3: Verify

Run: `cargo clippy -p poker-solver-core -p poker-solver-trainer`
Run: `cargo test -p poker-solver-core`
Expected: Clean clippy, all tests pass.
