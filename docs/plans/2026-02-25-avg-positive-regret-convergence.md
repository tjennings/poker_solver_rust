# Average Positive Regret Convergence Metric

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace strategy delta with average positive regret as the convergence metric and early-stopping criterion for both postflop MCCFR and preflop LCFR training.

**Architecture:** Add `avg_positive_regret()` methods to both `MccfrSolver` and `PreflopSolver`, wire them through the `TrainingSolver` trait, replace all strategy-delta-based convergence reporting and stopping with the new metric. Remove the previous-strategy snapshot machinery (no longer needed since the metric reads directly from regret buffers).

**Tech Stack:** Rust, serde (YAML config), indicatif (progress bars)

---

### Task 1: Replace `avg_regret` with positive-only regret in convergence.rs

**Files:**
- Modify: `crates/core/src/cfr/convergence.rs:79-102`

**Step 1: Update the `avg_regret` function to sum only positive regrets**

Change the body of `avg_regret` to filter `r > 0.0` instead of using `r.abs()`. Update the doc comment to reflect the new semantics.

```rust
/// Mean positive regret per iteration across all info sets and actions.
///
/// Sums only positive regrets (actions the player regrets not playing more),
/// then divides by the total number of regret entries and the iteration count.
/// This bounds exploitability: as avg positive regret → 0, strategy → Nash.
///
/// Returns 0.0 if the regret map is empty or iterations is 0.
#[must_use]
#[allow(clippy::implicit_hasher)]
pub fn avg_regret(regret_sum: &FxHashMap<u64, Vec<f64>>, iterations: u64) -> f64 {
    if iterations == 0 {
        return 0.0;
    }

    let mut total = 0.0;
    let mut count = 0u64;

    for regrets in regret_sum.values() {
        for &r in regrets {
            if r > 0.0 {
                total += r;
            }
            count += 1;
        }
    }

    if count == 0 {
        0.0
    } else {
        #[allow(clippy::cast_precision_loss)]
        {
            total / count as f64 / iterations as f64
        }
    }
}
```

**Step 2: Update tests for the new behavior**

Update `avg_regret_normalized` — with inputs `[10.0, 30.0]` at 10 iterations, old result was `(10+30)/2/10 = 2.0`. New result: `(10+30)/2/10 = 2.0` (both positive, same result).

Update `avg_regret_multiple_info_sets` — with `[(20.0, 40.0), (0.0, 60.0)]` at 10 iterations, old: `(20+40+0+60)/4/10 = 3.0`. New: positive only = `(20+40+0+60)/4/10 = 3.0` (all non-negative, same result).

Add a new test with negative regrets to verify filtering:

```rust
#[test]
fn avg_regret_ignores_negative_regrets() {
    let regrets = make_map(&[(1, vec![10.0, -50.0, 30.0])]);
    // positive sum = 10 + 30 = 40, count = 3, iterations = 10
    // result = 40 / 3 / 10 = 1.333...
    let result = avg_regret(&regrets, 10);
    let expected = 40.0 / 3.0 / 10.0;
    assert!(
        (result - expected).abs() < 1e-10,
        "expected {expected}, got {result}"
    );
}
```

**Step 3: Run tests to verify**

Run: `cargo test -p poker-solver-core cfr::convergence`
Expected: All tests pass.

**Step 4: Commit**

```
feat: change avg_regret to sum only positive regrets

Avg positive regret directly bounds exploitability and doesn't require
storing a previous strategy snapshot. This prepares for replacing
strategy delta as the convergence metric.
```

---

### Task 2: Add `avg_positive_regret()` to `MccfrSolver`

**Files:**
- Modify: `crates/core/src/cfr/mccfr.rs` (near line 455, next to `regret_sum()`)
- Modify: `crates/core/tests/convergence_metrics_test.rs`

**Step 1: Write a test that calls avg_positive_regret on a trained solver**

Add to `convergence_metrics_test.rs` (which already imports `MccfrSolver` and trains Kuhn poker):

```rust
#[test]
fn avg_positive_regret_decreases_over_training() {
    let game = KuhnPoker::new();
    let mut solver = MccfrSolver::new(game);
    solver.set_seed(42);

    solver.train_full(200);
    let early = solver.avg_positive_regret();

    solver.train_full(19800);
    let late = solver.avg_positive_regret();

    assert!(early > 0.0, "early regret should be positive, got {early}");
    assert!(
        late < early,
        "avg positive regret should decrease: early={early:.6}, late={late:.6}"
    );
}

#[test]
fn avg_positive_regret_zero_on_empty_solver() {
    let game = KuhnPoker::new();
    let solver = MccfrSolver::new(game);
    assert!((solver.avg_positive_regret()).abs() < 1e-10);
}
```

**Step 2: Run tests to verify they fail**

Run: `cargo test -p poker-solver-core avg_positive_regret`
Expected: FAIL — method does not exist yet.

**Step 3: Implement `avg_positive_regret()` on `MccfrSolver`**

Add after the `regret_sum()` method (~line 457):

```rust
/// Average positive regret per info-set-action per iteration.
///
/// Sums `max(R[I][a], 0)` across all info sets and actions, then divides
/// by `iterations * total_entries`. Directly bounds exploitability:
/// as this metric → 0, the average strategy → Nash equilibrium.
#[must_use]
pub fn avg_positive_regret(&self) -> f64 {
    if self.iterations == 0 {
        return 0.0;
    }

    let mut total = 0.0;
    let mut count = 0u64;

    for regrets in self.regret_sum.values() {
        for &r in regrets {
            if r > 0.0 {
                total += r;
            }
            count += 1;
        }
    }

    if count == 0 {
        return 0.0;
    }

    #[allow(clippy::cast_precision_loss)]
    {
        total / count as f64 / self.iterations as f64
    }
}
```

**Step 4: Run tests to verify they pass**

Run: `cargo test -p poker-solver-core avg_positive_regret`
Expected: PASS

**Step 5: Commit**

```
feat: add avg_positive_regret() to MccfrSolver
```

---

### Task 3: Add `avg_positive_regret()` to `PreflopSolver`

**Files:**
- Modify: `crates/core/src/preflop/solver.rs` (near line 263, next to `regret_at()`)
- Create: `crates/core/tests/preflop_avg_regret_test.rs`

**Step 1: Write a failing test**

Create `crates/core/tests/preflop_avg_regret_test.rs`:

```rust
use poker_solver_core::preflop::{PreflopConfig, PreflopSolver};

#[test]
fn preflop_avg_positive_regret_decreases() {
    let config = PreflopConfig::default();
    let mut solver = PreflopSolver::new(&config);

    solver.train(100);
    let early = solver.avg_positive_regret();

    solver.train(900);
    let late = solver.avg_positive_regret();

    assert!(early > 0.0, "early regret should be positive, got {early}");
    assert!(
        late < early,
        "avg positive regret should decrease: early={early:.6}, late={late:.6}"
    );
}

#[test]
fn preflop_avg_positive_regret_zero_before_training() {
    let config = PreflopConfig::default();
    let solver = PreflopSolver::new(&config);
    assert!((solver.avg_positive_regret()).abs() < 1e-10);
}
```

**Step 2: Run to verify failure**

Run: `cargo test -p poker-solver-core preflop_avg_positive_regret`
Expected: FAIL — method does not exist.

**Step 3: Implement on `PreflopSolver`**

Add after `strategy_sum_at()` (~line 275):

```rust
/// Average positive regret per slot per iteration.
///
/// Iterates the flat regret buffer, sums all positive values, divides by
/// `iteration * buffer_length`. Same metric as `MccfrSolver::avg_positive_regret`.
#[must_use]
pub fn avg_positive_regret(&self) -> f64 {
    if self.iteration == 0 || self.regret_sum.is_empty() {
        return 0.0;
    }

    let mut total = 0.0;
    for &r in &self.regret_sum {
        if r > 0.0 {
            total += r;
        }
    }

    #[allow(clippy::cast_precision_loss)]
    {
        total / self.regret_sum.len() as f64 / self.iteration as f64
    }
}
```

**Step 4: Run tests**

Run: `cargo test -p poker-solver-core preflop_avg_positive_regret`
Expected: PASS

**Step 5: Commit**

```
feat: add avg_positive_regret() to PreflopSolver
```

---

### Task 4: Rename config field `convergence_threshold` → `regret_threshold`

**Files:**
- Modify: `crates/trainer/src/main.rs` — `TrainingConfig` struct (~line 347-354), `PreflopTrainingConfig` struct (~line 728-739), all references
- Modify: `sample_configurations/ultra_fast.yaml`
- Modify: `sample_configurations/smoke.yaml`
- Modify: `sample_configurations/fast_buckets.yaml`
- Modify: `sample_configurations/AKQr_vs_234r.yaml`
- Modify: `sample_configurations/preflop_medium.yaml`

**Step 1: Rename the field in `TrainingConfig` with backward-compat alias**

In `TrainingConfig` (~line 347):

```rust
    /// Average positive regret threshold for convergence-based stopping.
    /// When set, training continues until avg positive regret drops below
    /// this value, overriding `iterations`.
    #[serde(default, alias = "convergence_threshold")]
    regret_threshold: Option<f64>,
    /// How often (in iterations) to check convergence (default: 100).
    #[serde(default = "default_convergence_check_interval")]
    convergence_check_interval: u64,
```

**Step 2: Rename in `PreflopTrainingConfig` (~line 728)**

```rust
    #[serde(default = "default_regret_threshold", alias = "convergence_threshold")]
    pub regret_threshold: f64,
```

Update `default_convergence_threshold` → `default_regret_threshold` (keep same value 0.0001).

**Step 3: Update all references in main.rs**

Search-and-replace `convergence_threshold` → `regret_threshold` in all the non-serde code: struct field accesses, print statements, `TrainingLoopConfig`, `StrategyReportCtx`, etc. Update user-facing messages like:

- `"Training: converge to delta < {:.6}"` → `"Training: converge to avg +regret < {:.6}"`
- `"target delta < {:.6}"` → `"target regret < {:.6}"`

**Step 4: Update sample configuration files**

Replace `convergence_threshold:` → `regret_threshold:` in all 5 YAML files.

**Step 5: Verify it compiles and old configs still work**

Run: `cargo build -p poker-solver-trainer`
Expected: Compiles. Old YAML files with `convergence_threshold` still parse due to serde `alias`.

**Step 6: Commit**

```
refactor: rename convergence_threshold to regret_threshold

Serde alias preserves backward compatibility with existing YAML configs.
```

---

### Task 5: Wire avg_positive_regret into TrainingSolver trait and checkpoint reporting

**Files:**
- Modify: `crates/trainer/src/main.rs` — `TrainingSolver` trait (~line 1242), `MccfrTrainingSolver` (~line 1408), `SimpleTrainingSolver` (~line 1484), `StrategyReportCtx` (~line 3616), `compute_and_print_convergence` (~line 3691), `print_strategy_report` (~line 3639)

**Step 1: Add `avg_positive_regret` to `TrainingSolver` trait**

```rust
trait TrainingSolver {
    fn train_batch(&mut self, iterations: u64, callback: &dyn Fn(u64));
    fn all_strategies(&self) -> FxHashMap<u64, Vec<f64>>;
    fn iterations(&self) -> u64;
    fn checkpoint_prefix(&self) -> &str;
    fn checkpoint_report(
        &mut self,
        checkpoint_num: u64,
        is_convergence: bool,
        total_checkpoints: u64,
        total_iterations: u64,
    ) -> Option<f64>;

    /// Average positive regret per info-set-action per iteration.
    fn avg_positive_regret(&self) -> f64;
}
```

**Step 2: Implement in `MccfrTrainingSolver`**

```rust
fn avg_positive_regret(&self) -> f64 {
    self.solver.avg_positive_regret()
}
```

**Step 3: Implement in `SimpleTrainingSolver`**

The `SimpleTrainingSolverBackend` trait will also need the method. Add to the trait and implement for `SequenceCfrSolver`. For GPU solvers that don't have regret access, return `f64::MAX` (or compute from strategies — but for now, these solvers already report `max_regret` from GPU).

For simplicity, add a default to `SimpleTrainingSolverBackend`:

```rust
fn avg_positive_regret(&self) -> f64 { 0.0 }
```

And implement it for `SequenceCfrSolver` by delegating to the convergence module's `avg_regret()` using the solver's regret map.

**Step 4: Simplify `StrategyReportCtx` — remove `previous`, add regret metric**

Remove `previous: &'a Option<FxHashMap<u64, Vec<f64>>>` field.
Add `avg_positive_regret: f64` field.
Replace `convergence_threshold: Option<f64>` with `regret_threshold: Option<f64>`.

**Step 5: Update `compute_and_print_convergence`**

Replace the function body. No longer needs `previous` strategies or `strategy_delta()`:

```rust
fn compute_and_print_convergence(ctx: &StrategyReportCtx) {
    let entropy = convergence::strategy_entropy(ctx.strategies);

    println!("\nConvergence Metrics:");
    if let Some(mr) = ctx.max_regret {
        println!("  Max regret:       {mr:.6}");
    }
    println!("  Avg +regret:      {:.6}", ctx.avg_positive_regret);
    if let Some(threshold) = ctx.regret_threshold {
        let status = if ctx.avg_positive_regret < threshold {
            "CONVERGED"
        } else {
            "not converged"
        };
        println!("  Target regret:    {threshold:.6} ({status})");
    }
    println!("  Strategy entropy: {entropy:.4}");
}
```

Change the return type of `print_strategy_report` from `Option<f64>` to `()` since we no longer return delta. The convergence check now uses `solver.avg_positive_regret()` directly in the loop (Task 6).

**Step 6: Update `checkpoint_report` implementations**

In `MccfrTrainingSolver::checkpoint_report` and `SimpleTrainingSolver::checkpoint_report`:
- Remove `self.previous_strategies` / `self.previous` field and its `Some(strategies)` assignment
- Pass `avg_positive_regret: self.avg_positive_regret()` to `StrategyReportCtx`
- Change return type from `Option<f64>` to `f64` (always available, no Option needed)

**Step 7: Remove `previous_strategies` field from solver wrappers**

Remove from `MccfrTrainingSolver`:
```rust
    previous_strategies: Option<FxHashMap<u64, Vec<f64>>>,
```

Remove from `SimpleTrainingSolver`:
```rust
    previous: Option<FxHashMap<u64, Vec<f64>>>,
```

**Step 8: Verify compilation**

Run: `cargo build -p poker-solver-trainer`

**Step 9: Commit**

```
feat: wire avg_positive_regret into training loop reporting

Replaces strategy delta with avg positive regret in checkpoint output.
Removes previous-strategy snapshot machinery from solver wrappers.
```

---

### Task 6: Replace convergence stopping criterion in training loops

**Files:**
- Modify: `crates/trainer/src/main.rs` — `run_convergence_loop` (~line 1297), `TrainingSolver::checkpoint_report` return type, preflop training loop (~line 1140-1195)

**Step 1: Update `TrainingSolver::checkpoint_report` return type**

Change from `Option<f64>` to `f64`:

```rust
fn checkpoint_report(
    &mut self,
    checkpoint_num: u64,
    is_convergence: bool,
    total_checkpoints: u64,
    total_iterations: u64,
) -> f64;
```

**Step 2: Update `run_convergence_loop` to use avg_positive_regret**

```rust
fn run_convergence_loop<S: TrainingSolver>(
    solver: &mut S,
    threshold: f64,
    config: &TrainingLoopConfig<'_>,
) {
    let pb = ProgressBar::new_spinner();
    pb.set_style(
        ProgressStyle::with_template(
            "{spinner:.green} [{elapsed_precise}] {pos} iters ({per_sec})",
        )
        .expect("valid template"),
    );

    let mut checkpoint_num = 0u64;

    loop {
        solver.train_batch(config.check_interval, &|_| pb.inc(1));
        checkpoint_num += 1;

        let converged = pb.suspend(|| {
            solver.checkpoint_report(checkpoint_num, true, 0, 0);

            save_checkpoint(
                solver.all_strategies(),
                solver.iterations(),
                &format!("{}{checkpoint_num}", solver.checkpoint_prefix()),
                config.output_dir,
                config.bundle_config,
                config.boundaries,
            );

            solver.avg_positive_regret() < threshold
        });

        if converged {
            pb.finish_with_message(format!(
                "Converged after {} iterations (regret {:.6})",
                solver.iterations(),
                solver.avg_positive_regret(),
            ));
            break;
        }
    }
}
```

**Step 3: Update preflop training loop early-stopping**

In the preflop training function (~line 1140-1195), replace the matrix-delta early stopping with avg_positive_regret:

Replace:
```rust
let delta_threshold = convergence_threshold;
```
with:
```rust
let regret_threshold = convergence_threshold;
```

Replace the inner `pb.suspend` block. Instead of computing matrix delta and checking it:

```rust
pb.suspend(|| {
    print_preflop_matrices(
        &solver.strategy(), &tree, bb_node, done, None,
    );
    let apr = solver.avg_positive_regret();
    println!("  Avg +regret: {apr:.6}");
    if apr < regret_threshold {
        println!("Avg +regret {apr:.6} < {regret_threshold} — stopping early at iteration {done}");
        early_stop = true;
    }
});
```

Remove the `prev_matrices` variable and the `PrevMatrices` type alias.

**Step 4: Simplify `print_preflop_matrices`**

Change signature to remove the `prev` parameter and `max_delta` return. It now just prints the matrices without delta computation:

```rust
fn print_preflop_matrices(
    strategy: &PreflopStrategy,
    tree: &PreflopTree,
    bb_node: Option<u32>,
    iteration: u64,
)
```

Remove `lhe_viz::matrix_delta` calls and delta title formatting.

**Step 5: Verify compilation and run tests**

Run: `cargo build -p poker-solver-trainer && cargo test -p poker-solver-trainer`
Expected: Compiles, tests pass.

**Step 6: Commit**

```
feat: replace strategy delta with avg positive regret for early stopping

Convergence loop and preflop training now stop when avg positive regret
drops below the configured threshold. No longer requires storing
previous strategy snapshots.
```

---

### Task 7: Remove dead code

**Files:**
- Modify: `crates/core/src/cfr/convergence.rs` — remove `strategy_delta()` and its tests
- Modify: `crates/trainer/src/main.rs` — remove `print_preflop_matrices` delta logic, `PrevMatrices` type alias
- Modify: `crates/core/tests/convergence_metrics_test.rs` — remove `strategy_delta` usage

**Step 1: Remove `strategy_delta` from convergence.rs**

Delete the `strategy_delta` function (lines 14-45) and its 4 tests (`delta_zero_for_identical_maps`, `delta_nonzero_for_changes`, `delta_ignores_non_overlapping_keys`, `delta_averages_across_info_sets`).

**Step 2: Update convergence_metrics_test.rs**

Remove the `prev_strategies` variable, `deltas` vector, the `strategy_delta` call, and the assertion that deltas decrease. The test should now track `avg_positive_regret` instead:

```rust
let mut avg_regrets = Vec::new();
// ... in loop:
avg_regrets.push(solver.avg_positive_regret());
// ... after loop:
assert!(
    avg_regrets.last().unwrap() < avg_regrets.first().unwrap(),
    "Avg positive regret should decrease: first={:.6}, last={:.6}",
    avg_regrets.first().unwrap(),
    avg_regrets.last().unwrap()
);
```

**Step 3: Remove `PrevMatrices` type alias from main.rs**

Delete the type alias at line 1198 if no longer used.

**Step 4: Run full test suite**

Run: `cargo test`
Expected: All tests pass (except the 4 known timer failures).

**Step 5: Commit**

```
refactor: remove strategy_delta and dead convergence code

strategy_delta is no longer used now that convergence checking uses
avg positive regret directly from the solver's regret buffers.
```

---

### Task 8: Update documentation

**Files:**
- Modify: `docs/training.md` — update convergence config documentation

**Step 1: Update training docs**

Find references to `convergence_threshold` and update to document `regret_threshold`:
- Explain the metric (avg positive regret per info-set-action per iteration)
- Note that it bounds exploitability
- Document that `convergence_threshold` still works as an alias
- Note that threshold values are not comparable to the old strategy delta values — users should tune starting from the default

**Step 2: Commit**

```
docs: update training docs for regret_threshold metric
```

---

## Scope Notes

**Out of scope (follow-up):** The per-flop postflop abstraction builder (`postflop_bucketed.rs`, `postflop_mccfr.rs`) uses its own `weighted_avg_strategy_delta` and `cfr_delta_threshold` for per-flop CFR early stopping. Converting this to avg positive regret is a separate change since it operates on flat buffers with a different layout (`PostflopLayout`) and has its own convergence dynamics.
