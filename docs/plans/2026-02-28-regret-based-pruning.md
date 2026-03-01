# Regret-Based Pruning Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add regret-based pruning to the exhaustive postflop CFR backend, skipping subtree traversal for negative-regret actions at hero decision nodes.

**Architecture:** Three config fields (`prune_warmup`, `prune_explore_freq`, `regret_floor`) gate pruning in `exhaustive_cfr_traverse`. Pruning skips children with negative cumulative regret at hero nodes when (a) past warmup, (b) not an exploration iteration, and (c) at least one action has positive regret. A regret floor clamp is applied after DCFR discounting to bound unprune delay.

**Tech Stack:** Rust, serde (YAML config), rayon (existing parallelism unchanged)

---

### Task 1: Add Config Fields to PostflopModelConfig

**Files:**
- Modify: `crates/core/src/preflop/postflop_model.rs`

**Step 1: Add default functions**

After the existing `default_ev_convergence_threshold` function (line 41), add:

```rust
fn default_prune_warmup() -> usize {
    200
}
fn default_prune_explore_freq() -> usize {
    20
}
fn default_regret_floor() -> f64 {
    1_000_000.0
}
```

**Step 2: Add fields to PostflopModelConfig struct**

After the `cfr_variant` field (line 144), add:

```rust
    /// Iterations before regret-based pruning activates. 0 = no pruning.
    /// Pruning skips subtree traversal for negative-regret actions at hero
    /// decision nodes, saving ~3-5x compute after convergence begins.
    #[serde(default = "default_prune_warmup")]
    pub prune_warmup: usize,

    /// Explore all actions (disable pruning) every N iterations. Default: 20
    /// (i.e. prune 95% of iterations, explore 5%).
    #[serde(default = "default_prune_explore_freq")]
    pub prune_explore_freq: usize,

    /// Maximum magnitude of negative cumulative regret. Regrets are clamped to
    /// `>= -regret_floor` after each iteration. Bounds the delay before a
    /// pruned action can recover. Default: 1,000,000.
    #[serde(default = "default_regret_floor")]
    pub regret_floor: f64,
```

**Step 3: Add defaults to `fast()` and `exhaustive_fast()` presets**

In each preset constructor (`fast()`, `exhaustive_fast()`), add:
```rust
            prune_warmup: 0,
            prune_explore_freq: 20,
            regret_floor: 1_000_000.0,
```

Note: `prune_warmup: 0` in fast presets means pruning is disabled by default in tests (warmup > total iterations).

**Step 4: Verify compilation**

Run: `cargo check -p poker-solver-core`
Expected: compiles with no errors

**Step 5: Commit**

```bash
git add crates/core/src/preflop/postflop_model.rs
git commit -m "feat: add RBP config fields to PostflopModelConfig"
```

---

### Task 2: Add Pruning to exhaustive_cfr_traverse

**Files:**
- Modify: `crates/core/src/preflop/postflop_exhaustive.rs`

**Step 1: Add pruning parameters to `exhaustive_cfr_traverse` signature**

Add two parameters after `dcfr: &DcfrParams`:
```rust
    prune_active: bool,     // true if past warmup AND not an exploration iteration
```

This is a single bool computed by the caller so the recursive function doesn't need the config or iteration number for the pruning decision (it already has `iteration` for DCFR weights).

**Step 2: Add pruning logic in the hero branch**

Replace the hero action loop (lines 188-205) with:

```rust
            if is_hero {
                let mut action_values = [0.0f64; MAX_POSTFLOP_ACTIONS];

                // RBP: skip negative-regret subtrees if pruning is active
                // and at least one action has positive regret.
                let any_positive = prune_active
                    && (0..num_actions).any(|i| snapshot[start + i] > 0.0);

                for (i, &child) in children.iter().enumerate() {
                    if any_positive && snapshot[start + i] < 0.0 {
                        // Pruned: strategy[i] is already 0 from regret matching,
                        // so action_values[i] = 0 doesn't affect node_value.
                        continue;
                    }
                    action_values[i] = exhaustive_cfr_traverse(
                        tree,
                        layout,
                        equity_table,
                        snapshot,
                        dr,
                        ds,
                        child,
                        hero_hand,
                        opp_hand,
                        hero_pos,
                        reach_hero * strategy[i],
                        reach_opp,
                        iteration,
                        dcfr,
                        prune_active,
                    );
                }
```

The rest of the hero branch (node_value, regret/strategy updates) is unchanged.

**Step 3: Update the opponent branch recursive call**

In the opponent branch (line ~226), add `prune_active` to the recursive call.

**Step 4: Update the Chance node recursive call**

At line ~157, add `prune_active` to the recursive call.

**Step 5: Update `PostflopCfrCtx` and `ParallelCfr` impl**

The `PostflopCfrCtx` struct (find it in the same file) needs a `prune_active: bool` field, and its `traverse_pair` method needs to pass it through to `exhaustive_cfr_traverse`.

**Step 6: Verify compilation**

Run: `cargo check -p poker-solver-core`
Expected: compiles (tests may fail since callers don't pass the new param yet — that's Task 3)

**Step 7: Commit**

```bash
git add crates/core/src/preflop/postflop_exhaustive.rs
git commit -m "feat: add RBP pruning logic to exhaustive_cfr_traverse"
```

---

### Task 3: Thread Config Through and Add Regret Floor

**Files:**
- Modify: `crates/core/src/preflop/postflop_exhaustive.rs`

**Step 1: Update `exhaustive_solve_one_flop` signature**

Add `config: &PostflopModelConfig` parameter (or just the three RBP fields — config is cleaner). The function is already called with `config` available in the outer scope.

**Step 2: Compute `prune_active` in the iteration loop**

Inside the `for iter in 0..num_iterations` loop (line 534), before creating `PostflopCfrCtx`:

```rust
        let prune_active = config.prune_warmup > 0
            && iter >= config.prune_warmup
            && (config.prune_explore_freq == 0 || iter % config.prune_explore_freq != 0);
```

Pass this into `PostflopCfrCtx`.

**Step 3: Add regret floor after DCFR discounting**

After line 561 (`dcfr.floor_regrets`), add:

```rust
        // RBP regret floor: clamp negative regrets to bound unprune delay.
        if config.regret_floor > 0.0 && config.prune_warmup > 0 {
            let floor = -config.regret_floor;
            for v in regret_sum.iter_mut() {
                if *v < floor {
                    *v = floor;
                }
            }
        }
```

**Step 4: Update all callers of `exhaustive_solve_one_flop`**

The main caller at line ~744 already has `config` in scope — pass `config` through.

For test callers (lines ~914, ~947, ~1134, ~1161, ~1171, ~1195), pass `&PostflopModelConfig::exhaustive_fast()` (which has `prune_warmup: 0`, meaning pruning disabled — no behavior change in tests).

**Step 5: Verify all tests pass**

Run: `cargo test -p poker-solver-core`
Expected: all tests pass (pruning disabled in tests via `prune_warmup: 0`)

**Step 6: Commit**

```bash
git add crates/core/src/preflop/postflop_exhaustive.rs
git commit -m "feat: thread RBP config through solve loop and add regret floor"
```

---

### Task 4: Add Tests for Pruning Behavior

**Files:**
- Modify: `crates/core/src/preflop/postflop_exhaustive.rs` (test module)

**Step 1: Test that pruning produces valid strategy**

```rust
    #[test]
    fn exhaustive_solve_with_pruning_produces_strategy() {
        let config = PostflopModelConfig {
            bet_sizes: vec![1.0],
            max_raises_per_street: 0,
            postflop_solve_iterations: 50,
            prune_warmup: 10,
            prune_explore_freq: 5,
            regret_floor: 1_000_000.0,
            ..PostflopModelConfig::exhaustive_fast()
        };
        let tree = PostflopTree::build_with_spr(&config, 3.5).unwrap();
        let node_streets = annotate_streets(&tree);
        let n = NUM_CANONICAL_HANDS;
        let layout = PostflopLayout::build(&tree, &node_streets, n, n, n);
        let equity_table = synthetic_equity_table();
        let dcfr = DcfrParams::linear();

        let result = exhaustive_solve_one_flop(
            &tree, &layout, &equity_table, 50, 0.001, "test", &dcfr, true, &|_| {}, &config,
        );

        assert!(result.iterations_used > 0);
        let has_nonzero = result.strategy_sum.iter().any(|&v| v.abs() > 1e-15);
        assert!(has_nonzero, "strategy_sum should have non-zero entries with pruning");
    }
```

**Step 2: Test that regret floor clamps negative regrets**

```rust
    #[test]
    fn regret_floor_clamps_negative_values() {
        let config = PostflopModelConfig {
            bet_sizes: vec![1.0],
            max_raises_per_street: 0,
            postflop_solve_iterations: 100,
            prune_warmup: 5,
            prune_explore_freq: 10,
            regret_floor: 100.0, // small floor for testing
            ..PostflopModelConfig::exhaustive_fast()
        };
        let tree = PostflopTree::build_with_spr(&config, 3.5).unwrap();
        let node_streets = annotate_streets(&tree);
        let n = NUM_CANONICAL_HANDS;
        let layout = PostflopLayout::build(&tree, &node_streets, n, n, n);
        let equity_table = synthetic_equity_table();
        let dcfr = DcfrParams::linear();

        let result = exhaustive_solve_one_flop(
            &tree, &layout, &equity_table, 100, 0.0001, "test", &dcfr, true, &|_| {}, &config,
        );

        // All regrets should be >= -regret_floor
        for &v in &result.regret_sum {
            assert!(v >= -100.0 - 1e-9, "regret {v} below floor -100");
        }
    }
```

Note: `FlopSolveResult` currently may not expose `regret_sum`. If not, the test should verify the strategy is valid (non-NaN, sums roughly to 1 per infoset). The developer should check `FlopSolveResult` fields and adapt.

**Step 3: Run tests**

Run: `cargo test -p poker-solver-core`
Expected: all pass

**Step 4: Commit**

```bash
git add crates/core/src/preflop/postflop_exhaustive.rs
git commit -m "test: add RBP pruning and regret floor tests"
```

---

## Agent Team & Execution Order

| Step | Agent | Task | Parallel? |
|-|-|-|-|
| 1 | `rust-developer` | Tasks 1-4 (sequential, single agent) | No — each task builds on previous |
| 2 | `idiomatic-rust-enforcer` | Review all changes | After Task 4 |
| 3 | `rust-perf-reviewer` | Review pruning hot path | After Task 4, parallel with step 2 |
| 4 | `rust-developer` | Address review findings | After steps 2-3 |
| 5 | `idiomatic-rust-enforcer` + `rust-perf-reviewer` | Second review pass | After step 4 |
