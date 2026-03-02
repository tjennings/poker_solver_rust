# Prune Threshold & Median Regret TUI Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a configurable negative regret threshold for pruning, and display median positive/negative regret per flop in the TUI.

**Architecture:** Add `prune_regret_threshold` to `PostflopModelConfig`, thread it through to the traversal pruning logic, compute median regrets from the regret buffer periodically, and pipe them through `FlopStage` → `FlopTuiState` → TUI display.

**Tech Stack:** Rust, serde (YAML config), ratatui (TUI)

---

## Agent Team & Execution Order

- **rust-developer (worktree):** All implementation tasks (sequential, one agent)
- **idiomatic-rust-enforcer:** Post-implementation review
- **rust-perf-reviewer:** Post-implementation review (median computation is in hot path)

---

### Task 1: Add `prune_regret_threshold` config field

**Files:**
- Modify: `crates/core/src/preflop/postflop_model.rs:43-48` (add default fn)
- Modify: `crates/core/src/preflop/postflop_model.rs:156-167` (add field after `prune_warmup`)
- Modify: `crates/core/src/preflop/postflop_model.rs:174-228` (add to all presets)

**Step 1: Add default function**

After `default_prune_explore_pct` (line 46-48), add:

```rust
fn default_prune_regret_threshold() -> f64 {
    0.0
}
```

**Step 2: Add field to struct**

After `prune_explore_pct` field (line 167), add:

```rust
    /// Cumulative regret threshold below which an action becomes a pruning
    /// candidate. Default: 0.0 (any negative regret triggers pruning).
    /// Set to e.g. -500.0 to only prune deeply negative actions.
    #[serde(default = "default_prune_regret_threshold")]
    pub prune_regret_threshold: f64,
```

**Step 3: Add to all presets**

In `fast()`, `standard()`, `exhaustive_fast()`, and `exhaustive_standard()` — add `prune_regret_threshold: 0.0` to each preset that explicitly lists fields. For `fast()` and `standard()` (which list all fields), add after `prune_explore_pct`. For `exhaustive_fast()` and `exhaustive_standard()` (which use `..Self::standard()`), no change needed since they inherit.

**Step 4: Run tests**

Run: `cargo test -p poker-solver-core --lib postflop_model`
Expected: PASS (serde default handles existing configs)

**Step 5: Commit**

```
feat(config): add prune_regret_threshold field
```

---

### Task 2: Use threshold in pruning logic

**Files:**
- Modify: `crates/core/src/preflop/postflop_exhaustive.rs:187-188` (add field to traverse fn)
- Modify: `crates/core/src/preflop/postflop_exhaustive.rs:243-263` (change pruning condition)
- Modify: `crates/core/src/preflop/postflop_exhaustive.rs:531-542` (add field to ctx struct)
- Modify: `crates/core/src/preflop/postflop_exhaustive.rs:650-658` (pass field in ctx)

**Step 1: Add `prune_regret_threshold` to `PostflopCfrCtx`**

In the `PostflopCfrCtx` struct (line ~532-542), add after `prune_explore_pct`:

```rust
    prune_regret_threshold: f64,
```

**Step 2: Pass it when constructing ctx**

In `exhaustive_solve_one_flop` (line ~650-658), add to the ctx construction:

```rust
                prune_regret_threshold: config.prune_regret_threshold,
```

**Step 3: Add to `exhaustive_cfr_traverse` signature**

Add `prune_regret_threshold: f64` parameter after `prune_explore_pct` (line ~188). Thread it through all recursive calls (lines ~224, ~291-292, ~331-332).

**Step 4: Change pruning condition**

In the pruning block (line 253), change:

```rust
} else if snapshot[start + i] < 0.0 {
```

to:

```rust
} else if snapshot[start + i] < prune_regret_threshold {
```

This way, with threshold=0.0, behavior is identical to current (any negative regret prunes). With threshold=-500.0, only actions with cumulative regret below -500 get pruned.

**Step 5: Run existing pruning tests**

Run: `cargo test -p poker-solver-core exhaustive_solve_with_pruning`
Run: `cargo test -p poker-solver-core pruning_does_not_break`
Expected: PASS (threshold defaults to 0.0, identical behavior)

**Step 6: Commit**

```
feat(pruning): use configurable prune_regret_threshold in traversal
```

---

### Task 3: Add median regret fields to `FlopStage` and `FlopTuiState`

**Files:**
- Modify: `crates/core/src/preflop/postflop_abstraction.rs:245-257` (add fields to `FlopStage::Solving`)
- Modify: `crates/trainer/src/tui_metrics.rs:7-12` (add fields to `FlopTuiState`)
- Modify: `crates/trainer/src/tui_metrics.rs:82-102` (add to `update_flop`)

**Step 1: Add fields to `FlopStage::Solving`**

After `pruned_action_slots` (line ~256), add:

```rust
        /// Median of positive regret values across all slots.
        median_positive_regret: f64,
        /// Median of negative regret values across all slots.
        median_negative_regret: f64,
```

**Step 2: Add fields to `FlopTuiState`**

After `pct_actions_pruned` (line 11), add:

```rust
    pub median_positive_regret: f64,
    pub median_negative_regret: f64,
```

**Step 3: Update `update_flop` signature and body**

Add `median_positive_regret: f64, median_negative_regret: f64` params to `update_flop` (line ~82-102). Set them on the entry alongside `pct_actions_pruned`. Initialize to 0.0 in `or_insert_with`.

**Step 4: Fix all compile errors**

Update all callers:
- `crates/trainer/src/main.rs:428-434` — destructure new fields from `FlopStage::Solving`, pass to `update_flop`
- `crates/trainer/src/main.rs:460` — destructure (ignore with `..`) in the log branch
- `crates/trainer/src/tui_metrics.rs` tests — add `0.0, 0.0` args to `update_flop` calls

**Step 5: Run tests**

Run: `cargo test -p poker-solver-trainer`
Run: `cargo test -p poker-solver-core postflop_abstraction`
Expected: PASS

**Step 6: Commit**

```
feat(tui): add median regret fields to FlopStage and FlopTuiState
```

---

### Task 4: Compute median regrets in solve loop

**Files:**
- Modify: `crates/core/src/preflop/postflop_exhaustive.rs:694-711` (compute + emit)

**Step 1: Add helper function**

Above `exhaustive_solve_one_flop`, add:

```rust
/// Compute median of positive and negative values in a regret buffer.
/// Returns (median_positive, median_negative). If no values exist for
/// a sign, returns 0.0.
fn median_regrets(regret_sum: &[f64]) -> (f64, f64) {
    let mut positives: Vec<f64> = Vec::new();
    let mut negatives: Vec<f64> = Vec::new();
    for &v in regret_sum {
        if v > 0.0 {
            positives.push(v);
        } else if v < 0.0 {
            negatives.push(v);
        }
    }
    let med_pos = if positives.is_empty() {
        0.0
    } else {
        positives.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
        positives[positives.len() / 2]
    };
    let med_neg = if negatives.is_empty() {
        0.0
    } else {
        negatives.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
        negatives[negatives.len() / 2]
    };
    (med_pos, med_neg)
}
```

**Step 2: Compute and pass in progress callback**

In the solve loop, around line 696-711, after computing exploitability and before the `on_progress` call, add median computation. Only compute every 10 iterations to avoid sorting overhead:

```rust
        let (med_pos, med_neg) = if iter % 10 == 0 || iter == num_iterations - 1 {
            median_regrets(&regret_sum)
        } else {
            (current_med_pos, current_med_neg)
        };
        current_med_pos = med_pos;
        current_med_neg = med_neg;
```

Add `current_med_pos` and `current_med_neg` as `let mut` variables initialized to `0.0` near the top of the function (with `current_exploitability`).

Add the median fields to the `FlopStage::Solving` emission:

```rust
                median_positive_regret: current_med_pos,
                median_negative_regret: current_med_neg,
```

**Step 3: Run tests**

Run: `cargo test -p poker-solver-core postflop_exhaustive`
Expected: PASS

**Step 4: Commit**

```
feat(exhaustive): compute median positive/negative regret per flop
```

---

### Task 5: Display median regrets in TUI

**Files:**
- Modify: `crates/trainer/src/tui.rs:208-212` (update prune_line format)

**Step 1: Update prune line**

Change the prune_line (lines 208-211) to include median regret:

```rust
            let prune_line = Line::from(Span::styled(
                format!(
                    "  pruned: {:.1}%  regret +{:.0} / {:.0}",
                    state.pct_actions_pruned,
                    state.median_positive_regret,
                    state.median_negative_regret,
                ),
                Style::default().fg(Color::DarkGray),
            ));
```

**Step 2: Run full build**

Run: `cargo build -p poker-solver-trainer`
Expected: compiles cleanly

**Step 3: Commit**

```
feat(tui): display median positive/negative regret per flop
```

---

### Task 6: Update sample config and add test

**Files:**
- Modify: `sample_configurations/20spr.yaml:14` (fix `prune_explore_freq` → add threshold)
- Modify: `sample_configurations/tiny.yaml` (add threshold example)

**Step 1: Update 20spr.yaml**

The config uses the old name `prune_explore_freq` which works via alias, but add the threshold:

```yaml
  prune_regret_threshold: 0  # 0 = prune any negative regret
```

**Step 2: Update tiny.yaml**

Add after `prune_explore_pct`:

```yaml
  prune_regret_threshold: 0  # cumulative regret below this triggers pruning
```

**Step 3: Run all tests**

Run: `cargo test -p poker-solver-core`
Run: `cargo test -p poker-solver-trainer`
Expected: PASS

**Step 4: Commit**

```
docs(config): add prune_regret_threshold to sample configs
```
