# EV-Based Postflop Rebucketing Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add post-solve EV rebucketing to the postflop abstraction so buckets capture strategy-dependent hand value (nut advantage, cooler dynamics), not just raw equity.

**Architecture:** The existing EHS-based postflop abstraction pipeline gains an outer loop: solve CFR with current buckets, extract per-hand EV histograms from the converged strategy, re-cluster on EV features, and re-solve. Each CFR solve uses early stopping via max-regret-delta. Progress uses `indicatif::MultiProgress` with per-flop bars.

**Tech Stack:** Rust, rayon (parallelism), indicatif 0.17 (MultiProgress), serde (config), existing k-means infra in `hand_buckets.rs`.

**Design doc:** `docs/plans/2026-02-23-ev-rebucketing-design.md`

---

## Task 1: Config — Add rebucketing fields to PostflopModelConfig

**Files:**
- Modify: `crates/core/src/preflop/postflop_model.rs`

**Step 1: Write failing tests**

Add to the existing `mod tests` block in `postflop_model.rs`:

```rust
#[timed_test]
fn rebucket_rounds_defaults_to_one() {
    let cfg = PostflopModelConfig::standard();
    assert_eq!(cfg.rebucket_rounds, 1);
}

#[timed_test]
fn rebucket_delta_threshold_defaults() {
    let cfg = PostflopModelConfig::standard();
    assert!((cfg.rebucket_delta_threshold - 0.001).abs() < 1e-9);
}

#[timed_test]
fn postflop_sprs_defaults_to_single_value() {
    let cfg = PostflopModelConfig::standard();
    assert_eq!(cfg.postflop_sprs.len(), 1);
    assert!((cfg.postflop_sprs[0] - 3.5).abs() < 1e-9);
}

#[timed_test]
fn postflop_spr_scalar_yaml_deserializes_as_vec() {
    let yaml = "postflop_spr: 4.0";
    let cfg: PostflopModelConfig = serde_yaml::from_str(yaml).unwrap();
    assert_eq!(cfg.postflop_sprs.len(), 1);
    assert!((cfg.postflop_sprs[0] - 4.0).abs() < 1e-9);
}

#[timed_test]
fn postflop_sprs_vec_yaml_deserializes() {
    let yaml = "postflop_sprs: [3.5, 6.0]";
    let cfg: PostflopModelConfig = serde_yaml::from_str(yaml).unwrap();
    assert_eq!(cfg.postflop_sprs.len(), 2);
    assert!((cfg.postflop_sprs[0] - 3.5).abs() < 1e-9);
    assert!((cfg.postflop_sprs[1] - 6.0).abs() < 1e-9);
}

#[timed_test]
fn rebucket_rounds_round_trip() {
    let mut cfg = PostflopModelConfig::fast();
    cfg.rebucket_rounds = 3;
    let yaml = serde_yaml::to_string(&cfg).unwrap();
    let restored: PostflopModelConfig = serde_yaml::from_str(&yaml).unwrap();
    assert_eq!(restored.rebucket_rounds, 3);
}
```

**Step 2: Run tests to verify they fail**

Run: `cargo test -p poker-solver-core postflop_model::tests::rebucket_rounds_defaults`
Expected: FAIL — field does not exist.

**Step 3: Implement config fields**

Add default functions:

```rust
fn default_rebucket_rounds() -> u16 { 1 }
fn default_rebucket_delta_threshold() -> f64 { 0.001 }
fn default_postflop_sprs() -> Vec<f64> { vec![3.5] }
```

Add fields to `PostflopModelConfig`:

```rust
#[serde(default = "default_rebucket_rounds")]
pub rebucket_rounds: u16,

#[serde(default = "default_rebucket_delta_threshold")]
pub rebucket_delta_threshold: f64,

/// SPR(s) for postflop solves. Supports scalar `postflop_spr: 3.5`
/// (backward compat) or list `postflop_sprs: [3.5, 6.0]`.
#[serde(default = "default_postflop_sprs", alias = "postflop_spr")]
pub postflop_sprs: Vec<f64>,
```

Remove the old `postflop_spr: f64` field. Update all presets (`fast()`, `medium()`, `standard()`, `accurate()`) to use `postflop_sprs: vec![3.5]` and include the new fields. Add a convenience method:

```rust
/// Primary SPR (first in the list). Used when a single SPR is needed.
pub fn primary_spr(&self) -> f64 {
    self.postflop_sprs.first().copied().unwrap_or(3.5)
}
```

**Step 4: Fix all compilation errors from `postflop_spr` → `postflop_sprs` rename**

Search for all references to `postflop_spr` and update:
- `postflop_abstraction.rs`: `config.postflop_spr` → `config.primary_spr()`
- `solve_cache.rs`: any references
- Existing tests referencing `postflop_spr`

**Step 5: Run tests to verify they pass**

Run: `cargo test -p poker-solver-core postflop_model`
Expected: all PASS.

**Step 6: Commit**

```
feat: add rebucket_rounds, delta_threshold, postflop_sprs config

Replaces scalar postflop_spr with Vec<f64> postflop_sprs (backward
compatible via serde alias). Adds rebucket_rounds (default 1, no
rebucketing) and rebucket_delta_threshold (default 0.001) for
EV-based postflop rebucketing.
```

---

## Task 2: Early stopping — Add max-regret-delta to solve_one_flop

**Files:**
- Modify: `crates/core/src/preflop/postflop_abstraction.rs`

**Step 1: Write failing tests**

Add to the `mod tests` block in `postflop_abstraction.rs`:

```rust
#[timed_test]
fn max_strategy_delta_detects_convergence() {
    // Two identical strategy buffers → delta should be 0
    let buf = vec![3.0, 1.0, 0.0, 0.5, 0.5, 0.0];
    let delta = max_strategy_delta(&buf, &buf, 3, 2);
    assert!(delta.abs() < 1e-12, "identical buffers should have zero delta");
}

#[timed_test]
fn max_strategy_delta_detects_change() {
    // First buffer: [3, 1, 0] for node with 3 actions → strategy [0.75, 0.25, 0.0]
    // Second buffer: [1, 1, 0] → strategy [0.5, 0.5, 0.0]
    let old = vec![3.0, 1.0, 0.0];
    let new = vec![1.0, 1.0, 0.0];
    let delta = max_strategy_delta(&old, &new, 3, 1);
    assert!((delta - 0.25).abs() < 1e-9, "delta should be 0.25, got {delta}");
}
```

**Step 2: Run tests to verify they fail**

Run: `cargo test -p poker-solver-core postflop_abstraction::tests::max_strategy_delta`
Expected: FAIL — function does not exist.

**Step 3: Implement max_strategy_delta**

Add a function that computes the max absolute change in regret-matched strategy between two regret buffers. It iterates over all decision nodes (using the layout), applies regret matching to both buffers, and returns the max absolute difference.

```rust
/// Compute maximum strategy probability change between two regret snapshots.
///
/// For each decision node, applies regret matching to both `old_regrets` and
/// `new_regrets`, then finds the max absolute difference across all
/// (node, bucket, action) triples. Returns 0.0 if buffers are identical.
fn max_strategy_delta(
    old_regrets: &[f64],
    new_regrets: &[f64],
    num_actions_max: usize,
    num_decision_nodes: usize,
    // Pass the layout to iterate over nodes
) -> f64 {
    // Implementation iterates over layout entries, applies regret_matching_into
    // to both buffers, computes max |old_strat[a] - new_strat[a]|
}
```

Actually, we need the layout to know slot offsets. Simpler approach: pass the full layout and iterate entries. The function signature should be:

```rust
fn max_strategy_delta_from_layout(
    old_regrets: &[f64],
    new_regrets: &[f64],
    layout: &PostflopLayout,
    tree: &PostflopTree,
) -> f64
```

For each decision node in tree, get the slot for bucket 0..num_buckets, apply regret matching, compare. But this requires knowing num_buckets per street. Simpler: just compare regret-matched strategies for ALL slots contiguously — iterate through the layout entries.

Simplest approach that avoids layout complexity in tests: compute delta inside `solve_one_flop` directly by snapshotting the regret buffer before and after each iteration and computing the max change in regret-matched output. This keeps the delta computation co-located with the solve loop.

**Revised implementation:** Modify `solve_one_flop` to:
1. Accept a `delta_threshold: f64` parameter
2. After each iteration, snapshot the new `regret_sum` and compare regret-matched strategies with the previous snapshot
3. Return `(strategy_sum, final_delta, iterations_used)` instead of just `strategy_sum`
4. Break early when delta < threshold

```rust
/// Result of a single flop CFR solve.
pub(crate) struct FlopSolveResult {
    pub strategy_sum: Vec<f64>,
    pub final_delta: f64,
    pub iterations_used: usize,
}
```

For the delta computation inside the loop: after each iteration, apply regret matching to each slot in the current `regret_sum` and compare with the previous iteration's regret-matched strategies. Track the max absolute change.

To avoid double-allocating a full strategy buffer each iteration, use a single `prev_strategy` buffer that gets swapped. Compute regret matching for all slots into `prev_strategy` at iteration end, compare with the previous one.

**Step 4: Run tests**

Run: `cargo test -p poker-solver-core postflop_abstraction::tests::max_strategy_delta`
Expected: PASS.

**Step 5: Update solve_one_flop with early stopping**

Modify `solve_one_flop` to accept `delta_threshold`, compute delta each iteration, and break when converged. Return `FlopSolveResult`.

**Step 6: Update solve_postflop_per_flop to pass threshold and collect results**

Change return type from `Vec<Vec<f64>>` to `Vec<FlopSolveResult>`. Update `compute_postflop_values` to accept `FlopSolveResult` (just uses `.strategy_sum`).

**Step 7: Run full test suite**

Run: `cargo test -p poker-solver-core`
Expected: all existing tests still pass.

**Step 8: Commit**

```
feat: add max-regret-delta early stopping to postflop CFR solve

solve_one_flop now tracks the max change in regret-matched strategy
between iterations and stops early when delta drops below threshold.
Returns FlopSolveResult with strategy_sum, final_delta, and
iterations_used.
```

---

## Task 3: Progress reporting — Extend BuildPhase and update trainer

**Files:**
- Modify: `crates/core/src/preflop/postflop_abstraction.rs` (BuildPhase enum)
- Modify: `crates/trainer/src/main.rs` (MultiProgress handling)

**Step 1: Extend BuildPhase enum**

Replace the current `SolvingPostflop(usize, usize)` variant with the richer version:

```rust
pub enum BuildPhase {
    HandBuckets(usize, usize),
    EquityTable,
    Trees,
    Layout,
    SolvingPostflop {
        round: u16,
        total_rounds: u16,
        flop_name: String,
        iteration: usize,
        max_iterations: usize,
        delta: f64,
    },
    ExtractingEv(usize, usize),
    Rebucketing(u16, u16),
    ComputingValues,
}
```

Update the `Display` impl for `BuildPhase` to format the new variants:
- `SolvingPostflop` → `"[1/2] Flop 'AhKd7s' iter 45/200 δ=0.0032"`
- `ExtractingEv` → `"EV histograms (120/169)"`
- `Rebucketing` → `"Rebucketing (round 2/3)"`

**Step 2: Update the on_progress callback in solve_postflop_per_flop**

Currently uses `Fn(usize, usize)` (step, total). Change to `Fn(BuildPhase)` so each per-flop solve can report its own flop name and delta. This requires:
- Passing flop names (as `Vec<String>`) into `solve_postflop_per_flop`
- Each `solve_one_flop` call reports via `BuildPhase::SolvingPostflop { .. }`

To format flop names: `rs_poker::core::Card` implements `Display`, so format as `format!("{}{}{}", flop[0], flop[1], flop[2])`.

**Step 3: Update trainer main.rs to use MultiProgress**

Replace the single `ProgressBar` with `indicatif::MultiProgress`. The progress callback from `PostflopAbstraction::build()` now receives `BuildPhase` variants. For `SolvingPostflop`, create/update a per-flop progress bar.

Key changes:
- `use indicatif::{MultiProgress, ProgressBar, ProgressStyle};`
- Create `MultiProgress` before the build call
- Maintain a `HashMap<String, ProgressBar>` for per-flop bars
- On `SolvingPostflop { flop_name, .. }`: get-or-create bar for that flop, update position and message
- On phase transitions (HandBuckets, ExtractingEv, etc.): clear flop bars, show phase bar

**Step 4: Run the trainer to verify progress output visually**

Run: `cargo run -p poker-solver-trainer --release -- solve-preflop -c sample_configurations/smoke.yaml`
Expected: multi-bar progress output with flop names and delta values.

**Step 5: Commit**

```
feat: rich progress reporting with per-flop bars and delta display

BuildPhase::SolvingPostflop now carries round, flop_name, iteration,
and current delta. Trainer uses indicatif::MultiProgress with parallel
bars per concurrent flop solve.
```

---

## Task 4: EV histogram extraction — build_ev_histograms

**Files:**
- Modify: `crates/core/src/preflop/hand_buckets.rs`

**Step 1: Write failing tests**

```rust
#[timed_test]
fn ev_histogram_produces_valid_cdf() {
    // Simulate: 2 flop buckets, hero bucket 0 gets EV [0.3, 0.7] vs opp buckets [0, 1]
    let ev_per_opp = vec![0.3, 0.7];
    let hist = ev_values_to_histogram(&ev_per_opp);
    // Should be valid CDF: monotonically non-decreasing, ends at 1.0
    for i in 1..HISTOGRAM_BINS {
        assert!(hist[i] >= hist[i - 1] - 1e-9, "CDF not monotonic at bin {i}");
    }
    assert!((hist[HISTOGRAM_BINS - 1] - 1.0).abs() < 1e-6, "CDF must end at 1.0");
}

#[timed_test]
fn ev_histogram_all_same_value() {
    let ev_per_opp = vec![0.5, 0.5, 0.5];
    let hist = ev_values_to_histogram(&ev_per_opp);
    assert!((hist[HISTOGRAM_BINS - 1] - 1.0).abs() < 1e-6);
}

#[timed_test]
fn ev_histogram_empty_returns_nan() {
    let ev_per_opp: Vec<f64> = vec![];
    let hist = ev_values_to_histogram(&ev_per_opp);
    assert!(hist[0].is_nan(), "empty input should produce NaN sentinel");
}
```

**Step 2: Run tests to verify they fail**

Run: `cargo test -p poker-solver-core hand_buckets::tests::ev_histogram`
Expected: FAIL — function does not exist.

**Step 3: Implement ev_values_to_histogram**

```rust
/// Convert a vector of EV values into a histogram CDF (same format as equity histograms).
///
/// EV values are normalized to [0, 1] range using min/max of the input,
/// then binned into HISTOGRAM_BINS equal-width bins. Returns NaN sentinel
/// if input is empty.
#[must_use]
pub fn ev_values_to_histogram(ev_values: &[f64]) -> HistogramFeatures {
    use crate::preflop::ehs::{HISTOGRAM_BINS, counts_to_cdf};

    if ev_values.is_empty() {
        return [f64::NAN; HISTOGRAM_BINS];
    }

    let min_ev = ev_values.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_ev = ev_values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let range = max_ev - min_ev;

    let mut counts = [0u32; HISTOGRAM_BINS];
    for &ev in ev_values {
        let normalized = if range > 1e-12 { (ev - min_ev) / range } else { 0.5 };
        let bin = (normalized * HISTOGRAM_BINS as f64) as usize;
        counts[bin.min(HISTOGRAM_BINS - 1)] += 1;
    }
    counts_to_cdf(&counts, ev_values.len())
}
```

**Step 4: Implement build_ev_histograms — extract EV distribution per (hand, flop)**

```rust
/// Extract per-hand EV histograms from converged per-flop strategy sums.
///
/// For each (hand, flop), looks up the hand's bucket assignment, then collects
/// the EV this bucket achieves against every opponent bucket (from the value table).
/// These EVs are binned into a histogram CDF for clustering.
///
/// Returns flat `Vec<HistogramFeatures>` indexed by `hand_idx * num_flops + flop_idx`.
#[must_use]
pub fn build_ev_histograms(
    buckets: &StreetBuckets,
    values: &super::postflop_abstraction::PostflopValues,
    num_hands: usize,
    num_flop_buckets: usize,
) -> Vec<HistogramFeatures>
```

For each (hand_idx, flop_idx):
1. Look up `hero_bucket = buckets.flop_bucket_for_hand(hand_idx, flop_idx)`
2. Collect EVs: for each `opp_bucket` in 0..num_flop_buckets, get `values.get_by_flop(flop_idx, 0, hero_bucket, opp_bucket)` (position 0; average both positions)
3. Call `ev_values_to_histogram(&evs)` to produce the CDF

**Step 5: Run tests**

Run: `cargo test -p poker-solver-core hand_buckets::tests`
Expected: PASS.

**Step 6: Commit**

```
feat: add EV histogram extraction for postflop rebucketing

ev_values_to_histogram converts per-opponent-bucket EV values into
histogram CDFs. build_ev_histograms extracts these for all (hand, flop)
pairs from converged PostflopValues, producing features for k-means
rebucketing.
```

---

## Task 5: Rebucketing loop — Wire it into PostflopAbstraction::build

**Files:**
- Modify: `crates/core/src/preflop/postflop_abstraction.rs`

**Step 1: Write failing test**

```rust
#[timed_test]
fn build_with_rebucket_rounds_1_matches_current_behavior() {
    let mut config = PostflopModelConfig::fast();
    config.rebucket_rounds = 1;
    config.postflop_solve_iterations = 10;
    let result = PostflopAbstraction::build(&config, None, None, |_| {});
    assert!(result.is_ok());
    let abs = result.unwrap();
    assert!(!abs.values.is_empty());
}
```

**Step 2: Implement the rebucketing outer loop**

Restructure `PostflopAbstraction::build()`:

```rust
pub fn build(config, equity_table, cache_base, on_progress) -> Result<Self, _> {
    // Phase 1: Initial EHS bucketing (same as current)
    let (mut buckets, mut street_equity, flops) = load_or_build_abstraction(config, &on_progress)?;

    // Build tree once (shared across all rounds)
    on_progress(BuildPhase::Trees);
    let tree = PostflopTree::build_with_spr(config, config.primary_spr())?;
    let node_streets = annotate_streets(&tree);
    let layout = PostflopLayout::build(/* ... */);

    let total_rounds = config.rebucket_rounds;

    for round in 1..=total_rounds {
        // Solve postflop per flop
        on_progress(BuildPhase::SolvingPostflop { round, total_rounds, .. });
        let solve_results = solve_postflop_per_flop(
            &tree, &layout, &street_equity,
            num_flop_b, config.postflop_solve_iterations,
            samples, config.rebucket_delta_threshold,
            round, total_rounds, &flop_names,
            &on_progress,
        );

        // If not the last round, extract EV histograms and rebucket
        if round < total_rounds {
            // Compute value table from this round's solve
            let values = compute_postflop_values(&tree, &layout, &street_equity, &solve_results, num_flop_b);

            // Extract EV histograms
            on_progress(BuildPhase::ExtractingEv(0, NUM_HANDS));
            let ev_histograms = build_ev_histograms(&buckets, &values, NUM_HANDS, num_flop_b);

            // Re-cluster flop buckets using EV histograms
            on_progress(BuildPhase::Rebucketing(round, total_rounds));
            let new_flop_assignments = recluster_flop_buckets(&ev_histograms, num_flop_b, num_flops);
            buckets.flop = new_flop_assignments;

            // Recompute bucket-pair equity with new assignments
            on_progress(BuildPhase::EquityTable);
            street_equity = recompute_street_equity(/* with new buckets */);
        }
    }

    // Final values from last round's solve
    let values = compute_postflop_values(/* from last solve_results */);

    Ok(Self { buckets, street_equity, tree, values, spr, flops })
}
```

The key insight: only flop buckets get rebucketed (turn/river buckets are unchanged since the rebucketing targets flop-level EV dynamics). This keeps the implementation focused and avoids recomputing turn/river features.

**Step 3: Add helper function recluster_flop_buckets**

In `hand_buckets.rs`:

```rust
/// Re-cluster flop buckets using EV histograms instead of equity histograms.
///
/// Same k-means infrastructure as initial clustering, but with EV histogram
/// features. Returns per-flop bucket assignments.
pub fn recluster_flop_buckets(
    ev_histograms: &[HistogramFeatures],
    num_flop_buckets: u16,
    num_flops: usize,
) -> Vec<Vec<u16>>
```

This calls `cluster_histograms()` per flop (same as initial clustering) but with the EV histogram features instead of equity histogram features.

**Step 4: Run tests**

Run: `cargo test -p poker-solver-core`
Expected: all PASS.

**Step 5: Commit**

```
feat: wire EV rebucketing loop into PostflopAbstraction::build

PostflopAbstraction::build now iterates rebucket_rounds times.
Round 1 uses EHS histograms (current behavior). Subsequent rounds
extract EV histograms from the converged strategy and recluster
flop buckets, then re-solve. Only flop buckets are rebucketed;
turn/river remain stable.
```

---

## Task 6: Integration test — Verify rebucketing changes bucket assignments

**Files:**
- Modify: `crates/core/src/preflop/postflop_abstraction.rs` (add integration test)

**Step 1: Write integration test**

```rust
#[timed_test(120)]
fn rebucketing_round_2_changes_flop_assignments() {
    let mut config = PostflopModelConfig::fast();
    config.rebucket_rounds = 1;
    config.postflop_solve_iterations = 20;
    let r1 = PostflopAbstraction::build(&config, None, None, |_| {}).unwrap();

    config.rebucket_rounds = 2;
    let r2 = PostflopAbstraction::build(&config, None, None, |_| {}).unwrap();

    // Bucket assignments should differ after EV rebucketing
    let mut any_different = false;
    for flop_idx in 0..r1.buckets.flop.len() {
        if r1.buckets.flop[flop_idx] != r2.buckets.flop[flop_idx] {
            any_different = true;
            break;
        }
    }
    assert!(any_different, "rebucketing should produce different assignments");
}
```

**Step 2: Run test**

Run: `cargo test -p poker-solver-core postflop_abstraction::tests::rebucketing_round_2 -- --ignored`
(May need `#[ignore = "slow"]` depending on timing with fast config.)

Expected: PASS — round 2 produces different flop bucket assignments.

**Step 3: Commit**

```
test: verify EV rebucketing changes flop bucket assignments

Integration test confirms that rebucket_rounds=2 produces different
flop bucket assignments than rounds=1, validating that EV histograms
provide distinct clustering signal from equity histograms.
```

---

## Task 7: Trainer progress — MultiProgress with per-flop bars

**Files:**
- Modify: `crates/trainer/src/main.rs`

**Step 1: Update the postflop build section in run_solve_preflop**

Replace the single `ProgressBar pf_pb` with a `MultiProgress`-based setup. Key structure:

```rust
let multi = MultiProgress::new();
let phase_bar = multi.add(ProgressBar::new_spinner()); // top-level phase
let mut flop_bars: HashMap<String, ProgressBar> = HashMap::new();

let abstraction = PostflopAbstraction::build(
    pf_config, Some(&equity), Some(cache_base),
    |phase| {
        match &phase {
            BuildPhase::HandBuckets(done, total) => {
                phase_bar.set_style(bar_style.clone());
                phase_bar.set_length(*total as u64);
                phase_bar.set_position(*done as u64);
                phase_bar.set_message("Hand buckets");
            }
            BuildPhase::SolvingPostflop { round, total_rounds, flop_name, iteration, max_iterations, delta } => {
                let bar = flop_bars.entry(flop_name.clone()).or_insert_with(|| {
                    let b = multi.add(ProgressBar::new(*max_iterations as u64));
                    b.set_style(bar_style.clone());
                    b
                });
                bar.set_position(*iteration as u64);
                bar.set_message(format!(
                    "[{round}/{total_rounds}] Flop '{flop_name}' δ={delta:.4}"
                ));
            }
            BuildPhase::ExtractingEv(done, total) => {
                // Clear flop bars, show extraction progress
                for (_, bar) in flop_bars.drain() {
                    bar.finish_and_clear();
                }
                phase_bar.set_length(*total as u64);
                phase_bar.set_position(*done as u64);
                phase_bar.set_message("EV histograms");
            }
            BuildPhase::Rebucketing(round, total) => {
                phase_bar.set_style(spinner_style.clone());
                phase_bar.set_message(format!("Rebucketing ({round}/{total})..."));
            }
            _ => {
                phase_bar.set_style(spinner_style.clone());
                phase_bar.set_message(format!("{phase}..."));
            }
        }
    },
)?;
```

**Step 2: Handle thread safety for MultiProgress**

The `on_progress` callback is called from rayon threads. `MultiProgress` is `Send + Sync`, and individual `ProgressBar`s are `Send + Sync`, so this works. However, the `HashMap<String, ProgressBar>` needs to be wrapped in a `Mutex` since it's mutated from multiple threads:

```rust
let flop_bars: Arc<Mutex<HashMap<String, ProgressBar>>> = Arc::new(Mutex::new(HashMap::new()));
```

**Step 3: Test visually**

Run: `cargo run -p poker-solver-trainer --release -- solve-preflop -c sample_configurations/smoke.yaml`
Expected: multi-bar output showing per-flop progress with delta values.

**Step 4: Commit**

```
feat: MultiProgress per-flop bars with delta display in trainer

Postflop build progress now shows parallel bars per concurrent flop
solve with flop name, round indicator, and current regret delta.
Phase transitions clear flop bars and show the new phase.
```

---

## Task 8: Update docs and sample configs

**Files:**
- Modify: `docs/architecture.md` — document rebucketing loop
- Modify: `docs/training.md` — document new config fields
- Modify: `sample_configurations/smoke.yaml` — add rebucket fields (commented out)

**Step 1: Update architecture.md**

Add a section under postflop abstraction describing the rebucketing loop, EV histograms, and early stopping.

**Step 2: Update training.md**

Document new config fields: `rebucket_rounds`, `rebucket_delta_threshold`, `postflop_sprs`. Include example YAML.

**Step 3: Add commented example to smoke.yaml**

```yaml
# EV rebucketing (default: 1 round = EHS only)
# rebucket_rounds: 2
# rebucket_delta_threshold: 0.001
# postflop_sprs: [3.5]
```

**Step 4: Commit**

```
docs: document EV rebucketing config and architecture
```

---

## Task 9: Final validation — Full pipeline test

**Step 1: Run full test suite**

Run: `cargo test`
Expected: all existing tests pass, no regressions.

**Step 2: Run clippy**

Run: `cargo clippy`
Expected: clean (pedantic enabled in core).

**Step 3: Run a real solve with rebucketing**

Create a test config with `rebucket_rounds: 2` and run:
```
cargo run -p poker-solver-trainer --release -- solve-preflop -c <test_config.yaml>
```
Verify: progress bars show two rounds, second round shows "Rebucketing (EV)", bucket assignments change between rounds.

**Step 4: Commit any fixes, bd sync, push**

```bash
bd sync
git push
```
