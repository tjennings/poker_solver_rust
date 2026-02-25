# Postflop Validation: Bucket Equity Fix + Layered Smoke Test & Diagnostics

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix the broken bucket-pair equity computation (root cause of bad strategy), then add layered validation (smoke config + diagnostic tests) to catch pipeline bugs quickly.

**Architecture:** The postflop abstraction pipeline runs: hand bucketing → street equity tables → postflop CFR solve → value table → preflop terminal evaluation. The street equity table is currently a placeholder (all 0.5), causing showdowns to be uninformative. We fix that first, then add a smoke test config and phase-level diagnostic tests.

**Tech Stack:** Rust, rayon (parallelism), serde_yaml (config), existing histogram CDF infrastructure.

---

## Task 0: Compute Real Bucket-Pair Equity

The root cause of bad postflop strategy output. `build_street_equity_from_buckets()` in `postflop_abstraction.rs:417` returns 0.5 for every bucket pair, making all showdowns return EV=0 for both players. The CFR can only learn fold equity, never showdown equity.

**Approach:** Reuse data already computed during bucketing — no second EHS pass needed.

- **Flop/turn:** `compute_flop_histograms` and `compute_turn_histograms` produce CDF histograms per (hand, board) situation. Extract average equity from each CDF: `avg_eq = (1 / HISTOGRAM_BINS) * sum_i(1 - cdf[i])`. Group by bucket to get per-bucket centroid equity.
- **River:** `compute_river_equities` already produces raw scalar equity per (hand, board). Use directly.
- **Pair equity:** For bucket pair (a, b): `equity(a, b) = centroid[a] / (centroid[a] + centroid[b])`. This approximation correctly orders buckets by strength and breaks the placeholder symmetry.

**Key structural change:** `build_street_buckets_independent` currently discards intermediate features after clustering. Return them alongside the buckets in a `BucketingResult` struct so `load_or_build_abstraction` can compute real equity without recomputation.

**Files:**
- Modify: `crates/core/src/preflop/hand_buckets.rs`
  - Add `BucketingResult` struct
  - Change `build_street_buckets_independent` return type to `BucketingResult`
  - Add `cdf_to_avg_equity()` helper
  - Add `compute_bucket_pair_equity()` function
- Modify: `crates/core/src/preflop/postflop_abstraction.rs:373-411`
  - Update `load_or_build_abstraction` to use `BucketingResult`
  - Replace placeholder with real equity computation
  - Remove `build_street_equity_from_buckets()`
- Test: inline `#[cfg(test)]` in `hand_buckets.rs`

### Step 1: Write the failing test

Add to the `#[cfg(test)]` module at the bottom of `crates/core/src/preflop/hand_buckets.rs`:

```rust
#[timed_test]
fn cdf_to_avg_equity_extracts_mean() {
    // A CDF that's a step function at 0.7 equity (bin 7 of 10)
    // should yield avg_equity ≈ 0.7
    let cdf = crate::preflop::ehs::single_value_cdf(0.7);
    let avg = cdf_to_avg_equity(&cdf);
    assert!(
        (avg - 0.7).abs() < 0.15,
        "avg equity from CDF of 0.7-equity hand should be near 0.7, got {avg}"
    );
}

#[timed_test]
fn bucket_pair_equity_distinguishes_strong_from_weak() {
    let hands: Vec<_> = crate::hands::all_hands().collect();
    let flops = crate::preflop::ehs::sample_canonical_flops(3);
    let result = build_street_buckets_independent(
        &hands, &flops, 5, 5, 5, &|_| {},
    );

    // Extract per-situation average equity from flop histograms
    let flop_ehs: Vec<f64> = result.flop_histograms.iter()
        .map(|cdf| cdf_to_avg_equity(cdf))
        .collect();
    let equity = compute_bucket_pair_equity(
        &result.buckets.flop, result.buckets.num_flop_buckets as usize, &flop_ehs,
    );

    // Self-equity should be ~0.5
    for a in 0..5 {
        let self_eq = equity.get(a, a);
        assert!(
            (self_eq - 0.5).abs() < 0.05,
            "self-equity for bucket {a} should be ~0.5, got {self_eq}"
        );
    }

    // Find strongest and weakest buckets
    let mut best = (0, 0.0f32);
    let mut worst = (0, 1.0f32);
    for a in 0..5 {
        let eq_vs_0 = equity.get(a, 0);
        if eq_vs_0 > best.1 { best = (a, eq_vs_0); }
        if eq_vs_0 < worst.1 { worst = (a, eq_vs_0); }
    }

    let strong_vs_weak = equity.get(best.0, worst.0);
    assert!(
        strong_vs_weak > 0.55,
        "strongest bucket vs weakest should have equity > 0.55, got {strong_vs_weak}"
    );
    let reverse = equity.get(worst.0, best.0);
    assert!(
        (strong_vs_weak + reverse - 1.0).abs() < 0.01,
        "equity not zero-sum: {strong_vs_weak} + {reverse} != 1.0"
    );
}
```

### Step 2: Run test to verify it fails

```bash
cargo test -p poker-solver-core bucket_pair_equity_distinguishes -- --nocapture 2>&1 | tail -20
```

Expected: FAIL — `BucketingResult`, `cdf_to_avg_equity`, `compute_bucket_pair_equity` don't exist.

### Step 3: Implement the data structures and helpers

Add to `crates/core/src/preflop/hand_buckets.rs`:

**`BucketingResult` struct** (near `StreetBuckets`):

```rust
/// Result of the independent per-street bucketing pipeline.
///
/// Contains both the bucket assignments and the intermediate features
/// used during clustering, so downstream code can compute bucket-pair
/// equity without recomputation.
pub struct BucketingResult {
    pub buckets: StreetBuckets,
    /// Flat histogram CDF features per (hand, board) for flop.
    /// Indexed as `hand_idx * num_flops + flop_idx`.
    pub flop_histograms: Vec<HistogramFeatures>,
    /// Flat histogram CDF features per (hand, board) for turn.
    pub turn_histograms: Vec<HistogramFeatures>,
    /// Flat scalar equity per (hand, board) for river.
    pub river_equities: Vec<f64>,
}
```

**`cdf_to_avg_equity` helper:**

```rust
/// Extract average equity from a histogram CDF.
///
/// For a CDF over `N` equal-width bins on [0, 1], the mean of the
/// underlying distribution is: `(1/N) * sum_i(1 - cdf[i])`.
///
/// Intuition: each `1 - cdf[i]` is the probability mass above bin `i`,
/// and summing these scaled by bin width gives the expected value.
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn cdf_to_avg_equity(cdf: &HistogramFeatures) -> f64 {
    let n = cdf.len() as f64;
    cdf.iter().map(|&c| 1.0 - c).sum::<f64>() / n
}
```

**`compute_bucket_pair_equity`:**

```rust
/// Compute bucket-pair equity from bucket assignments and per-situation
/// average equity values.
///
/// Groups situations by bucket, computes per-bucket centroid equity,
/// then derives pairwise equity: `equity(a, b) = c_a / (c_a + c_b)`.
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn compute_bucket_pair_equity(
    assignments: &[u16],
    num_buckets: usize,
    avg_equities: &[f64],
) -> BucketEquity {
    assert_eq!(assignments.len(), avg_equities.len());

    let mut sums = vec![0.0f64; num_buckets];
    let mut counts = vec![0u32; num_buckets];
    for (idx, &bucket) in assignments.iter().enumerate() {
        let eq = avg_equities[idx];
        if eq.is_nan() { continue; }
        sums[bucket as usize] += eq;
        counts[bucket as usize] += 1;
    }
    let centroids: Vec<f64> = sums.iter().zip(&counts)
        .map(|(&s, &c)| if c > 0 { s / f64::from(c) } else { 0.5 })
        .collect();

    let mut equity = vec![vec![0.5f32; num_buckets]; num_buckets];
    for a in 0..num_buckets {
        for b in 0..num_buckets {
            let denom = centroids[a] + centroids[b];
            equity[a][b] = if denom > 1e-12 {
                (centroids[a] / denom) as f32
            } else {
                0.5
            };
        }
    }

    BucketEquity { equity, num_buckets }
}
```

### Step 4: Change `build_street_buckets_independent` to return `BucketingResult`

The function already computes `flop_features`, `turn_features`, and `river_equities` internally — it just discards them. Change the return type and keep them:

```rust
// Change signature:
pub fn build_street_buckets_independent(
    hands: &[CanonicalHand],
    flops: &[[Card; 3]],
    num_flop_buckets: u16,
    num_turn_buckets: u16,
    num_river_buckets: u16,
    on_progress: &(impl Fn(BuildProgress) + Sync + Send),
) -> BucketingResult {
    // ... existing clustering code unchanged ...

    BucketingResult {
        buckets: StreetBuckets {
            flop: flop_assignments,
            num_flop_buckets,
            num_flop_boards: flops.len(),
            turn: turn_assignments,
            num_turn_buckets,
            river: river_assignments,
            num_river_buckets,
        },
        flop_histograms: flop_features,
        turn_histograms: turn_features,
        river_equities: river_equities,
    }
}
```

### Step 5: Update all callers of `build_street_buckets_independent`

All callers currently expect `StreetBuckets`. Update them to destructure `BucketingResult`. The main caller is `load_or_build_abstraction` in `postflop_abstraction.rs`.

### Step 6: Wire real equity into `load_or_build_abstraction`

In `postflop_abstraction.rs`, replace lines 387-410:

```rust
let result = hand_buckets::build_street_buckets_independent(
    &hands, &flops,
    config.num_hand_buckets_flop,
    config.num_hand_buckets_turn,
    config.num_hand_buckets_river,
    &|progress| { /* ... existing progress callback ... */ },
);

let buckets = result.buckets;

on_progress(BuildPhase::EquityTable);

// Compute real bucket-pair equity from histogram CDFs
let flop_ehs: Vec<f64> = result.flop_histograms.iter()
    .map(|cdf| hand_buckets::cdf_to_avg_equity(cdf))
    .collect();
let turn_ehs: Vec<f64> = result.turn_histograms.iter()
    .map(|cdf| hand_buckets::cdf_to_avg_equity(cdf))
    .collect();

let street_equity = StreetEquity {
    flop: hand_buckets::compute_bucket_pair_equity(
        &buckets.flop, buckets.num_flop_buckets as usize, &flop_ehs,
    ),
    turn: hand_buckets::compute_bucket_pair_equity(
        &buckets.turn, buckets.num_turn_buckets as usize, &turn_ehs,
    ),
    river: hand_buckets::compute_bucket_pair_equity(
        &buckets.river, buckets.num_river_buckets as usize, &result.river_equities,
    ),
};

Ok((buckets, street_equity))
```

Delete the now-unused `build_street_equity_from_buckets` function.

### Step 7: Run tests

```bash
cargo test -p poker-solver-core --lib -- postflop 2>&1 | tail -20
cargo test -p poker-solver-core bucket_pair_equity -- --nocapture 2>&1 | tail -20
cargo test -p poker-solver-core cdf_to_avg -- --nocapture 2>&1 | tail -20
```

Expected: All pass.

### Step 8: Run clippy

```bash
cargo clippy -p poker-solver-core 2>&1 | tail -20
```

Fix any warnings. Confirm `build_street_equity_from_buckets` is removed.

### Step 9: Commit

```bash
git add crates/core/src/preflop/hand_buckets.rs crates/core/src/preflop/postflop_abstraction.rs
git commit -m "fix: compute real bucket-pair equity from histogram CDFs

The postflop CFR was using equity=0.5 for all bucket pairs at showdown
terminals, making every showdown return EV=0. Now extracts average
equity from the CDF histograms already computed during bucketing,
groups by bucket centroid, and derives pair equity as
c_a / (c_a + c_b). No second EHS pass needed.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

## Task 1: Create Smoke Test Config

A tweakable YAML config that exercises the full bucketed pipeline in ~1-2 minutes.

**Files:**
- Create: `sample_configurations/smoke.yaml`

### Step 1: Create the config file

Create `sample_configurations/smoke.yaml`:

```yaml
# Smoke test: exercises the full postflop bucketed pipeline in ~1-2 minutes.
# Tweak these values to debug specific phases.
#
# Usage:
#   cargo run -p poker-solver-trainer --release -- solve-preflop \
#     -c sample_configurations/smoke.yaml -o smoke_test
#
# Validation: AA (hand 0) should raise more than 72o (hand 168).

# ── Training parameters ──────────────────────────────────────────────────────

iterations: 2000
equity_samples: 10000
print_every: 500

# ── Game structure ───────────────────────────────────────────────────────────

positions:
  - name: Small Blind
    short_name: SB
  - name: Big Blind
    short_name: BB

blinds:
  - [0, 1]
  - [1, 2]

antes: []

stacks: [50, 50]  # 25 BB

raise_sizes:
  - [2.5]
  - [3.0]

raise_cap: 3

# ── DCFR discounting ─────────────────────────────────────────────────────────

dcfr_alpha: 1.5
dcfr_beta: 0.5
dcfr_gamma: 2.0
dcfr_warmup: 0
exploration: 0.05

# ── Postflop model (minimal for fast validation) ─────────────────────────────

postflop_model:
  num_hand_buckets_flop: 10
  num_hand_buckets_turn: 10
  num_hand_buckets_river: 10
  max_flop_boards: 3
  bet_sizes: [0.5, 1.0]
  max_raises_per_street: 1
  postflop_solve_iterations: 50
  postflop_solve_samples: 0
  canonical_sprs: [1.0, 5.0]
  flop_samples_per_iter: 1
```

### Step 2: Verify it parses

```bash
cargo run -p poker-solver-trainer --release -- solve-preflop -c sample_configurations/smoke.yaml -o /tmp/smoke_test 2>&1 | head -5
```

Expected: Should start training. YAML parses without error.

### Step 3: Commit

```bash
git add sample_configurations/smoke.yaml
git commit -m "feat: add smoke test config for fast postflop pipeline validation

Exercises the full bucketed pipeline (bucketing → CFR → value table →
preflop solve) with minimal parameters. Trains in ~1-2 minutes.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

## Task 2: Diagnostic Test — Bucket Monotonicity

Validate that bucket assignments correlate with hand strength.

**Files:**
- Create: `crates/core/tests/postflop_diagnostics.rs`

### Step 1: Write the test

```rust
//! Phase-level diagnostic tests for the postflop bucketed abstraction pipeline.
//!
//! Each test validates one phase independently with small inputs and runs in seconds.
//! When the smoke test fails, run these to pinpoint the broken phase.
//!
//! Run: `cargo test -p poker-solver-core --test postflop_diagnostics --release -- --nocapture`

use poker_solver_core::hands;
use poker_solver_core::preflop::ehs::sample_canonical_flops;
use poker_solver_core::preflop::hand_buckets::{
    build_street_buckets_independent, cdf_to_avg_equity, compute_bucket_pair_equity,
};

/// Bucket assignments should correlate with hand strength.
/// Buckets should have meaningful equity spread (strong != weak).
#[test]
fn diag_bucket_monotonicity() {
    let hands: Vec<_> = hands::all_hands().collect();
    let flops = sample_canonical_flops(3);
    let result = build_street_buckets_independent(
        &hands, &flops, 10, 10, 10, &|_| {},
    );

    let flop_ehs: Vec<f64> = result.flop_histograms.iter()
        .map(|cdf| cdf_to_avg_equity(cdf))
        .collect();
    let equity = compute_bucket_pair_equity(
        &result.buckets.flop, result.buckets.num_flop_buckets as usize, &flop_ehs,
    );

    let n = result.buckets.num_flop_buckets as usize;
    let mut min_eq = 1.0f32;
    let mut max_eq = 0.0f32;
    for a in 0..n {
        for b in 0..n {
            if a == b { continue; }
            let eq = equity.get(a, b);
            if eq < min_eq { min_eq = eq; }
            if eq > max_eq { max_eq = eq; }
        }
    }

    let spread = max_eq - min_eq;
    eprintln!("Bucket equity spread: min={min_eq:.3}, max={max_eq:.3}, spread={spread:.3}");
    assert!(
        spread > 0.1,
        "Bucket equity spread should be > 0.1, got {spread}"
    );
}
```

### Step 2: Run test

```bash
cargo test -p poker-solver-core --test postflop_diagnostics diag_bucket_monotonicity --release -- --nocapture 2>&1 | tail -20
```

Expected: PASS

### Step 3: Commit

```bash
git add crates/core/tests/postflop_diagnostics.rs
git commit -m "test: add bucket monotonicity diagnostic for postflop pipeline

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

## Task 3: Diagnostic Test — Street Equity Sanity

Validate the street equity table independently of CFR.

**Files:**
- Modify: `crates/core/tests/postflop_diagnostics.rs`

### Step 1: Write the test

Add to `postflop_diagnostics.rs`:

```rust
use poker_solver_core::preflop::hand_buckets::StreetEquity;

/// Street equity table should have non-trivial values, correct self-equity,
/// and zero-sum symmetry.
#[test]
fn diag_street_equity_sanity() {
    let hands: Vec<_> = hands::all_hands().collect();
    let flops = sample_canonical_flops(3);
    let result = build_street_buckets_independent(
        &hands, &flops, 10, 10, 10, &|_| {},
    );

    // Build equity the same way the pipeline does
    let flop_ehs: Vec<f64> = result.flop_histograms.iter()
        .map(|cdf| cdf_to_avg_equity(cdf))
        .collect();
    let flop_equity = compute_bucket_pair_equity(
        &result.buckets.flop, result.buckets.num_flop_buckets as usize, &flop_ehs,
    );

    let n = result.buckets.num_flop_buckets as usize;

    // Should not be all 0.5
    let mut all_half = true;
    for a in 0..n {
        for b in 0..n {
            let eq = flop_equity.get(a, b);
            assert!(eq.is_finite(), "equity({a},{b}) not finite: {eq}");
            assert!((0.0..=1.0).contains(&eq), "equity({a},{b}) out of range: {eq}");
            if (eq - 0.5).abs() > 0.01 { all_half = false; }
        }
    }
    assert!(!all_half, "Equity is all 0.5 — placeholder not replaced");

    // Self-equity ≈ 0.5
    for a in 0..n {
        let self_eq = flop_equity.get(a, a);
        assert!(
            (self_eq - 0.5).abs() < 0.05,
            "self-equity bucket {a} should be ~0.5, got {self_eq}"
        );
    }

    // Symmetry: eq(a,b) + eq(b,a) ≈ 1.0
    for a in 0..n {
        for b in (a+1)..n {
            let ab = flop_equity.get(a, b);
            let ba = flop_equity.get(b, a);
            assert!(
                (ab + ba - 1.0).abs() < 0.01,
                "equity({a},{b})={ab} + equity({b},{a})={ba} should ≈ 1.0"
            );
        }
    }
}
```

### Step 2: Run test

```bash
cargo test -p poker-solver-core --test postflop_diagnostics diag_street_equity --release -- --nocapture 2>&1 | tail -20
```

Expected: PASS

### Step 3: Commit

```bash
git add crates/core/tests/postflop_diagnostics.rs
git commit -m "test: add street equity sanity diagnostic

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

## Task 4: Diagnostic Test — Value Table Extraction

Validate the full postflop abstraction produces a value table where strong buckets win.

**Files:**
- Modify: `crates/core/tests/postflop_diagnostics.rs`

### Step 1: Write the test

```rust
use poker_solver_core::preflop::postflop_abstraction::PostflopAbstraction;
use poker_solver_core::preflop::postflop_model::PostflopModelConfig;

/// After building the full postflop abstraction (buckets + CFR + values),
/// the value table should show strong buckets winning against weak buckets.
#[test]
fn diag_value_table_strong_beats_weak() {
    let config = PostflopModelConfig {
        num_hand_buckets_flop: 5,
        num_hand_buckets_turn: 5,
        num_hand_buckets_river: 5,
        canonical_sprs: vec![5.0],
        postflop_solve_iterations: 100,
        postflop_solve_samples: 0,
        bet_sizes: vec![1.0],
        max_raises_per_street: 0,
        max_flop_boards: 3,
        flop_samples_per_iter: 1,
    };

    let abstraction = PostflopAbstraction::build(
        &config, None, None, &|phase| eprintln!("  [build] {phase}"),
    ).expect("build should succeed");

    let n = config.num_hand_buckets_flop as usize;
    let mut best_bucket = 0;
    let mut worst_bucket = 0;
    let mut best_eq = 0.0f32;
    let mut worst_eq = 1.0f32;
    for b in 0..n {
        let avg: f32 = (0..n)
            .map(|o| abstraction.street_equity.flop.get(b, o))
            .sum::<f32>() / n as f32;
        if avg > best_eq { best_eq = avg; best_bucket = b; }
        if avg < worst_eq { worst_eq = avg; worst_bucket = b; }
    }

    eprintln!("strongest bucket: {best_bucket} (avg eq {best_eq:.3})");
    eprintln!("weakest bucket: {worst_bucket} (avg eq {worst_eq:.3})");

    let strong_ev = abstraction.values.get_by_spr(
        0, 0, best_bucket as u16, worst_bucket as u16,
    );
    let weak_ev = abstraction.values.get_by_spr(
        0, 0, worst_bucket as u16, best_bucket as u16,
    );

    eprintln!("strong_hero EV={strong_ev:.4}, weak_hero EV={weak_ev:.4}");
    assert!(
        strong_ev > weak_ev,
        "strong bucket should have higher EV: {strong_ev} vs {weak_ev}"
    );
}
```

### Step 2: Run test

```bash
cargo test -p poker-solver-core --test postflop_diagnostics diag_value_table --release -- --nocapture 2>&1 | tail -20
```

Expected: PASS

### Step 3: Commit

```bash
git add crates/core/tests/postflop_diagnostics.rs
git commit -m "test: add value table diagnostic — strong buckets should win

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

## Task 5: Diagnostic Test — End-to-End Mini Integration

Validate the full pipeline: postflop abstraction → preflop solve → AA beats 72o.

**Files:**
- Modify: `crates/core/tests/postflop_diagnostics.rs`

### Step 1: Write the test

```rust
use poker_solver_core::preflop::config::PreflopConfig;
use poker_solver_core::preflop::equity::EquityTable;
use poker_solver_core::preflop::solver::PreflopSolver;

/// Full pipeline: build postflop abstraction, run preflop solver,
/// AA should fold less than 72o.
#[test]
fn diag_end_to_end_aa_beats_72o() {
    let pf_config = PostflopModelConfig {
        num_hand_buckets_flop: 10,
        num_hand_buckets_turn: 10,
        num_hand_buckets_river: 10,
        canonical_sprs: vec![1.0, 5.0],
        postflop_solve_iterations: 50,
        postflop_solve_samples: 0,
        bet_sizes: vec![0.5, 1.0],
        max_raises_per_street: 1,
        max_flop_boards: 3,
        flop_samples_per_iter: 1,
    };

    let mut config = PreflopConfig::heads_up(25);
    config.postflop_model = Some(pf_config);

    let equity = EquityTable::new_computed(5000, |_| {});
    let mut solver = PreflopSolver::new(config.clone(), equity);

    let abstraction = PostflopAbstraction::build(
        config.postflop_model.as_ref().unwrap(),
        None, None,
        &|phase| eprintln!("  [build] {phase}"),
    ).expect("postflop build should succeed");

    solver.attach_postflop(abstraction, &config);

    for i in 0..500 {
        solver.run_iteration();
        if (i + 1) % 100 == 0 {
            eprintln!("  preflop iteration {}", i + 1);
        }
    }

    let strategy = solver.average_strategy();

    let aa_idx = hands::CanonicalHand::parse("AA").unwrap().index();
    let seven_two_idx = hands::CanonicalHand::parse("72o").unwrap().index();

    let aa_probs = strategy.get_root_probs(aa_idx);
    let seven_two_probs = strategy.get_root_probs(seven_two_idx);

    eprintln!("AA root probs: {aa_probs:?}");
    eprintln!("72o root probs: {seven_two_probs:?}");

    // AA should fold less than 72o
    let aa_fold = aa_probs.first().copied().unwrap_or(1.0);
    let seven_two_fold = seven_two_probs.first().copied().unwrap_or(0.0);

    assert!(
        aa_fold < seven_two_fold,
        "AA should fold less than 72o: AA fold={aa_fold:.3}, 72o fold={seven_two_fold:.3}"
    );
}
```

**Note:** Check visibility of `PreflopSolver::run_iteration()`, `average_strategy()`, and `attach_postflop()`. Verify action ordering (fold is first action) against the actual tree structure. Adjust imports as needed.

### Step 2: Run test

```bash
cargo test -p poker-solver-core --test postflop_diagnostics diag_end_to_end --release -- --nocapture 2>&1 | tail -30
```

Expected: PASS (may take 1-2 minutes).

### Step 3: Commit

```bash
git add crates/core/tests/postflop_diagnostics.rs
git commit -m "test: add end-to-end diagnostic — AA should fold less than 72o

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

## Task 6: Update Architecture Docs

**Files:**
- Modify: `docs/architecture.md` — document bucket equity computation

### Step 1: Update architecture doc

Add to the postflop abstraction section:
- Bucket-pair equity derived from histogram CDF centroids
- Formula: `avg_eq = (1/BINS) * sum(1 - cdf[i])`, then `equity(a,b) = centroid[a] / (centroid[a] + centroid[b])`
- This is an approximation; future work may use pairwise hand evaluation

### Step 2: Commit

```bash
git add docs/
git commit -m "docs: document bucket equity computation

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```
