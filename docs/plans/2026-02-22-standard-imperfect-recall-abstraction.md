# Standard Imperfect-Recall Abstraction Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace the pseudo-hierarchical texture+transition-table bucketing with standard independent per-street clustering on equity histograms (EMD via L2-on-CDF), matching the Pluribus approach.

**Architecture:** Each street clusters independently — flop, turn, river each run k-means on 10-bin equity CDF feature vectors over exhaustively enumerated future boards. No transition tables, no texture grouping. Imperfect recall by design. `HandBucketMapping` replaced by `StreetBuckets` with flat per-situation lookups.

**Tech Stack:** Rust, rayon (parallelism), serde (serialization), existing `HandStrengthCalculator` for equity computation.

**Design doc:** `docs/plans/2026-02-22-standard-imperfect-recall-abstraction-design.md`

---

### Task 1: Add equity histogram feature type and CDF computation

**Files:**
- Modify: `crates/core/src/preflop/ehs.rs`
- Test: `crates/core/src/preflop/ehs.rs` (inline tests)

**Context:** Currently `EhsFeatures = [f64; 3]` (EHS, ppot, npot). We need a new 10-bin CDF feature vector for clustering. The existing `EhsFeatures` type is used throughout k-means — we'll add the new type alongside it, then migrate clustering in later tasks.

**Step 1: Write failing tests for equity histogram**

Add to the `#[cfg(test)] mod tests` block in `ehs.rs`:

```rust
#[timed_test]
fn equity_histogram_flop_has_10_bins() {
    let hole = [
        make_card(Value::Ace, Suit::Spade),
        make_card(Value::King, Suit::Spade),
    ];
    let board = [
        make_card(Value::Two, Suit::Spade),
        make_card(Value::Five, Suit::Heart),
        make_card(Value::Eight, Suit::Club),
    ];
    let hist = equity_histogram(&hole, &board);
    assert_eq!(hist.len(), HISTOGRAM_BINS);
    // CDF must be non-decreasing and end at 1.0
    for i in 1..hist.len() {
        assert!(hist[i] >= hist[i - 1], "CDF not monotonic at bin {i}");
    }
    assert!((hist[HISTOGRAM_BINS - 1] - 1.0).abs() < 1e-6, "CDF must end at 1.0");
}

#[timed_test]
fn equity_histogram_strong_hand_skews_right() {
    // AA on low board — most future equities should be high
    let hole = [
        make_card(Value::Ace, Suit::Spade),
        make_card(Value::Ace, Suit::Heart),
    ];
    let board = [
        make_card(Value::Two, Suit::Diamond),
        make_card(Value::Seven, Suit::Club),
        make_card(Value::Three, Suit::Heart),
    ];
    let hist = equity_histogram(&hole, &board);
    // CDF at bin 5 (equity < 0.5) should be low — most mass is above 0.5
    assert!(hist[4] < 0.3, "AA should have most equity mass above 0.5, CDF at 0.5 = {}", hist[4]);
}

#[timed_test]
fn equity_histogram_river_is_degenerate() {
    // River: no future cards. Histogram should be a single spike.
    let hole = [
        make_card(Value::Ace, Suit::Spade),
        make_card(Value::King, Suit::Heart),
    ];
    let board = [
        make_card(Value::Two, Suit::Diamond),
        make_card(Value::Five, Suit::Heart),
        make_card(Value::Eight, Suit::Club),
        make_card(Value::Ten, Suit::Spade),
        make_card(Value::Queen, Suit::Diamond),
    ];
    let hist = equity_histogram(&hole, &board);
    // All mass in one bin — CDF jumps from 0 to 1 at the equity bin
    let nonzero_steps: Vec<usize> = (0..HISTOGRAM_BINS)
        .filter(|&i| {
            let prev = if i == 0 { 0.0 } else { hist[i - 1] };
            (hist[i] - prev).abs() > 1e-6
        })
        .collect();
    assert_eq!(nonzero_steps.len(), 1, "River histogram should have exactly one step, got {nonzero_steps:?}");
}
```

**Step 2: Run tests to verify they fail**

Run: `cargo test -p poker-solver-core equity_histogram -- --nocapture 2>&1 | head -30`
Expected: Compilation error — `equity_histogram` and `HISTOGRAM_BINS` not defined.

**Step 3: Implement equity histogram**

Add to `ehs.rs` above the existing `ehs_features` function:

```rust
/// Number of bins in the equity histogram CDF feature vector.
pub const HISTOGRAM_BINS: usize = 10;

/// Equity histogram CDF feature vector for k-means clustering.
///
/// For flop/turn: enumerates all future cards, computes equity for each,
/// bins into `HISTOGRAM_BINS` equal-width bins over [0, 1], returns the CDF.
/// For river: single equity value placed in one bin → degenerate CDF.
///
/// L2 distance between CDFs equals Earth Mover's Distance for 1D distributions.
#[must_use]
pub fn equity_histogram(hole: &[Card; 2], board: &[Card]) -> [f64; HISTOGRAM_BINS] {
    let live = live_deck(*hole, board);
    if board.len() >= 5 || live.is_empty() {
        // River or no live cards: degenerate histogram from single equity value
        let eq = ehs_features(*hole, board)[0];
        return single_value_cdf(eq);
    }

    // Enumerate all future single-card runouts
    let mut counts = [0u32; HISTOGRAM_BINS];
    let mut total = 0u32;
    for &card in &live {
        let mut future_board = board.to_vec();
        future_board.push(card);
        let eq = ehs_features(*hole, &future_board)[0];
        let bin = equity_to_bin(eq);
        counts[bin] += 1;
        total += 1;
    }

    counts_to_cdf(&counts, total)
}

/// Map an equity value in [0, 1] to a bin index in [0, HISTOGRAM_BINS).
fn equity_to_bin(eq: f64) -> usize {
    let bin = (eq * HISTOGRAM_BINS as f64) as usize;
    bin.min(HISTOGRAM_BINS - 1)
}

/// Convert bin counts to a CDF (cumulative distribution function).
#[allow(clippy::cast_precision_loss)]
fn counts_to_cdf(counts: &[u32; HISTOGRAM_BINS], total: u32) -> [f64; HISTOGRAM_BINS] {
    let mut cdf = [0.0f64; HISTOGRAM_BINS];
    if total == 0 {
        // Uniform fallback
        for (i, v) in cdf.iter_mut().enumerate() {
            *v = (i + 1) as f64 / HISTOGRAM_BINS as f64;
        }
        return cdf;
    }
    let n = f64::from(total);
    let mut cumsum = 0.0;
    for (i, &c) in counts.iter().enumerate() {
        cumsum += f64::from(c) / n;
        cdf[i] = cumsum;
    }
    // Ensure last bin is exactly 1.0 (floating point)
    cdf[HISTOGRAM_BINS - 1] = 1.0;
    cdf
}

/// CDF for a single equity value: 0 below the bin, 1 at and above.
fn single_value_cdf(eq: f64) -> [f64; HISTOGRAM_BINS] {
    let bin = equity_to_bin(eq);
    let mut cdf = [0.0f64; HISTOGRAM_BINS];
    for i in bin..HISTOGRAM_BINS {
        cdf[i] = 1.0;
    }
    cdf
}
```

Also add `HISTOGRAM_BINS` and `equity_histogram` to the test imports.

**Step 4: Run tests to verify they pass**

Run: `cargo test -p poker-solver-core equity_histogram -- --nocapture`
Expected: All 3 new tests pass.

**Step 5: Commit**

```bash
git add crates/core/src/preflop/ehs.rs
git commit -m "feat: add equity histogram CDF feature vectors for EMD-based clustering"
```

---

### Task 2: Generalize k-means to work with variable-length feature vectors

**Files:**
- Modify: `crates/core/src/preflop/hand_buckets.rs`
- Test: `crates/core/src/preflop/hand_buckets.rs` (inline tests)

**Context:** Current `kmeans()` operates on `EhsFeatures = [f64; 3]`. We need it to work with `[f64; 10]` (CDF histograms). Rather than duplicating, make k-means generic over a trait.

**Step 1: Write failing test for k-means on 10-element vectors**

Add to `hand_buckets.rs` test module:

```rust
#[timed_test]
fn kmeans_works_with_10d_vectors() {
    // Two clearly separated clusters in 10D
    let low: [f64; 10] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0];
    let high: [f64; 10] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.2, 0.3, 0.4, 1.0];
    let points = vec![low, low, low, high, high, high];
    let assignments = kmeans_generic(&points, 2, 100);
    assert_eq!(assignments.len(), 6);
    // First 3 should be in one cluster, last 3 in another
    assert_eq!(assignments[0], assignments[1]);
    assert_eq!(assignments[1], assignments[2]);
    assert_eq!(assignments[3], assignments[4]);
    assert_eq!(assignments[4], assignments[5]);
    assert_ne!(assignments[0], assignments[3]);
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test -p poker-solver-core kmeans_works_with_10d -- --nocapture 2>&1 | head -20`
Expected: Compilation error — `kmeans_generic` not defined.

**Step 3: Implement generic k-means**

Add a `KmeansPoint` trait and `kmeans_generic` function. Keep existing `kmeans()` as a thin wrapper for backward compatibility.

```rust
/// Trait for types that can be used as k-means feature vectors.
pub trait KmeansPoint: Copy + Send + Sync + PartialEq {
    /// Number of dimensions.
    fn dims(&self) -> usize;
    /// Get the value at dimension `i`.
    fn get(&self, i: usize) -> f64;
    /// Create a zero vector with the same dimensionality.
    fn zero() -> Self;
    /// Set the value at dimension `i`.
    fn set(&mut self, i: usize, val: f64);
}

impl KmeansPoint for [f64; 3] {
    fn dims(&self) -> usize { 3 }
    fn get(&self, i: usize) -> f64 { self[i] }
    fn zero() -> Self { [0.0; 3] }
    fn set(&mut self, i: usize, val: f64) { self[i] = val; }
}

impl KmeansPoint for [f64; 10] {
    fn dims(&self) -> usize { 10 }
    fn get(&self, i: usize) -> f64 { self[i] }
    fn zero() -> Self { [0.0; 10] }
    fn set(&mut self, i: usize, val: f64) { self[i] = val; }
}

/// Generic k-means clustering for any point type.
#[must_use]
pub fn kmeans_generic<P: KmeansPoint>(points: &[P], k: usize, max_iter: usize) -> Vec<u16> {
    if points.is_empty() || k == 0 {
        return vec![];
    }
    let k = k.min(points.len());
    let mut centroids = kmeans_pp_init_generic(points, k);
    let mut assignments = assign_clusters_generic(points, &centroids);

    for _ in 0..max_iter {
        let new_centroids = recompute_centroids_generic(points, &assignments, k);
        let new_assignments = assign_clusters_generic(points, &new_centroids);
        if new_assignments == assignments {
            break;
        }
        assignments = new_assignments;
        centroids = new_centroids;
    }
    drop(centroids);
    assignments
}
```

Then implement `kmeans_pp_init_generic`, `assign_clusters_generic`, `recompute_centroids_generic`, `sq_dist_generic` as generic versions of the existing functions. Finally, update the existing `kmeans()` to delegate:

```rust
pub fn kmeans(points: &[EhsFeatures], k: usize, max_iter: usize) -> Vec<u16> {
    kmeans_generic(points, k, max_iter)
}
```

**Step 4: Run tests to verify pass**

Run: `cargo test -p poker-solver-core kmeans -- --nocapture`
Expected: All k-means tests pass (old + new).

**Step 5: Commit**

```bash
git add crates/core/src/preflop/hand_buckets.rs
git commit -m "feat: generalize k-means to work with variable-dimension feature vectors"
```

---

### Task 3: Add `StreetBuckets` data model (independent per-street)

**Files:**
- Modify: `crates/core/src/preflop/hand_buckets.rs`
- Test: `crates/core/src/preflop/hand_buckets.rs` (inline tests)

**Context:** Replace `HandBucketMapping` (hierarchical with transition tables) with `StreetBuckets` (flat per-street lookups). Keep `HandBucketMapping` temporarily for backward compat while we migrate callers.

**Step 1: Write failing test**

```rust
#[timed_test]
fn street_buckets_lookup_returns_correct_bucket() {
    let sb = StreetBuckets {
        flop: vec![0, 1, 2, 0, 1],   // 5 situations
        num_flop_buckets: 3,
        turn: vec![1, 0, 2, 1],       // 4 situations
        num_turn_buckets: 3,
        river: vec![0, 0, 1, 1, 2, 2], // 6 situations
        num_river_buckets: 3,
    };
    assert_eq!(sb.flop_bucket(0), 0);
    assert_eq!(sb.flop_bucket(2), 2);
    assert_eq!(sb.turn_bucket(1), 0);
    assert_eq!(sb.river_bucket(5), 2);
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test -p poker-solver-core street_buckets_lookup -- --nocapture 2>&1 | head -20`
Expected: Compilation error.

**Step 3: Implement `StreetBuckets`**

```rust
/// Per-street bucket assignments with imperfect recall.
///
/// Each street is clustered independently. The player "forgets" which bucket
/// they were in on the previous street, spending the full bucket budget on
/// present-state resolution.
///
/// Situation indexing is caller-defined — typically `(hand_idx, board_idx)` mapped
/// to a flat index.
#[derive(Serialize, Deserialize)]
pub struct StreetBuckets {
    /// `flop[situation_idx]` → flop bucket ID
    pub flop: Vec<u16>,
    pub num_flop_buckets: u16,
    /// `turn[situation_idx]` → turn bucket ID
    pub turn: Vec<u16>,
    pub num_turn_buckets: u16,
    /// `river[situation_idx]` → river bucket ID
    pub river: Vec<u16>,
    pub num_river_buckets: u16,
}

impl StreetBuckets {
    /// Look up flop bucket for a given situation index.
    #[must_use]
    pub fn flop_bucket(&self, situation_idx: usize) -> u16 {
        self.flop[situation_idx]
    }

    /// Look up turn bucket for a given situation index.
    #[must_use]
    pub fn turn_bucket(&self, situation_idx: usize) -> u16 {
        self.turn[situation_idx]
    }

    /// Look up river bucket for a given situation index.
    #[must_use]
    pub fn river_bucket(&self, situation_idx: usize) -> u16 {
        self.river[situation_idx]
    }
}
```

**Step 4: Run tests**

Run: `cargo test -p poker-solver-core street_buckets -- --nocapture`
Expected: Pass.

**Step 5: Commit**

```bash
git add crates/core/src/preflop/hand_buckets.rs
git commit -m "feat: add StreetBuckets data model for independent per-street bucketing"
```

---

### Task 4: Build canonical board enumeration

**Files:**
- Modify: `crates/core/src/preflop/ehs.rs` (add `canonical_flops`, `canonical_turn_boards`, `canonical_river_boards`)
- Test: `crates/core/src/preflop/ehs.rs` (inline tests)

**Context:** Replace texture-based board sampling with suit-isomorphic canonical board enumeration. We need functions that enumerate all boards for a given street, reduced by suit isomorphism. These will feed into the equity histogram computation.

**Step 1: Write failing tests**

```rust
#[timed_test]
fn canonical_flops_count_is_correct() {
    // C(52,3) = 22100 raw flops. Suit isomorphism reduces to ~1755.
    // Exact count depends on canonicalization; just verify reasonable range.
    let flops = canonical_flops();
    assert!(flops.len() > 1000, "too few canonical flops: {}", flops.len());
    assert!(flops.len() < 3000, "too many canonical flops: {}", flops.len());
    // All flops should have 3 distinct cards
    for f in &flops {
        assert_ne!(f[0], f[1]);
        assert_ne!(f[1], f[2]);
        assert_ne!(f[0], f[2]);
    }
}

#[timed_test]
fn live_turn_cards_excludes_board_and_hole() {
    let hole = [
        make_card(Value::Ace, Suit::Spade),
        make_card(Value::King, Suit::Heart),
    ];
    let flop = [
        make_card(Value::Two, Suit::Diamond),
        make_card(Value::Five, Suit::Club),
        make_card(Value::Eight, Suit::Spade),
    ];
    let turns = live_turn_cards(&hole, &flop);
    assert_eq!(turns.len(), 47); // 52 - 2 hole - 3 flop
    assert!(!turns.contains(&hole[0]));
    assert!(!turns.contains(&hole[1]));
    for c in &flop {
        assert!(!turns.contains(c));
    }
}
```

**Step 2: Run tests to verify failure**

Run: `cargo test -p poker-solver-core canonical_flops_count -- --nocapture 2>&1 | head -20`
Expected: Compilation error.

**Step 3: Implement canonical board enumeration**

```rust
/// Generate all suit-isomorphic canonical flops (3-card boards).
///
/// Uses a simple canonical form: sort by value, then assign suits in first-seen order.
/// This is not the most compact canonicalization but is correct and fast enough for
/// one-time enumeration.
#[must_use]
pub fn canonical_flops() -> Vec<[Card; 3]> {
    let all: Vec<Card> = all_cards().collect();
    let mut seen = std::collections::HashSet::new();
    let mut result = Vec::new();

    for i in 0..all.len() {
        for j in (i + 1)..all.len() {
            for k in (j + 1)..all.len() {
                let flop = [all[i], all[j], all[k]];
                let key = canonical_board_key(&flop);
                if seen.insert(key) {
                    result.push(flop);
                }
            }
        }
    }
    result
}

/// All cards not in `hole` or `board`.
#[must_use]
pub fn live_turn_cards(hole: &[Card; 2], board: &[Card]) -> Vec<Card> {
    live_deck(*hole, board)
}

/// Canonical key for a board: sort by value, remap suits to first-seen order.
fn canonical_board_key(board: &[Card]) -> Vec<(Value, u8)> {
    let mut sorted: Vec<Card> = board.to_vec();
    sorted.sort_by_key(|c| (c.value, c.suit));

    let mut suit_map = [u8::MAX; 4]; // Suit -> canonical index
    let mut next_suit = 0u8;

    sorted
        .iter()
        .map(|c| {
            let suit_idx = c.suit as usize;
            if suit_map[suit_idx] == u8::MAX {
                suit_map[suit_idx] = next_suit;
                next_suit += 1;
            }
            (c.value, suit_map[suit_idx])
        })
        .collect()
}
```

**Step 4: Run tests**

Run: `cargo test -p poker-solver-core canonical_flops -- --nocapture && cargo test -p poker-solver-core live_turn_cards -- --nocapture`
Expected: Pass.

**Step 5: Commit**

```bash
git add crates/core/src/preflop/ehs.rs
git commit -m "feat: add canonical board enumeration for texture-free abstraction"
```

---

### Task 5: Build per-street histogram feature computation

**Files:**
- Modify: `crates/core/src/preflop/hand_buckets.rs` (add `compute_flop_histograms`, `compute_turn_histograms`, `compute_river_histograms`)
- Test: `crates/core/src/preflop/hand_buckets.rs` (inline tests)

**Context:** Replace the texture-based `compute_all_flop_features` etc. with new functions that compute equity histogram CDFs over enumerated canonical boards. Each returns a flat `Vec<[f64; 10]>` indexed by situation.

**Step 1: Write failing test**

```rust
#[timed_test(120)]
#[ignore = "slow"]
fn compute_flop_histograms_produces_valid_cdfs() {
    use crate::preflop::ehs::{canonical_flops, HISTOGRAM_BINS};
    let hands: Vec<CanonicalHand> = all_hands().collect();
    let flops = canonical_flops();
    // Use a small subset for testing
    let small_flops = &flops[..5.min(flops.len())];
    let histograms = compute_flop_histograms(&hands, small_flops, &|_| {});

    assert_eq!(histograms.len(), hands.len() * small_flops.len());

    // Every non-NaN histogram should be a valid CDF
    for hist in &histograms {
        if hist[0].is_nan() {
            continue; // blocked hand
        }
        for i in 1..HISTOGRAM_BINS {
            assert!(hist[i] >= hist[i - 1] - 1e-9,
                "CDF not monotonic: {:?}", hist);
        }
        assert!((hist[HISTOGRAM_BINS - 1] - 1.0).abs() < 1e-6,
            "CDF must end at 1.0: {:?}", hist);
    }
}
```

**Step 2: Run test to verify failure**

Run: `cargo test -p poker-solver-core compute_flop_histograms -- --ignored --nocapture 2>&1 | head -20`
Expected: Compilation error.

**Step 3: Implement per-street histogram computation**

```rust
use crate::preflop::ehs::{equity_histogram, HISTOGRAM_BINS};

/// Feature type for histogram-based k-means clustering.
pub type HistogramFeatures = [f64; HISTOGRAM_BINS];

/// Compute equity histogram CDF features for all (hand, flop) situations.
///
/// Returns flat `Vec<HistogramFeatures>` indexed by `hand_idx * num_flops + flop_idx`.
/// Blocked hands (hole cards conflict with board) get NaN sentinel.
pub fn compute_flop_histograms(
    hands: &[CanonicalHand],
    flops: &[[Card; 3]],
    on_hand_done: &(impl Fn(usize) + Sync + Send),
) -> Vec<HistogramFeatures> {
    let num_flops = flops.len();
    let done = AtomicUsize::new(0);

    let per_hand: Vec<Vec<HistogramFeatures>> = hands
        .par_iter()
        .map(|hand| {
            let feats: Vec<HistogramFeatures> = flops
                .iter()
                .map(|flop| {
                    let combos = hand.combos();
                    // Use first non-conflicting combo
                    if let Some(&(c1, c2)) = combos.iter().find(|&&(c1, c2)| {
                        !flop.contains(&c1) && !flop.contains(&c2)
                    }) {
                        equity_histogram(&[c1, c2], flop.as_slice())
                    } else {
                        [f64::NAN; HISTOGRAM_BINS] // blocked
                    }
                })
                .collect();
            let count = done.fetch_add(1, Ordering::Relaxed) + 1;
            on_hand_done(count);
            feats
        })
        .collect();

    // Flatten to flat indexing: hand_idx * num_flops + flop_idx
    per_hand.into_iter().flatten().collect()
}

/// Compute equity histogram CDF features for all (hand, flop, turn_card) situations.
///
/// Returns flat `Vec<HistogramFeatures>`. Situation index = hand_idx * num_boards + board_idx
/// where boards are enumerated as (flop, turn_card) pairs.
pub fn compute_turn_histograms(
    hands: &[CanonicalHand],
    turn_boards: &[[Card; 4]],
) -> Vec<HistogramFeatures> {
    let per_hand: Vec<Vec<HistogramFeatures>> = hands
        .par_iter()
        .map(|hand| {
            turn_boards
                .iter()
                .map(|board| {
                    let combos = hand.combos();
                    if let Some(&(c1, c2)) = combos.iter().find(|&&(c1, c2)| {
                        !board.contains(&c1) && !board.contains(&c2)
                    }) {
                        equity_histogram(&[c1, c2], board.as_slice())
                    } else {
                        [f64::NAN; HISTOGRAM_BINS]
                    }
                })
                .collect()
        })
        .collect();

    per_hand.into_iter().flatten().collect()
}

/// Compute equity (scalar) features for all (hand, board_5) situations on the river.
///
/// River has no future cards, so feature = raw equity. Returns flat Vec.
pub fn compute_river_equities(
    hands: &[CanonicalHand],
    river_boards: &[[Card; 5]],
) -> Vec<f64> {
    let per_hand: Vec<Vec<f64>> = hands
        .par_iter()
        .map(|hand| {
            river_boards
                .iter()
                .map(|board| {
                    let combos = hand.combos();
                    if let Some(&(c1, c2)) = combos.iter().find(|&&(c1, c2)| {
                        !board.contains(&c1) && !board.contains(&c2)
                    }) {
                        crate::preflop::ehs::ehs_features([c1, c2], board.as_slice())[0]
                    } else {
                        f64::NAN
                    }
                })
                .collect()
        })
        .collect();

    per_hand.into_iter().flatten().collect()
}
```

**Step 4: Run tests**

Run: `cargo test -p poker-solver-core compute_flop_histograms -- --ignored --nocapture`
Expected: Pass.

**Step 5: Commit**

```bash
git add crates/core/src/preflop/hand_buckets.rs
git commit -m "feat: add per-street equity histogram computation for canonical boards"
```

---

### Task 6: Build independent per-street clustering pipeline

**Files:**
- Modify: `crates/core/src/preflop/hand_buckets.rs`
- Test: `crates/core/src/preflop/hand_buckets.rs` (inline tests)

**Context:** Wire up histogram features → k-means → `StreetBuckets`. This is the core function that replaces the old `build_flop_buckets`/`build_turn_buckets`/`build_river_buckets` + transition table pipeline.

**Step 1: Write failing test**

```rust
#[timed_test(120)]
#[ignore = "slow"]
fn build_street_buckets_independent_produces_valid_assignments() {
    use crate::preflop::ehs::canonical_flops;
    let hands: Vec<CanonicalHand> = all_hands().collect();
    let flops = canonical_flops();
    let small_flops = &flops[..3.min(flops.len())];

    let buckets = build_street_buckets_independent(
        &hands, small_flops, 5, 5, 5, &|_| {},
    );

    // All bucket IDs should be in range
    for &b in &buckets.flop {
        assert!(b < buckets.num_flop_buckets, "flop bucket {b} >= {}", buckets.num_flop_buckets);
    }
    // Flop situation count = hands × flops
    assert_eq!(buckets.flop.len(), hands.len() * small_flops.len());
}
```

**Step 2: Run test to verify failure**

Run: `cargo test -p poker-solver-core build_street_buckets_independent -- --ignored --nocapture 2>&1 | head -20`
Expected: Compilation error.

**Step 3: Implement `build_street_buckets_independent`**

```rust
/// Build independent per-street bucket assignments from canonical boards.
///
/// Each street runs k-means independently on equity histogram CDFs (flop/turn)
/// or raw equity (river). No cross-street dependencies. This implements standard
/// imperfect-recall abstraction as used by Pluribus.
#[allow(clippy::too_many_arguments)]
pub fn build_street_buckets_independent(
    hands: &[CanonicalHand],
    flops: &[[Card; 3]],
    num_flop_buckets: u16,
    num_turn_buckets: u16,
    num_river_buckets: u16,
    on_progress: &(impl Fn(BuildProgress) + Sync + Send),
) -> StreetBuckets {
    // ── Flop ──
    on_progress(BuildProgress::FlopFeatures);
    let flop_features = compute_flop_histograms(hands, flops, &|_| {});

    on_progress(BuildProgress::FlopClustering);
    let flop_assignments = cluster_histograms(&flop_features, num_flop_buckets);

    // ── Turn ──
    // Enumerate turn boards: for each flop, all live turn cards
    on_progress(BuildProgress::TurnFeatures);
    let turn_boards = enumerate_turn_boards(hands, flops);
    let turn_features = compute_turn_histograms(hands, &turn_boards);

    on_progress(BuildProgress::TurnClustering);
    let turn_assignments = cluster_histograms(&turn_features, num_turn_buckets);

    // ── River ──
    on_progress(BuildProgress::RiverFeatures);
    let river_boards = enumerate_river_boards(hands, &turn_boards);
    let river_equities = compute_river_equities(hands, &river_boards);

    on_progress(BuildProgress::RiverClustering);
    let river_assignments = cluster_river_equities(&river_equities, num_river_buckets);

    StreetBuckets {
        flop: flop_assignments,
        num_flop_buckets,
        turn: turn_assignments,
        num_turn_buckets,
        river: river_assignments,
        num_river_buckets,
    }
}

/// Progress phases for build_street_buckets_independent.
pub enum BuildProgress {
    FlopFeatures,
    FlopClustering,
    TurnFeatures,
    TurnClustering,
    RiverFeatures,
    RiverClustering,
}

/// Cluster histogram features into k buckets using k-means with L2 on CDFs.
fn cluster_histograms(features: &[HistogramFeatures], k: u16) -> Vec<u16> {
    // Filter out NaN (blocked) features, cluster valid ones, assign blocked to nearest
    let valid_indices: Vec<usize> = features.iter().enumerate()
        .filter(|(_, f)| !f[0].is_nan())
        .map(|(i, _)| i)
        .collect();
    let valid_points: Vec<HistogramFeatures> = valid_indices.iter()
        .map(|&i| features[i])
        .collect();

    let valid_assignments = kmeans_generic(&valid_points, k as usize, 100);

    // Build full assignment vec
    let centroids = recompute_centroids_generic(&valid_points, &valid_assignments, k as usize);
    let mut full = vec![0u16; features.len()];
    for (vi, &orig_idx) in valid_indices.iter().enumerate() {
        full[orig_idx] = valid_assignments[vi];
    }
    // Assign blocked to nearest centroid
    for (i, feat) in features.iter().enumerate() {
        if feat[0].is_nan() {
            full[i] = nearest_centroid_generic(
                &[0.5; HISTOGRAM_BINS], // uniform CDF fallback
                &centroids,
            );
        }
    }
    full
}
```

Also add `enumerate_turn_boards` and `enumerate_river_boards` helper functions, and `cluster_river_equities` for scalar river features.

**Step 4: Run tests**

Run: `cargo test -p poker-solver-core build_street_buckets_independent -- --ignored --nocapture`
Expected: Pass.

**Step 5: Commit**

```bash
git add crates/core/src/preflop/hand_buckets.rs
git commit -m "feat: add independent per-street clustering pipeline (imperfect recall)"
```

---

### Task 7: Update `PostflopModelConfig` to remove texture fields, add histogram config

**Files:**
- Modify: `crates/core/src/preflop/postflop_model.rs`
- Test: `crates/core/src/preflop/postflop_model.rs` (inline tests)

**Context:** Remove `num_flop_textures`, `num_turn_transitions`, `num_river_transitions`, `ehs_samples`. Add `histogram_bins` (default 10). Update defaults to 500/500/500. Update presets.

**Step 1: Write failing test**

```rust
#[timed_test]
fn standard_preset_defaults_to_500_buckets() {
    let cfg = PostflopModelConfig::standard();
    assert_eq!(cfg.num_hand_buckets_flop, 500);
    assert_eq!(cfg.num_hand_buckets_turn, 500);
    assert_eq!(cfg.num_hand_buckets_river, 500);
}
```

**Step 2: Run test to verify failure**

Expected: Fails — current default is 2000.

**Step 3: Update config**

Remove texture fields, update defaults, update presets. Keep `ehs_samples` for backward-compat deserialization but mark deprecated. Change defaults:
- fast: 50/50/50
- medium: 200/200/200
- standard: 500/500/500
- accurate: 1000/1000/1000

**Step 4: Run tests**

Run: `cargo test -p poker-solver-core postflop_model -- --nocapture`
Expected: All tests pass (update test expectations to match new defaults).

**Step 5: Commit**

```bash
git add crates/core/src/preflop/postflop_model.rs
git commit -m "feat: update PostflopModelConfig defaults to 500/500/500, remove texture fields"
```

---

### Task 8: Migrate `PostflopAbstraction` to use `StreetBuckets`

**Files:**
- Modify: `crates/core/src/preflop/postflop_abstraction.rs`
- Modify: `crates/core/src/preflop/abstraction_cache.rs`
- Test: existing tests in both files

**Context:** This is the major integration task. Replace `HandBucketMapping` with `StreetBuckets` throughout `PostflopAbstraction`. Replace the `build_hand_buckets_and_equity` function. Remove `transition_buckets`. Update the CFR traversal to use direct per-street bucket lookups instead of transition tables. Remove the `BoardAbstraction` dependency.

**Key changes:**
1. `PostflopAbstraction` stores `StreetBuckets` instead of `HandBucketMapping`
2. `build_hand_buckets_and_equity` → calls `build_street_buckets_independent` + builds `StreetEquity` from the new buckets
3. `transition_buckets` → removed. At chance nodes, look up the new street's bucket directly from the hand's situation index.
4. The CFR traversal carries `(hand_idx, board_state)` alongside bucket IDs so it can look up the new bucket at each chance node
5. Update `abstraction_cache` to serialize/deserialize `StreetBuckets` instead of `HandBucketMapping`
6. Remove `BoardAbstraction` from `PostflopAbstraction` struct (no longer needed for bucketing)

**Step 1: Update `PostflopAbstraction` struct**

Replace `buckets: HandBucketMapping` with `buckets: StreetBuckets`. Remove `board: BoardAbstraction` field.

**Step 2: Update `load_or_build_abstraction`**

Replace the board abstraction build + texture-based bucketing with `canonical_flops()` + `build_street_buckets_independent()`.

**Step 3: Update CFR traversal**

In `solve_cfr_traverse`, replace `transition_buckets(buckets, street, hero_bucket, opp_bucket, tex_id)` with direct lookups into `StreetBuckets` using the hand's situation index for the new street.

**Step 4: Update `abstraction_cache.rs`**

Change `HandBucketMapping` → `StreetBuckets` in cache key computation, serialization, deserialization, and the `bincode_clone_buckets` helper.

**Step 5: Run all tests**

Run: `cargo test -p poker-solver-core -- --nocapture`
Expected: All tests pass. Some old tests referencing `HandBucketMapping` will need updating.

**Step 6: Commit**

```bash
git add crates/core/src/preflop/postflop_abstraction.rs crates/core/src/preflop/abstraction_cache.rs
git commit -m "feat: migrate PostflopAbstraction to StreetBuckets with imperfect recall"
```

---

### Task 9: Update sample configs and trainer

**Files:**
- Modify: `sample_configurations/fast_buckets.yaml`
- Modify: `sample_configurations/preflop_medium.yaml`
- Modify: `crates/trainer/src/main.rs` (if it references texture config)

**Step 1: Update YAML configs**

Remove `num_flop_textures`, `num_turn_transitions`, `num_river_transitions`, `ehs_samples` fields. Update bucket counts to new defaults.

**Step 2: Update trainer if needed**

Check if `main.rs` references any removed config fields and update.

**Step 3: Run full test suite**

Run: `cargo test`
Expected: All tests pass.

**Step 4: Commit**

```bash
git add sample_configurations/ crates/trainer/src/main.rs
git commit -m "chore: update sample configs and trainer for new abstraction pipeline"
```

---

### Task 10: Clean up dead code

**Files:**
- Modify: `crates/core/src/preflop/hand_buckets.rs` — remove `HandBucketMapping`, `build_flop_buckets`, `build_turn_buckets`, `build_river_buckets`, `build_transition_table`, `cluster_per_texture`, `cluster_global`, old `compute_all_*_features`
- Modify: `crates/core/src/preflop/postflop_abstraction.rs` — remove `transition_buckets`, any remaining texture references
- Modify: `crates/core/src/preflop/board_abstraction.rs` — evaluate if still needed; if only used for textures, mark for future removal
- Modify: `crates/core/src/preflop/mod.rs` — remove `board_abstraction` if fully unused

**Step 1: Remove dead functions and types**

Delete all functions/types that are no longer referenced after the migration.

**Step 2: Run clippy**

Run: `cargo clippy -p poker-solver-core -- -D warnings`
Expected: Clean.

**Step 3: Run full test suite**

Run: `cargo test`
Expected: All pass.

**Step 4: Commit**

```bash
git add -A
git commit -m "chore: remove dead texture-based bucketing code"
```

---

### Task 11: End-to-end integration test

**Files:**
- Modify: `crates/core/tests/` (add or update an integration test)

**Context:** Verify the full pipeline works: canonical board enumeration → histogram features → independent clustering → equity tables → postflop CFR solve → value extraction.

**Step 1: Write integration test**

```rust
#[test]
#[ignore = "slow"]
fn postflop_abstraction_with_imperfect_recall_converges() {
    let config = PostflopModelConfig::fast(); // 50/50/50 buckets
    let abstraction = PostflopAbstraction::build(&config, None, None, &|_| {}).unwrap();

    // Basic sanity: values should be in reasonable range
    let val = abstraction.evaluate(0, 0, 1, 0);
    assert!(val.abs() < 2.0, "EV out of range: {val}");
}
```

**Step 2: Run integration test**

Run: `cargo test -p poker-solver-core postflop_abstraction_with_imperfect_recall -- --ignored --nocapture`
Expected: Pass.

**Step 3: Commit**

```bash
git add crates/core/tests/
git commit -m "test: add integration test for imperfect-recall abstraction pipeline"
```
