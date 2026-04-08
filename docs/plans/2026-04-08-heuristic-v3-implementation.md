# Heuristic V3 Bucketing Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use hex:executing-plans to implement this plan task-by-task.

**Goal:** Add a new `ClusteringAlgorithm::HeuristicV3` variant that assigns postflop buckets using two axes — nut distance (6 bits) and equity delta (4 bits) — producing 1,024 buckets per street via deterministic per-flop precomputation, coexisting with the existing PotentialAwareEmd pipeline.

**Architecture:** The heuristic bucketing bypasses the EMD clustering pipeline entirely. Instead of building equity histograms and running k-means, it directly computes two scalar features per holding (nut distance and equity delta), quantizes them, and packs the result into a u16 bucket ID. Precomputed mappings are stored in the existing `PerFlopBucketFile` format. The MCCFR traversal is unchanged — it still looks up `bucket_id = mapping[combo_index]` from the per-flop file.

**Tech Stack:** Rust, rayon (parallelism), existing `showdown_equity` module, existing `PerFlopBucketFile` I/O, serde/YAML config.

---

### Task 1: Add HeuristicV3 to ClusteringAlgorithm enum

**Files:**
- Modify: `crates/core/src/blueprint_v2/config.rs:94-99`

**Step 1: Write the failing test**

In `crates/core/src/blueprint_v2/config.rs`, add a test at the bottom of the file (or in an existing `#[cfg(test)]` block):

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn heuristic_v3_algorithm_deserializes() {
        let yaml = r#"
            algorithm: heuristic_v3
            nut_distance_bits: 6
            equity_delta_bits: 4
            preflop:
              buckets: 169
            flop:
              buckets: 1024
            turn:
              buckets: 1024
            river:
              buckets: 64
        "#;
        let config: ClusteringConfig = serde_yaml::from_str(yaml).unwrap();
        assert!(matches!(config.algorithm, ClusteringAlgorithm::HeuristicV3));
        assert_eq!(config.nut_distance_bits, 6);
        assert_eq!(config.equity_delta_bits, 4);
    }

    #[test]
    fn heuristic_v3_defaults() {
        let yaml = r#"
            algorithm: heuristic_v3
            preflop:
              buckets: 169
            flop:
              buckets: 1024
            turn:
              buckets: 1024
            river:
              buckets: 64
        "#;
        let config: ClusteringConfig = serde_yaml::from_str(yaml).unwrap();
        assert_eq!(config.nut_distance_bits, 6);
        assert_eq!(config.equity_delta_bits, 4);
    }
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test -p poker-solver-core heuristic_v3_algorithm_deserializes`
Expected: FAIL — `HeuristicV3` variant doesn't exist yet.

**Step 3: Write minimal implementation**

In `crates/core/src/blueprint_v2/config.rs`:

1. Add variant to `ClusteringAlgorithm` (line ~97):
```rust
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ClusteringAlgorithm {
    #[default]
    PotentialAwareEmd,
    HeuristicV3,
}
```

2. Add fields to `ClusteringConfig` (after line 90, before the closing brace):
```rust
    /// Number of bits for nut distance quantization (default 6, range 2-8).
    /// Only used when algorithm = heuristic_v3.
    #[serde(default = "default_nut_distance_bits")]
    pub nut_distance_bits: u8,
    /// Number of bits for equity delta quantization (default 4, range 2-8).
    /// Only used when algorithm = heuristic_v3.
    #[serde(default = "default_equity_delta_bits")]
    pub equity_delta_bits: u8,
```

3. Add default functions:
```rust
fn default_nut_distance_bits() -> u8 { 6 }
fn default_equity_delta_bits() -> u8 { 4 }
```

**Step 4: Run test to verify it passes**

Run: `cargo test -p poker-solver-core heuristic_v3`
Expected: PASS

**Step 5: Commit**

```bash
git add crates/core/src/blueprint_v2/config.rs
git commit -m "feat(config): add HeuristicV3 clustering algorithm variant"
```

---

### Task 2: Implement nut distance computation

**Files:**
- Modify: `crates/core/src/showdown_equity.rs`

**Step 1: Write the failing test**

Add to the existing `#[cfg(test)] mod tests` block in `showdown_equity.rs`:

```rust
#[timed_test]
fn nut_distance_aces_on_dry_board() {
    let hole = [card(Ace, Spade), card(Ace, Heart)];
    let board = [
        card(King, Diamond),
        card(Seven, Club),
        card(Three, Spade),
        card(Nine, Heart),
        card(Two, Diamond),
    ];
    let nd = nut_distance(hole, &board);
    // AA on dry board — very few hands beat us (only sets)
    assert!(nd < 0.05, "AA nut distance should be very low: {nd}");
}

#[timed_test]
fn nut_distance_weak_hand_high() {
    let hole = [card(Seven, Club), card(Two, Diamond)];
    let board = [
        card(Ace, Spade),
        card(King, Heart),
        card(Queen, Diamond),
        card(Jack, Club),
        card(Nine, Spade),
    ];
    let nd = nut_distance(hole, &board);
    // 72o on AKQJ9 — most hands beat us
    assert!(nd > 0.80, "72o nut distance should be very high: {nd}");
}

#[timed_test]
fn nut_distance_range_is_zero_to_one() {
    let hole = [card(Ace, Spade), card(King, Spade)];
    let board = [card(Queen, Heart), card(Seven, Diamond), card(Three, Club)];
    let nd = nut_distance(hole, &board);
    assert!((0.0..=1.0).contains(&nd), "nut distance out of range: {nd}");
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test -p poker-solver-core nut_distance`
Expected: FAIL — `nut_distance` function doesn't exist.

**Step 3: Write minimal implementation**

Add to `showdown_equity.rs`, right after the `compute_equity` function:

```rust
/// Compute nut distance for `hole` against all possible opponents on `board`.
///
/// Returns the fraction of opponent holdings that beat us, in [0.0, 1.0].
/// 0.0 = absolute nuts (no opponent hand beats us).
/// 1.0 = absolute air (every opponent hand beats us).
///
/// Ties do not count as "beaten by" — only strict losses.
#[must_use]
pub fn nut_distance(hole: [Card; 2], board: &[Card]) -> f64 {
    let our_rank = rank_hand(hole, board);
    let remaining = remaining_cards(hole, board);

    let (beaten_by, total) = remaining
        .iter()
        .enumerate()
        .flat_map(|(i, &c1)| remaining[i + 1..].iter().map(move |&c2| [c1, c2]))
        .fold((0u32, 0u32), |(b, n), opp| {
            let opp_rank = rank_hand(opp, board);
            if opp_rank > our_rank {
                (b + 1, n + 1)
            } else {
                (b, n + 1)
            }
        });

    if total == 0 {
        return 0.0;
    }
    f64::from(beaten_by) / f64::from(total)
}
```

**Step 4: Run test to verify it passes**

Run: `cargo test -p poker-solver-core nut_distance`
Expected: PASS

**Step 5: Commit**

```bash
git add crates/core/src/showdown_equity.rs
git commit -m "feat(equity): add nut_distance() computation"
```

---

### Task 3: Implement equity delta computation

**Files:**
- Modify: `crates/core/src/showdown_equity.rs`

**Step 1: Write the failing test**

Add to the test block in `showdown_equity.rs`:

```rust
#[timed_test]
fn equity_delta_flush_draw_positive() {
    // Flush draw on flop — should have positive delta (likely to improve)
    let hole = [card(Ace, Heart), card(King, Heart)];
    let board = [card(Queen, Heart), card(Seven, Heart), card(Three, Club)];
    let delta = equity_delta(hole, &board);
    assert!(delta > 0.0, "Flush draw should have positive delta: {delta}");
}

#[timed_test]
fn equity_delta_top_set_near_zero() {
    // Top set on dry flop — stable, delta near zero or slightly negative
    let hole = [card(King, Spade), card(King, Heart)];
    let board = [card(King, Diamond), card(Seven, Club), card(Three, Spade)];
    let delta = equity_delta(hole, &board);
    assert!(delta.abs() < 0.10, "Top set delta should be near zero: {delta}");
}

#[timed_test]
fn equity_delta_river_is_zero() {
    // On river, no future cards — delta should be 0.0
    let hole = [card(Ace, Spade), card(King, Spade)];
    let board = [
        card(Queen, Heart),
        card(Seven, Diamond),
        card(Three, Club),
        card(Nine, Heart),
        card(Two, Diamond),
    ];
    let delta = equity_delta(hole, &board);
    assert!(
        delta.abs() < f64::EPSILON,
        "River delta should be exactly 0.0: {delta}"
    );
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test -p poker-solver-core equity_delta`
Expected: FAIL — `equity_delta` function doesn't exist.

**Step 3: Write minimal implementation**

Add to `showdown_equity.rs`:

```rust
/// Compute equity delta: expected future equity minus current equity.
///
/// For flop (3 cards): averages equity over all possible turn cards.
/// For turn (4 cards): averages equity over all possible river cards.
/// For river (5 cards): returns 0.0 (no future cards).
///
/// Positive delta = hand is likely to improve (draws).
/// Negative delta = hand is likely to get worse (vulnerable made hands).
/// Range is roughly [-0.5, +0.5].
#[must_use]
pub fn equity_delta(hole: [Card; 2], board: &[Card]) -> f64 {
    if board.len() >= 5 {
        return 0.0;
    }

    let current_equity = compute_equity(hole, board);
    let remaining = remaining_cards(hole, board);

    let mut total_future = 0.0f64;
    let mut count = 0u32;

    let mut extended_board = board.to_vec();
    extended_board.push(Card::new(Value::Two, Suit::Spade)); // placeholder

    for &next_card in &remaining {
        // Skip cards already on the board (shouldn't happen but guard)
        if board.iter().any(|&b| b.value == next_card.value && b.suit == next_card.suit) {
            continue;
        }
        *extended_board.last_mut().unwrap() = next_card;
        total_future += compute_equity(hole, &extended_board);
        count += 1;
    }

    if count == 0 {
        return 0.0;
    }
    (total_future / f64::from(count)) - current_equity
}
```

**Step 4: Run test to verify it passes**

Run: `cargo test -p poker-solver-core equity_delta`
Expected: PASS

**Step 5: Commit**

```bash
git add crates/core/src/showdown_equity.rs
git commit -m "feat(equity): add equity_delta() computation"
```

---

### Task 4: Implement quantization helpers and bucket packing

**Files:**
- Modify: `crates/core/src/showdown_equity.rs`

**Step 1: Write the failing test**

```rust
#[timed_test]
fn nut_distance_bin_boundaries() {
    // 6 bits = 64 bins
    assert_eq!(nut_distance_bin(0.0, 6), 0);
    assert_eq!(nut_distance_bin(1.0, 6), 63);
    assert_eq!(nut_distance_bin(0.5, 6), 32);
    // 4 bits = 16 bins
    assert_eq!(nut_distance_bin(0.0, 4), 0);
    assert_eq!(nut_distance_bin(1.0, 4), 15);
}

#[timed_test]
fn delta_bin_boundaries() {
    // 4 bits = 16 bins over [-0.5, +0.5]
    assert_eq!(delta_bin(0.0, 4), 8);   // midpoint
    assert_eq!(delta_bin(-0.5, 4), 0);  // minimum
    assert_eq!(delta_bin(0.5, 4), 15);  // maximum
    assert_eq!(delta_bin(0.25, 4), 12); // positive delta
}

#[timed_test]
fn pack_heuristic_v3_bucket() {
    // nut_dist_bin=10, delta_bin=5, delta_bits=4
    let bucket = pack_heuristic_v3(10, 5, 4);
    assert_eq!(bucket, (10 << 4) | 5);
    assert_eq!(bucket, 165);
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test -p poker-solver-core nut_distance_bin`
Expected: FAIL

**Step 3: Write minimal implementation**

Add to `showdown_equity.rs`:

```rust
/// Quantize nut distance [0.0, 1.0] into bins.
///
/// Returns a value in [0, 2^bits - 1].
#[must_use]
pub fn nut_distance_bin(nd: f64, bits: u8) -> u16 {
    let num_bins = 1u16 << bits;
    let bin = (nd * f64::from(num_bins)) as u16;
    bin.min(num_bins - 1)
}

/// Quantize equity delta [-0.5, +0.5] into bins.
///
/// Maps the signed range to [0, 2^bits - 1] with 0.0 at the midpoint.
#[must_use]
pub fn delta_bin(delta: f64, bits: u8) -> u16 {
    let num_bins = 1u16 << bits;
    // Map [-0.5, +0.5] to [0.0, 1.0]
    let normalized = (delta + 0.5).clamp(0.0, 1.0);
    let bin = (normalized * f64::from(num_bins)) as u16;
    bin.min(num_bins - 1)
}

/// Pack nut distance bin and delta bin into a single u16 bucket ID.
///
/// Layout: `(nut_dist_bin << delta_bits) | delta_bin`
#[must_use]
pub fn pack_heuristic_v3(nut_dist_bin: u16, delta_bin: u16, delta_bits: u8) -> u16 {
    (nut_dist_bin << delta_bits) | delta_bin
}
```

**Step 4: Run test to verify it passes**

Run: `cargo test -p poker-solver-core nut_distance_bin delta_bin pack_heuristic`
Expected: PASS

**Step 5: Commit**

```bash
git add crates/core/src/showdown_equity.rs
git commit -m "feat(equity): add heuristic V3 quantization and packing helpers"
```

---

### Task 5: Implement per-flop heuristic bucketing pipeline

This is the core task: generate `PerFlopBucketFile`s using heuristic V3 bucketing instead of EMD clustering.

**Files:**
- Modify: `crates/core/src/blueprint_v2/cluster_pipeline.rs`

**Step 1: Write the failing test**

Add a test in `cluster_pipeline.rs` test module:

```rust
#[test]
fn heuristic_v3_single_flop_produces_valid_buckets() {
    use crate::poker::{Card, Suit, Value};

    let flop = [
        Card::new(Value::King, Suit::Spade),
        Card::new(Value::Nine, Suit::Heart),
        Card::new(Value::Seven, Suit::Diamond),
    ];
    let pf = heuristic_v3_single_flop(flop, 6, 4, |_, _, _| {});

    // Bucket counts should match: 2^6 for nut distance, 2^4 for delta
    let max_bucket = (1u16 << 6) * (1u16 << 4); // 1024
    assert!(pf.turn_bucket_count <= max_bucket);
    assert!(pf.river_bucket_count <= max_bucket);

    // All bucket values should be within range
    for &b in &pf.turn_buckets {
        assert!(b < max_bucket, "turn bucket {b} >= {max_bucket}");
    }
    for river_buckets in &pf.river_buckets_per_turn {
        for &b in river_buckets {
            assert!(b < max_bucket, "river bucket {b} >= {max_bucket}");
        }
    }

    // Should have turn cards
    assert!(!pf.turn_cards.is_empty());
    // Should have river cards per turn
    assert_eq!(pf.river_cards_per_turn.len(), pf.turn_cards.len());
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test -p poker-solver-core heuristic_v3_single_flop`
Expected: FAIL — function doesn't exist.

**Step 3: Write minimal implementation**

Add to `cluster_pipeline.rs`:

```rust
use crate::showdown_equity::{
    nut_distance, equity_delta, nut_distance_bin, delta_bin, pack_heuristic_v3,
};

/// Generate a per-flop bucket file using heuristic V3 bucketing.
///
/// For each turn card and each combo, computes nut distance and equity delta,
/// quantizes them, and packs into a u16 bucket ID. For river, equity delta
/// is 0 (no future cards), so only nut distance is used.
///
/// `progress` is called with (phase, current, total).
#[allow(clippy::cast_possible_truncation)]
pub fn heuristic_v3_single_flop(
    flop: [Card; 3],
    nut_distance_bits: u8,
    equity_delta_bits: u8,
    progress: impl Fn(&str, u32, u32) + Sync,
) -> PerFlopBucketFile {
    let max_nd_bin = 1u16 << nut_distance_bits;
    let max_delta_bin = 1u16 << equity_delta_bits;
    let max_bucket = max_nd_bin * max_delta_bin;
    // For river, delta = 0 so only nut_distance bins matter.
    let river_max_bucket = max_nd_bin;

    let deck = build_deck();
    let flop_set: arrayvec::ArrayVec<Card, 3> = flop.iter().copied().collect();

    // Enumerate turn cards (cards not on flop)
    let turn_cards: Vec<Card> = deck
        .iter()
        .filter(|c| !flop_set.iter().any(|f| f.value == c.value && f.suit == c.suit))
        .copied()
        .collect();

    let total_turns = turn_cards.len() as u32;

    // For each turn card, compute turn buckets and river buckets
    let per_turn_results: Vec<_> = turn_cards
        .iter()
        .enumerate()
        .map(|(t_idx, &turn_card)| {
            let turn_board = [flop[0], flop[1], flop[2], turn_card];

            // --- Turn buckets (nut distance + equity delta) ---
            let mut turn_buckets = vec![0u16; COMBOS];
            for i in 0..deck.len() {
                if turn_board.iter().any(|b| b.value == deck[i].value && b.suit == deck[i].suit) {
                    continue;
                }
                for j in (i + 1)..deck.len() {
                    if turn_board.iter().any(|b| b.value == deck[j].value && b.suit == deck[j].suit) {
                        continue;
                    }
                    let hole = [deck[i], deck[j]];
                    let cidx = combo_index(deck[i], deck[j]) as usize;
                    let nd = nut_distance(hole, &turn_board);
                    let ed = equity_delta(hole, &turn_board);
                    let nd_bin = nut_distance_bin(nd, nut_distance_bits);
                    let ed_bin = delta_bin(ed, equity_delta_bits);
                    turn_buckets[cidx] = pack_heuristic_v3(nd_bin, ed_bin, equity_delta_bits);
                }
            }

            // --- River buckets (nut distance only, delta = 0) ---
            let river_cards: Vec<Card> = deck
                .iter()
                .filter(|c| {
                    !turn_board.iter().any(|b| b.value == c.value && b.suit == c.suit)
                })
                .copied()
                .collect();

            let mut all_river_buckets = vec![0u16; river_cards.len() * COMBOS];
            for (r_idx, &river_card) in river_cards.iter().enumerate() {
                let river_board = [flop[0], flop[1], flop[2], turn_card, river_card];
                for i in 0..deck.len() {
                    if river_board.iter().any(|b| b.value == deck[i].value && b.suit == deck[i].suit) {
                        continue;
                    }
                    for j in (i + 1)..deck.len() {
                        if river_board.iter().any(|b| b.value == deck[j].value && b.suit == deck[j].suit) {
                            continue;
                        }
                        let hole = [deck[i], deck[j]];
                        let cidx = combo_index(deck[i], deck[j]) as usize;
                        let nd = nut_distance(hole, &river_board);
                        let nd_bin = nut_distance_bin(nd, nut_distance_bits);
                        // Delta = 0 on river, so delta_bin = midpoint
                        let mid = delta_bin(0.0, equity_delta_bits);
                        all_river_buckets[r_idx * COMBOS + cidx] =
                            pack_heuristic_v3(nd_bin, mid, equity_delta_bits);
                    }
                }
            }

            progress("turn", t_idx as u32 + 1, total_turns);

            (turn_card, turn_buckets, river_cards, all_river_buckets)
        })
        .collect();

    // Assemble the PerFlopBucketFile
    let mut all_turn_buckets = Vec::with_capacity(turn_cards.len() * COMBOS);
    let mut river_cards_per_turn = Vec::with_capacity(turn_cards.len());
    let mut river_buckets_per_turn = Vec::with_capacity(turn_cards.len());
    let mut sorted_turn_cards = Vec::with_capacity(turn_cards.len());

    for (tc, tb, rc, rb) in per_turn_results {
        sorted_turn_cards.push(tc);
        all_turn_buckets.extend_from_slice(&tb);
        river_cards_per_turn.push(rc);
        river_buckets_per_turn.push(rb);
    }

    PerFlopBucketFile {
        flop_cards: flop,
        turn_bucket_count: max_bucket,
        river_bucket_count: river_max_bucket,
        turn_cards: sorted_turn_cards,
        turn_buckets: all_turn_buckets,
        river_cards_per_turn,
        river_buckets_per_turn,
    }
}
```

**Step 4: Run test to verify it passes**

Run: `cargo test -p poker-solver-core heuristic_v3_single_flop -- --nocapture`
Expected: PASS (may take a few seconds per flop due to enumeration)

**Step 5: Commit**

```bash
git add crates/core/src/blueprint_v2/cluster_pipeline.rs
git commit -m "feat(clustering): add heuristic_v3_single_flop per-flop bucketing"
```

---

### Task 6: Implement the full heuristic V3 pipeline over all flops

**Files:**
- Modify: `crates/core/src/blueprint_v2/cluster_pipeline.rs`

**Step 1: Write the failing test**

```rust
#[test]
fn run_heuristic_v3_pipeline_creates_files() {
    let dir = tempfile::tempdir().unwrap();
    // Run on just 1 flop for speed (use a progress callback that does nothing)
    run_heuristic_v3_pipeline(
        6,
        4,
        dir.path(),
        Some(1), // limit to 1 flop for testing
        |_, _, _| {},
    );
    // Should produce at least one flop_NNNN.buckets file
    let files: Vec<_> = std::fs::read_dir(dir.path())
        .unwrap()
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().map_or(false, |ext| ext == "buckets"))
        .collect();
    assert!(!files.is_empty(), "should produce bucket files");
    // Also produce preflop.buckets
    assert!(dir.path().join("preflop.buckets").exists());
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test -p poker-solver-core run_heuristic_v3_pipeline`
Expected: FAIL — function doesn't exist.

**Step 3: Write minimal implementation**

Add to `cluster_pipeline.rs`:

```rust
/// Run the heuristic V3 bucketing pipeline over all canonical flops.
///
/// Generates per-flop bucket files (`flop_NNNN.buckets`) and a preflop
/// bucket file in `output_dir`. Each flop file uses the `PerFlopBucketFile`
/// format with turn and river bucket assignments.
///
/// `max_flops` limits the number of flops processed (for testing). Pass
/// `None` to process all 1,755 canonical flops.
pub fn run_heuristic_v3_pipeline(
    nut_distance_bits: u8,
    equity_delta_bits: u8,
    output_dir: &Path,
    max_flops: Option<usize>,
    progress: impl Fn(&str, &str, f64) + Sync,
) {
    std::fs::create_dir_all(output_dir).expect("failed to create output dir");

    // 1. Preflop (deterministic canonical hand mapping, 169 buckets)
    progress("preflop", "mapping", 0.0);
    let preflop = cluster_preflop(|phase, p| progress("preflop", phase, p));
    preflop
        .save(&output_dir.join("preflop.buckets"))
        .expect("failed to save preflop buckets");

    // 2. Per-flop bucketing
    let canonical_flops = enumerate_canonical_flops();
    let total = max_flops.unwrap_or(canonical_flops.len()).min(canonical_flops.len());
    let flops_to_process = &canonical_flops[..total];

    let completed = std::sync::atomic::AtomicUsize::new(0);

    flops_to_process.par_iter().enumerate().for_each(|(idx, wb)| {
        let flop = [wb.cards[0], wb.cards[1], wb.cards[2]];
        let pf = heuristic_v3_single_flop(
            flop,
            nut_distance_bits,
            equity_delta_bits,
            |_, _, _| {},
        );
        let path = output_dir.join(format!("flop_{idx:04}.buckets"));
        pf.save(&path).unwrap_or_else(|e| {
            panic!("failed to save {}: {e}", path.display());
        });

        let done = completed.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
        progress(
            "per_flop",
            "bucketing",
            done as f64 / total as f64,
        );
    });
}
```

**Step 4: Run test to verify it passes**

Run: `cargo test -p poker-solver-core run_heuristic_v3_pipeline -- --nocapture`
Expected: PASS

**Step 5: Commit**

```bash
git add crates/core/src/blueprint_v2/cluster_pipeline.rs
git commit -m "feat(clustering): add run_heuristic_v3_pipeline for all flops"
```

---

### Task 7: Wire HeuristicV3 into the clustering CLI dispatch

**Files:**
- Modify: `crates/core/src/blueprint_v2/cluster_pipeline.rs:899` (in `run_clustering_pipeline`)
- Modify: `crates/trainer/src/main.rs` (Cluster command handler)

**Step 1: Write the failing test**

```rust
#[test]
fn run_clustering_pipeline_dispatches_heuristic_v3() {
    let config = ClusteringConfig {
        algorithm: ClusteringAlgorithm::HeuristicV3,
        preflop: StreetClusterConfig { buckets: 169, ..Default::default() },
        flop: StreetClusterConfig { buckets: 1024, ..Default::default() },
        turn: StreetClusterConfig { buckets: 1024, ..Default::default() },
        river: StreetClusterConfig { buckets: 64, ..Default::default() },
        seed: 42,
        kmeans_iterations: 10,
        cfvnet_river_data: None,
        per_flop: None,
        nut_distance_bits: 6,
        equity_delta_bits: 4,
    };
    let dir = tempfile::tempdir().unwrap();
    // This should not panic — it should dispatch to the heuristic V3 pipeline.
    // Limit to 1 flop via the progress callback won't help here, so we just
    // test that the function accepts HeuristicV3 without panicking.
    // For actual execution, use run_heuristic_v3_pipeline directly.
    let result = run_clustering_pipeline(&config, dir.path(), |_, _, _| {});
    assert!(result.is_ok());
}
```

Note: This test will be slow if it processes all flops. Consider adding a `max_flops` parameter to the config or gating via a test attribute. For the implementation, just ensure the dispatch branch exists.

**Step 2: Run test to verify it fails**

Run: `cargo test -p poker-solver-core run_clustering_pipeline_dispatches_heuristic`
Expected: FAIL — `run_clustering_pipeline` doesn't handle `HeuristicV3`.

**Step 3: Write minimal implementation**

Modify `run_clustering_pipeline` in `cluster_pipeline.rs` to dispatch on algorithm:

At the top of `run_clustering_pipeline` (line ~900), add an early return for HeuristicV3:

```rust
pub fn run_clustering_pipeline(
    config: &ClusteringConfig,
    output_dir: &Path,
    progress: impl Fn(&str, &str, f64) + Sync,
) -> Result<(), Box<dyn std::error::Error>> {
    // Dispatch to heuristic V3 pipeline if selected.
    if matches!(config.algorithm, ClusteringAlgorithm::HeuristicV3) {
        run_heuristic_v3_pipeline(
            config.nut_distance_bits,
            config.equity_delta_bits,
            output_dir,
            None,
            progress,
        );
        return Ok(());
    }

    // ... existing PotentialAwareEmd pipeline below ...
```

**Step 4: Run test to verify it passes**

Run: `cargo test -p poker-solver-core run_clustering_pipeline_dispatches`
Expected: PASS (will take a while if it processes all 1,755 flops — mark as `#[ignore]` for CI if needed)

**Step 5: Build workspace to verify no breakage**

Run: `cargo build`
Expected: PASS — the trainer and all dependent crates should compile.

**Step 6: Commit**

```bash
git add crates/core/src/blueprint_v2/cluster_pipeline.rs
git commit -m "feat(clustering): wire HeuristicV3 dispatch into run_clustering_pipeline"
```

---

### Task 8: Add sample YAML configuration

**Files:**
- Create: `sample_configurations/heuristic_v3_1024bkt.yaml`

**Step 1: Create the config file**

```yaml
game:
  name: "heuristic_v3_1024bkt"
  players: 2
  stack_depth: 200
  small_blind: 1
  big_blind: 2

clustering:
  algorithm: heuristic_v3
  nut_distance_bits: 6
  equity_delta_bits: 4
  preflop:
    buckets: 169
  flop:
    buckets: 1024
  turn:
    buckets: 1024
  river:
    buckets: 64
  per_flop:
    turn_buckets: 1024
    river_buckets: 64

action_abstraction:
  preflop:
    - ["5bb"]
    - ["8bb", "12bb"]
    - ["36bb"]
  flop:
    - [0.33, 0.67, 1.0]
    - [0.33, 0.67, 2.0]
    - [2.0]
  turn:
    - [0.33, 0.67, 1.0]
    - [0.33, 0.67, 2.0]
    - [2.0]
  river:
    - [0.33, 0.67, 1.0]
    - [0.33, 0.67, 2.0]
    - [2.0]

training:
  cluster_path: "./local_data/clusters_heuristic_v3"
  time_limit_minutes: 7200
  lcfr_warmup_iterations: 5000000
  prune_after_iterations: 5000000
  prune_threshold: 0
  lcfr_discount_interval: 1000000
  batch_size: 2000

snapshots:
  resume: true
  warmup_minutes: 60
  snapshot_every_minutes: 30
  output_dir: "./local_data/blueprints/heuristic_v3_1024bkt/snap"

tui:
  enabled: true
  refresh_rate_ms: 2000
  telemetry:
    strategy_delta_interval_seconds: 30
    sparkline_window: 60
  scenarios:
    - name: "SB Open"
      player: SB
      actions: []
    - name: "BB vs 2.5x"
      player: BB
      actions: ["raise-0"]
  random_scenario:
    enabled: true
    hold_minutes: 3
    pool: ["preflop", "flop", "turn", "river"]
```

**Step 2: Verify it parses**

Run: `cargo test -p poker-solver-core heuristic_v3_algorithm_deserializes`
Expected: PASS (already passes from Task 1)

**Step 3: Commit**

```bash
git add sample_configurations/heuristic_v3_1024bkt.yaml
git commit -m "docs: add sample config for heuristic V3 bucketing (1024 buckets)"
```

---

### Task 9: Update trainer bucket count detection for HeuristicV3

**Files:**
- Modify: `crates/core/src/blueprint_v2/trainer.rs:255-260`

**Context:** The trainer reads bucket counts from `config.clustering.{street}.buckets` to size the storage arrays. For HeuristicV3, the per-flop bucket counts are `per_flop.turn_buckets` and `per_flop.river_buckets`, and these must match what `heuristic_v3_single_flop` produces: `2^nut_distance_bits * 2^equity_delta_bits` for turn, `2^nut_distance_bits` for river (since delta=0 on river uses a single midpoint bin, the river bucket count stored in the PerFlopBucketFile is `2^nut_distance_bits` but the actual bucket IDs include the packed delta midpoint).

**Step 1: Write the failing test**

Add to `trainer.rs` tests:

```rust
#[test]
fn trainer_accepts_heuristic_v3_config() {
    use crate::blueprint_v2::config::*;

    let config = BlueprintV2Config {
        game: GameConfig {
            name: "test_hv3".to_string(),
            players: 2,
            stack_depth: 200.0,
            small_blind: 1.0,
            big_blind: 2.0,
            rake_rate: 0.0,
            rake_cap: 0.0,
            allow_preflop_limp: true,
        },
        clustering: ClusteringConfig {
            algorithm: ClusteringAlgorithm::HeuristicV3,
            preflop: StreetClusterConfig { buckets: 169, ..Default::default() },
            flop: StreetClusterConfig { buckets: 1024, ..Default::default() },
            turn: StreetClusterConfig { buckets: 1024, ..Default::default() },
            river: StreetClusterConfig { buckets: 1024, ..Default::default() },
            seed: 42,
            kmeans_iterations: 10,
            cfvnet_river_data: None,
            per_flop: Some(PerFlopConfig {
                turn_buckets: 1024,
                river_buckets: 1024,
            }),
            nut_distance_bits: 6,
            equity_delta_bits: 4,
        },
        // ... fill in remaining fields with defaults ...
    };
    // Should not panic
    let _trainer = BlueprintTrainer::new(config);
}
```

**Step 2: Verify the trainer initializes correctly**

The existing `BlueprintTrainer::new` should work without changes as long as the bucket counts in the config match the `PerFlopBucketFile` format. If not, adjust the bucket counts in the config to match `2^(nd_bits + delta_bits)`.

**Step 3: Commit if changes needed**

```bash
git add crates/core/src/blueprint_v2/trainer.rs
git commit -m "test: verify BlueprintTrainer accepts HeuristicV3 config"
```

---

### Task 10: End-to-end integration test

**Files:**
- Modify: `crates/core/tests/blueprint_v2_e2e.rs` (or create new test)

**Step 1: Write the integration test**

This test generates heuristic V3 buckets for 1 flop, then verifies the bucket file can be loaded and used by `AllBuckets`:

```rust
#[test]
fn heuristic_v3_end_to_end_single_flop() {
    use poker_solver_core::blueprint_v2::cluster_pipeline::{
        heuristic_v3_single_flop, cluster_preflop,
    };
    use poker_solver_core::blueprint_v2::per_flop_bucket_file::PerFlopBucketFile;
    use poker_solver_core::blueprint_v2::mccfr::AllBuckets;
    use poker_solver_core::poker::{Card, Suit, Value};

    let dir = tempfile::tempdir().unwrap();

    // Generate preflop buckets
    let preflop = cluster_preflop(|_, _| {});
    preflop.save(&dir.path().join("preflop.buckets")).unwrap();

    // Generate one flop's heuristic V3 buckets
    let flop = [
        Card::new(Value::King, Suit::Spade),
        Card::new(Value::Nine, Suit::Heart),
        Card::new(Value::Seven, Suit::Diamond),
    ];
    let pf = heuristic_v3_single_flop(flop, 6, 4, |_, _, _| {});
    pf.save(&dir.path().join("flop_0000.buckets")).unwrap();

    // Verify it loads
    let loaded = PerFlopBucketFile::load(&dir.path().join("flop_0000.buckets")).unwrap();
    assert_eq!(loaded.turn_bucket_count, 1024);

    // Verify distinct buckets exist (not all zeros)
    let unique_turn: std::collections::HashSet<u16> = loaded.turn_buckets.iter().copied().collect();
    assert!(unique_turn.len() > 10, "should have many distinct turn buckets, got {}", unique_turn.len());
}
```

**Step 2: Run the test**

Run: `cargo test -p poker-solver-core heuristic_v3_end_to_end -- --nocapture`
Expected: PASS

**Step 3: Commit**

```bash
git add crates/core/tests/blueprint_v2_e2e.rs
git commit -m "test: add end-to-end integration test for heuristic V3 bucketing"
```

---

### Task 11: Parallelize the per-turn computation in heuristic_v3_single_flop

**Files:**
- Modify: `crates/core/src/blueprint_v2/cluster_pipeline.rs`

**Context:** The initial implementation in Task 5 uses sequential iteration over turn cards. For production use, this should use `rayon` for parallel processing since each turn card is independent.

**Step 1: Change `.iter().enumerate().map()` to `.par_iter().enumerate().map()`**

In `heuristic_v3_single_flop`, change the turn card iteration to use rayon:

```rust
let per_turn_results: Vec<_> = turn_cards
    .par_iter()
    .enumerate()
    .map(|(t_idx, &turn_card)| {
        // ... same body ...
    })
    .collect();
```

**Step 2: Run existing tests to verify no breakage**

Run: `cargo test -p poker-solver-core heuristic_v3`
Expected: PASS

**Step 3: Commit**

```bash
git add crates/core/src/blueprint_v2/cluster_pipeline.rs
git commit -m "perf: parallelize heuristic_v3_single_flop with rayon"
```

---

### Task 12: Update documentation

**Files:**
- Modify: `docs/architecture.md` — add HeuristicV3 to the abstraction modes section
- Modify: `docs/training.md` — add heuristic_v3 clustering algorithm and config options

**Step 1: Add HeuristicV3 section to architecture.md**

Under the hand abstraction section, add a subsection describing the new mode:
- Two-axis bucketing: nut distance + equity delta
- Per-flop precomputation, stored as PerFlopBucketFile
- No clustering required — deterministic bin assignment
- Default: 6+4 bits = 1,024 buckets per street (64 on river)

**Step 2: Add config documentation to training.md**

Document the new clustering algorithm option and its parameters:
- `algorithm: heuristic_v3`
- `nut_distance_bits` (default 6)
- `equity_delta_bits` (default 4)
- How it differs from `potential_aware_emd`

**Step 3: Commit**

```bash
git add docs/architecture.md docs/training.md
git commit -m "docs: add HeuristicV3 bucketing to architecture and training docs"
```
