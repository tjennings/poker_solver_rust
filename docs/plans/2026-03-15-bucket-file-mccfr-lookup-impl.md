# Bucket File MCCFR Lookup — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use hex:executing-plans to implement this plan task-by-task.

**Goal:** Replace runtime `compute_equity()` calls in MCCFR with O(1) precomputed bucket file lookups (Pluribus-style), and fix the clustering pipeline to produce exhaustive canonical bucket files via two-phase clustering: sample-based k-means to find centroids, then exhaustive nearest-centroid assignment for all canonical boards.

**Architecture:** (A) Modify k-means functions to return centroids alongside assignments. (B) Add exhaustive assignment passes to each street's clustering function: enumerate all canonical boards, compute features, assign to nearest centroid. (C) Rewrite `AllBuckets::precompute_buckets()` to use O(1) bucket file lookups via board canonicalization + combo indexing.

**Tech Stack:** Rust, `rs_poker::core::Card`, existing `CanonicalBoard` from `abstraction/isomorphism.rs`, existing `BucketFile` v2 format, existing `combo_index()` and `canonical_key()` from `cluster_pipeline.rs`

---

### Task 1: Return centroids from k-means functions

**Files:**
- Modify: `crates/core/src/blueprint_v2/clustering.rs`

The k-means functions currently return only `Vec<u16>` (assignments). They need to also return the final centroids so the clustering pipeline can use them for exhaustive assignment.

**Step 1: Change `kmeans_1d_weighted` return type**

Change signature from:
```rust
pub fn kmeans_1d_weighted(...) -> Vec<u16>
```
to:
```rust
pub fn kmeans_1d_weighted(...) -> (Vec<u16>, Vec<f64>)
```

Return `(assignments, centroids)` at the end (line ~414):
```rust
    // Final assignment pass (parallel).
    let assignments = data.par_iter()
        .map(|&val| nearest_centroid_1d(val, &centroids))
        .collect();
    (assignments, centroids)
```

**Step 2: Change `kmeans_emd_weighted_u8` return type**

Change signature from:
```rust
pub fn kmeans_emd_weighted_u8(...) -> Vec<u16>
```
to:
```rust
pub fn kmeans_emd_weighted_u8(...) -> (Vec<u16>, Vec<Vec<f64>>)
```

Return `(assignments, centroids)` at the end (line ~611):
```rust
    // Final assignment pass (parallel).
    data.par_iter()
        .zip(assignments.par_iter_mut())
        .for_each(|(point, assign)| {
            *assign = nearest_centroid_u8(point, &centroids);
        });
    (assignments, centroids)
```

**Step 3: Make `nearest_centroid_u8` pub(crate)**

Change `fn nearest_centroid_u8` (line 741) to `pub(crate) fn nearest_centroid_u8`. It will be called from `cluster_pipeline.rs` during the exhaustive assignment pass.

`nearest_centroid_1d` is already `pub(crate)` (line 695).

**Step 4: Fix all call sites**

Grep for `kmeans_1d_weighted` and `kmeans_emd_weighted_u8` across the codebase. Each call currently expects `Vec<u16>` — update to destructure:

```rust
// Before:
let cluster_labels = kmeans_1d_weighted(&all_equities, &all_weights, k, iters);
// After:
let (cluster_labels, _centroids) = kmeans_1d_weighted(&all_equities, &all_weights, k, iters);
```

For the clustering pipeline functions that will use centroids (Task 2), keep the centroids. For all other callers (tests, cfvnet), discard with `_`.

**Step 5: Run tests**

```bash
cargo test -p poker-solver-core -- clustering
```

**Step 6: Commit**

```bash
git add crates/core/src/blueprint_v2/clustering.rs
git commit -m "feat: return centroids from k-means functions for exhaustive assignment"
```

---

### Task 2: Add exhaustive assignment to river clustering

**Files:**
- Modify: `crates/core/src/blueprint_v2/cluster_pipeline.rs`

Two-phase river clustering: (1) k-means on sampled boards to find centroids, (2) exhaustive assignment for all canonical rivers.

**Step 1: Create `cluster_river_exhaustive` function**

```rust
/// Two-phase river clustering: k-means on sampled boards, then exhaustive
/// nearest-centroid assignment for all canonical rivers.
///
/// Phase 1: Sample `sample_boards` canonical rivers, compute equity for each
/// (board, combo) pair, run weighted 1-D k-means to find centroids.
///
/// Phase 2: Enumerate ALL canonical rivers, compute equity for each
/// (board, combo) pair, assign to nearest centroid. Streams boards in
/// parallel batches to avoid holding all ~700K boards in memory at once.
pub fn cluster_river_exhaustive(
    bucket_count: u16,
    kmeans_iterations: u32,
    seed: u64,
    sample_boards: usize,
    progress: impl Fn(f64) + Sync,
) -> BucketFile {
    let combos = enumerate_combos(&build_deck());

    // Phase 1: K-means on sampled boards to find centroids.
    let all_canonical = enumerate_canonical_rivers();
    let (sampled_boards, board_weights) = sample_canonical(&all_canonical, sample_boards, seed);
    let num_sampled = sampled_boards.len();

    let board_equities: Vec<Vec<Option<f64>>> = sampled_boards
        .par_iter()
        .enumerate()
        .map(|(i, board)| {
            let eq = compute_board_equities(*board, &combos);
            #[allow(clippy::cast_precision_loss)]
            progress((i + 1) as f64 / num_sampled as f64 * 0.4);
            eq
        })
        .collect();

    let mut all_equities: Vec<f64> = Vec::new();
    let mut all_weights: Vec<f64> = Vec::new();
    for (board_idx, eqs) in board_equities.iter().enumerate() {
        let w = board_weights[board_idx];
        for eq in eqs.iter().flatten() {
            all_equities.push(*eq);
            all_weights.push(w);
        }
    }

    let (_sample_labels, centroids) = kmeans_1d_weighted(
        &all_equities, &all_weights, bucket_count as usize, kmeans_iterations,
    );
    drop(board_equities); // free memory before phase 2

    // Phase 2: Exhaustive assignment over all canonical rivers.
    let num_all = all_canonical.len();
    let all_buckets: Vec<Vec<u16>> = all_canonical
        .par_iter()
        .enumerate()
        .map(|(i, wb)| {
            let equities = compute_board_equities(wb.cards, &combos);
            let board_buckets: Vec<u16> = equities
                .iter()
                .map(|eq| {
                    match eq {
                        Some(e) => nearest_centroid_1d(*e, &centroids),
                        None => 0, // overlapping cards — bucket 0 as placeholder
                    }
                })
                .collect();
            #[allow(clippy::cast_precision_loss)]
            progress(0.4 + (i + 1) as f64 / num_all as f64 * 0.6);
            board_buckets
        })
        .collect();

    // Flatten into BucketFile.
    let mut buckets = Vec::with_capacity(num_all * TOTAL_COMBOS as usize);
    for board_buckets in &all_buckets {
        buckets.extend_from_slice(board_buckets);
    }

    let packed_boards: Vec<PackedBoard> = all_canonical
        .iter()
        .map(|wb| canonical_key(&wb.cards))
        .collect();

    #[allow(clippy::cast_possible_truncation)]
    BucketFile {
        header: BucketFileHeader {
            street: Street::River,
            bucket_count,
            board_count: num_all as u32,
            combos_per_board: TOTAL_COMBOS,
            version: 2,
        },
        boards: packed_boards,
        buckets,
    }
}
```

**Important:** `nearest_centroid_1d` is in `clustering.rs` — import it:
```rust
use super::clustering::nearest_centroid_1d;
```

**Step 2: Add import for `nearest_centroid_1d`**

At the top of `cluster_pipeline.rs`, add to the existing clustering imports:
```rust
use super::clustering::nearest_centroid_1d;
```

**Step 3: Run tests**

```bash
cargo test -p poker-solver-core -- cluster_pipeline
```

**Step 4: Commit**

```bash
git add crates/core/src/blueprint_v2/cluster_pipeline.rs
git commit -m "feat: two-phase river clustering with exhaustive centroid assignment"
```

---

### Task 3: Add exhaustive assignment to turn and flop clustering

**Files:**
- Modify: `crates/core/src/blueprint_v2/cluster_pipeline.rs`

Same two-phase pattern for turn and flop, using EMD centroids.

**Step 1: Create `cluster_turn_exhaustive` function**

```rust
/// Two-phase turn clustering: EMD k-means on sampled boards, then exhaustive
/// nearest-centroid assignment for all canonical turns.
pub fn cluster_turn_exhaustive(
    river_buckets: &BucketFile,
    bucket_count: u16,
    kmeans_iterations: u32,
    seed: u64,
    sample_boards: usize,
    progress: impl Fn(f64) + Sync,
) -> BucketFile {
    let deck = build_deck();
    let combos = enumerate_combos(&deck);
    let board_map = river_buckets.board_index_map();
    let all_canonical = enumerate_canonical_turns();

    // Phase 1: K-means on sampled boards.
    let (sampled_boards, board_weights) = sample_canonical(&all_canonical, sample_boards, seed);
    let num_sampled = sampled_boards.len();

    let board_features: Vec<Vec<Option<Vec<u8>>>> = sampled_boards
        .par_iter()
        .enumerate()
        .map(|(board_idx, board)| {
            let features: Vec<Option<Vec<u8>>> = combos
                .iter()
                .map(|combo| {
                    if cards_overlap(*combo, board) {
                        return None;
                    }
                    Some(build_bucket_histogram_u8(
                        *combo, board, &deck, river_buckets, &board_map,
                    ))
                })
                .collect();
            #[allow(clippy::cast_precision_loss)]
            progress((board_idx + 1) as f64 / num_sampled as f64 * 0.4);
            features
        })
        .collect();

    let mut all_features: Vec<Vec<u8>> = Vec::new();
    let mut all_weights: Vec<f64> = Vec::new();
    for (board_feats, wb_weight) in board_features.into_iter().zip(board_weights.iter()) {
        for feat in board_feats.into_iter().flatten() {
            all_features.push(feat);
            all_weights.push(*wb_weight);
        }
    }

    let (_sample_labels, centroids) = kmeans_emd_weighted_u8(
        &all_features, &all_weights, bucket_count as usize, kmeans_iterations, seed,
        |iter, max_iter| {
            #[allow(clippy::cast_precision_loss)]
            progress(0.4 + 0.2 * f64::from(iter) / f64::from(max_iter));
        },
    );
    drop(all_features); // free memory before phase 2

    // Phase 2: Exhaustive assignment over all canonical turns.
    let num_all = all_canonical.len();
    let all_buckets: Vec<Vec<u16>> = all_canonical
        .par_iter()
        .enumerate()
        .map(|(board_idx, wb)| {
            let board_buckets: Vec<u16> = combos
                .iter()
                .map(|combo| {
                    if cards_overlap(*combo, &wb.cards) {
                        return 0;
                    }
                    let hist = build_bucket_histogram_u8(
                        *combo, &wb.cards, &deck, river_buckets, &board_map,
                    );
                    nearest_centroid_u8(&hist, &centroids)
                })
                .collect();
            #[allow(clippy::cast_precision_loss)]
            progress(0.6 + (board_idx + 1) as f64 / num_all as f64 * 0.4);
            board_buckets
        })
        .collect();

    let mut buckets = Vec::with_capacity(num_all * TOTAL_COMBOS as usize);
    for board_buckets in &all_buckets {
        buckets.extend_from_slice(board_buckets);
    }

    let packed_boards: Vec<PackedBoard> = all_canonical
        .iter()
        .map(|wb| canonical_key(&wb.cards))
        .collect();

    #[allow(clippy::cast_possible_truncation)]
    BucketFile {
        header: BucketFileHeader {
            street: Street::Turn,
            bucket_count,
            board_count: num_all as u32,
            combos_per_board: TOTAL_COMBOS,
            version: 2,
        },
        boards: packed_boards,
        buckets,
    }
}
```

**Step 2: Create `cluster_flop_exhaustive` function**

Same pattern as turn but with `enumerate_canonical_flops()` and turn_buckets. Flop is already small (~1,755 boards) so the exhaustive pass is trivial, but the two-phase approach is consistent.

```rust
pub fn cluster_flop_exhaustive(
    turn_buckets: &BucketFile,
    bucket_count: u16,
    kmeans_iterations: u32,
    seed: u64,
    sample_boards: usize,
    progress: impl Fn(f64) + Sync,
) -> BucketFile {
    // Same structure as cluster_turn_exhaustive but:
    // - Uses enumerate_canonical_flops() (1,755 boards)
    // - Builds histograms over turn_buckets
    // - Since flop is small, sample_boards >= 1755 means phase 1 = phase 2
    // ... (same pattern)
}
```

Since flop only has 1,755 canonical boards, `sample_boards >= 1755` means k-means runs on all boards and the exhaustive assignment produces identical results. The function is still useful for consistency.

**Step 3: Add import for `nearest_centroid_u8`**

```rust
use super::clustering::nearest_centroid_u8;
```

**Step 4: Run tests**

```bash
cargo test -p poker-solver-core -- cluster_pipeline
```

**Step 5: Commit**

```bash
git add crates/core/src/blueprint_v2/cluster_pipeline.rs
git commit -m "feat: two-phase turn/flop clustering with exhaustive centroid assignment"
```

---

### Task 4: Wire exhaustive functions into `run_clustering_pipeline`

**Files:**
- Modify: `crates/core/src/blueprint_v2/cluster_pipeline.rs` (lines 1144-1226)

**Step 1: Update `run_clustering_pipeline` to use exhaustive variants**

```rust
pub fn run_clustering_pipeline(
    config: &ClusteringConfig,
    output_dir: &Path,
    progress: impl Fn(&str, f64) + Sync,
) -> Result<(), Box<dyn std::error::Error>> {
    // 1. River
    progress("river", 0.0);
    let river = if let Some(ref cfvnet_dir) = config.cfvnet_river_data {
        cluster_river_from_cfvnet(
            cfvnet_dir,
            config.river.buckets,
            config.kmeans_iterations,
            |p| progress("river", p),
        )?
    } else {
        let sample = config.river.sample_boards.unwrap_or(DEFAULT_NUM_BOARDS);
        cluster_river_exhaustive(
            config.river.buckets,
            config.kmeans_iterations,
            config.seed,
            sample,
            |p| progress("river", p),
        )
    };
    river.save(&output_dir.join("river.buckets"))?;

    // 2. Turn
    progress("turn", 0.0);
    let sample_turn = config.turn.sample_boards.unwrap_or(DEFAULT_TURN_BOARDS);
    let turn = cluster_turn_exhaustive(
        &river,
        config.turn.buckets,
        config.kmeans_iterations,
        config.seed,
        sample_turn,
        |p| progress("turn", p),
    );
    turn.save(&output_dir.join("turn.buckets"))?;

    // 3. Flop
    progress("flop", 0.0);
    let sample_flop = config.flop.sample_boards.unwrap_or(usize::MAX);
    let flop = cluster_flop_exhaustive(
        &turn,
        config.flop.buckets,
        config.kmeans_iterations,
        config.seed,
        sample_flop,
        |p| progress("flop", p),
    );
    flop.save(&output_dir.join("flop.buckets"))?;

    // 4. Preflop (unchanged)
    progress("preflop", 0.0);
    let preflop = cluster_preflop(|p| progress("preflop", p));
    preflop.save(&output_dir.join("preflop.buckets"))?;

    Ok(())
}
```

**Step 2: Mark old clustering functions as `#[allow(dead_code)]`**

Grep for which old functions are still used in tests. Mark unused ones. Don't delete yet — they may be useful for fast iteration during development.

**Step 3: Run tests**

```bash
cargo test -p poker-solver-core -- cluster_pipeline
```

**Step 4: Commit**

```bash
git add crates/core/src/blueprint_v2/cluster_pipeline.rs
git commit -m "feat: wire exhaustive clustering into run_clustering_pipeline"
```

---

### Task 5: Add board index maps to `AllBuckets`

**Files:**
- Modify: `crates/core/src/blueprint_v2/mccfr.rs` (lines 85-129)
- Modify: `crates/core/src/blueprint_v2/bucket_file.rs`

**Step 1: Add `board_index_map_fx()` to `BucketFile`**

In `bucket_file.rs`, add:
```rust
/// Build an FxHashMap from PackedBoard to board index for O(1) lookups.
#[must_use]
pub fn board_index_map_fx(&self) -> rustc_hash::FxHashMap<PackedBoard, u32> {
    self.boards
        .iter()
        .enumerate()
        .map(|(i, &board)| (board, i as u32))
        .collect()
}
```

**Step 2: Add imports to `mccfr.rs`**

```rust
use rustc_hash::FxHashMap;
use super::bucket_file::PackedBoard;
use super::cluster_pipeline::{canonical_key, combo_index};
use crate::abstraction::isomorphism::CanonicalBoard;
```

Note: `canonical_key` is `pub(crate)` — it's in the same crate, so this works.

**Step 3: Rewrite `AllBuckets` struct**

```rust
pub struct AllBuckets {
    pub bucket_counts: [u16; 4],
    pub bucket_files: [Option<BucketFile>; 4],
    /// Board index lookup tables for O(1) bucket file lookups.
    board_maps: [Option<FxHashMap<PackedBoard, u32>>; 4],
}
```

Remove `delta_bins`, `expected_delta`, `equity_cache` fields.

**Step 4: Rewrite `AllBuckets::new()`**

Remove the `street_configs` parameter:

```rust
impl AllBuckets {
    #[must_use]
    pub fn new(
        bucket_counts: [u16; 4],
        bucket_files: [Option<BucketFile>; 4],
    ) -> Self {
        let board_maps = std::array::from_fn(|i| {
            bucket_files[i].as_ref().and_then(|bf| {
                if bf.boards.is_empty() {
                    None
                } else {
                    Some(bf.board_index_map_fx())
                }
            })
        });
        Self { bucket_counts, bucket_files, board_maps }
    }
}
```

**Step 5: Update `equity_only()` constructor**

```rust
pub fn equity_only(bucket_counts: [u16; 4], bucket_files: [Option<BucketFile>; 4]) -> Self {
    Self::new(bucket_counts, bucket_files)
}
```

**Step 6: Remove `set_equity_cache()`**

Delete the method entirely.

**Step 7: Run tests (expect compilation errors — fix in Task 6)**

```bash
cargo test -p poker-solver-core 2>&1 | head -50
```

**Step 8: Commit**

```bash
git add crates/core/src/blueprint_v2/mccfr.rs crates/core/src/blueprint_v2/bucket_file.rs
git commit -m "feat: add board index maps to AllBuckets, remove delta/cache fields"
```

---

### Task 6: Rewrite `precompute_buckets()` and `get_bucket()` to use bucket file lookups

**Files:**
- Modify: `crates/core/src/blueprint_v2/mccfr.rs`

**Step 1: Write the `lookup_bucket` helper**

```rust
/// Look up a bucket for a postflop street via bucket file.
///
/// Canonicalizes the board, looks up the board index in the hash map,
/// applies the same suit permutation to hole cards, and reads the bucket
/// from the flat array. Falls back to equity binning if no bucket file
/// or board not found.
fn lookup_bucket(&self, street_idx: usize, hole: [Card; 2], board: &[Card]) -> u16 {
    if let (Some(bf), Some(board_map)) = (
        &self.bucket_files[street_idx],
        &self.board_maps[street_idx],
    ) {
        if let Ok(canonical) = CanonicalBoard::from_cards(board) {
            let packed = canonical_key(&canonical.cards);
            if let Some(&board_idx) = board_map.get(&packed) {
                let (c0, c1) = canonical.canonicalize_holding(hole[0], hole[1]);
                let ci = combo_index(c0, c1);
                return bf.get_bucket(board_idx, ci);
            }
        }
    }

    // Fallback: equity-based bucketing.
    let equity = crate::showdown_equity::compute_equity(hole, board);
    let k = self.bucket_counts[street_idx];
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let bucket = (equity * f64::from(k)) as u16;
    bucket.min(k - 1)
}
```

**Step 2: Rewrite `precompute_buckets()`**

```rust
#[must_use]
pub fn precompute_buckets(&self, deal: &Deal) -> [[u16; 4]; 2] {
    let mut result = [[0u16; 4]; 2];
    for (player, row) in result.iter_mut().enumerate() {
        let hole = deal.hole_cards[player];

        // Preflop: canonical hand index.
        let hand = crate::hands::CanonicalHand::from_cards(hole[0], hole[1]);
        let idx = hand.index() as u16;
        row[0] = idx.min(self.bucket_counts[0] - 1);

        // Postflop: bucket file lookup with equity fallback.
        row[1] = self.lookup_bucket(1, hole, &deal.board[..3]); // flop
        row[2] = self.lookup_bucket(2, hole, &deal.board[..4]); // turn
        row[3] = self.lookup_bucket(3, hole, &deal.board[..5]); // river
    }
    result
}
```

**Step 3: Rewrite `get_bucket()`**

```rust
#[must_use]
pub fn get_bucket(&self, street: Street, hole_cards: [Card; 2], board: &[Card]) -> u16 {
    if street == Street::Preflop {
        let hand = crate::hands::CanonicalHand::from_cards(hole_cards[0], hole_cards[1]);
        let idx = hand.index() as u16;
        return idx.min(self.bucket_counts[0] - 1);
    }
    self.lookup_bucket(street as usize, hole_cards, board)
}
```

**Step 4: Remove dead code**

Remove:
- `compute_bucket()` method
- `expected_next_equity()` free function
- `delta_bin()` free function (if it exists)
- Unused imports (`EquityDeltaCache`, `Arc`, `StreetClusterConfig` from mccfr.rs)

Keep `board_for_street()` — still used.

**Step 5: Run tests**

```bash
cargo test -p poker-solver-core -- mccfr
```

**Step 6: Commit**

```bash
git add crates/core/src/blueprint_v2/mccfr.rs
git commit -m "feat: precompute_buckets uses bucket file lookups instead of compute_equity"
```

---

### Task 7: Update trainer to match new `AllBuckets` API

**Files:**
- Modify: `crates/core/src/blueprint_v2/trainer.rs`

**Step 1: Update `AllBuckets::new()` call in `BlueprintTrainer::new()`**

```rust
// Remove street_configs construction (lines 228-233)
// Replace line 234:
let buckets = AllBuckets::new(bucket_counts, bucket_files);
```

**Step 2: Remove equity cache loading block in `train()`**

Delete the entire `needs_cache` block (lines 405-448).

**Step 3: Remove unused imports**

Remove `EquityDeltaCache` import and any unused `Arc` imports.

**Step 4: Run tests**

```bash
cargo test -p poker-solver-core -- trainer
cargo test -p poker-solver-trainer
```

**Step 5: Commit**

```bash
git add crates/core/src/blueprint_v2/trainer.rs
git commit -m "feat: simplify trainer init, remove equity cache loading"
```

---

### Task 8: Add tests for bucket file lookup path

**Files:**
- Modify: `crates/core/src/blueprint_v2/mccfr.rs` (tests module)

**Step 1: Test bucket file lookup produces correct bucket**

```rust
#[test]
fn precompute_buckets_uses_bucket_file() {
    use super::bucket_file::{BucketFile, BucketFileHeader, PackedBoard};
    use crate::abstraction::isomorphism::CanonicalBoard;
    use crate::blueprint_v2::cluster_pipeline::{canonical_key, combo_index};

    let board = [
        Card::new(Value::Ace, Suit::Spade),
        Card::new(Value::King, Suit::Spade),
        Card::new(Value::Queen, Suit::Heart),
        Card::new(Value::Jack, Suit::Diamond),
        Card::new(Value::Ten, Suit::Club),
    ];
    let canonical = CanonicalBoard::from_cards(&board).unwrap();
    let packed = canonical_key(&canonical.cards);

    let hole = [
        Card::new(Value::Two, Suit::Spade),
        Card::new(Value::Three, Suit::Heart),
    ];
    let (c0, c1) = canonical.canonicalize_holding(hole[0], hole[1]);
    let ci = combo_index(c0, c1) as usize;

    let mut buckets_data = vec![0_u16; 1326];
    buckets_data[ci] = 42;

    let bf = BucketFile {
        header: BucketFileHeader {
            street: Street::River,
            bucket_count: 500,
            board_count: 1,
            combos_per_board: 1326,
            version: 2,
        },
        boards: vec![packed],
        buckets: buckets_data,
    };

    let all = AllBuckets::new([169, 500, 500, 500], [None, None, None, Some(bf)]);

    let deal = Deal {
        hole_cards: [hole, [Card::new(Value::Four, Suit::Club), Card::new(Value::Five, Suit::Club)]],
        board,
    };
    let result = all.precompute_buckets(&deal);
    assert_eq!(result[0][3], 42, "river bucket should come from bucket file");
}
```

**Step 2: Test equity fallback when no bucket file**

```rust
#[test]
fn precompute_buckets_equity_fallback() {
    let all = AllBuckets::new([169, 10, 10, 10], [None, None, None, None]);
    let deal = Deal {
        hole_cards: [
            [Card::new(Value::Ace, Suit::Spade), Card::new(Value::Ace, Suit::Heart)],
            [Card::new(Value::Two, Suit::Club), Card::new(Value::Three, Suit::Club)],
        ],
        board: [
            Card::new(Value::King, Suit::Diamond),
            Card::new(Value::Seven, Suit::Spade),
            Card::new(Value::Two, Suit::Diamond),
            Card::new(Value::Jack, Suit::Heart),
            Card::new(Value::Nine, Suit::Club),
        ],
    };
    let result = all.precompute_buckets(&deal);
    assert!(result[0][3] >= 7, "AA should get a high river bucket, got {}", result[0][3]);
}
```

**Step 3: Run tests**

```bash
cargo test -p poker-solver-core -- mccfr
```

**Step 4: Commit**

```bash
git add crates/core/src/blueprint_v2/mccfr.rs
git commit -m "test: bucket file lookup and equity fallback in precompute_buckets"
```

---

### Task 9: Full test suite + clippy + cleanup

**Step 1: Run full test suite**

```bash
cargo test
```

**Step 2: Run clippy**

```bash
cargo clippy
```

**Step 3: Verify tests complete in under 1 minute**

```bash
time cargo test 2>&1 | tail -5
```

**Step 4: Remove dead code**

Grep for unused functions: `expected_next_equity`, `delta_bin`, `compute_bucket`, old clustering variants. Remove or mark `#[allow(dead_code)]`.

Keep `EquityDeltaCache` if it's used by CLI commands or other tools. Keep config fields (`delta_bins`, `expected_delta`) for serde backward compat.

**Step 5: Commit**

```bash
git add -u
git commit -m "chore: cleanup dead code from bucket file refactor"
```
