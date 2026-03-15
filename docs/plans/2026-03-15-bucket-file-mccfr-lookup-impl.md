# Bucket File MCCFR Lookup — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use hex:executing-plans to implement this plan task-by-task.

**Goal:** Replace runtime `compute_equity()` calls in MCCFR with O(1) precomputed bucket file lookups (Pluribus-style), and fix the clustering pipeline to produce exhaustive canonical bucket files.

**Architecture:** Two changes: (A) modify `run_clustering_pipeline` to always use exhaustive canonical board enumeration for all streets, and (B) rewrite `AllBuckets::precompute_buckets()` to look up buckets from loaded `BucketFile`s via board canonicalization + combo indexing, eliminating all runtime equity computation.

**Tech Stack:** Rust, `rs_poker::core::Card`, existing `CanonicalBoard` from `abstraction/isomorphism.rs`, existing `BucketFile` v2 format, existing `combo_index()` and `canonical_key()` from `cluster_pipeline.rs`

---

### Task 1: Make `run_clustering_pipeline` use exhaustive canonical enumeration

**Files:**
- Modify: `crates/core/src/blueprint_v2/cluster_pipeline.rs` (lines 1144-1226)

Currently, `run_clustering_pipeline` dispatches to sampled variants when `sample_boards` is set. The canonical variants (`cluster_turn_canonical`, `cluster_flop_canonical`) already exist but aren't wired in. The default `cluster_river` samples 10K boards via `DEFAULT_NUM_BOARDS`.

**Step 1: Change river default to exhaustive**

In `cluster_river()` (line 74), `sample_canonical` is called with `DEFAULT_NUM_BOARDS` (10,000). Change to `usize::MAX` so all canonical rivers are used:

```rust
// line 74: change DEFAULT_NUM_BOARDS to usize::MAX
let (boards, board_weights) = sample_canonical(&all_canonical, usize::MAX, seed);
```

Alternatively, keep `DEFAULT_NUM_BOARDS` for the default function but update `run_clustering_pipeline` to always pass a large number. The simplest fix: change the pipeline dispatch.

**Step 2: Rewrite `run_clustering_pipeline` to always use canonical variants**

Replace the branching logic. The `sample_boards` config field is still respected — if `sample_boards` is set AND less than the total canonical count, it subsamples. Otherwise, exhaustive.

```rust
pub fn run_clustering_pipeline(
    config: &ClusteringConfig,
    output_dir: &Path,
    progress: impl Fn(&str, f64) + Sync,
) -> Result<(), Box<dyn std::error::Error>> {
    // 1. River (1-D equity clustering)
    progress("river", 0.0);
    let river = if let Some(ref cfvnet_dir) = config.cfvnet_river_data {
        cluster_river_from_cfvnet(
            cfvnet_dir,
            config.river.buckets,
            config.kmeans_iterations,
            |p| progress("river", p),
        )?
    } else {
        // Use canonical enumeration; sample_boards caps the count if set.
        let max_boards = config.river.sample_boards.unwrap_or(usize::MAX);
        cluster_river_sampled(
            config.river.buckets,
            config.kmeans_iterations,
            config.seed,
            max_boards,
            |p| progress("river", p),
        )
    };
    river.save(&output_dir.join("river.buckets"))?;

    // 2. Turn (potential-aware EMD, canonical enumeration)
    progress("turn", 0.0);
    let max_turn = config.turn.sample_boards.unwrap_or(usize::MAX);
    let turn = cluster_turn_sampled(
        &river,
        config.turn.buckets,
        config.kmeans_iterations,
        config.seed,
        max_turn,
        |p| progress("turn", p),
    );
    turn.save(&output_dir.join("turn.buckets"))?;

    // 3. Flop (potential-aware EMD, canonical enumeration)
    progress("flop", 0.0);
    let max_flop = config.flop.sample_boards.unwrap_or(usize::MAX);
    let flop = cluster_flop_sampled(
        &turn,
        config.flop.buckets,
        config.kmeans_iterations,
        config.seed,
        max_flop,
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

Here `cluster_river_sampled`, `cluster_turn_sampled`, `cluster_flop_sampled` are unified versions that take a `max_boards` parameter. They use `enumerate_canonical_*()` + `sample_canonical(all, max_boards, seed)`. When `max_boards >= all.len()`, all boards are used (exhaustive). This unifies the existing `cluster_*` and `cluster_*_canonical` variants.

**Step 3: Create unified `cluster_river_sampled` function**

Merge `cluster_river` and `cluster_river_with_boards` into a single function that takes `max_boards`. The existing `cluster_river` body is almost right — just parameterize the sample count:

```rust
pub fn cluster_river_sampled(
    bucket_count: u16,
    kmeans_iterations: u32,
    seed: u64,
    max_boards: usize,
    progress: impl Fn(f64) + Sync,
) -> BucketFile {
    let combos = enumerate_combos(&build_deck());
    let all_canonical = enumerate_canonical_rivers();
    let (boards, board_weights) = sample_canonical(&all_canonical, max_boards, seed);
    // ... rest identical to cluster_river body (lines 76-128)
}
```

**Step 4: Create unified `cluster_turn_sampled` function**

Same pattern — merge `cluster_turn` and `cluster_turn_canonical`. Take `max_boards`, enumerate all canonical turns, then `sample_canonical`:

```rust
pub fn cluster_turn_sampled(
    river_buckets: &BucketFile,
    bucket_count: u16,
    kmeans_iterations: u32,
    seed: u64,
    max_boards: usize,
    progress: impl Fn(f64) + Sync,
) -> BucketFile {
    let deck = build_deck();
    let combos = enumerate_combos(&deck);
    let all_canonical = enumerate_canonical_turns();
    let (boards_raw, board_weights) = sample_canonical(&all_canonical, max_boards, seed);
    let boards: Vec<[Card; 4]> = boards_raw;
    let num_boards = boards.len();
    let board_map = river_buckets.board_index_map();

    // Build histograms (same as cluster_turn body lines 214-267)
    // Use board_weights for k-means weighting
    // ...
}
```

Note: the existing `cluster_turn` (lines 198-293) uses `sample_canonical` with `DEFAULT_TURN_BOARDS`. The existing `cluster_turn_canonical` (lines 672-763) enumerates all and doesn't use `sample_canonical`. The unified version should use `sample_canonical` with configurable `max_boards`, defaulting to `usize::MAX` for exhaustive.

**Step 5: Create unified `cluster_flop_sampled` function**

Same pattern for flop. The existing `cluster_flop` is already exhaustive (uses all 1,755 canonical flops), but wrap it to accept `max_boards` for consistency.

**Step 6: Mark old functions as `#[allow(dead_code)]` or remove them**

The old `cluster_river`, `cluster_turn`, `cluster_river_with_boards`, `cluster_turn_with_boards`, `cluster_flop_with_boards`, `cluster_turn_canonical`, `cluster_flop_canonical` can be removed or marked dead_code. Prefer removal if no tests reference them — grep first.

**Step 7: Run tests**

```bash
cargo test -p poker-solver-core -- cluster_pipeline
```

**Step 8: Commit**

```bash
git add crates/core/src/blueprint_v2/cluster_pipeline.rs
git commit -m "feat: unify clustering pipeline with exhaustive canonical enumeration"
```

---

### Task 2: Add board index maps to `AllBuckets`

**Files:**
- Modify: `crates/core/src/blueprint_v2/mccfr.rs` (lines 85-129)

Add `HashMap<PackedBoard, u32>` lookup maps so `precompute_buckets` can find a board's index in O(1).

**Step 1: Add imports**

At the top of `mccfr.rs`, add:

```rust
use rustc_hash::FxHashMap;
use super::bucket_file::PackedBoard;
use super::cluster_pipeline::{canonical_key, combo_index};
use crate::abstraction::isomorphism::CanonicalBoard;
```

Note: `canonical_key` is currently `pub(crate)` — it needs to be `pub` since it's used from `mccfr.rs` in the same crate. Same for `combo_index` (already `pub`). Check visibility and adjust if needed.

**Step 2: Add `board_maps` field to `AllBuckets`**

```rust
pub struct AllBuckets {
    pub bucket_counts: [u16; 4],
    pub bucket_files: [Option<BucketFile>; 4],
    /// Board index lookup tables for O(1) bucket file lookups.
    /// Built from bucket file board tables at construction time.
    board_maps: [Option<FxHashMap<PackedBoard, u32>>; 4],
}
```

Remove the `delta_bins`, `expected_delta`, and `equity_cache` fields — they're no longer needed.

**Step 3: Rewrite `AllBuckets::new()`**

Remove the `street_configs` parameter. Build board maps from bucket files:

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
        Self {
            bucket_counts,
            bucket_files,
            board_maps,
        }
    }
}
```

This needs `board_index_map_fx()` on `BucketFile` — add it in Task 2b.

**Step 4: Update `equity_only()` constructor**

```rust
pub fn equity_only(bucket_counts: [u16; 4], bucket_files: [Option<BucketFile>; 4]) -> Self {
    Self::new(bucket_counts, bucket_files)
}
```

**Step 5: Remove `set_equity_cache()`, `compute_bucket()`, and `expected_next_equity()` function**

These are no longer needed. Remove them. Also remove the `delta_bin()` helper if it exists and is unused.

Remove the unused imports: `EquityDeltaCache`, `Arc`, `StreetClusterConfig`.

**Step 6: Add `board_index_map_fx()` to `BucketFile`**

In `crates/core/src/blueprint_v2/bucket_file.rs`, add:

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

**Step 7: Run tests (expect some failures from removed API)**

```bash
cargo test -p poker-solver-core 2>&1 | head -50
```

Fix any compilation errors from removed fields/methods. Tests in `mccfr.rs` that use `equity_only()` should still work since it delegates to `new()`.

**Step 8: Commit**

```bash
git add crates/core/src/blueprint_v2/mccfr.rs crates/core/src/blueprint_v2/bucket_file.rs
git commit -m "feat: add board index maps to AllBuckets for O(1) lookups"
```

---

### Task 3: Rewrite `precompute_buckets()` to use bucket file lookups

**Files:**
- Modify: `crates/core/src/blueprint_v2/mccfr.rs` (lines 143-222)

This is the core change. Replace all `compute_equity` calls with bucket file lookups.

**Step 1: Write the new `precompute_buckets` implementation**

```rust
#[must_use]
pub fn precompute_buckets(&self, deal: &Deal) -> [[u16; 4]; 2] {
    let mut result = [[0u16; 4]; 2];
    for (player, row) in result.iter_mut().enumerate() {
        let hole = deal.hole_cards[player];

        // Preflop: canonical hand index (unchanged).
        let hand = crate::hands::CanonicalHand::from_cards(hole[0], hole[1]);
        let idx = hand.index() as u16;
        row[0] = idx.min(self.bucket_counts[0] - 1);

        // Postflop: bucket file lookup with equity fallback.
        let streets: [(usize, &[Card]); 3] = [
            (1, &deal.board[..3]), // flop
            (2, &deal.board[..4]), // turn
            (3, &deal.board[..5]), // river
        ];

        for (street_idx, board) in streets {
            row[street_idx] = self.lookup_bucket(street_idx, hole, board);
        }
    }
    result
}

/// Look up a bucket for a postflop street.
///
/// Tries the bucket file first (O(1) canonicalize + hash lookup + array index).
/// Falls back to equity binning if no bucket file or board not found.
fn lookup_bucket(&self, street_idx: usize, hole: [Card; 2], board: &[Card]) -> u16 {
    if let (Some(bf), Some(board_map)) = (
        &self.bucket_files[street_idx],
        &self.board_maps[street_idx],
    ) {
        // Canonicalize the board (suit isomorphism).
        if let Ok(canonical) = CanonicalBoard::from_cards(board) {
            let packed = canonical_key(&canonical.cards);
            if let Some(&board_idx) = board_map.get(&packed) {
                // Apply the same suit permutation to hole cards.
                let (c0, c1) = canonical.canonicalize_holding(hole[0], hole[1]);
                let ci = combo_index(c0, c1);
                return bf.get_bucket(board_idx, ci);
            }
        }
    }

    // Fallback: equity-based bucketing.
    let equity = crate::showdown_equity::compute_equity(hole, board);
    let k = self.bucket_counts[street_idx];
    let bucket = (equity * f64::from(k)) as u16;
    bucket.min(k - 1)
}
```

**Step 2: Update `get_bucket()` to also use bucket file lookups**

The `get_bucket` method (line 230) is used by the TUI scenario display (line 741). Update it to use the same lookup path:

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

**Step 3: Remove dead code**

Remove:
- `compute_bucket()` method
- `expected_next_equity()` free function
- `delta_bin()` free function (grep for it first)
- Any unused imports (`showdown_equity::compute_equity` may still be needed for fallback — keep it)

**Step 4: Run tests**

```bash
cargo test -p poker-solver-core -- mccfr
```

**Step 5: Commit**

```bash
git add crates/core/src/blueprint_v2/mccfr.rs
git commit -m "feat: precompute_buckets uses bucket file lookups instead of compute_equity"
```

---

### Task 4: Update trainer to match new `AllBuckets` API

**Files:**
- Modify: `crates/core/src/blueprint_v2/trainer.rs`

**Step 1: Update `AllBuckets::new()` call site**

In `BlueprintTrainer::new()` (line 234), remove `street_configs` parameter:

```rust
// Before:
let street_configs = [
    &config.clustering.preflop,
    &config.clustering.flop,
    &config.clustering.turn,
    &config.clustering.river,
];
let buckets = AllBuckets::new(bucket_counts, bucket_files, street_configs);

// After:
let buckets = AllBuckets::new(bucket_counts, bucket_files);
```

**Step 2: Remove equity cache loading in `train()`**

Delete the entire `needs_cache` block in `train()` (lines 405-448):

```rust
// DELETE this entire block:
let needs_cache = (self.config.clustering.flop.expected_delta ...
if needs_cache {
    if let Some(cache_path) = ...
    ...
    self.buckets.set_equity_cache(Arc::new(cache));
}
```

**Step 3: Remove unused imports**

Remove `EquityDeltaCache` import (line 29) and any unused `Arc` imports.

**Step 4: Run tests**

```bash
cargo test -p poker-solver-core -- trainer
cargo test -p poker-solver-trainer
```

**Step 5: Commit**

```bash
git add crates/core/src/blueprint_v2/trainer.rs
git commit -m "feat: remove equity cache, use simplified AllBuckets API"
```

---

### Task 5: Add tests for bucket file lookup path

**Files:**
- Modify: `crates/core/src/blueprint_v2/mccfr.rs` (tests module)

**Step 1: Test that bucket file lookup produces correct bucket**

Build a minimal `BucketFile` with one known board, set a specific combo to a known bucket, and verify `precompute_buckets` returns it:

```rust
#[test]
fn precompute_buckets_uses_bucket_file() {
    use super::bucket_file::{BucketFile, BucketFileHeader, PackedBoard};
    use crate::abstraction::isomorphism::CanonicalBoard;
    use crate::blueprint_v2::cluster_pipeline::{canonical_key, combo_index};

    // Create a known board and canonicalize it.
    let board = [
        Card::new(Value::Ace, Suit::Spade),
        Card::new(Value::King, Suit::Spade),
        Card::new(Value::Queen, Suit::Heart),
        Card::new(Value::Jack, Suit::Diamond),
        Card::new(Value::Ten, Suit::Club),
    ];
    let canonical = CanonicalBoard::from_cards(&board).unwrap();
    let packed = canonical_key(&canonical.cards);

    // Pick a hole-card combo and find its canonical combo index.
    let hole = [
        Card::new(Value::Two, Suit::Spade),
        Card::new(Value::Three, Suit::Heart),
    ];
    let (c0, c1) = canonical.canonicalize_holding(hole[0], hole[1]);
    let ci = combo_index(c0, c1) as usize;

    // Build a bucket file with this one board.
    let mut buckets_data = vec![0_u16; 1326];
    buckets_data[ci] = 42; // known bucket value

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

    let all = AllBuckets::new(
        [169, 500, 500, 500],
        [None, None, None, Some(bf)],
    );

    let deal = Deal {
        hole_cards: [hole, [Card::new(Value::Four, Suit::Club), Card::new(Value::Five, Suit::Club)]],
        board,
    };
    let result = all.precompute_buckets(&deal);
    assert_eq!(result[0][3], 42, "river bucket should come from bucket file");
}
```

**Step 2: Test fallback to equity when no bucket file**

```rust
#[test]
fn precompute_buckets_equity_fallback() {
    let all = AllBuckets::new(
        [169, 10, 10, 10],
        [None, None, None, None],
    );

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
    // AA should have high equity → high bucket
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

### Task 6: Full test suite + clippy

**Step 1: Run full test suite**

```bash
cargo test
```

Fix any compilation errors or test failures.

**Step 2: Run clippy**

```bash
cargo clippy
```

Fix any warnings.

**Step 3: Verify test suite completes in under 1 minute**

```bash
time cargo test 2>&1 | tail -5
```

**Step 4: Commit any fixes**

```bash
git add -u
git commit -m "chore: fix clippy and test issues from bucket file refactor"
```

---

### Task 7: Clean up dead code

**Files:**
- Modify: `crates/core/src/blueprint_v2/mccfr.rs`
- Modify: `crates/core/src/blueprint_v2/equity_cache.rs`
- Modify: `crates/core/src/blueprint_v2/config.rs`

**Step 1: Check if `EquityDeltaCache` is still used anywhere**

```bash
cargo grep EquityDeltaCache
```

If only used in tests or the `generate-equity-cache` CLI command, keep it. If completely unused, consider marking with `#[allow(dead_code)]`. Do NOT delete it — it may be useful for other tools.

**Step 2: Check if `delta_bins` and `expected_delta` config fields are still used**

```bash
cargo grep delta_bins
cargo grep expected_delta
```

Keep config fields for backward compatibility (serde deserialization of existing configs). Just stop using them in `AllBuckets`.

**Step 3: Remove unused functions in `mccfr.rs`**

Grep for `expected_next_equity`, `delta_bin`, `compute_bucket` — if truly unused after the refactor, remove them.

**Step 4: Run tests + clippy**

```bash
cargo test && cargo clippy
```

**Step 5: Commit**

```bash
git add -u
git commit -m "chore: remove dead equity computation code from MCCFR hot path"
```
