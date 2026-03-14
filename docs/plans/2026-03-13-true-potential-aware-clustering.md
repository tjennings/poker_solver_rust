# True Potential-Aware Clustering Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use hex:executing-plans to implement this plan task-by-task.

**Goal:** Replace uniform equity binning with actual bucket ID lookups in the clustering pipeline so turn/flop/preflop feature vectors are distributions over the previous street's bucket assignments (true Pluribus-style potential-aware abstraction).

**Architecture:** The fix targets `cluster_pipeline.rs`. Replace `build_next_street_histogram` / `build_next_street_histogram_u8` with versions that look up bucket IDs from the previous street's `BucketFile`. Add a `board_index_map()` helper to `BucketFile`. Change preflop from 1-D equity k-means to EMD over flop bucket distributions. Remove dead uniform-binning code. Move file writes to end of pipeline.

**Tech Stack:** Rust, rayon (parallelism), rs_poker (card types)

**Design doc:** `docs/plans/2026-03-13-true-potential-aware-clustering-design.md`

---

### Task 1: Add `board_index_map()` to `BucketFile`

**Files:**
- Modify: `crates/core/src/blueprint_v2/bucket_file.rs:247` (after `get_bucket`)

**Step 1: Write the failing test**

Add to `bucket_file.rs` tests module:

```rust
#[test]
fn board_index_map_round_trip() {
    let boards = vec![
        PackedBoard::from_cards(&[
            Card::new(Value::Ace, Suit::Spade),
            Card::new(Value::King, Suit::Spade),
            Card::new(Value::Queen, Suit::Heart),
        ]),
        PackedBoard::from_cards(&[
            Card::new(Value::Two, Suit::Spade),
            Card::new(Value::Three, Suit::Heart),
            Card::new(Value::Four, Suit::Diamond),
        ]),
    ];
    let bf = BucketFile {
        header: BucketFileHeader {
            street: Street::Flop,
            bucket_count: 10,
            board_count: 2,
            combos_per_board: 1326,
            version: VERSION,
        },
        boards: boards.clone(),
        buckets: vec![0; 2 * 1326],
    };
    let map = bf.board_index_map();
    assert_eq!(map.len(), 2);
    assert_eq!(map[&boards[0]], 0);
    assert_eq!(map[&boards[1]], 1);
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test -p poker-solver-core board_index_map_round_trip`
Expected: FAIL — method `board_index_map` not found

**Step 3: Write implementation**

Add to `impl BucketFile` in `bucket_file.rs` after `get_bucket`:

```rust
/// Build a lookup map from `PackedBoard` to board index.
///
/// Used by the clustering pipeline to find a board's index for bucket
/// lookups when building potential-aware histograms.
#[must_use]
pub fn board_index_map(&self) -> std::collections::HashMap<PackedBoard, u32> {
    self.boards
        .iter()
        .enumerate()
        .map(|(i, &board)| (board, i as u32))
        .collect()
}
```

**Step 4: Run test to verify it passes**

Run: `cargo test -p poker-solver-core board_index_map_round_trip`
Expected: PASS

**Step 5: Commit**

```
git add crates/core/src/blueprint_v2/bucket_file.rs
git commit -m "feat(clustering): add board_index_map() to BucketFile"
```

---

### Task 2: Replace histogram builders with bucket-lookup versions

**Files:**
- Modify: `crates/core/src/blueprint_v2/cluster_pipeline.rs:370-452` (replace `build_next_street_histogram`, `build_next_street_histogram_u8`, delete `equity_to_bucket`)

**Step 1: Write the failing test**

Add to `cluster_pipeline.rs` tests module. This test builds a small river `BucketFile` by hand, then calls the new histogram builder and verifies it produces a histogram over actual bucket IDs:

```rust
#[test]
fn test_build_bucket_histogram_uses_actual_buckets() {
    let deck = build_deck();
    let combo = [deck[0], deck[1]]; // first two cards as hole cards
    // 4-card turn board using cards not in combo
    let board: [Card; 4] = [deck[10], deck[20], deck[30], deck[40]];

    // Build a fake river BucketFile where every (board, combo) maps to
    // bucket 0 or bucket 1 based on combo_index parity.
    let num_river_buckets: u16 = 3;
    let mut river_boards = Vec::new();
    let mut river_buckets = Vec::new();

    // Enumerate all possible river cards for this turn board
    for &river_card in &deck {
        if board.contains(&river_card)
            || river_card == combo[0]
            || river_card == combo[1]
        {
            continue;
        }
        let mut river_board = board.to_vec();
        river_board.push(river_card);
        let packed = PackedBoard::from_cards(&river_board);
        river_boards.push(packed);
        // Assign all combos on this board to bucket (combo_index % 3)
        let mut board_buckets = Vec::with_capacity(1326);
        for ci in 0..1326_u16 {
            board_buckets.push(ci % num_river_buckets);
        }
        river_buckets.extend(board_buckets);
    }

    let river_bf = BucketFile {
        header: BucketFileHeader {
            street: Street::River,
            bucket_count: num_river_buckets,
            board_count: river_boards.len() as u32,
            combos_per_board: 1326,
            version: 2,
        },
        boards: river_boards,
        buckets: river_buckets,
    };
    let board_map = river_bf.board_index_map();

    let histogram = build_bucket_histogram_u8(
        combo,
        &board,
        &deck,
        &river_bf,
        &board_map,
    );

    assert_eq!(histogram.len(), num_river_buckets as usize);
    // combo_index(deck[0], deck[1]) == 0, so 0 % 3 == 0
    // Every river card maps our combo to bucket 0
    let total: u8 = histogram.iter().sum();
    assert_eq!(total, 46); // 52 - 4 board - 2 hole = 46 river cards
    assert_eq!(histogram[0], 46); // all in bucket 0
    assert_eq!(histogram[1], 0);
    assert_eq!(histogram[2], 0);
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test -p poker-solver-core test_build_bucket_histogram_uses_actual_buckets`
Expected: FAIL — `build_bucket_histogram_u8` not found

**Step 3: Write implementation**

Replace the three functions (`build_next_street_histogram`, `build_next_street_histogram_u8`, `equity_to_bucket`) with:

```rust
/// Build a histogram of bucket IDs from the previous street's clustering.
///
/// For each possible next-street card, extends the board, looks up the
/// resulting board in `prev_buckets` via `board_map`, and increments the
/// histogram bin for that combo's bucket ID.
///
/// This is the core of true potential-aware abstraction: feature vectors
/// at street S are distributions over bucket assignments at street S+1.
fn build_bucket_histogram_u8(
    combo: [Card; 2],
    board: &[Card],
    deck: &[Card],
    prev_buckets: &BucketFile,
    board_map: &std::collections::HashMap<PackedBoard, u32>,
) -> Vec<u8> {
    let num_buckets = prev_buckets.header.bucket_count;
    let mut histogram = vec![0_u8; num_buckets as usize];
    let ci = combo_index(combo[0], combo[1]);

    let mut extended = Vec::with_capacity(board.len() + 1);
    extended.extend_from_slice(board);
    extended.push(Card::new(crate::poker::Value::Two, crate::poker::Suit::Club));

    for &next_card in deck {
        if board.contains(&next_card)
            || next_card == combo[0]
            || next_card == combo[1]
        {
            continue;
        }

        *extended.last_mut().expect("non-empty") = next_card;
        let packed = PackedBoard::from_cards(&extended);

        if let Some(&board_idx) = board_map.get(&packed) {
            let bucket = prev_buckets.get_bucket(board_idx, ci);
            debug_assert!((bucket as usize) < histogram.len());
            // Max per-bin count is 47 (flop: 52 - 3 board - 2 hole), fits in u8.
            debug_assert!(histogram[bucket as usize] < 255);
            histogram[bucket as usize] += 1;
        }
        // If board not in map, this next-street card was not in the
        // previous street's enumeration (can happen with sampling).
        // Skip silently — the histogram is over observed transitions only.
    }

    histogram
}
```

Also delete `build_next_street_histogram`, `build_next_street_histogram_u8`, and `equity_to_bucket`.

**Step 4: Run test to verify it passes**

Run: `cargo test -p poker-solver-core test_build_bucket_histogram_uses_actual_buckets`
Expected: PASS

**Step 5: Commit**

```
git add crates/core/src/blueprint_v2/cluster_pipeline.rs
git commit -m "feat(clustering): replace uniform equity binning with actual bucket lookups"
```

---

### Task 3: Update turn clustering functions to use `build_bucket_histogram_u8`

**Files:**
- Modify: `crates/core/src/blueprint_v2/cluster_pipeline.rs`
  - `cluster_turn` (~line 201)
  - `cluster_turn_with_boards` (~line 290)
  - `cluster_turn_canonical` (~line 719)

**Step 1: Update `cluster_turn`**

In `cluster_turn`, after `let num_river_buckets = river_buckets.header.bucket_count;`, add:

```rust
let board_map = river_buckets.board_index_map();
```

Then in the inner closure where each combo's feature is computed, replace:

```rust
Some(build_next_street_histogram(*combo, board, &deck, num_river_buckets))
```

with:

```rust
Some(build_bucket_histogram_u8(*combo, board, &deck, river_buckets, &board_map))
```

Since this changes the feature type from `Vec<f64>` to `Vec<u8>`, the outer type changes from `Vec<Vec<Option<Vec<f64>>>>` to `Vec<Vec<Option<Vec<u8>>>>`, and the downstream k-means call changes from `kmeans_emd_weighted` to `kmeans_emd_weighted_u8`. Follow the pattern already used in `cluster_turn_canonical`.

Remove the `num_river_buckets` local variable (it's accessed via `river_buckets.header.bucket_count` when needed).

**Step 2: Update `cluster_turn_with_boards`**

Same change as `cluster_turn`: add `board_map`, replace histogram builder call, change types from `f64` to `u8`, switch to `kmeans_emd_weighted_u8`.

**Step 3: Update `cluster_turn_canonical`**

Add `board_map`, replace `build_next_street_histogram_u8` call with `build_bucket_histogram_u8`. The types are already `Vec<u8>` here, so only the function call and its arguments change.

**Step 4: Run existing turn tests**

Run: `cargo test -p poker-solver-core cluster_turn`
Expected: PASS (tests use small bucket files that the pipeline builds — the test structure chains river → turn, so the river `BucketFile` produced by `cluster_river_with_boards` will be used for real lookups)

Note: if tests fail because the sampling-based `cluster_river_with_boards` doesn't produce a `BucketFile` with a `boards` vec (v1 format), you'll need to ensure `cluster_river_with_boards` populates `boards`. Check this during implementation. The canonical variants already do.

**Step 5: Commit**

```
git add crates/core/src/blueprint_v2/cluster_pipeline.rs
git commit -m "feat(clustering): wire turn clustering to actual river bucket lookups"
```

---

### Task 4: Update flop clustering functions to use `build_bucket_histogram_u8`

**Files:**
- Modify: `crates/core/src/blueprint_v2/cluster_pipeline.rs`
  - `cluster_flop` (~line 466)
  - `cluster_flop_with_boards` (~line 555)
  - `cluster_flop_canonical` (~line 818)

**Step 1: Apply same pattern as Task 3**

In each flop function:
- Add `let board_map = turn_buckets.board_index_map();`
- Replace `build_next_street_histogram` / `build_next_street_histogram_u8` calls with `build_bucket_histogram_u8(*combo, board, &deck, turn_buckets, &board_map)`
- Update types from `f64` to `u8` where needed (sampling variants)
- Switch sampling variants from `kmeans_emd_weighted` to `kmeans_emd_weighted_u8`

**Step 2: Run existing flop tests**

Run: `cargo test -p poker-solver-core cluster_flop`
Expected: PASS

**Step 3: Commit**

```
git add crates/core/src/blueprint_v2/cluster_pipeline.rs
git commit -m "feat(clustering): wire flop clustering to actual turn bucket lookups"
```

---

### Task 5: Restructure preflop clustering to use EMD over flop buckets

**Files:**
- Modify: `crates/core/src/blueprint_v2/cluster_pipeline.rs`
  - `cluster_preflop` (~line 927)
  - `cluster_preflop_with_boards` (~line 943)

**Step 1: Write the failing test**

```rust
#[test]
#[ignore] // slow: equity enumeration in debug mode
fn test_cluster_preflop_potential_aware() {
    let river = cluster_river_with_boards(5, 30, 42, 10, |_| {});
    let turn = cluster_turn_with_boards(&river, 5, 20, 42, 10, |_| {});
    let flop = cluster_flop_with_boards(&turn, 5, 20, 42, 5, |_| {});
    let preflop = cluster_preflop_with_boards(&flop, 4, 20, 42, 5, |_| {});

    assert_eq!(preflop.header.street, Street::Preflop);
    assert_eq!(preflop.header.bucket_count, 4);
    assert_eq!(preflop.header.board_count, 1);
    assert_eq!(preflop.header.combos_per_board, 1326);
    assert_eq!(preflop.buckets.len(), 1326);

    for &b in &preflop.buckets {
        assert!(b < 4, "bucket {b} out of range");
    }

    // Should see at least 2 distinct buckets.
    let mut seen = std::collections::HashSet::new();
    for &b in &preflop.buckets {
        seen.insert(b);
    }
    assert!(seen.len() >= 2, "expected at least 2 distinct preflop buckets");
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test -p poker-solver-core test_cluster_preflop_potential_aware -- --ignored`
Expected: FAIL — signature mismatch (`cluster_preflop_with_boards` doesn't take a `BucketFile`)

**Step 3: Rewrite `cluster_preflop` and `cluster_preflop_with_boards`**

Change signatures to accept `flop_buckets: &BucketFile`:

```rust
pub fn cluster_preflop(
    flop_buckets: &BucketFile,
    bucket_count: u16,
    kmeans_iterations: u32,
    seed: u64,
    progress: impl Fn(f64) + Sync,
) -> BucketFile {
    cluster_preflop_with_boards(
        flop_buckets,
        bucket_count,
        kmeans_iterations,
        seed,
        DEFAULT_PREFLOP_BOARDS,
        progress,
    )
}

pub fn cluster_preflop_with_boards(
    flop_buckets: &BucketFile,
    bucket_count: u16,
    kmeans_iterations: u32,
    seed: u64,
    num_boards: usize,
    progress: impl Fn(f64) + Sync,
) -> BucketFile {
    let deck = build_deck();
    let combos = enumerate_combos(&deck);
    let num_flop_buckets = flop_buckets.header.bucket_count;
    let board_map = flop_buckets.board_index_map();

    // Sample 3-card flop boards.
    let boards = sample_flop_boards(&deck, num_boards, seed);

    // For each combo, build a histogram over flop bucket IDs across sampled boards.
    let features: Vec<Vec<u8>> = combos
        .par_iter()
        .enumerate()
        .map(|(combo_idx, &combo)| {
            let mut histogram = vec![0_u16; num_flop_buckets as usize];
            let ci = combo_index(combo[0], combo[1]);

            for board in &boards {
                if cards_overlap(combo, board) {
                    continue;
                }
                let packed = PackedBoard::from_cards(board);
                if let Some(&board_idx) = board_map.get(&packed) {
                    let bucket = flop_buckets.get_bucket(board_idx, ci);
                    histogram[bucket as usize] += 1;
                }
            }

            #[allow(clippy::cast_precision_loss)]
            progress((combo_idx + 1) as f64 / combos.len() as f64 * 0.8);

            // Truncate to u8 (max count per bin bounded by num_boards which is small)
            histogram.iter().map(|&c| c.min(255) as u8).collect()
        })
        .collect();

    // All combos get equal weight for preflop.
    let weights: Vec<f64> = vec![1.0; features.len()];
    let cluster_labels = kmeans_emd_weighted_u8(
        &features,
        &weights,
        bucket_count as usize,
        kmeans_iterations,
        seed,
        |iter, max| {
            #[allow(clippy::cast_precision_loss)]
            progress(0.8 + 0.2 * iter as f64 / max as f64);
        },
    );

    let header = BucketFileHeader {
        street: Street::Preflop,
        bucket_count,
        board_count: 1,
        combos_per_board: TOTAL_COMBOS,
        version: 1,
    };

    BucketFile {
        header,
        boards: Vec::new(),
        buckets: cluster_labels,
    }
}
```

Delete `compute_combo_avg_equities` (now dead code).

**Step 4: Run test**

Run: `cargo test -p poker-solver-core test_cluster_preflop_potential_aware -- --ignored`
Expected: PASS

**Step 5: Commit**

```
git add crates/core/src/blueprint_v2/cluster_pipeline.rs
git commit -m "feat(clustering): change preflop to EMD over flop bucket distributions"
```

---

### Task 6: Update `run_clustering_pipeline` — in-memory chaining, write at end

**Files:**
- Modify: `crates/core/src/blueprint_v2/cluster_pipeline.rs:1024-1072`

**Step 1: Rewrite `run_clustering_pipeline`**

```rust
pub fn run_clustering_pipeline(
    config: &ClusteringConfig,
    output_dir: &Path,
    progress: impl Fn(&str, f64) + Sync,
) -> Result<(), Box<dyn std::error::Error>> {
    // 1. River (equity-based, no previous street)
    progress("river", 0.0);
    let river = cluster_river(
        config.river.buckets,
        config.kmeans_iterations,
        config.seed,
        |p| progress("river", p),
    );

    // 2. Turn (potential-aware: histograms over river bucket IDs)
    progress("turn", 0.0);
    let turn = cluster_turn(
        &river,
        config.turn.buckets,
        config.kmeans_iterations,
        config.seed,
        |p| progress("turn", p),
    );

    // 3. Flop (potential-aware: histograms over turn bucket IDs)
    progress("flop", 0.0);
    let flop = cluster_flop(
        &turn,
        config.flop.buckets,
        config.kmeans_iterations,
        config.seed,
        |p| progress("flop", p),
    );

    // 4. Preflop (potential-aware: histograms over flop bucket IDs)
    progress("preflop", 0.0);
    let preflop = cluster_preflop(
        &flop,
        config.preflop.buckets,
        config.kmeans_iterations,
        config.seed,
        |p| progress("preflop", p),
    );

    // 5. Write all bucket files at the end.
    river.save(&output_dir.join("river.buckets"))?;
    turn.save(&output_dir.join("turn.buckets"))?;
    flop.save(&output_dir.join("flop.buckets"))?;
    preflop.save(&output_dir.join("preflop.buckets"))?;

    Ok(())
}
```

**Step 2: Ensure river `BucketFile` has `boards` populated**

Check that `cluster_river` (and `cluster_river_with_boards`) populates the `boards: Vec<PackedBoard>` field with actual `PackedBoard` entries — not an empty vec. The sampling-based river functions may currently produce v1 files without boards. If so, fix them to pack boards. Without this, `board_index_map()` returns an empty map and turn histograms will be all zeros.

Look at `cluster_river_canonical` (line 643) for the pattern — it builds `WeightedBoard` entries with packed boards. The sampling variants need the same treatment.

**Step 3: Fix compilation — update all callers of `cluster_preflop`**

The signature change (new `flop_buckets` parameter) may break:
- `run_clustering_pipeline` (updated in this task)
- Any test that calls `cluster_preflop` or `cluster_preflop_with_boards` directly

Update tests to chain river → turn → flop → preflop.

**Step 4: Run the full test suite**

Run: `cargo test -p poker-solver-core`
Expected: PASS (all non-`#[ignore]` tests)

Run: `cargo clippy -p poker-solver-core`
Expected: No warnings

**Step 5: Commit**

```
git add crates/core/src/blueprint_v2/cluster_pipeline.rs
git commit -m "feat(clustering): pipeline runs in-memory, writes all files at end"
```

---

### Task 7: Ensure sampling-based river/turn/flop produce `BucketFile` with `boards` field

**Files:**
- Modify: `crates/core/src/blueprint_v2/cluster_pipeline.rs`
  - `cluster_river` / `cluster_river_with_boards` (~line 68, 137)
  - `cluster_turn` / `cluster_turn_with_boards` (~line 201, 290)
  - `cluster_flop` / `cluster_flop_with_boards` (~line 466, 555)

**Context:** The sampling-based variants currently use raw `[Card; N]` boards, not canonical `WeightedBoard` entries. They may produce `BucketFile` with `boards: Vec::new()` (v1 format). For `board_index_map()` to work in downstream stages, the `boards` field must be populated with `PackedBoard` entries matching the boards used during clustering.

**Step 1: Check each sampling variant**

For each function, find where the `BucketFile` is constructed and verify the `boards` field. If it's empty, populate it:

```rust
boards: boards.iter().map(|b| PackedBoard::from_cards(b)).collect(),
```

And set `version: 2` in the header.

**Step 2: Run tests**

Run: `cargo test -p poker-solver-core`
Expected: PASS

**Step 3: Commit**

```
git add crates/core/src/blueprint_v2/cluster_pipeline.rs
git commit -m "fix(clustering): populate boards field in sampling-based BucketFiles"
```

---

### Task 8: Clean up dead code and update tests

**Files:**
- Modify: `crates/core/src/blueprint_v2/cluster_pipeline.rs`

**Step 1: Remove dead code**

- Delete `equity_to_bucket` (if not already deleted in Task 2)
- Delete `compute_combo_avg_equities` (if not already deleted in Task 5)
- Delete any remaining `build_next_street_histogram` / `build_next_street_histogram_u8`
- Remove unused imports (`kmeans_1d` if no longer used by preflop)

**Step 2: Update tests that reference deleted functions**

Tests like `test_build_next_street_histogram_*` and `test_equity_to_bucket` reference deleted functions. Either:
- Delete them (they tested the old uniform-binning behavior)
- Replace with equivalent tests for `build_bucket_histogram_u8`

The test from Task 2 (`test_build_bucket_histogram_uses_actual_buckets`) already covers the new function.

**Step 3: Run full test suite and clippy**

Run: `cargo test -p poker-solver-core && cargo clippy -p poker-solver-core`
Expected: PASS, no warnings

**Step 4: Commit**

```
git add crates/core/src/blueprint_v2/cluster_pipeline.rs
git commit -m "refactor(clustering): remove dead uniform-binning code and update tests"
```

---

### Task 9: Full integration test — run pipeline end to end

**Step 1: Run the full test suite including slow tests**

Run: `cargo test -p poker-solver-core -- --ignored 2>&1 | head -100`

This runs the `#[ignore]` tests that chain river → turn → flop → preflop. They should all pass with the new potential-aware pipeline.

**Step 2: Run clippy on the whole workspace**

Run: `cargo clippy`
Expected: No new warnings

**Step 3: Commit any remaining fixes**

```
git commit -m "test(clustering): verify full potential-aware pipeline"
```
