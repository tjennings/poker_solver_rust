# CFVnet-Based River Clustering Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use hex:executing-plans to implement this plan task-by-task.

**Goal:** Replace random-hand equity river clustering with CFV-based equity from pre-solved cfvnet data, using streaming histogram k-means for constant memory usage.

**Architecture:** Three tasks in dependency order: (1) fix kmeans_1d_weighted bugs, (2) add cluster_river_from_cfvnet streaming function, (3) wire into pipeline via config. The cfvnet storage crate provides the record reader; cluster_pipeline consumes it. Card encoding translation (cfvnet uses range-solver's `card_id = 4*rank + suit` with Club=0; core uses `value*4 + suit` with Spade=0) must be handled at the boundary.

**Tech Stack:** Rust, rayon, bytemuck, serde (YAML config)

---

### Task 1: Fix kmeans_1d_weighted Empty Cluster and Initialization Bugs

**Files:**
- Modify: `crates/core/src/blueprint_v2/clustering.rs:296-378`
- Test: `crates/core/src/blueprint_v2/clustering.rs` (inline tests)

**Step 1: Write failing test for empty cluster re-seeding**

Add to the `#[cfg(test)]` module in `clustering.rs`:

```rust
#[test]
fn kmeans_1d_weighted_no_empty_clusters() {
    // Data heavily concentrated at 0.0 and 1.0 with a few points at 0.5.
    // With bad initialization, middle centroids starve and become empty.
    let mut data = Vec::new();
    let mut weights = Vec::new();
    for _ in 0..1000 {
        data.push(0.0);
        weights.push(1.0);
    }
    for _ in 0..10 {
        data.push(0.5);
        weights.push(1.0);
    }
    for _ in 0..1000 {
        data.push(1.0);
        weights.push(1.0);
    }
    let labels = kmeans_1d_weighted(&data, &weights, 50, 100);
    let used: std::collections::HashSet<u16> = labels.iter().copied().collect();
    // All 50 clusters should be occupied (no empty buckets).
    assert_eq!(
        used.len(),
        50,
        "expected 50 occupied clusters, got {}",
        used.len()
    );
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test -p poker-solver-core kmeans_1d_weighted_no_empty_clusters`
Expected: FAIL — some clusters will be empty with current implementation.

**Step 3: Implement weighted percentile init and empty cluster re-seeding**

Replace the body of `kmeans_1d_weighted` in `clustering.rs:296-378`:

```rust
pub fn kmeans_1d_weighted(
    data: &[f64],
    weights: &[f64],
    k: usize,
    max_iterations: u32,
) -> Vec<u16> {
    assert_eq!(data.len(), weights.len());
    assert!(!data.is_empty(), "data must not be empty");
    assert!(k > 0, "k must be positive");

    let n = data.len();
    if k >= n {
        return (0..n).map(|i| i as u16).collect();
    }

    // Build (value, weight) pairs sorted by value for weighted percentile init.
    let mut indexed: Vec<(f64, f64)> = data.iter().copied().zip(weights.iter().copied()).collect();
    indexed.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    // Weighted percentile initialisation: place centroids at evenly-spaced
    // quantiles of the weighted CDF.
    let total_weight: f64 = indexed.iter().map(|(_, w)| w).sum();
    let mut centroids: Vec<f64> = Vec::with_capacity(k);
    let mut cumulative = 0.0_f64;
    let mut idx = 0;
    for ci in 0..k {
        #[allow(clippy::cast_precision_loss)]
        let target = (ci as f64 + 0.5) / k as f64 * total_weight;
        while idx < indexed.len() - 1 && cumulative + indexed[idx].1 < target {
            cumulative += indexed[idx].1;
            idx += 1;
        }
        centroids.push(indexed[idx].0);
    }
    // Deduplicate centroids that landed on the same value by nudging.
    centroids.dedup_by(|a, b| (*a - *b).abs() < f64::EPSILON);
    while centroids.len() < k {
        // Fill remaining with spread values from the data range.
        let min = indexed.first().unwrap().0;
        let max = indexed.last().unwrap().0;
        #[allow(clippy::cast_precision_loss)]
        let val = min + (max - min) * (centroids.len() as f64 / k as f64);
        centroids.push(val);
    }

    let mut assignments = vec![0_u16; n];

    for _iter in 0..max_iterations {
        // -- Assignment step (parallel) ------------------------------------
        let new_assignments: Vec<u16> = data
            .par_iter()
            .map(|&val| nearest_centroid_1d(val, &centroids))
            .collect();

        let changed = assignments
            .par_iter()
            .zip(new_assignments.par_iter())
            .any(|(old, new)| old != new);
        assignments = new_assignments;

        if !changed {
            break;
        }

        // -- Weighted update step (parallel fold/reduce) -------------------
        let (sums, weight_sums) = data
            .par_iter()
            .zip(weights.par_iter())
            .zip(assignments.par_iter())
            .fold(
                || (vec![0.0_f64; k], vec![0.0_f64; k]),
                |(mut sums, mut wsums), ((&val, &w), &ci)| {
                    let ci = ci as usize;
                    sums[ci] += val * w;
                    wsums[ci] += w;
                    (sums, wsums)
                },
            )
            .reduce(
                || (vec![0.0_f64; k], vec![0.0_f64; k]),
                |(mut s1, mut w1), (s2, w2)| {
                    for i in 0..k {
                        s1[i] += s2[i];
                        w1[i] += w2[i];
                    }
                    (s1, w1)
                },
            );

        for ci in 0..k {
            if weight_sums[ci] > 0.0 {
                centroids[ci] = sums[ci] / weight_sums[ci];
            } else {
                // Re-seed empty cluster: find the data point farthest from
                // its assigned centroid.
                let farthest = data
                    .iter()
                    .zip(assignments.iter())
                    .enumerate()
                    .map(|(i, (&val, &a))| {
                        let dist = (val - centroids[a as usize]).abs();
                        (i, dist)
                    })
                    .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|(i, _)| i)
                    .unwrap();
                centroids[ci] = data[farthest];
                assignments[farthest] = ci as u16;
            }
        }
    }

    // Final assignment pass (parallel).
    data.par_iter()
        .map(|&val| nearest_centroid_1d(val, &centroids))
        .collect()
}
```

**Step 4: Run test to verify it passes**

Run: `cargo test -p poker-solver-core kmeans_1d_weighted`
Expected: ALL kmeans_1d tests PASS, including the new one.

**Step 5: Run full test suite**

Run: `cargo test`
Expected: All tests pass.

**Step 6: Commit**

```bash
git add crates/core/src/blueprint_v2/clustering.rs
git commit -m "fix(clustering): add empty cluster re-seeding and weighted percentile init to kmeans_1d_weighted"
```

---

### Task 2: Add cluster_river_from_cfvnet Streaming Function

**Files:**
- Modify: `crates/core/src/blueprint_v2/cluster_pipeline.rs` (add function)
- Modify: `crates/core/Cargo.toml` (add cfvnet dependency or keep I/O adapter approach)

**Important design decision:** The cfvnet `TrainingRecord` reader lives in the `cfvnet` crate. To avoid a circular dependency (core cannot depend on cfvnet), we use a lightweight I/O adapter approach: the new function accepts a pre-parsed stream of `(board: [u8; 5], cfvs: [f32; 1326], valid_mask: [u8; 1326])` tuples, or we duplicate the minimal record-reading logic (~40 lines) in core. The recommended approach is to define the streaming function in `cluster_pipeline.rs` as taking a directory path and inlining the binary record reading (it's just sequential reads of fixed-size fields).

**Card encoding bridge:** The cfvnet data uses range-solver card encoding (`card_id = 4 * rank + suit`, Club=0/Diamond=1/Heart=2/Spade=3). The core crate uses `card_to_deck_index` (`value * 4 + suit`, Spade=0/Heart=1/Diamond=2/Club=3). The rank ordering is identical (2=0..A=12), but suit mapping differs:

| Suit    | range-solver | core   |
|---------|-------------|--------|
| Club    | 0           | 3      |
| Diamond | 1           | 2      |
| Heart   | 2           | 1      |
| Spade   | 3           | 0      |

Translation: `core_card_id = rank * 4 + (3 - cfvnet_suit)`. The combo index also uses different formulas, but both produce a triangular enumeration of the same 1326 pairs — they just order them differently. Since cfvs are indexed by cfvnet's combo ordering (0..1326), we need to map cfvnet combo index → core combo index when building the BucketFile.

**Step 1: Write failing test for cfvnet equity conversion**

Add a helper and test:

```rust
/// Convert a pot-relative CFV to an equity value in [0, 1].
///
/// CFVs range from approximately -1.0 (lose the pot) to +1.0 (win the pot).
fn cfv_to_equity(cfv: f32) -> f64 {
    ((cfv as f64 + 1.0) / 2.0).clamp(0.0, 1.0)
}

#[test]
fn test_cfv_to_equity() {
    assert!((cfv_to_equity(-1.0) - 0.0).abs() < 1e-9);
    assert!((cfv_to_equity(0.0) - 0.5).abs() < 1e-9);
    assert!((cfv_to_equity(1.0) - 1.0).abs() < 1e-9);
    assert!((cfv_to_equity(-1.5) - 0.0).abs() < 1e-9); // clamped
    assert!((cfv_to_equity(1.5) - 1.0).abs() < 1e-9);  // clamped
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test -p poker-solver-core test_cfv_to_equity`
Expected: FAIL — function not defined.

**Step 3: Implement cfv_to_equity**

Add the function to `cluster_pipeline.rs`.

**Step 4: Run test to verify it passes**

Run: `cargo test -p poker-solver-core test_cfv_to_equity`
Expected: PASS.

**Step 5: Write failing test for card encoding bridge**

```rust
/// Map a cfvnet card_id (4*rank + suit where C=0,D=1,H=2,S=3)
/// to a core Card.
fn cfvnet_card_to_core(card_id: u8) -> Card {
    let rank = card_id / 4;
    let cfvnet_suit = card_id % 4;
    let value = match rank {
        0 => Value::Two, 1 => Value::Three, 2 => Value::Four,
        3 => Value::Five, 4 => Value::Six, 5 => Value::Seven,
        6 => Value::Eight, 7 => Value::Nine, 8 => Value::Ten,
        9 => Value::Jack, 10 => Value::Queen, 11 => Value::King,
        12 => Value::Ace, _ => panic!("invalid rank {rank}"),
    };
    let suit = match cfvnet_suit {
        0 => Suit::Club, 1 => Suit::Diamond, 2 => Suit::Heart,
        3 => Suit::Spade, _ => unreachable!(),
    };
    Card::new(value, suit)
}

#[test]
fn test_cfvnet_card_to_core() {
    // card_id 0 = 2c
    let c = cfvnet_card_to_core(0);
    assert_eq!(c.value, Value::Two);
    assert_eq!(c.suit, Suit::Club);
    // card_id 51 = As
    let c = cfvnet_card_to_core(51);
    assert_eq!(c.value, Value::Ace);
    assert_eq!(c.suit, Suit::Spade);
    // card_id 48 = Ac
    let c = cfvnet_card_to_core(48);
    assert_eq!(c.value, Value::Ace);
    assert_eq!(c.suit, Suit::Club);
}
```

**Step 6: Write the cfvnet combo-index-to-core-combo-index mapping**

The cfvnet data stores cfvs[0..1326] using range-solver's `card_pair_to_index`. We need a lookup table mapping cfvnet combo index → core combo index.

```rust
/// Build a lookup table mapping cfvnet combo index (range-solver ordering)
/// to core combo index (cluster_pipeline ordering).
///
/// Both are triangular enumerations of C(52,2) = 1326 pairs, but with
/// different card-to-integer mappings.
fn build_cfvnet_to_core_combo_map() -> [u16; 1326] {
    let mut map = [0u16; 1326];
    let mut cfvnet_idx = 0usize;
    for c0 in 0u8..52 {
        for c1 in (c0 + 1)..52 {
            let card0 = cfvnet_card_to_core(c0);
            let card1 = cfvnet_card_to_core(c1);
            map[cfvnet_idx] = combo_index(card0, card1);
            cfvnet_idx += 1;
        }
    }
    map
}

#[test]
fn test_cfvnet_to_core_combo_map_is_permutation() {
    let map = build_cfvnet_to_core_combo_map();
    let mut sorted: Vec<u16> = map.to_vec();
    sorted.sort_unstable();
    let expected: Vec<u16> = (0..1326).collect();
    assert_eq!(sorted, expected, "combo map must be a permutation of 0..1326");
}
```

**Step 7: Implement the streaming function**

```rust
const HISTOGRAM_BINS: usize = 10_000;

/// Cluster river hands from pre-solved cfvnet training data.
///
/// Streams binary record files from `data_dir`, converts pot-relative CFVs
/// to equity values, and clusters using histogram-based 1-D k-means.
///
/// # Errors
/// Returns an error if no valid `.bin` files are found or on I/O failure.
pub fn cluster_river_from_cfvnet(
    data_dir: &Path,
    bucket_count: u16,
    kmeans_iterations: u32,
    progress: impl Fn(f64) + Sync,
) -> Result<BucketFile, Box<dyn std::error::Error>> {
    use std::io::{BufReader, Read};

    let combo_map = build_cfvnet_to_core_combo_map();
    let record_size: usize = 1 + 5 + 4 + 4 + 1 + 4 + 1326 * 4 + 1326 * 4 + 1326 * 4 + 1326;

    // Find all .bin files.
    let mut bin_files: Vec<PathBuf> = std::fs::read_dir(data_dir)?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|e| e == "bin"))
        .filter(|p| std::fs::metadata(p).map_or(false, |m| m.len() > 0))
        .collect();
    bin_files.sort();
    if bin_files.is_empty() {
        return Err("no .bin files found in cfvnet data directory".into());
    }

    // --- Pass 1: build equity histogram + collect canonical boards ----------
    let mut histogram = [0u64; HISTOGRAM_BINS];
    let mut board_set: HashMap<PackedBoard, u32> = HashMap::new();
    let mut total_records = 0u64;

    for (file_idx, path) in bin_files.iter().enumerate() {
        let file = std::fs::File::open(path)?;
        let mut reader = BufReader::with_capacity(1 << 20, file);

        loop {
            // Read one record manually (minimal inlined reader).
            let mut header = [0u8; 1 + 5 + 4 + 4 + 1 + 4]; // board_size + board + pot + stack + player + game_value
            if reader.read_exact(&mut header).is_err() {
                break;
            }
            let board_size = header[0] as usize;
            if board_size != 5 {
                // Skip non-river records.
                let skip = 1326 * 4 * 3 + 1326; // oop_range + ip_range + cfvs + valid_mask
                std::io::copy(&mut reader.by_ref().take(skip as u64), &mut std::io::sink())?;
                continue;
            }

            let board_bytes: [u8; 5] = header[1..6].try_into().unwrap();

            // Convert board to core Cards and get canonical key.
            let board_cards: [Card; 5] = std::array::from_fn(|i| cfvnet_card_to_core(board_bytes[i]));
            let packed = canonical_key(&board_cards);
            let next_idx = board_set.len() as u32;
            board_set.entry(packed).or_insert(next_idx);

            // Skip oop_range and ip_range (1326 * 4 * 2 bytes).
            let mut skip_buf = vec![0u8; 1326 * 4 * 2];
            reader.read_exact(&mut skip_buf)?;

            // Read cfvs (1326 * f32).
            let mut cfv_bytes = [0u8; 1326 * 4];
            reader.read_exact(&mut cfv_bytes)?;

            // Read valid_mask (1326 * u8).
            let mut valid_mask = [0u8; 1326];
            reader.read_exact(&mut valid_mask)?;

            // Accumulate into histogram.
            for cfvnet_ci in 0..1326 {
                if valid_mask[cfvnet_ci] == 0 {
                    continue;
                }
                let cfv_bytes_slice = &cfv_bytes[cfvnet_ci * 4..(cfvnet_ci + 1) * 4];
                let cfv = f32::from_le_bytes(cfv_bytes_slice.try_into().unwrap());
                let equity = cfv_to_equity(cfv);
                let bin = (equity * (HISTOGRAM_BINS - 1) as f64).round() as usize;
                histogram[bin.min(HISTOGRAM_BINS - 1)] += 1;
            }

            total_records += 1;
        }

        #[allow(clippy::cast_precision_loss)]
        progress(0.3 * (file_idx + 1) as f64 / bin_files.len() as f64);
    }

    if total_records == 0 {
        return Err("no valid river records found in cfvnet data".into());
    }

    // --- Pass 2: k-means on histogram bins ----------------------------------
    let bin_midpoints: Vec<f64> = (0..HISTOGRAM_BINS)
        .map(|i| (i as f64 + 0.5) / HISTOGRAM_BINS as f64)
        .collect();
    let bin_weights: Vec<f64> = histogram.iter().map(|&c| c as f64).collect();

    // Filter out empty bins for efficiency.
    let (active_vals, active_weights): (Vec<f64>, Vec<f64>) = bin_midpoints
        .iter()
        .zip(bin_weights.iter())
        .filter(|(_, &w)| w > 0.0)
        .map(|(&v, &w)| (v, w))
        .unzip();

    let centroid_labels =
        kmeans_1d_weighted(&active_vals, &active_weights, bucket_count as usize, kmeans_iterations);

    // Recover centroid positions from the labels.
    let mut centroid_positions = vec![0.0_f64; bucket_count as usize];
    let mut centroid_weights = vec![0.0_f64; bucket_count as usize];
    for (i, (&val, &w)) in active_vals.iter().zip(active_weights.iter()).enumerate() {
        let ci = centroid_labels[i] as usize;
        centroid_positions[ci] += val * w;
        centroid_weights[ci] += w;
    }
    for ci in 0..bucket_count as usize {
        if centroid_weights[ci] > 0.0 {
            centroid_positions[ci] /= centroid_weights[ci];
        }
    }

    progress(0.5);

    // --- Pass 3: assign bucket labels per (board, combo) --------------------
    let num_boards = board_set.len();
    let total_entries = num_boards * TOTAL_COMBOS as usize;
    let mut buckets = vec![0u16; total_entries];

    // Re-build board_set for deterministic ordering (sorted by packed key).
    let mut boards_sorted: Vec<(PackedBoard, [Card; 5])> = Vec::new();
    // We need to re-derive the card arrays. Re-scan is needed.
    let mut board_index_map: HashMap<PackedBoard, u32> = HashMap::new();

    // Second scan to build ordered board list and assign labels.
    for (file_idx, path) in bin_files.iter().enumerate() {
        let file = std::fs::File::open(path)?;
        let mut reader = BufReader::with_capacity(1 << 20, file);

        loop {
            let mut header = [0u8; 1 + 5 + 4 + 4 + 1 + 4];
            if reader.read_exact(&mut header).is_err() {
                break;
            }
            let board_size = header[0] as usize;
            if board_size != 5 {
                let skip = 1326 * 4 * 3 + 1326;
                std::io::copy(&mut reader.by_ref().take(skip as u64), &mut std::io::sink())?;
                continue;
            }

            let board_bytes: [u8; 5] = header[1..6].try_into().unwrap();
            let board_cards: [Card; 5] = std::array::from_fn(|i| cfvnet_card_to_core(board_bytes[i]));
            let packed = canonical_key(&board_cards);

            let next_idx = board_index_map.len() as u32;
            let board_idx = *board_index_map.entry(packed).or_insert(next_idx);

            // Skip oop_range + ip_range.
            let mut skip_buf = vec![0u8; 1326 * 4 * 2];
            reader.read_exact(&mut skip_buf)?;

            // Read cfvs.
            let mut cfv_bytes = [0u8; 1326 * 4];
            reader.read_exact(&mut cfv_bytes)?;

            // Read valid_mask.
            let mut valid_mask = [0u8; 1326];
            reader.read_exact(&mut valid_mask)?;

            // Assign bucket for each valid combo.
            for cfvnet_ci in 0..1326 {
                if valid_mask[cfvnet_ci] == 0 {
                    continue;
                }
                let cfv = f32::from_le_bytes(
                    cfv_bytes[cfvnet_ci * 4..(cfvnet_ci + 1) * 4].try_into().unwrap(),
                );
                let equity = cfv_to_equity(cfv);
                let bucket = nearest_centroid_1d(equity, &centroid_positions);
                let core_ci = combo_map[cfvnet_ci] as usize;
                buckets[board_idx as usize * TOTAL_COMBOS as usize + core_ci] = bucket;
            }
        }

        #[allow(clippy::cast_precision_loss)]
        progress(0.5 + 0.5 * (file_idx + 1) as f64 / bin_files.len() as f64);
    }

    // Build sorted board list for BucketFile.
    let mut board_entries: Vec<(u32, PackedBoard)> = board_index_map
        .iter()
        .map(|(&packed, &idx)| (idx, packed))
        .collect();
    board_entries.sort_by_key(|(idx, _)| *idx);
    let packed_boards: Vec<PackedBoard> = board_entries.iter().map(|(_, p)| *p).collect();

    #[allow(clippy::cast_possible_truncation)]
    Ok(BucketFile {
        header: BucketFileHeader {
            street: Street::River,
            bucket_count,
            board_count: num_boards as u32,
            combos_per_board: TOTAL_COMBOS,
            version: 2,
        },
        boards: packed_boards,
        buckets,
    })
}
```

**Step 8: Write integration test**

```rust
#[test]
fn test_cluster_river_from_cfvnet_with_synthetic_data() {
    use std::io::Write;
    use tempfile::TempDir;

    let dir = TempDir::new().unwrap();

    // Write a minimal synthetic bin file with 2 records (OOP + IP).
    let path = dir.path().join("river_test.bin");
    let mut f = std::fs::File::create(&path).unwrap();

    for player in 0u8..2 {
        // board_size = 5
        f.write_all(&[5u8]).unwrap();
        // board: 2c 3c 4c 5c 6c = [0, 4, 8, 12, 16]
        f.write_all(&[0, 4, 8, 12, 16]).unwrap();
        // pot
        f.write_all(&100.0_f32.to_le_bytes()).unwrap();
        // stack
        f.write_all(&200.0_f32.to_le_bytes()).unwrap();
        // player
        f.write_all(&[player]).unwrap();
        // game_value
        f.write_all(&0.0_f32.to_le_bytes()).unwrap();
        // oop_range (1326 f32)
        f.write_all(&[0u8; 1326 * 4]).unwrap();
        // ip_range (1326 f32)
        f.write_all(&[0u8; 1326 * 4]).unwrap();
        // cfvs: set linearly from -1 to +1
        for i in 0..1326 {
            let cfv = -1.0 + 2.0 * (i as f32 / 1325.0);
            f.write_all(&cfv.to_le_bytes()).unwrap();
        }
        // valid_mask: all valid except board-card combos (simplify: all valid)
        f.write_all(&[1u8; 1326]).unwrap();
    }
    drop(f);

    let result = cluster_river_from_cfvnet(dir.path(), 10, 50, |_| {});
    let bf = result.expect("clustering should succeed");
    assert_eq!(bf.header.street, Street::River);
    assert_eq!(bf.header.bucket_count, 10);
    assert_eq!(bf.header.board_count, 1); // same board, both records
    assert_eq!(bf.buckets.len(), 1326); // 1 board * 1326 combos
    for &b in &bf.buckets {
        assert!(b < 10, "bucket {b} out of range");
    }
    // With linear CFVs, buckets should be roughly ordered.
    // Check that the first combo has a lower bucket than the last.
    // (Not guaranteed due to combo index remapping, but the distribution should span buckets.)
    let used: std::collections::HashSet<u16> = bf.buckets.iter().copied().collect();
    assert!(used.len() > 1, "expected multiple buckets used");
}
```

**Step 9: Run tests**

Run: `cargo test -p poker-solver-core cluster_river_from_cfvnet`
Expected: PASS.

**Step 10: Run full test suite**

Run: `cargo test`
Expected: All tests pass.

**Step 11: Commit**

```bash
git add crates/core/src/blueprint_v2/cluster_pipeline.rs
git commit -m "feat(clustering): add cluster_river_from_cfvnet with streaming histogram k-means"
```

---

### Task 3: Wire into Pipeline via Config

**Files:**
- Modify: `crates/core/src/blueprint_v2/config.rs` (add `cfvnet_data_dir` field)
- Modify: `crates/core/src/blueprint_v2/cluster_pipeline.rs` (update `run_clustering_pipeline`)
- Modify: `sample_configurations/blueprint_v2_500bkt.yaml` (add example field)

**Step 1: Add config field**

In `config.rs`, add to `ClusteringConfig`:

```rust
pub struct ClusteringConfig {
    pub algorithm: ClusteringAlgorithm,
    pub preflop: StreetClusterConfig,
    pub flop: StreetClusterConfig,
    pub turn: StreetClusterConfig,
    pub river: StreetClusterConfig,
    #[serde(default = "default_seed")]
    pub seed: u64,
    #[serde(default = "default_kmeans_iterations")]
    pub kmeans_iterations: u32,
    /// Optional path to cfvnet river training data directory.
    /// When set, river clustering uses pre-solved CFV equity instead of
    /// random-hand showdown equity.
    #[serde(default)]
    pub cfvnet_river_data: Option<std::path::PathBuf>,
}
```

**Step 2: Update run_clustering_pipeline**

Replace the river section:

```rust
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
        cluster_river(
            config.river.buckets,
            config.kmeans_iterations,
            config.seed,
            |p| progress("river", p),
        )
    };
```

**Step 3: Update sample config**

Add to `sample_configurations/blueprint_v2_500bkt.yaml`:

```yaml
clustering:
  # cfvnet_river_data: ./local_data/cfvnet/river/v1  # uncomment to use CFV-based river clustering
```

**Step 4: Run full test suite**

Run: `cargo test`
Expected: All tests pass (existing configs without the field use `default` = None).

**Step 5: Commit**

```bash
git add crates/core/src/blueprint_v2/config.rs crates/core/src/blueprint_v2/cluster_pipeline.rs sample_configurations/blueprint_v2_500bkt.yaml
git commit -m "feat(clustering): wire cfvnet river data into pipeline via config"
```

---

Plan complete and saved to `docs/plans/2026-03-14-cfvnet-river-clustering-impl.md`. Two execution options:

**1. Subagent-Driven (this session)** - Agent teams with parallel implementer-reviewer streams, fast throughput

**2. Parallel Session (separate)** - Open new session with executing-plans, batch execution with checkpoints

Which approach?
