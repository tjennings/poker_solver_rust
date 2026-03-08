# U8 Histogram Quantization Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use hex:executing-plans to implement this plan task-by-task.

**Goal:** Reduce canonical clustering peak memory 8x by storing histograms as `Vec<u8>` raw integer counts instead of `Vec<f64>` normalized probabilities.

**Architecture:** Add dedicated u8 EMD distance and k-means functions in `clustering.rs`, add a u8 histogram builder in `cluster_pipeline.rs`, then rewire the two canonical clustering functions to use them. Existing f64 functions remain untouched (used by dead-code sampling paths).

**Tech Stack:** Rust, rayon (parallel iterators), rand (k-means++ seeding)

---

### Task 1: Add u8 EMD distance functions

**Files:**
- Modify: `crates/core/src/blueprint_v2/clustering.rs`

**Step 1: Write failing tests**

Add these tests inside the existing `mod tests` block at the bottom of `clustering.rs` (before the closing `}`):

```rust
    #[test]
    fn test_emd_u8_identical() {
        let a = vec![10, 10, 10, 10];
        assert!(emd_u8(&a, &a).abs() < 1e-10);
    }

    #[test]
    fn test_emd_u8_opposite_ends() {
        let a = vec![40, 0, 0, 0];
        let b = vec![0, 0, 0, 40];
        assert!((emd_u8(&a, &b) - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_emd_u8_matches_f64() {
        // Same distribution expressed as u8 counts and f64 probabilities
        let counts_a: Vec<u8> = vec![20, 15, 10, 1];
        let counts_b: Vec<u8> = vec![5, 20, 15, 6];
        let total_a: f64 = counts_a.iter().map(|&c| f64::from(c)).sum();
        let total_b: f64 = counts_b.iter().map(|&c| f64::from(c)).sum();
        let prob_a: Vec<f64> = counts_a.iter().map(|&c| f64::from(c) / total_a).collect();
        let prob_b: Vec<f64> = counts_b.iter().map(|&c| f64::from(c) / total_b).collect();
        let d_u8 = emd_u8(&counts_a, &counts_b);
        let d_f64 = emd(&prob_a, &prob_b);
        assert!((d_u8 - d_f64).abs() < 1e-10, "u8={d_u8} f64={d_f64}");
    }

    #[test]
    fn test_emd_u8_vs_f64_matches() {
        let counts: Vec<u8> = vec![20, 15, 10, 1];
        let centroid: Vec<f64> = vec![0.1, 0.4, 0.3, 0.2];
        let total: f64 = counts.iter().map(|&c| f64::from(c)).sum();
        let prob: Vec<f64> = counts.iter().map(|&c| f64::from(c) / total).collect();
        let d_mixed = emd_u8_vs_f64(&counts, &centroid);
        let d_f64 = emd(&prob, &centroid);
        assert!((d_mixed - d_f64).abs() < 1e-10, "mixed={d_mixed} f64={d_f64}");
    }
```

**Step 2: Run tests to verify they fail**

Run: `cargo test -p poker-solver-core emd_u8 -- --nocapture 2>&1 | tail -5`
Expected: compile errors — `emd_u8` and `emd_u8_vs_f64` not found.

**Step 3: Implement the functions**

Add these after the existing `emd` function (after line 33 in `clustering.rs`), before the k-means section:

```rust
/// EMD between two unnormalized u8 count histograms.
///
/// Normalizes each histogram on-the-fly by dividing by its sum. Both slices
/// must have the same length. For zero-sum histograms the result is 0.
///
/// Numerically identical to [`emd`] on the corresponding normalized `f64`
/// distributions, but avoids storing 8-byte floats per bin.
#[must_use]
pub fn emd_u8(a: &[u8], b: &[u8]) -> f64 {
    debug_assert_eq!(a.len(), b.len());
    let total_a: f64 = a.iter().map(|&c| f64::from(c)).sum();
    let total_b: f64 = b.iter().map(|&c| f64::from(c)).sum();
    let inv_a = if total_a > 0.0 { 1.0 / total_a } else { 0.0 };
    let inv_b = if total_b > 0.0 { 1.0 / total_b } else { 0.0 };
    let mut cdf_diff = 0.0_f64;
    let mut distance = 0.0_f64;
    for (&ai, &bi) in a.iter().zip(b.iter()) {
        cdf_diff += f64::from(ai) * inv_a - f64::from(bi) * inv_b;
        distance += cdf_diff.abs();
    }
    distance
}

/// EMD between an unnormalized u8 count histogram and a normalized f64
/// centroid.
///
/// Used in the assignment step of u8 k-means: data points are u8 counts,
/// centroids are f64 probability vectors.
#[must_use]
pub fn emd_u8_vs_f64(counts: &[u8], centroid: &[f64]) -> f64 {
    debug_assert_eq!(counts.len(), centroid.len());
    let total: f64 = counts.iter().map(|&c| f64::from(c)).sum();
    let inv = if total > 0.0 { 1.0 / total } else { 0.0 };
    let mut cdf_diff = 0.0_f64;
    let mut distance = 0.0_f64;
    for (&ci, &qi) in counts.iter().zip(centroid.iter()) {
        cdf_diff += f64::from(ci) * inv - qi;
        distance += cdf_diff.abs();
    }
    distance
}
```

**Step 4: Run tests to verify they pass**

Run: `cargo test -p poker-solver-core emd_u8 -- --nocapture 2>&1 | tail -10`
Expected: all 4 tests PASS.

**Step 5: Commit**

```bash
git add crates/core/src/blueprint_v2/clustering.rs
git commit -m "feat: add u8 EMD distance functions for quantized histograms"
```

---

### Task 2: Add u8 weighted k-means

**Files:**
- Modify: `crates/core/src/blueprint_v2/clustering.rs`

**Step 1: Write failing test**

Add inside `mod tests`:

```rust
    #[test]
    fn weighted_emd_u8_separable() {
        let mut data: Vec<Vec<u8>> = Vec::new();
        let mut weights: Vec<f64> = Vec::new();
        // Cluster A: mass in first two bins
        for _ in 0..50 {
            data.push(vec![36, 4, 0, 0]);
            weights.push(5.0);
        }
        // Cluster B: mass in last two bins
        for _ in 0..50 {
            data.push(vec![0, 0, 4, 36]);
            weights.push(1.0);
        }
        let assignments = kmeans_emd_weighted_u8(&data, &weights, 2, 100, 42, |_, _| {});
        assert!(assignments[0..50].iter().all(|&a| a == assignments[0]));
        assert!(assignments[50..100].iter().all(|&a| a == assignments[50]));
        assert_ne!(assignments[0], assignments[50]);
    }

    #[test]
    fn emd_u8_kmeans_three_clusters() {
        let mut data: Vec<Vec<u8>> = Vec::new();
        let mut weights: Vec<f64> = Vec::new();
        for _ in 0..30 {
            data.push(vec![36, 10, 0, 0, 0]);
            weights.push(1.0);
        }
        for _ in 0..30 {
            data.push(vec![0, 0, 36, 10, 0]);
            weights.push(1.0);
        }
        for _ in 0..30 {
            data.push(vec![0, 0, 0, 10, 36]);
            weights.push(1.0);
        }
        let assignments = kmeans_emd_weighted_u8(&data, &weights, 3, 100, 42, |_, _| {});
        let c0 = assignments[0];
        let c1 = assignments[30];
        let c2 = assignments[60];
        assert_ne!(c0, c1);
        assert_ne!(c1, c2);
        assert_ne!(c0, c2);
        assert!(assignments[0..30].iter().all(|&a| a == c0));
        assert!(assignments[30..60].iter().all(|&a| a == c1));
        assert!(assignments[60..90].iter().all(|&a| a == c2));
    }
```

**Step 2: Run tests to verify they fail**

Run: `cargo test -p poker-solver-core emd_u8_kmeans -- --nocapture 2>&1 | tail -5`
Expected: compile error — `kmeans_emd_weighted_u8` not found.

**Step 3: Implement helpers and main function**

Add these in the "Internal helpers" section of `clustering.rs` (after `farthest_point`, before `#[cfg(test)]`):

```rust
/// Normalize a u8 count histogram to a f64 probability distribution.
fn normalize_u8(counts: &[u8]) -> Vec<f64> {
    let total: f64 = counts.iter().map(|&c| f64::from(c)).sum();
    let inv = if total > 0.0 { 1.0 / total } else { 0.0 };
    counts.iter().map(|&c| f64::from(c) * inv).collect()
}

/// Index of the nearest centroid to a u8 histogram `point` by EMD.
#[allow(clippy::cast_possible_truncation)]
fn nearest_centroid_u8(point: &[u8], centroids: &[Vec<f64>]) -> u16 {
    let mut best_idx = 0_u16;
    let mut best_dist = f64::MAX;
    for (ci, centroid) in centroids.iter().enumerate() {
        let d = emd_u8_vs_f64(point, centroid);
        if d < best_dist {
            best_dist = d;
            best_idx = ci as u16;
        }
    }
    best_idx
}

/// Find the u8 data point with the largest EMD distance to its assigned
/// centroid. Used to re-seed empty clusters.
fn farthest_point_u8(data: &[Vec<u8>], assignments: &[u16], centroids: &[Vec<f64>]) -> usize {
    let mut best_idx = 0;
    let mut best_dist = -1.0_f64;
    for (i, point) in data.iter().enumerate() {
        let ci = assignments[i] as usize;
        let d = emd_u8_vs_f64(point, &centroids[ci]);
        if d > best_dist {
            best_dist = d;
            best_idx = i;
        }
    }
    best_idx
}

/// K-means++ seeding with EMD distance for u8 count histograms.
///
/// Returns f64 centroids (normalized probability distributions).
#[allow(clippy::cast_possible_truncation)]
fn kmeanspp_init_u8(data: &[Vec<u8>], k: usize, rng: &mut StdRng) -> Vec<Vec<f64>> {
    let n = data.len();
    let mut centroids: Vec<Vec<f64>> = Vec::with_capacity(k);

    let first = rng.random_range(0..n);
    centroids.push(normalize_u8(&data[first]));

    let mut dists = vec![f64::MAX; n];

    for _ in 1..k {
        let newest = centroids.last().expect("centroids is non-empty");
        let mut total = 0.0_f64;
        for (i, point) in data.iter().enumerate() {
            let d = emd_u8_vs_f64(point, newest);
            let d2 = d * d;
            if d2 < dists[i] {
                dists[i] = d2;
            }
            total += dists[i];
        }

        if total <= 0.0 {
            let idx = rng.random_range(0..n);
            centroids.push(normalize_u8(&data[idx]));
            continue;
        }

        let threshold = rng.random::<f64>() * total;
        let mut cumulative = 0.0_f64;
        let mut chosen = n - 1;
        for (i, &d2) in dists.iter().enumerate() {
            cumulative += d2;
            if cumulative >= threshold {
                chosen = i;
                break;
            }
        }
        centroids.push(normalize_u8(&data[chosen]));
    }

    centroids
}
```

Now add the main public function. Place it after `kmeans_emd_weighted` (after line 426), before the "Internal helpers" section comment:

```rust
/// Weighted k-means using EMD on u8 count histograms.
///
/// Data points are unnormalized `u8` integer counts (e.g. how many
/// next-street cards landed in each equity bucket). Centroids are `f64`
/// probability vectors. This reduces per-histogram storage from 8 bytes
/// (f64) to 1 byte (u8) per bin — an 8x memory reduction.
///
/// Numerically equivalent to [`kmeans_emd_weighted`] on the corresponding
/// normalized distributions.
///
/// # Panics
/// Panics if `data` is empty, `k` is zero, or `data.len() != weights.len()`.
#[allow(clippy::cast_possible_truncation)]
pub fn kmeans_emd_weighted_u8(
    data: &[Vec<u8>],
    weights: &[f64],
    k: usize,
    max_iterations: u32,
    seed: u64,
    progress: impl Fn(u32, u32),
) -> Vec<u16> {
    assert_eq!(data.len(), weights.len());
    assert!(!data.is_empty(), "data must not be empty");
    assert!(k > 0, "k must be positive");

    let n = data.len();
    if k >= n {
        return (0..n).map(|i| i as u16).collect();
    }

    let mut rng = StdRng::seed_from_u64(seed);

    let mut centroids = kmeanspp_init_u8(data, k, &mut rng);

    let mut assignments = vec![0_u16; n];

    for iter in 0..max_iterations {
        progress(iter, max_iterations);

        // -- Assignment step (parallel) ----------------------------------------
        let new_assignments: Vec<u16> = data
            .par_iter()
            .map(|point| nearest_centroid_u8(point, &centroids))
            .collect();

        let changed = assignments
            .iter()
            .zip(new_assignments.iter())
            .any(|(old, new)| old != new);
        assignments = new_assignments;

        if !changed {
            break;
        }

        // -- Weighted update step ----------------------------------------------
        for c in &mut centroids {
            c.fill(0.0);
        }
        let mut weight_sums = vec![0.0_f64; k];

        for (i, point) in data.iter().enumerate() {
            let ci = assignments[i] as usize;
            let w = weights[i];
            let total: f64 = point.iter().map(|&c| f64::from(c)).sum();
            let inv = if total > 0.0 { 1.0 / total } else { 0.0 };
            weight_sums[ci] += w;
            for (j, &val) in point.iter().enumerate() {
                centroids[ci][j] += f64::from(val) * inv * w;
            }
        }

        // Average and handle empty clusters.
        for ci in 0..k {
            if weight_sums[ci] <= 0.0 {
                let farthest = farthest_point_u8(data, &assignments, &centroids);
                centroids[ci] = normalize_u8(&data[farthest]);
                assignments[farthest] = ci as u16;
            } else {
                let inv = 1.0 / weight_sums[ci];
                for v in &mut centroids[ci] {
                    *v *= inv;
                }
            }
        }
    }

    // Final assignment pass (parallel).
    data.par_iter()
        .zip(assignments.par_iter_mut())
        .for_each(|(point, assign)| {
            *assign = nearest_centroid_u8(point, &centroids);
        });

    assignments
}
```

**Step 4: Run tests to verify they pass**

Run: `cargo test -p poker-solver-core emd_u8 -- --nocapture 2>&1 | tail -15`
Expected: all 6 u8 tests PASS.

**Step 5: Commit**

```bash
git add crates/core/src/blueprint_v2/clustering.rs
git commit -m "feat: add weighted k-means for u8 count histograms"
```

---

### Task 3: Add `build_next_street_histogram_u8` and tests

**Files:**
- Modify: `crates/core/src/blueprint_v2/cluster_pipeline.rs`

**Step 1: Write failing tests**

Add inside the existing `mod tests` block in `cluster_pipeline.rs`:

```rust
    #[test]
    fn test_build_next_street_histogram_u8_river_counts() {
        let deck = build_deck();
        let combo = [deck[0], deck[1]];
        let board = [deck[2], deck[3], deck[4], deck[5]];
        let hist = build_next_street_histogram_u8(combo, &board, &deck, 10);

        assert_eq!(hist.len(), 10);
        let total: u32 = hist.iter().map(|&c| u32::from(c)).sum();
        assert_eq!(total, 46, "4 board + 2 hole = 6 used, 46 river cards");
    }

    #[test]
    fn test_build_next_street_histogram_u8_matches_f64() {
        let deck = build_deck();
        let combo = [deck[0], deck[1]];
        let board = [deck[2], deck[3], deck[4], deck[5]];
        let hist_u8 = build_next_street_histogram_u8(combo, &board, &deck, 10);
        let hist_f64 = build_next_street_histogram(combo, &board, &deck, 10);

        // u8 counts / total should equal f64 values
        let total: f64 = hist_u8.iter().map(|&c| f64::from(c)).sum();
        for (i, (&cu, &cf)) in hist_u8.iter().zip(hist_f64.iter()).enumerate() {
            let normalized = f64::from(cu) / total;
            assert!(
                (normalized - cf).abs() < 1e-10,
                "bin {i}: u8 normalized={normalized}, f64={cf}"
            );
        }
    }

    #[test]
    fn test_build_next_street_histogram_u8_turn_counts() {
        let deck = build_deck();
        let combo = [deck[0], deck[1]];
        let board = [deck[10], deck[20], deck[30]];
        let hist = build_next_street_histogram_u8(combo, &board, &deck, 5);

        assert_eq!(hist.len(), 5);
        let total: u32 = hist.iter().map(|&c| u32::from(c)).sum();
        assert_eq!(total, 47, "3 board + 2 hole = 5 used, 47 turn cards");
    }
```

**Step 2: Run tests to verify they fail**

Run: `cargo test -p poker-solver-core build_next_street_histogram_u8 -- --nocapture 2>&1 | tail -5`
Expected: compile error — `build_next_street_histogram_u8` not found.

**Step 3: Implement the function**

Add directly after the existing `build_next_street_histogram` function (after line 313 in `cluster_pipeline.rs`), before `equity_to_bucket`:

```rust
/// Like [`build_next_street_histogram`] but returns raw u8 counts instead of
/// normalized f64 probabilities.
///
/// Maximum per-bin count is 47 (flop with 3 board + 2 hole = 5 dead cards),
/// which fits comfortably in u8. Normalization happens on-the-fly during
/// EMD distance computation in [`kmeans_emd_weighted_u8`].
fn build_next_street_histogram_u8(
    combo: [Card; 2],
    board: &[Card],
    deck: &[Card],
    num_buckets: u16,
) -> Vec<u8> {
    let mut histogram = vec![0_u8; num_buckets as usize];
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
        let eq = compute_equity(combo, &extended);
        let bucket = equity_to_bucket(eq, num_buckets);
        histogram[bucket as usize] = histogram[bucket as usize].saturating_add(1);
    }

    histogram
}
```

**Step 4: Run tests to verify they pass**

Run: `cargo test -p poker-solver-core build_next_street_histogram_u8 -- --nocapture 2>&1 | tail -10`
Expected: all 3 tests PASS.

**Step 5: Commit**

```bash
git add crates/core/src/blueprint_v2/cluster_pipeline.rs
git commit -m "feat: add u8 histogram builder for quantized clustering"
```

---

### Task 4: Convert `cluster_turn_canonical` and `cluster_flop_canonical` to u8

**Files:**
- Modify: `crates/core/src/blueprint_v2/cluster_pipeline.rs`

**Step 1: Modify `cluster_turn_canonical` (starts at line 533)**

Change the import at top of file — add `kmeans_emd_weighted_u8` to the import from `super::clustering`:

Replace:
```rust
use super::clustering::{kmeans_1d, kmeans_1d_weighted, kmeans_emd_weighted, kmeans_emd_with_progress};
```
With:
```rust
use super::clustering::{kmeans_1d, kmeans_1d_weighted, kmeans_emd_weighted, kmeans_emd_weighted_u8, kmeans_emd_with_progress};
```

In `cluster_turn_canonical`, replace the histogram computation block (lines 547-571):
```rust
    // For each board, compute a histogram feature vector for every combo.
    let board_features: Vec<Vec<Option<Vec<u8>>>> = boards
        .par_iter()
        .enumerate()
        .map(|(board_idx, wb)| {
            let features: Vec<Option<Vec<u8>>> = combos
                .iter()
                .map(|combo| {
                    if cards_overlap(*combo, &wb.cards) {
                        return None;
                    }
                    Some(build_next_street_histogram_u8(
                        *combo,
                        &wb.cards,
                        &deck,
                        num_river_buckets,
                    ))
                })
                .collect();

            #[allow(clippy::cast_precision_loss)]
            progress((board_idx + 1) as f64 / num_boards as f64 * 0.8);

            features
        })
        .collect();
```

Replace the flatten block (lines 573-588):
```rust
    // Collect valid feature vectors with weights and positions.
    let mut all_features: Vec<Vec<u8>> = Vec::new();
    let mut all_weights: Vec<f64> = Vec::new();
    let mut feature_positions: Vec<(usize, usize)> = Vec::new();

    for (board_idx, (board_feats, wb)) in
        board_features.into_iter().zip(boards.iter()).enumerate()
    {
        for (combo_idx, feat) in board_feats.into_iter().enumerate() {
            if let Some(histogram) = feat {
                all_features.push(histogram);
                all_weights.push(f64::from(wb.weight));
                feature_positions.push((board_idx, combo_idx));
            }
        }
    }
```

Replace the k-means call (lines 590-599):
```rust
    let cluster_labels = kmeans_emd_weighted_u8(
        &all_features,
        &all_weights,
        bucket_count as usize,
        kmeans_iterations,
        seed,
        |iter, max_iter| {
            progress(0.8 + 0.2 * f64::from(iter) / f64::from(max_iter));
        },
    );
```

**Step 2: Modify `cluster_flop_canonical` (starts at line 632)**

Apply identical changes: `Vec<f64>` -> `Vec<u8>`, `build_next_street_histogram` -> `build_next_street_histogram_u8`, `kmeans_emd_weighted` -> `kmeans_emd_weighted_u8`.

In `cluster_flop_canonical`, replace the histogram computation block (lines 646-670):
```rust
    // For each board, compute a histogram feature vector for every combo.
    let board_features: Vec<Vec<Option<Vec<u8>>>> = boards
        .par_iter()
        .enumerate()
        .map(|(board_idx, wb)| {
            let features: Vec<Option<Vec<u8>>> = combos
                .iter()
                .map(|combo| {
                    if cards_overlap(*combo, &wb.cards) {
                        return None;
                    }
                    Some(build_next_street_histogram_u8(
                        *combo,
                        &wb.cards,
                        &deck,
                        num_turn_buckets,
                    ))
                })
                .collect();

            #[allow(clippy::cast_precision_loss)]
            progress((board_idx + 1) as f64 / num_boards as f64 * 0.8);

            features
        })
        .collect();
```

Replace the flatten block (lines 672-687):
```rust
    // Collect valid feature vectors with weights and positions.
    let mut all_features: Vec<Vec<u8>> = Vec::new();
    let mut all_weights: Vec<f64> = Vec::new();
    let mut feature_positions: Vec<(usize, usize)> = Vec::new();

    for (board_idx, (board_feats, wb)) in
        board_features.into_iter().zip(boards.iter()).enumerate()
    {
        for (combo_idx, feat) in board_feats.into_iter().enumerate() {
            if let Some(histogram) = feat {
                all_features.push(histogram);
                all_weights.push(f64::from(wb.weight));
                feature_positions.push((board_idx, combo_idx));
            }
        }
    }
```

Replace the k-means call (lines 689-698):
```rust
    let cluster_labels = kmeans_emd_weighted_u8(
        &all_features,
        &all_weights,
        bucket_count as usize,
        kmeans_iterations,
        seed,
        |iter, max_iter| {
            progress(0.8 + 0.2 * f64::from(iter) / f64::from(max_iter));
        },
    );
```

**Step 3: Run full test suite**

Run: `cargo test -p poker-solver-core 2>&1 | tail -5`
Expected: all tests PASS.

Run: `cargo clippy -p poker-solver-core 2>&1 | tail -10`
Expected: no warnings on changed code.

**Step 4: Commit**

```bash
git add crates/core/src/blueprint_v2/cluster_pipeline.rs
git commit -m "perf: use u8 histograms in canonical clustering for 8x memory reduction

Turn and flop canonical clustering now store histograms as Vec<u8> raw
integer counts instead of Vec<f64> normalized probabilities. The max
per-bin count is 47 (fits in u8). Normalization happens on-the-fly
during EMD distance computation.

Peak memory for 600-bucket turn clustering drops from ~90 GB to ~11 GB."
```
