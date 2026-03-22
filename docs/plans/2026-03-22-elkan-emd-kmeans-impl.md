# Elkan EMD K-Means Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use hex:executing-plans to implement this plan task-by-task.

**Goal:** Replace all L2 k-means (`fast_kmeans_histogram`) calls in the clustering pipeline with Elkan-accelerated EMD k-means, fixing the metric mismatch that degrades bucket quality.

**Architecture:** Port robopoker's Elkan algorithm into `clustering.rs` as a runtime-sized struct with EMD distance. Adapt from const-generic `<K, N>` to heap-allocated `Vec`s. Replace 5 call sites in `cluster_pipeline.rs`. TDD with naive-equivalence test as the anchor.

**Tech Stack:** Rust, Rayon (parallelism), existing EMD functions in `clustering.rs`

**Design doc:** `docs/plans/2026-03-22-elkan-emd-kmeans-design.md`

---

### Task 1: Add `ElkanBounds` struct

**Files:**
- Modify: `crates/core/src/blueprint_v2/clustering.rs` (insert after line 116, before the "K-Means with EMD distance" section at line 118)
- Test: same file, `mod tests` at line 1038

**Step 1: Write the failing test**

Add to the `mod tests` block:

```rust
#[test]
fn elkan_bounds_witness_reassigns_when_closer() {
    let mut b = ElkanBounds::new(3);
    // Simulate initial assignment to centroid 0 with distance 1.0
    b.assign(1.0, 0);
    assert_eq!(b.j(), 0);
    assert!((b.upper() - 1.0).abs() < 1e-6);

    // Witness centroid 2 at distance 0.5 — should reassign
    b.witness(0.5, 2);
    assert_eq!(b.j(), 2);
    assert!((b.upper() - 0.5).abs() < 1e-6);

    // Witness centroid 1 at distance 0.8 — should NOT reassign
    b.witness(0.8, 1);
    assert_eq!(b.j(), 2);
    assert!((b.upper() - 0.5).abs() < 1e-6);
}

#[test]
fn elkan_bounds_update_shifts_bounds() {
    let k = 3;
    let mut b = ElkanBounds::new(k);
    b.assign(1.0, 1);
    b.lower[0] = 2.0;
    b.lower[1] = 1.0;
    b.lower[2] = 3.0;

    let drifts = vec![0.1, 0.2, 0.5];
    b.update(&drifts);

    // Lower bounds decrease by drift amounts (clamped to 0)
    assert!((b.lower[0] - 1.9).abs() < 1e-6);
    assert!((b.lower[1] - 0.8).abs() < 1e-6);
    assert!((b.lower[2] - 2.5).abs() < 1e-6);
    // Upper bound increases by assigned centroid's drift
    assert!((b.upper() - 1.2).abs() < 1e-6);
    assert!(b.stale());
}

#[test]
fn elkan_bounds_has_shifted_triangle_inequality() {
    let k = 3;
    let mut b = ElkanBounds::new(k);
    b.assign(1.0, 0);
    b.lower[1] = 0.5;
    b.lower[2] = 1.5;

    // Pairwise distances: centroid 0-1 = 3.0, centroid 0-2 = 0.5
    let pairwise = vec![
        vec![0.0, 3.0, 0.5],
        vec![3.0, 0.0, 2.0],
        vec![0.5, 2.0, 0.0],
    ];

    // Centroid 1: upper(1.0) > lower(0.5) but upper(1.0) <= d(0,1)/2 = 1.5 -> skip
    assert!(!b.has_shifted(&pairwise, 1));
    // Centroid 2: upper(1.0) > lower(1.5)? No -> skip
    assert!(!b.has_shifted(&pairwise, 2));
    // Same centroid -> always skip
    assert!(!b.has_shifted(&pairwise, 0));
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test -p poker-solver-core elkan_bounds -- --nocapture 2>&1 | head -30`
Expected: FAIL — `ElkanBounds` not found

**Step 3: Write minimal implementation**

Insert after line 116 in `clustering.rs` (before the "K-Means with EMD distance" comment):

```rust
// ---------------------------------------------------------------------------
// Elkan bounds for triangle-inequality accelerated k-means
// ---------------------------------------------------------------------------

/// Per-point distance bounds for Elkan's accelerated k-means.
///
/// Maintains upper/lower bounds on point-centroid distances to skip
/// expensive EMD computations via the triangle inequality.
/// Runtime-sized adaptation of robopoker's `Bounds<K>`.
#[derive(Debug, Clone)]
pub(crate) struct ElkanBounds {
    /// Currently assigned centroid index.
    j: usize,
    /// Lower bounds on distance to each centroid.
    lower: Vec<f32>,
    /// Upper bound on distance to assigned centroid.
    error: f32,
    /// Whether the upper bound is potentially stale.
    stale: bool,
}

impl ElkanBounds {
    /// Create bounds for `k` centroids, initially unassigned.
    fn new(k: usize) -> Self {
        Self {
            j: 0,
            lower: vec![0.0; k],
            error: f32::MAX,
            stale: false,
        }
    }

    /// Currently assigned centroid index.
    fn j(&self) -> usize {
        self.j
    }

    /// Upper bound on distance to assigned centroid.
    fn upper(&self) -> f32 {
        self.error
    }

    /// Whether the upper bound may be outdated.
    fn stale(&self) -> bool {
        self.stale
    }

    /// Checks if centroid `j` could be closer than current assignment.
    ///
    /// Returns false (skip) if triangle inequality proves it can't be closer:
    /// 1. j == assigned centroid -> skip
    /// 2. upper <= lower[j] -> skip
    /// 3. upper <= d(assigned, j) / 2 -> skip
    fn has_shifted(&self, pairwise: &[Vec<f32>], j: usize) -> bool {
        self.j != j
            && self.error > self.lower[j]
            && self.error > 0.5 * pairwise[self.j][j]
    }

    /// Direct assignment (for initialization).
    fn assign(&mut self, distance: f32, j: usize) {
        self.j = j;
        self.error = distance;
    }

    /// Refreshes stale upper bound with actual distance.
    fn refresh(&mut self, distance: f32) {
        self.lower[self.j] = distance;
        self.error = distance;
        self.stale = false;
    }

    /// Records distance to centroid `j`, reassigning if closer.
    fn witness(&mut self, distance: f32, j: usize) {
        self.lower[j] = distance;
        if distance < self.error {
            self.j = j;
            self.error = distance;
        }
    }

    /// Updates bounds after centroids move.
    /// Lower bounds decrease by drift; upper bound increases by assigned drift.
    fn update(&mut self, drifts: &[f32]) {
        for (lower, &drift) in self.lower.iter_mut().zip(drifts.iter()) {
            *lower = (*lower - drift).max(0.0);
        }
        self.error += drifts[self.j];
        self.stale = true;
    }
}
```

**Step 4: Run test to verify it passes**

Run: `cargo test -p poker-solver-core elkan_bounds -- --nocapture 2>&1 | head -30`
Expected: 3 tests PASS

**Step 5: Commit**

```bash
git add crates/core/src/blueprint_v2/clustering.rs
git commit -m "feat: add ElkanBounds struct for triangle-inequality accelerated k-means"
```

---

### Task 2: Add `elkan_emd_weighted_u8` function

**Files:**
- Modify: `crates/core/src/blueprint_v2/clustering.rs` (insert after `ElkanBounds` impl, before the existing `kmeans_emd_weighted_u8` at line 571)
- Test: same file, `mod tests`

**Step 1: Write the failing test (naive-equivalence)**

Add to `mod tests`:

```rust
#[test]
fn elkan_emd_naive_equivalence() {
    // Two well-separated clusters of u8 histograms
    let mut data: Vec<Vec<u8>> = Vec::new();
    let mut weights: Vec<f64> = Vec::new();
    for _ in 0..50 {
        data.push(vec![36, 4, 0, 0]);
        weights.push(1.0);
    }
    for _ in 0..50 {
        data.push(vec![0, 0, 4, 36]);
        weights.push(1.0);
    }

    let seed = 42;
    let max_iter = 50;

    let (naive_labels, naive_centroids) =
        kmeans_emd_weighted_u8(&data, &weights, 2, max_iter, seed, |_, _| {});
    let (elkan_labels, elkan_centroids) =
        elkan_emd_weighted_u8(&data, &weights, 2, max_iter, seed, |_, _| {});

    // Same assignments
    assert_eq!(naive_labels, elkan_labels, "labels differ");
    // Same centroids (within float tolerance)
    assert_eq!(naive_centroids.len(), elkan_centroids.len());
    for (nc, ec) in naive_centroids.iter().zip(elkan_centroids.iter()) {
        for (nv, ev) in nc.iter().zip(ec.iter()) {
            assert!(
                (nv - ev).abs() < 1e-10,
                "centroid mismatch: naive={nv}, elkan={ev}"
            );
        }
    }
}

#[test]
fn elkan_emd_three_clusters() {
    let mut data: Vec<Vec<u8>> = Vec::new();
    let mut weights: Vec<f64> = Vec::new();
    for _ in 0..30 {
        data.push(vec![40, 0, 0, 0, 0]);
        weights.push(1.0);
    }
    for _ in 0..30 {
        data.push(vec![0, 0, 40, 0, 0]);
        weights.push(1.0);
    }
    for _ in 0..30 {
        data.push(vec![0, 0, 0, 0, 40]);
        weights.push(1.0);
    }

    let (labels, centroids) =
        elkan_emd_weighted_u8(&data, &weights, 3, 50, 42, |_, _| {});

    assert_eq!(labels.len(), 90);
    assert_eq!(centroids.len(), 3);
    // Each cluster should be internally consistent
    assert!(labels[0..30].iter().all(|&a| a == labels[0]));
    assert!(labels[30..60].iter().all(|&a| a == labels[30]));
    assert!(labels[60..90].iter().all(|&a| a == labels[60]));
    // All three clusters should be different
    let unique: std::collections::HashSet<u16> = labels.iter().copied().collect();
    assert_eq!(unique.len(), 3);
}

#[test]
fn elkan_emd_k_equals_n() {
    let data = vec![vec![10, 0], vec![0, 10], vec![5, 5]];
    let weights = vec![1.0, 1.0, 1.0];
    let (labels, centroids) = elkan_emd_weighted_u8(&data, &weights, 3, 10, 42, |_, _| {});
    assert_eq!(labels.len(), 3);
    assert_eq!(centroids.len(), 3);
    // Each point should be in its own cluster
    let unique: std::collections::HashSet<u16> = labels.iter().copied().collect();
    assert_eq!(unique.len(), 3);
}

#[test]
fn elkan_emd_k_exceeds_n() {
    let data = vec![vec![10, 0], vec![0, 10]];
    let weights = vec![1.0, 1.0];
    let (labels, centroids) = elkan_emd_weighted_u8(&data, &weights, 5, 10, 42, |_, _| {});
    assert_eq!(labels.len(), 2);
    assert_eq!(centroids.len(), 2);
}

#[test]
fn elkan_emd_weighted_nonuniform() {
    // Heavy weight on first cluster should not change assignments
    // for well-separated data, but centroids should reflect weights.
    let mut data: Vec<Vec<u8>> = Vec::new();
    let mut weights: Vec<f64> = Vec::new();
    for _ in 0..50 {
        data.push(vec![36, 4, 0, 0]);
        weights.push(10.0); // heavy weight
    }
    for _ in 0..50 {
        data.push(vec![0, 0, 4, 36]);
        weights.push(1.0);
    }

    let seed = 42;
    let (naive_labels, _) =
        kmeans_emd_weighted_u8(&data, &weights, 2, 50, seed, |_, _| {});
    let (elkan_labels, _) =
        elkan_emd_weighted_u8(&data, &weights, 2, 50, seed, |_, _| {});

    assert_eq!(naive_labels, elkan_labels, "weighted labels differ");
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test -p poker-solver-core elkan_emd -- --nocapture 2>&1 | head -30`
Expected: FAIL — `elkan_emd_weighted_u8` not found

**Step 3: Write the implementation**

Insert after the `ElkanBounds` impl block, before `kmeans_emd_weighted_u8`:

```rust
// ---------------------------------------------------------------------------
// Elkan-accelerated EMD k-means
// ---------------------------------------------------------------------------

/// Triangle-inequality accelerated k-means with EMD distance for u8 histograms.
///
/// Produces identical results to [`kmeans_emd_weighted_u8`] but skips ~80-95%
/// of EMD computations via Elkan (2003) bound maintenance.
///
/// Ported from robopoker's `crates/clustering/src/elkan.rs`, adapted from
/// const-generic `<K, N>` to runtime-sized `Vec`s.
#[allow(clippy::cast_possible_truncation)]
pub fn elkan_emd_weighted_u8(
    data: &[Vec<u8>],
    weights: &[f64],
    k: usize,
    max_iterations: u32,
    seed: u64,
    progress: impl Fn(u32, u32),
) -> (Vec<u16>, Vec<Vec<f64>>) {
    assert_eq!(data.len(), weights.len());
    assert!(!data.is_empty(), "data must not be empty");
    assert!(k > 0, "k must be positive");

    let n = data.len();
    if k >= n {
        let assignments: Vec<u16> = (0..n).map(|i| i as u16).collect();
        let centroids: Vec<Vec<f64>> = data.iter().map(|d| normalize_u8(d)).collect();
        return (assignments, centroids);
    }

    let mut rng = StdRng::seed_from_u64(seed);
    let mut centroids = kmeanspp_init_u8(data, k, &mut rng);

    // Initialize bounds by computing all N x K distances.
    let mut bounds: Vec<ElkanBounds> = data
        .par_iter()
        .map(|point| {
            let mut b = ElkanBounds::new(k);
            let mut best_j = 0;
            let mut best_d = f32::MAX;
            for (j, centroid) in centroids.iter().enumerate() {
                let d = emd_u8_vs_f64(point, centroid) as f32;
                b.lower[j] = d;
                if d < best_d {
                    best_d = d;
                    best_j = j;
                }
            }
            b.assign(best_d, best_j);
            b
        })
        .collect();

    for iter in 0..max_iterations {
        progress(iter, max_iterations);

        // Check convergence: if no assignments changed since last centroid
        // update, we're done. (On iter 0, we always continue.)
        if iter > 0 {
            let any_changed = bounds
                .par_iter()
                .enumerate()
                .any(|(_, b)| b.stale());
            // After first iteration, if no bounds are stale AND no
            // assignments changed, we rely on the drift check below.
        }

        // Compute K x K pairwise centroid distances.
        let pairwise: Vec<Vec<f32>> = (0..k)
            .into_par_iter()
            .map(|i| {
                (0..k)
                    .map(|j| {
                        if i == j {
                            0.0
                        } else {
                            emd(&centroids[i], &centroids[j]) as f32
                        }
                    })
                    .collect()
            })
            .collect();

        // Compute midpoints: s(c) = min_{c' != c} d(c, c') / 2
        let midpoints: Vec<f32> = (0..k)
            .map(|i| {
                pairwise[i]
                    .iter()
                    .enumerate()
                    .filter(|(j, _)| *j != i)
                    .map(|(_, &d)| d * 0.5)
                    .fold(f32::MAX, f32::min)
            })
            .collect();

        // Elkan assignment: only check points where upper > s(assigned).
        bounds
            .par_iter_mut()
            .enumerate()
            .filter(|(_, b)| b.upper() > midpoints[b.j()])
            .for_each(|(i, b)| {
                let point = &data[i];
                // Refresh stale upper bound.
                if b.stale() {
                    let d = emd_u8_vs_f64(point, &centroids[b.j()]) as f32;
                    b.refresh(d);
                }
                // Check all other centroids via triangle inequality.
                for j in 0..k {
                    if b.has_shifted(&pairwise, j) {
                        let d = emd_u8_vs_f64(point, &centroids[j]) as f32;
                        b.witness(d, j);
                    }
                }
            });

        // Weighted centroid update.
        let mut new_centroids = vec![vec![0.0_f64; centroids[0].len()]; k];
        let mut weight_sums = vec![0.0_f64; k];

        for (i, point) in data.iter().enumerate() {
            let ci = bounds[i].j();
            let w = weights[i];
            let total: f64 = point.iter().map(|&c| f64::from(c)).sum();
            let inv = if total > 0.0 { 1.0 / total } else { 0.0 };
            weight_sums[ci] += w;
            for (j, &val) in point.iter().enumerate() {
                new_centroids[ci][j] += f64::from(val) * inv * w;
            }
        }

        // Average and handle empty clusters.
        for ci in 0..k {
            if weight_sums[ci] <= 0.0 {
                // Re-seed: find point farthest from its assigned centroid.
                let farthest = farthest_point_u8(
                    data,
                    &bounds.iter().map(|b| b.j() as u16).collect::<Vec<_>>(),
                    &centroids,
                );
                new_centroids[ci] = normalize_u8(&data[farthest]);
                bounds[farthest].assign(0.0, ci);
            } else {
                let inv = 1.0 / weight_sums[ci];
                for v in &mut new_centroids[ci] {
                    *v *= inv;
                }
            }
        }

        // Compute drift and update bounds.
        let drifts: Vec<f32> = (0..k)
            .map(|i| emd(&centroids[i], &new_centroids[i]) as f32)
            .collect();

        let max_drift: f32 = drifts.iter().copied().fold(0.0, f32::max);
        centroids = new_centroids;

        if max_drift < 1e-8 {
            break; // Converged
        }

        bounds.par_iter_mut().for_each(|b| b.update(&drifts));
    }

    // Final assignment pass to ensure labels match centroids exactly.
    let assignments: Vec<u16> = data
        .par_iter()
        .map(|point| nearest_centroid_u8(point, &centroids))
        .collect();

    (assignments, centroids)
}
```

**Step 4: Run test to verify it passes**

Run: `cargo test -p poker-solver-core elkan_emd -- --nocapture 2>&1 | head -50`
Expected: All 5 new tests PASS

**Step 5: Run full test suite**

Run: `cargo test -p poker-solver-core 2>&1 | tail -5`
Expected: All tests PASS, under 60 seconds

**Step 6: Commit**

```bash
git add crates/core/src/blueprint_v2/clustering.rs
git commit -m "feat: add elkan_emd_weighted_u8 — Elkan-accelerated EMD k-means"
```

---

### Task 3: Replace `fast_kmeans_histogram` in `cluster_histogram_exhaustive`

**Files:**
- Modify: `crates/core/src/blueprint_v2/cluster_pipeline.rs:44` (import)
- Modify: `crates/core/src/blueprint_v2/cluster_pipeline.rs:496-512` (main call site)

**Step 1: Update the import**

At line 44, change the import to include `elkan_emd_weighted_u8` and remove `fast_kmeans_histogram`:

```rust
// Before:
use super::clustering::{
    compute_centroid_ev, compute_centroid_gaps, fast_kmeans_1d, fast_kmeans_histogram,
    ...
};

// After:
use super::clustering::{
    compute_centroid_ev, compute_centroid_gaps, elkan_emd_weighted_u8, fast_kmeans_1d,
    ...
};
```

Note: `fast_kmeans_histogram` is still used at 4 other sites. Remove it from the import only after all sites are converted. For now, keep it in the import.

**Step 2: Replace lines 496-512**

Replace the L2 k-means call and f32→f64 conversion with a single `elkan_emd_weighted_u8` call:

```rust
// Before (lines 496-512):
    let (_labels, centroids) = fast_kmeans_histogram(
        &all_features,
        bucket_count as usize,
        kmeans_iterations,
        seed,
    );
    progress("k-means", 1.0);

    // Convert f32 centroids to f64 normalized probability distributions.
    let centroids_f64: Vec<Vec<f64>> = centroids
        .iter()
        .map(|c| {
            let total: f64 = c.iter().map(|&v| f64::from(v)).sum();
            let inv = if total > 0.0 { 1.0 / total } else { 0.0 };
            c.iter().map(|&v| f64::from(v) * inv).collect()
        })
        .collect();

// After:
    let (_labels, centroids_f64) = elkan_emd_weighted_u8(
        &all_features,
        &all_weights,
        bucket_count as usize,
        kmeans_iterations,
        seed,
        |iter, total| {
            #[allow(clippy::cast_precision_loss)]
            progress("k-means", (iter + 1) as f64 / total as f64);
        },
    );
```

**Step 3: Run tests**

Run: `cargo test -p poker-solver-core 2>&1 | tail -5`
Expected: All tests PASS

**Step 4: Commit**

```bash
git add crates/core/src/blueprint_v2/cluster_pipeline.rs
git commit -m "fix: use Elkan EMD k-means in cluster_histogram_exhaustive"
```

---

### Task 4: Replace `fast_kmeans_histogram` at remaining 4 call sites

**Files:**
- Modify: `crates/core/src/blueprint_v2/cluster_pipeline.rs:172` (global turn-only)
- Modify: `crates/core/src/blueprint_v2/cluster_pipeline.rs:294` (global flop-only)
- Modify: `crates/core/src/blueprint_v2/cluster_pipeline.rs:1136` (per-flop turn)
- Modify: `crates/core/src/blueprint_v2/cluster_pipeline.rs:1299` (per-flop flop)

**Step 1: Replace line 172**

```rust
// Before:
    let (cluster_labels, _centroids) = fast_kmeans_histogram(
        &all_features,
        bucket_count as usize,
        kmeans_iterations,
        seed,
    );
    progress("k-means", 1.0);

// After:
    let (cluster_labels, _centroids) = elkan_emd_weighted_u8(
        &all_features,
        &all_weights,
        bucket_count as usize,
        kmeans_iterations,
        seed,
        |iter, total| {
            #[allow(clippy::cast_precision_loss)]
            progress("k-means", (iter + 1) as f64 / total as f64);
        },
    );
```

**Step 2: Replace line 294**

Same pattern as Step 1. The `all_weights` vec already exists at this call site.

**Step 3: Replace line 1136**

```rust
// Before:
        fast_kmeans_histogram(&all_features, turn_k, kmeans_iterations, seed)

// After:
        elkan_emd_weighted_u8(&all_features, &all_weights, turn_k, kmeans_iterations, seed, |_, _| {})
```

Note: This returns `(Vec<u16>, Vec<Vec<f64>>)` but the call site expects `(Vec<u16>, Vec<Vec<f32>>)` from `fast_kmeans_histogram`. The `_` binding discards centroids, so the type change is fine. If the compiler complains about the label type (it shouldn't — both return `Vec<u16>`), adapt accordingly.

**Step 4: Replace line 1299**

```rust
// Before:
        fast_kmeans_histogram(&all_features, flop_k, config.kmeans_iterations, config.seed)

// After:
        elkan_emd_weighted_u8(&all_features, &all_weights, flop_k, config.kmeans_iterations, config.seed, |_, _| {})
```

**Step 5: Run tests**

Run: `cargo test -p poker-solver-core 2>&1 | tail -5`
Expected: All tests PASS

**Step 6: Commit**

```bash
git add crates/core/src/blueprint_v2/cluster_pipeline.rs
git commit -m "fix: replace all fast_kmeans_histogram calls with Elkan EMD k-means"
```

---

### Task 5: Remove `fast_kmeans_histogram` and clean up imports

**Files:**
- Modify: `crates/core/src/blueprint_v2/clustering.rs:942-959` (delete function)
- Modify: `crates/core/src/blueprint_v2/clustering.rs:961-983` (delete `nearest_centroid_l2` if unused)
- Modify: `crates/core/src/blueprint_v2/cluster_pipeline.rs:44` (remove from import)

**Step 1: Check for remaining callers**

Run: `grep -rn 'fast_kmeans_histogram\|nearest_centroid_l2' crates/core/src/`

Expected: Only the function definition itself, no callers.

**Step 2: Delete `fast_kmeans_histogram` (lines 942-959)**

Remove the entire function.

**Step 3: Delete `nearest_centroid_l2` if no callers remain**

Check: `grep -rn 'nearest_centroid_l2' crates/core/src/`
If only the definition, delete it.

**Step 4: Remove `fast_kmeans_histogram` from import in `cluster_pipeline.rs`**

Update the import line at line 44 to remove `fast_kmeans_histogram`.

**Step 5: Check if `fastkmeans-rs` is still needed**

`fast_kmeans` and `fast_kmeans_1d` still use it for river clustering. Keep the dependency.

**Step 6: Run tests**

Run: `cargo test -p poker-solver-core 2>&1 | tail -5`
Expected: All tests PASS

Run: `cargo clippy -p poker-solver-core 2>&1 | tail -10`
Expected: No new warnings

**Step 7: Commit**

```bash
git add crates/core/src/blueprint_v2/clustering.rs crates/core/src/blueprint_v2/cluster_pipeline.rs
git commit -m "chore: remove unused fast_kmeans_histogram and nearest_centroid_l2"
```

---

### Task 6: Final verification

**Files:** None (verification only)

**Step 1: Run full test suite**

Run: `cargo test 2>&1 | tail -10`
Expected: All tests PASS across all crates, under 60 seconds

**Step 2: Run clippy**

Run: `cargo clippy --all-targets 2>&1 | tail -10`
Expected: No errors

**Step 3: Verify no remaining L2 histogram clustering**

Run: `grep -rn 'fast_kmeans_histogram' crates/`
Expected: No matches

**Step 4: Update bean**

```bash
beans update poker_solver_rust-438r -s completed --body-append "## Summary of Changes

- Added ElkanBounds struct for triangle-inequality accelerated k-means
- Added elkan_emd_weighted_u8 function (Elkan 2003 algorithm with EMD distance)
- Replaced all 5 fast_kmeans_histogram (L2) call sites with Elkan EMD k-means
- Removed fast_kmeans_histogram and nearest_centroid_l2 (dead code)
- Verified naive-equivalence: Elkan produces identical results to brute-force EMD k-means"
```

**Step 5: Commit bean**

```bash
git add .beans/
git commit -m "chore: complete L2/EMD clustering fix bean"
```
