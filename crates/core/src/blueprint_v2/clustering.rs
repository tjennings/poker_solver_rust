//! Earth Mover's Distance and k-means clustering for the potential-aware
//! abstraction pipeline.
//!
//! The EMD implementation exploits the fact that our distributions live on
//! ordered 1-D buckets, so EMD reduces to the L1 norm of the CDF difference
//! (linear time, no LP needed).

use rand::prelude::*;
use rand::rngs::StdRng;
use rayon::prelude::*;

use fastkmeans_rs::FastKMeans;
use ndarray::Array2;

// ---------------------------------------------------------------------------
// Earth Mover's Distance
// ---------------------------------------------------------------------------

/// Earth Mover's Distance between two probability distributions over ordered
/// buckets.
///
/// For ordered 1-D histograms this equals the L1 distance of the CDFs.
/// Both `p` and `q` must have the same length and each should sum to ~1.0.
///
/// Runs in O(K) where K = `p.len()`.
#[must_use]
pub fn emd(p: &[f64], q: &[f64]) -> f64 {
    debug_assert_eq!(p.len(), q.len());
    let mut cdf_diff = 0.0_f64;
    let mut distance = 0.0_f64;
    for (&pi, &qi) in p.iter().zip(q) {
        cdf_diff += pi - qi;
        distance += cdf_diff.abs();
    }
    distance
}

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
/// centroids are f64 probability vectors. If all counts are zero the
/// normalized point is a zero vector and the result equals the centroid's
/// self-EMD (not meaningful — callers should ensure non-zero totals).
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

/// Weighted EMD between a u8 count histogram and an f64 centroid.
///
/// Like [`emd_u8_vs_f64`] but each CDF step is weighted by `gaps[i]`.
/// `gaps` has length K-1 where K = `counts.len()`.
/// With uniform gaps of 1.0 this equals the unweighted variant.
#[must_use]
pub fn emd_u8_vs_f64_weighted(counts: &[u8], centroid: &[f64], gaps: &[f64]) -> f64 {
    debug_assert_eq!(counts.len(), centroid.len());
    debug_assert_eq!(gaps.len(), counts.len() - 1);
    let total: f64 = counts.iter().map(|&c| f64::from(c)).sum();
    let inv = if total > 0.0 { 1.0 / total } else { 0.0 };
    let mut cdf_diff = 0.0_f64;
    let mut distance = 0.0_f64;
    for i in 0..counts.len() - 1 {
        cdf_diff += f64::from(counts[i]) * inv - centroid[i];
        distance += cdf_diff.abs() * gaps[i];
    }
    distance
}

/// Assign a u8 histogram to the nearest centroid using weighted EMD.
#[allow(clippy::cast_possible_truncation)]
#[must_use]
pub fn nearest_centroid_u8_weighted(point: &[u8], centroids: &[Vec<f64>], gaps: &[f64]) -> u16 {
    let mut best_idx = 0_u16;
    let mut best_dist = f64::MAX;
    for (ci, centroid) in centroids.iter().enumerate() {
        let d = emd_u8_vs_f64_weighted(point, centroid, gaps);
        if d < best_dist {
            best_dist = d;
            best_idx = ci as u16;
        }
    }
    best_idx
}

// ---------------------------------------------------------------------------
// K-Means with EMD distance
// ---------------------------------------------------------------------------

/// Run k-means clustering using EMD as the distance metric.
///
/// # Arguments
/// * `data`  -- feature vectors (probability distributions, all the same length).
/// * `k`     -- number of clusters.
/// * `max_iterations` -- hard cap on refinement rounds.
/// * `seed`  -- RNG seed for reproducibility.
///
/// # Returns
/// Cluster assignment for every data point (one `u16` per row).
///
/// # Panics
/// Panics if `data` is empty or `k` is zero.
#[allow(clippy::cast_possible_truncation)] // k and n fit in u16 for any realistic clustering
#[must_use]
pub fn kmeans_emd(data: &[Vec<f64>], k: usize, max_iterations: u32, seed: u64) -> Vec<u16> {
    kmeans_emd_with_progress(data, k, max_iterations, seed, |_, _| {})
}

/// Like [`kmeans_emd`] but with a progress callback `(iteration, max_iterations)`.
///
/// # Panics
/// Panics if `data` is empty or `k` is zero.
#[allow(clippy::cast_possible_truncation)]
pub fn kmeans_emd_with_progress(
    data: &[Vec<f64>],
    k: usize,
    max_iterations: u32,
    seed: u64,
    progress: impl Fn(u32, u32),
) -> Vec<u16> {
    assert!(!data.is_empty(), "data must not be empty");
    assert!(k > 0, "k must be positive");

    let n = data.len();

    // Trivial case: every point gets its own cluster.
    if k >= n {
        return (0..n).map(|i| i as u16).collect();
    }

    let mut rng = StdRng::seed_from_u64(seed);

    // -- k-means++ initialisation ------------------------------------------
    let mut centroids = kmeanspp_init(data, k, &mut rng);

    let mut assignments = vec![0_u16; n];
    let mut counts = vec![0_usize; k];

    for iter in 0..max_iterations {
        progress(iter, max_iterations);

        // -- Assignment step (parallel) ------------------------------------
        let new_assignments: Vec<u16> = data
            .par_iter()
            .map(|point| nearest_centroid(point, &centroids))
            .collect();

        let changed = assignments
            .iter()
            .zip(new_assignments.iter())
            .any(|(old, new)| old != new);
        assignments = new_assignments;

        if !changed {
            break;
        }

        // -- Update step (sequential — accumulates into centroids) ---------
        for c in &mut centroids {
            c.fill(0.0);
        }
        counts.fill(0);

        for (i, point) in data.iter().enumerate() {
            let ci = assignments[i] as usize;
            counts[ci] += 1;
            for (j, &val) in point.iter().enumerate() {
                centroids[ci][j] += val;
            }
        }

        // Average and handle empty clusters.
        for ci in 0..k {
            if counts[ci] == 0 {
                let farthest = farthest_point(data, &assignments, &centroids);
                centroids[ci].clone_from(&data[farthest]);
                assignments[farthest] = ci as u16;
            } else {
                #[allow(clippy::cast_precision_loss)]
                let inv = 1.0 / counts[ci] as f64;
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
            *assign = nearest_centroid(point, &centroids);
        });

    assignments
}

// ---------------------------------------------------------------------------
// 1-D k-means (for river EHS clustering)
// ---------------------------------------------------------------------------

/// Optimised k-means for 1-D scalar values.
///
/// Initialisation uses evenly-spaced percentile boundaries from the sorted
/// values.
///
/// # Panics
/// Panics if `data` is empty or `k` is zero.
#[allow(clippy::cast_possible_truncation)] // k fits in u16
#[must_use]
pub fn kmeans_1d(data: &[f64], k: usize, max_iterations: u32) -> Vec<u16> {
    assert!(!data.is_empty(), "data must not be empty");
    assert!(k > 0, "k must be positive");

    let n = data.len();
    if k >= n {
        return (0..n).map(|i| i as u16).collect();
    }

    // Sorted copy for percentile initialisation.
    let mut sorted = data.to_vec();
    sorted.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    // Initialise centroids at evenly-spaced percentiles.
    let mut centroids: Vec<f64> = (0..k)
        .map(|i| {
            let idx = (i * (n - 1)) / (k.max(2) - 1).max(1);
            sorted[idx.min(n - 1)]
        })
        .collect();

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

        // -- Update step (parallel fold/reduce) ----------------------------
        let (sums, counts) = data
            .par_iter()
            .zip(assignments.par_iter())
            .fold(
                || (vec![0.0_f64; k], vec![0_usize; k]),
                |(mut sums, mut counts), (&val, &ci)| {
                    let ci = ci as usize;
                    sums[ci] += val;
                    counts[ci] += 1;
                    (sums, counts)
                },
            )
            .reduce(
                || (vec![0.0_f64; k], vec![0_usize; k]),
                |(mut s1, mut c1), (s2, c2)| {
                    for i in 0..k {
                        s1[i] += s2[i];
                        c1[i] += c2[i];
                    }
                    (s1, c1)
                },
            );

        for ci in 0..k {
            if counts[ci] > 0 {
                #[allow(clippy::cast_precision_loss)]
                let count_f = counts[ci] as f64;
                centroids[ci] = sums[ci] / count_f;
            }
        }
    }

    // Final assignment pass (parallel).
    data.par_iter()
        .map(|&val| nearest_centroid_1d(val, &centroids))
        .collect()
}

// ---------------------------------------------------------------------------
// Weighted 1-D k-means
// ---------------------------------------------------------------------------

/// Weighted k-means for 1-D scalar values.
///
/// Identical to [`kmeans_1d`] except centroid updates use weighted averages,
/// so points with higher weight pull centroids more strongly.
///
/// # Panics
/// Panics if `data` is empty, `k` is zero, or `data.len() != weights.len()`.
#[allow(clippy::cast_possible_truncation)]
#[must_use]
pub fn kmeans_1d_weighted(
    data: &[f64],
    weights: &[f64],
    k: usize,
    max_iterations: u32,
) -> (Vec<u16>, Vec<f64>) {
    assert_eq!(data.len(), weights.len());
    assert!(!data.is_empty(), "data must not be empty");
    assert!(k > 0, "k must be positive");

    let n = data.len();
    if k >= n {
        let assignments: Vec<u16> = (0..n).map(|i| i as u16).collect();
        let centroids: Vec<f64> = data.to_vec();
        return (assignments, centroids);
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
        let min = indexed.first().unwrap().0;
        let max = indexed.last().unwrap().0;
        #[allow(clippy::cast_precision_loss)]
        let val = min + (max - min) * (centroids.len() as f64 / k as f64);
        centroids.push(val);
    }
    // Sort centroids so nearest_centroid_1d binary search works correctly.
    centroids.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

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
        // Re-sort centroids after updates/re-seeding so binary search works.
        centroids.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    }

    // Final assignment pass (parallel).
    let assignments = data.par_iter()
        .map(|&val| nearest_centroid_1d(val, &centroids))
        .collect();
    (assignments, centroids)
}

// ---------------------------------------------------------------------------
// Weighted EMD k-means
// ---------------------------------------------------------------------------

/// Weighted k-means using EMD as the distance metric.
///
/// Identical to [`kmeans_emd_with_progress`] except centroid updates use
/// weighted averages.
///
/// # Panics
/// Panics if `data` is empty, `k` is zero, or `data.len() != weights.len()`.
#[allow(clippy::cast_possible_truncation)]
pub fn kmeans_emd_weighted(
    data: &[Vec<f64>],
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

    // -- k-means++ initialisation ------------------------------------------
    let mut centroids = kmeanspp_init(data, k, &mut rng);

    let mut assignments = vec![0_u16; n];

    for iter in 0..max_iterations {
        progress(iter, max_iterations);

        // -- Assignment step (parallel) ------------------------------------
        let new_assignments: Vec<u16> = data
            .par_iter()
            .map(|point| nearest_centroid(point, &centroids))
            .collect();

        let changed = assignments
            .iter()
            .zip(new_assignments.iter())
            .any(|(old, new)| old != new);
        assignments = new_assignments;

        if !changed {
            break;
        }

        // -- Weighted update step ------------------------------------------
        for c in &mut centroids {
            c.fill(0.0);
        }
        let mut weight_sums = vec![0.0_f64; k];

        for (i, point) in data.iter().enumerate() {
            let ci = assignments[i] as usize;
            let w = weights[i];
            weight_sums[ci] += w;
            for (j, &val) in point.iter().enumerate() {
                centroids[ci][j] += val * w;
            }
        }

        // Average and handle empty clusters.
        for ci in 0..k {
            if weight_sums[ci] <= 0.0 {
                let farthest = farthest_point(data, &assignments, &centroids);
                centroids[ci].clone_from(&data[farthest]);
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
            *assign = nearest_centroid(point, &centroids);
        });

    assignments
}

// ---------------------------------------------------------------------------
// Weighted EMD k-means for u8 count histograms
// ---------------------------------------------------------------------------

/// Weighted k-means using EMD on u8 count histograms.
///
/// Data points are unnormalized `u8` integer counts (e.g. how many
/// next-street cards landed in each equity bucket). Centroids are `f64`
/// probability vectors. This reduces per-histogram storage from 8 bytes
/// (f64) to 1 byte (u8) per bin -- an 8x memory reduction.
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

    (assignments, centroids)
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// K-means++ seeding with EMD distance.
#[allow(clippy::cast_possible_truncation)]
fn kmeanspp_init(data: &[Vec<f64>], k: usize, rng: &mut StdRng) -> Vec<Vec<f64>> {
    let n = data.len();

    let mut centroids: Vec<Vec<f64>> = Vec::with_capacity(k);

    // Pick the first centroid uniformly at random.
    let first = rng.random_range(0..n);
    centroids.push(data[first].clone());

    // Squared-distance buffer (EMD can be used directly; squaring gives
    // the D^2 weighting prescribed by k-means++).
    let mut dists = vec![f64::MAX; n];

    for _ in 1..k {
        // Update distances: min with distance to newest centroid.
        let newest = centroids.last().expect("centroids is non-empty");

        // Parallel: compute EMD to newest centroid, update min-distances, sum.
        let total: f64 = data
            .par_iter()
            .zip(dists.par_iter_mut())
            .map(|(point, dist)| {
                let d = emd(point, newest);
                let d2 = d * d;
                if d2 < *dist {
                    *dist = d2;
                }
                *dist
            })
            .sum();

        if total <= 0.0 {
            // All remaining distances are zero; pick randomly.
            let idx = rng.random_range(0..n);
            centroids.push(data[idx].clone());
            continue;
        }

        // Weighted random selection proportional to D^2 (sequential — cheap prefix-sum scan).
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
        centroids.push(data[chosen].clone());
    }

    centroids
}

/// Index of the nearest centroid to `point` by EMD.
#[allow(clippy::cast_possible_truncation)]
fn nearest_centroid(point: &[f64], centroids: &[Vec<f64>]) -> u16 {
    let mut best_idx = 0_u16;
    let mut best_dist = f64::MAX;
    for (ci, centroid) in centroids.iter().enumerate() {
        let d = emd(point, centroid);
        if d < best_dist {
            best_dist = d;
            best_idx = ci as u16;
        }
    }
    best_idx
}

/// Index of the nearest centroid to a scalar `val`.
///
/// Uses binary search on sorted centroids (O(log k)) instead of linear scan.
/// Centroids are always sorted after percentile initialisation and stay sorted
/// because the weighted-average update preserves ordering in 1-D.
#[allow(clippy::cast_possible_truncation)]
pub(crate) fn nearest_centroid_1d(val: f64, centroids: &[f64]) -> u16 {
    let k = centroids.len();
    debug_assert!(k > 0);
    // Binary search for the insertion point.
    let pos = centroids.partition_point(|&c| c < val);
    if pos == 0 {
        return 0;
    }
    if pos >= k {
        return (k - 1) as u16;
    }
    // Compare the two neighbours.
    let d_left = (val - centroids[pos - 1]).abs();
    let d_right = (centroids[pos] - val).abs();
    if d_left <= d_right {
        (pos - 1) as u16
    } else {
        pos as u16
    }
}

/// Find the point with the largest EMD distance to its currently assigned
/// centroid. Used to re-seed empty clusters.
fn farthest_point(data: &[Vec<f64>], assignments: &[u16], centroids: &[Vec<f64>]) -> usize {
    let mut best_idx = 0;
    let mut best_dist = -1.0_f64;
    for (i, point) in data.iter().enumerate() {
        let ci = assignments[i] as usize;
        let d = emd(point, &centroids[ci]);
        if d > best_dist {
            best_dist = d;
            best_idx = i;
        }
    }
    best_idx
}

/// Normalize a u8 count histogram to a f64 probability distribution.
fn normalize_u8(counts: &[u8]) -> Vec<f64> {
    let total: f64 = counts.iter().map(|&c| f64::from(c)).sum();
    let inv = if total > 0.0 { 1.0 / total } else { 0.0 };
    counts.iter().map(|&c| f64::from(c) * inv).collect()
}

/// Index of the nearest centroid to a u8 histogram `point` by EMD.
#[allow(clippy::cast_possible_truncation)]
pub(crate) fn nearest_centroid_u8(point: &[u8], centroids: &[Vec<f64>]) -> u16 {
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

        // Parallel: compute EMD to newest centroid, update min-distances, sum.
        let total: f64 = data
            .par_iter()
            .zip(dists.par_iter_mut())
            .map(|(point, dist)| {
                let d = emd_u8_vs_f64(point, newest);
                let d2 = d * d;
                if d2 < *dist {
                    *dist = d2;
                }
                *dist
            })
            .sum();

        if total <= 0.0 {
            let idx = rng.random_range(0..n);
            centroids.push(normalize_u8(&data[idx]));
            continue;
        }

        // Weighted random selection (sequential — cheap prefix-sum scan).
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

// ---------------------------------------------------------------------------
// fastkmeans-rs wrappers
// ---------------------------------------------------------------------------

/// Run BLAS-accelerated L2 k-means via `fastkmeans-rs`.
///
/// Takes feature vectors as `&[Vec<f32>]` (n_samples × n_features),
/// returns `(labels, centroids)` in our internal format.
///
/// This replaces `kmeans_1d_weighted` and `kmeans_emd_weighted_u8` with
/// a single L2-distance implementation backed by Apple Accelerate BLAS.
#[allow(clippy::cast_possible_truncation)]
#[must_use]
pub fn fast_kmeans(
    features: &[Vec<f32>],
    k: usize,
    max_iters: u32,
    seed: u64,
) -> (Vec<u16>, Vec<Vec<f32>>) {
    let n = features.len();
    if n == 0 || k == 0 {
        return (vec![], vec![]);
    }
    let k = k.min(n);
    let dim = features[0].len();

    // Build Array2<f32> from flat data.
    let mut flat = Vec::with_capacity(n * dim);
    for f in features {
        flat.extend_from_slice(f);
    }
    let data = Array2::from_shape_vec((n, dim), flat)
        .expect("shape mismatch building Array2 for k-means");

    // Configure and run.
    let config = fastkmeans_rs::KMeansConfig::new(k)
        .with_max_iters(max_iters as usize)
        .with_seed(seed)
        .with_max_points_per_centroid(None); // no subsampling

    let mut km = FastKMeans::with_config(config);
    let labels = km
        .fit_predict(&data.view())
        .expect("fastkmeans fit_predict failed");

    // Extract labels as Vec<u16>.
    let label_vec: Vec<u16> = labels.iter().map(|&l| l as u16).collect();

    // Extract centroids.
    let centroids_arr = km.centroids().expect("centroids not set after fit");
    let centroid_vecs: Vec<Vec<f32>> = (0..k)
        .map(|i| centroids_arr.row(i).to_vec())
        .collect();

    (label_vec, centroid_vecs)
}

/// Convenience wrapper for 1D equity clustering.
///
/// Converts `f64` equities to single-feature `f32` vectors, runs fast_kmeans,
/// and returns labels + centroids as `f64`.
#[allow(clippy::cast_possible_truncation)]
#[must_use]
pub fn fast_kmeans_1d(
    data: &[f64],
    k: usize,
    max_iters: u32,
    seed: u64,
) -> (Vec<u16>, Vec<f64>) {
    let features: Vec<Vec<f32>> = data.iter().map(|&v| vec![v as f32]).collect();
    let (labels, centroids) = fast_kmeans(&features, k, max_iters, seed);
    let centroid_vals: Vec<f64> = centroids.iter().map(|c| f64::from(c[0])).collect();
    (labels, centroid_vals)
}

/// Convenience wrapper for histogram clustering.
///
/// Converts `u8` histograms to `f32` feature vectors and runs fast_kmeans.
/// Returns labels + centroids as `Vec<Vec<f32>>`.
#[allow(clippy::cast_possible_truncation)]
#[must_use]
pub fn fast_kmeans_histogram(
    histograms: &[Vec<u8>],
    k: usize,
    max_iters: u32,
    seed: u64,
) -> (Vec<u16>, Vec<Vec<f32>>) {
    let features: Vec<Vec<f32>> = histograms
        .iter()
        .map(|h| h.iter().map(|&v| f32::from(v)).collect())
        .collect();
    fast_kmeans(&features, k, max_iters, seed)
}

/// Assign a single u8 histogram point to the nearest centroid by L2 distance.
///
/// Used in the exhaustive assignment phase after `fast_kmeans_histogram`
/// finds centroids.
#[allow(clippy::cast_possible_truncation)]
#[must_use]
pub fn nearest_centroid_l2(point: &[u8], centroids: &[Vec<f32>]) -> u16 {
    let mut best_idx = 0_u16;
    let mut best_dist = f32::MAX;
    for (ci, centroid) in centroids.iter().enumerate() {
        let d: f32 = point
            .iter()
            .zip(centroid.iter())
            .map(|(&p, &c)| {
                let diff = f32::from(p) - c;
                diff * diff
            })
            .sum();
        if d < best_dist {
            best_dist = d;
            best_idx = ci as u16;
        }
    }
    best_idx
}

// ---------------------------------------------------------------------------
// Centroid EV computation, sorting, and label remapping
// ---------------------------------------------------------------------------

/// Compute the expected equity of a centroid histogram given child-street EVs.
/// Returns `centroid[i] * child_evs[i]`.
#[must_use]
pub fn compute_centroid_ev(centroid: &[f64], child_evs: &[f64]) -> f64 {
    debug_assert_eq!(centroid.len(), child_evs.len());
    centroid.iter().zip(child_evs).map(|(w, ev)| w * ev).sum()
}

/// Sort centroids by expected equity (ascending). Returns `(sorted_centroids, remap)`
/// where `remap[old_id] = new_id`.
#[must_use]
pub fn sort_centroids_by_ev(
    centroids: &[Vec<f64>],
    child_evs: &[f64],
) -> (Vec<Vec<f64>>, Vec<u16>) {
    let mut indexed_evs: Vec<(usize, f64)> = centroids
        .iter()
        .enumerate()
        .map(|(i, c)| (i, compute_centroid_ev(c, child_evs)))
        .collect();
    indexed_evs.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    let mut remap = vec![0_u16; centroids.len()];
    let mut sorted = Vec::with_capacity(centroids.len());
    #[allow(clippy::cast_possible_truncation)]
    for (new_id, (old_id, _ev)) in indexed_evs.iter().enumerate() {
        remap[*old_id] = new_id as u16;
        sorted.push(centroids[*old_id].clone());
    }
    (sorted, remap)
}

/// Compute equity gaps between adjacent sorted centroids. Returns vec of length K-1.
#[must_use]
pub fn compute_centroid_gaps(sorted_evs: &[f64]) -> Vec<f64> {
    sorted_evs.windows(2).map(|w| w[1] - w[0]).collect()
}

/// Apply a remap permutation to bucket labels. `remap[old_id] = new_id`.
#[must_use]
pub fn remap_labels(labels: &[u16], remap: &[u16]) -> Vec<u16> {
    labels.iter().map(|&old| remap[old as usize]).collect()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kmeans_1d_weighted_returns_centroids() {
        let data = vec![0.1, 0.2, 0.9, 1.0];
        let weights = vec![1.0, 1.0, 1.0, 1.0];
        let (labels, centroids) = kmeans_1d_weighted(&data, &weights, 2, 100);
        assert_eq!(labels.len(), 4);
        assert_eq!(centroids.len(), 2);
        // Centroids should be sorted (1-D k-means maintains sorted centroids).
        assert!(centroids[0] < centroids[1]);
        // Low cluster centroid should be near 0.15, high near 0.95.
        assert!(centroids[0] < 0.5);
        assert!(centroids[1] > 0.5);
    }

    #[test]
    fn kmeans_emd_weighted_u8_returns_centroids() {
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
        let (labels, centroids) = kmeans_emd_weighted_u8(&data, &weights, 2, 100, 42, |_, _| {});
        assert_eq!(labels.len(), 100);
        assert_eq!(centroids.len(), 2);
        // Each centroid should be a probability distribution (4 bins).
        for c in &centroids {
            assert_eq!(c.len(), 4);
            let sum: f64 = c.iter().sum();
            assert!((sum - 1.0).abs() < 0.01, "centroid should sum to ~1.0, got {sum}");
        }
    }

    #[test]
    fn nearest_centroid_u8_is_accessible() {
        // Verify nearest_centroid_u8 is pub(crate) by calling it directly.
        let point = vec![36, 4, 0, 0];
        let centroids = vec![
            vec![0.9, 0.1, 0.0, 0.0],
            vec![0.0, 0.0, 0.1, 0.9],
        ];
        let idx = nearest_centroid_u8(&point, &centroids);
        assert_eq!(idx, 0); // point is close to first centroid
    }

    #[test]
    fn test_emd_identical() {
        let p = vec![0.25, 0.25, 0.25, 0.25];
        assert!(emd(&p, &p).abs() < 1e-10);
    }

    #[test]
    fn test_emd_opposite_ends() {
        let p = vec![1.0, 0.0, 0.0, 0.0];
        let q = vec![0.0, 0.0, 0.0, 1.0];
        assert!((emd(&p, &q) - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_emd_adjacent() {
        let p = vec![1.0, 0.0, 0.0, 0.0];
        let q = vec![0.0, 1.0, 0.0, 0.0];
        assert!((emd(&p, &q) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_emd_symmetric() {
        let p = vec![0.5, 0.3, 0.2];
        let q = vec![0.1, 0.4, 0.5];
        assert!((emd(&p, &q) - emd(&q, &p)).abs() < 1e-10);
    }

    #[test]
    fn test_kmeans_separable_clusters() {
        let mut data = Vec::new();
        for _ in 0..50 {
            data.push(vec![0.9, 0.1, 0.0, 0.0]);
        }
        for _ in 0..50 {
            data.push(vec![0.0, 0.0, 0.1, 0.9]);
        }
        let assignments = kmeans_emd(&data, 2, 100, 42);
        assert!(assignments[0..50].iter().all(|&a| a == assignments[0]));
        assert!(assignments[50..100].iter().all(|&a| a == assignments[50]));
        assert_ne!(assignments[0], assignments[50]);
    }

    #[test]
    fn test_kmeans_three_clusters() {
        let mut data = Vec::new();
        for _ in 0..30 {
            data.push(vec![0.8, 0.2, 0.0, 0.0, 0.0]);
        }
        for _ in 0..30 {
            data.push(vec![0.0, 0.0, 0.8, 0.2, 0.0]);
        }
        for _ in 0..30 {
            data.push(vec![0.0, 0.0, 0.0, 0.2, 0.8]);
        }
        let assignments = kmeans_emd(&data, 3, 100, 42);
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

    #[test]
    fn test_kmeans_1d_basic() {
        let data: Vec<f64> = (0..50)
            .map(|_| 0.1)
            .chain((0..50).map(|_| 0.9))
            .collect();
        let assignments = kmeans_1d(&data, 2, 100);
        assert!(assignments[0..50].iter().all(|&a| a == assignments[0]));
        assert!(assignments[50..100].iter().all(|&a| a == assignments[50]));
        assert_ne!(assignments[0], assignments[50]);
    }

    #[test]
    fn test_kmeans_single_cluster() {
        let data = vec![vec![0.5, 0.5]; 10];
        let assignments = kmeans_emd(&data, 1, 100, 42);
        assert!(assignments.iter().all(|&a| a == 0));
    }

    #[test]
    fn test_kmeans_k_equals_n() {
        let data = vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![0.5, 0.5]];
        let assignments = kmeans_emd(&data, 3, 100, 42);
        // Each point should be its own cluster.
        assert_ne!(assignments[0], assignments[1]);
        assert_ne!(assignments[1], assignments[2]);
        assert_ne!(assignments[0], assignments[2]);
    }

    #[test]
    fn test_kmeans_1d_single_value() {
        let data = vec![0.5; 20];
        let assignments = kmeans_1d(&data, 3, 100);
        // All points are identical; they should end up in the same cluster.
        let first = assignments[0];
        assert!(assignments.iter().all(|&a| a == first));
    }

    #[test]
    fn weighted_1d_kmeans_respects_weights() {
        let data = vec![0.1, 0.2, 0.9, 1.0];
        let weights = vec![10.0, 10.0, 1.0, 1.0];
        let (labels, _centroids) = kmeans_1d_weighted(&data, &weights, 2, 100);
        assert_eq!(labels[0], labels[1]);
        assert_eq!(labels[2], labels[3]);
        assert_ne!(labels[0], labels[2]);
    }

    #[test]
    fn weighted_emd_kmeans_separable() {
        let mut data = Vec::new();
        let mut weights = Vec::new();
        for _ in 0..50 {
            data.push(vec![0.9, 0.1, 0.0, 0.0]);
            weights.push(5.0);
        }
        for _ in 0..50 {
            data.push(vec![0.0, 0.0, 0.1, 0.9]);
            weights.push(1.0);
        }
        let assignments = kmeans_emd_weighted(&data, &weights, 2, 100, 42, |_, _| {});
        assert!(assignments[0..50].iter().all(|&a| a == assignments[0]));
        assert!(assignments[50..100].iter().all(|&a| a == assignments[50]));
        assert_ne!(assignments[0], assignments[50]);
    }

    #[test]
    fn weighted_emd_u8_separable() {
        let mut data: Vec<Vec<u8>> = Vec::new();
        let mut weights: Vec<f64> = Vec::new();
        for _ in 0..50 {
            data.push(vec![36, 4, 0, 0]);
            weights.push(5.0);
        }
        for _ in 0..50 {
            data.push(vec![0, 0, 4, 36]);
            weights.push(1.0);
        }
        let (assignments, _centroids) = kmeans_emd_weighted_u8(&data, &weights, 2, 100, 42, |_, _| {});
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
        let (assignments, _centroids) = kmeans_emd_weighted_u8(&data, &weights, 3, 100, 42, |_, _| {});
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
        // With only 3 distinct values, we can have at most 3 occupied clusters.
        // Use k=3 to test that all clusters stay occupied (no empty buckets)
        // despite the highly skewed distribution.
        let (labels, _centroids) = kmeans_1d_weighted(&data, &weights, 3, 100);
        let used: std::collections::HashSet<u16> = labels.iter().copied().collect();
        assert_eq!(
            used.len(),
            3,
            "expected 3 occupied clusters, got {}",
            used.len()
        );

        // Also test with many distinct values and high k: use 200 distinct
        // points spread across [0,1] with heavily skewed weights, and k=20.
        // Without re-seeding, clusters in the low-weight region starve.
        let mut data2 = Vec::new();
        let mut weights2 = Vec::new();
        for i in 0..200 {
            #[allow(clippy::cast_precision_loss)]
            let v = i as f64 / 199.0;
            data2.push(v);
            // Heavy weight at ends, tiny weight in the middle.
            let w = if v < 0.1 || v > 0.9 { 100.0 } else { 0.01 };
            weights2.push(w);
        }
        let (labels2, _centroids2) = kmeans_1d_weighted(&data2, &weights2, 20, 100);
        let used2: std::collections::HashSet<u16> = labels2.iter().copied().collect();
        assert_eq!(
            used2.len(),
            20,
            "expected 20 occupied clusters, got {}",
            used2.len()
        );
    }

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

    #[test]
    fn test_compute_centroid_ev() {
        let child_evs = [0.1, 0.5, 0.9];
        let centroid = [0.5, 0.0, 0.5];
        let ev = compute_centroid_ev(&centroid, &child_evs);
        assert!((ev - 0.5).abs() < 1e-10, "expected 0.5, got {ev}");
    }

    #[test]
    fn test_sort_centroids_by_ev() {
        // Centroid 0: high EV, centroid 1: low EV, centroid 2: mid EV
        let centroids = vec![
            vec![0.0, 0.0, 1.0], // EV = 0.9
            vec![1.0, 0.0, 0.0], // EV = 0.1
            vec![0.0, 1.0, 0.0], // EV = 0.5
        ];
        let child_evs = [0.1, 0.5, 0.9];
        let (sorted, remap) = sort_centroids_by_ev(&centroids, &child_evs);

        // Sorted by ascending EV: old[1]=0.1, old[2]=0.5, old[0]=0.9
        assert_eq!(remap[0], 2); // old 0 -> new 2 (highest)
        assert_eq!(remap[1], 0); // old 1 -> new 0 (lowest)
        assert_eq!(remap[2], 1); // old 2 -> new 1 (middle)

        // Verify sorted centroids are in ascending EV order
        let ev0 = compute_centroid_ev(&sorted[0], &child_evs);
        let ev1 = compute_centroid_ev(&sorted[1], &child_evs);
        let ev2 = compute_centroid_ev(&sorted[2], &child_evs);
        assert!(ev0 < ev1);
        assert!(ev1 < ev2);
    }

    #[test]
    fn test_compute_centroid_gaps() {
        let sorted_evs = [0.1, 0.3, 0.9];
        let gaps = compute_centroid_gaps(&sorted_evs);
        assert_eq!(gaps.len(), 2);
        assert!((gaps[0] - 0.2).abs() < 1e-10, "expected 0.2, got {}", gaps[0]);
        assert!((gaps[1] - 0.6).abs() < 1e-10, "expected 0.6, got {}", gaps[1]);
    }

    #[test]
    fn test_remap_labels() {
        let labels: Vec<u16> = vec![0, 1, 2, 0, 2, 1];
        let remap: Vec<u16> = vec![2, 0, 1];
        let result = remap_labels(&labels, &remap);
        assert_eq!(result, vec![2, 0, 1, 2, 1, 0]);
    }

    #[test]
    fn test_emd_u8_vs_f64_weighted_uniform_gaps() {
        // With uniform gaps of 1.0, weighted EMD must equal unweighted EMD.
        let counts: Vec<u8> = vec![20, 15, 10, 1];
        let centroid: Vec<f64> = vec![0.1, 0.4, 0.3, 0.2];
        let gaps = vec![1.0, 1.0, 1.0]; // K-1 = 3 gaps
        let d_weighted = emd_u8_vs_f64_weighted(&counts, &centroid, &gaps);
        let d_unweighted = emd_u8_vs_f64(&counts, &centroid);
        assert!(
            (d_weighted - d_unweighted).abs() < 1e-10,
            "weighted={d_weighted} unweighted={d_unweighted}"
        );
    }

    #[test]
    fn test_emd_u8_vs_f64_weighted_nonuniform_gaps() {
        // All mass at bucket 0. Compare distance to centroids at bucket 1 vs bucket 3.
        // With gaps [0.1, 0.3, 0.5], bucket 3 is farther than bucket 1.
        let counts: Vec<u8> = vec![40, 0, 0, 0];
        let gaps = vec![0.1, 0.3, 0.5];

        // Centroid concentrated at bucket 1
        let near_centroid = vec![0.0, 1.0, 0.0, 0.0];
        // Centroid concentrated at bucket 3
        let far_centroid = vec![0.0, 0.0, 0.0, 1.0];

        let d_near = emd_u8_vs_f64_weighted(&counts, &near_centroid, &gaps);
        let d_far = emd_u8_vs_f64_weighted(&counts, &far_centroid, &gaps);

        assert!(
            d_far > d_near,
            "far={d_far} should be > near={d_near}"
        );
    }

    #[test]
    fn test_nearest_centroid_u8_weighted() {
        // Point with mass concentrated at bucket 0.
        let point: Vec<u8> = vec![36, 4, 0, 0];
        let centroids = vec![
            vec![0.9, 0.1, 0.0, 0.0], // close to point
            vec![0.0, 0.0, 0.1, 0.9], // far from point
        ];
        let gaps = vec![1.0, 1.0, 1.0];
        let idx = nearest_centroid_u8_weighted(&point, &centroids, &gaps);
        assert_eq!(idx, 0, "should pick centroid 0 (closer)");
    }
}
