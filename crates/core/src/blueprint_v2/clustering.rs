//! Earth Mover's Distance and k-means clustering for the potential-aware
//! abstraction pipeline.
//!
//! The EMD implementation exploits the fact that our distributions live on
//! ordered 1-D buckets, so EMD reduces to the L1 norm of the CDF difference
//! (linear time, no LP needed).

use rand::prelude::*;
use rand::rngs::StdRng;

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

    for _iter in 0..max_iterations {
        // -- Assignment step -----------------------------------------------
        let mut changed = false;
        for (i, point) in data.iter().enumerate() {
            let nearest = nearest_centroid(point, &centroids);
            if assignments[i] != nearest {
                assignments[i] = nearest;
                changed = true;
            }
        }

        if !changed {
            break;
        }

        // -- Update step ---------------------------------------------------
        // Zero-out centroids and counts.
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
                // Reinitialise from the point with the largest EMD to its
                // current centroid.
                let farthest = farthest_point(data, &assignments, &centroids);
                centroids[ci].clone_from(&data[farthest]);
                assignments[farthest] = ci as u16;
            } else {
                #[allow(clippy::cast_precision_loss)] // count fits in f64 mantissa
                let inv = 1.0 / counts[ci] as f64;
                for v in &mut centroids[ci] {
                    *v *= inv;
                }
            }
        }
    }

    // Final assignment pass to ensure consistency after last update.
    for (i, point) in data.iter().enumerate() {
        assignments[i] = nearest_centroid(point, &centroids);
    }

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
        // -- Assignment step -----------------------------------------------
        let mut changed = false;
        for (i, &val) in data.iter().enumerate() {
            let nearest = nearest_centroid_1d(val, &centroids);
            if assignments[i] != nearest {
                assignments[i] = nearest;
                changed = true;
            }
        }

        if !changed {
            break;
        }

        // -- Update step ---------------------------------------------------
        let mut sums = vec![0.0_f64; k];
        let mut counts = vec![0_usize; k];

        for (i, &val) in data.iter().enumerate() {
            let ci = assignments[i] as usize;
            sums[ci] += val;
            counts[ci] += 1;
        }

        for ci in 0..k {
            if counts[ci] > 0 {
                #[allow(clippy::cast_precision_loss)]
                let count_f = counts[ci] as f64;
                centroids[ci] = sums[ci] / count_f;
            }
            // Empty cluster: keep old centroid (rare with percentile init).
        }
    }

    // Final assignment pass.
    for (i, &val) in data.iter().enumerate() {
        assignments[i] = nearest_centroid_1d(val, &centroids);
    }

    assignments
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
        // INVARIANT: centroids is non-empty after pushing above.
        let newest = centroids.last().expect("centroids is non-empty");
        let mut total = 0.0_f64;
        for (i, point) in data.iter().enumerate() {
            let d = emd(point, newest);
            let d2 = d * d;
            if d2 < dists[i] {
                dists[i] = d2;
            }
            total += dists[i];
        }

        if total <= 0.0 {
            // All remaining distances are zero; pick randomly.
            let idx = rng.random_range(0..n);
            centroids.push(data[idx].clone());
            continue;
        }

        // Weighted random selection proportional to D^2.
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
#[allow(clippy::cast_possible_truncation)]
fn nearest_centroid_1d(val: f64, centroids: &[f64]) -> u16 {
    let mut best_idx = 0_u16;
    let mut best_dist = f64::MAX;
    for (ci, &c) in centroids.iter().enumerate() {
        let d = (val - c).abs();
        if d < best_dist {
            best_dist = d;
            best_idx = ci as u16;
        }
    }
    best_idx
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

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

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
}
