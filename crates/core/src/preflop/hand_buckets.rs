//! Hand bucketing via EHS k-means clustering for postflop abstraction.
//!
//! Maps `(canonical_hand_idx, board_texture_id)` pairs to bucket IDs at each street.
//! River `BucketEquity` provides average equity between bucket pairs for terminal values.

use std::sync::atomic::{AtomicUsize, Ordering};

use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::hands::{CanonicalHand, all_hands};
use crate::poker::{Card, Suit, Value};
use crate::preflop::ehs::{EhsFeatures, ehs_features};
use crate::preflop::equity::EquityTable;

/// Number of canonical preflop hands.
pub const NUM_HANDS: usize = 169;

/// Errors arising during bucket construction.
#[derive(Debug, Error)]
pub enum BucketError {
    #[error("num_buckets ({0}) must be > 0")]
    ZeroBuckets(u16),
    #[error("num_textures ({0}) must be > 0")]
    ZeroTextures(u16),
    #[error("feature vector is empty — no valid boards were sampled")]
    EmptyFeatures,
}

/// Mapping from `(hand_idx, texture_id)` to bucket IDs across all three streets.
///
/// Indices: `flop_buckets[hand_idx][texture_id]`, similarly for turn and river.
#[derive(Serialize, Deserialize)]
pub struct HandBucketMapping {
    /// `flop_buckets[hand_idx][texture_id]` → bucket ID
    pub flop_buckets: Vec<Vec<u16>>,
    /// `turn_buckets[flop_bucket_id][turn_texture_id]` → bucket ID
    pub turn_buckets: Vec<Vec<u16>>,
    /// `river_buckets[turn_bucket_id][river_texture_id]` → bucket ID
    pub river_buckets: Vec<Vec<u16>>,
    pub num_flop_buckets: u16,
    pub num_turn_buckets: u16,
    pub num_river_buckets: u16,
}

/// River equity table: `equity[bucket_a][bucket_b]` = average equity of `a` vs `b`.
#[derive(Serialize, Deserialize)]
pub struct BucketEquity {
    pub equity: Vec<Vec<f32>>,
    pub num_buckets: usize,
}

impl BucketEquity {
    /// Look up equity of `bucket_a` vs `bucket_b`.
    #[must_use]
    pub fn get(&self, bucket_a: usize, bucket_b: usize) -> f32 {
        self.equity
            .get(bucket_a)
            .and_then(|row| row.get(bucket_b))
            .copied()
            .unwrap_or(0.5)
    }
}

// ---------------------------------------------------------------------------
// Public orchestration
// ---------------------------------------------------------------------------

/// Build flop bucket assignments for all `(hand_idx, texture_id)` pairs.
///
/// `board_samples[texture_id]` supplies representative flop boards for that texture.
/// Each entry is a 3-card flop. Hands are clustered by their EHS feature vectors.
///
/// `on_hand_done` is called after each hand's EHS features are computed (169 total).
///
/// # Errors
///
/// Returns `BucketError::ZeroBuckets` if `num_buckets == 0`, or
/// `BucketError::ZeroTextures` if `board_samples` is empty.
#[allow(clippy::cast_possible_truncation)]
pub fn build_flop_buckets(
    num_buckets: u16,
    board_samples: &[Vec<[Card; 3]>],
    on_hand_done: impl Fn(usize) + Sync + Send,
) -> Result<Vec<Vec<u16>>, BucketError> {
    // board_samples.len() ≤ num_flop_textures which is always small (≤ 65535)
    validate_buckets(num_buckets, board_samples.len() as u16)?;
    let hands: Vec<CanonicalHand> = all_hands().collect();
    let features = compute_all_flop_features(&hands, board_samples, &on_hand_done);
    Ok(cluster_global(&features, num_buckets, board_samples.len()))
}

/// Build turn bucket assignments from flop bucket IDs and turn transition boards.
///
/// `turn_board_samples[flop_bucket][trans_id]` supplies representative turn boards (4 cards).
///
/// # Errors
///
/// Returns `BucketError::ZeroBuckets` if `num_buckets == 0`, or
/// `BucketError::ZeroTextures` if `turn_board_samples` is empty.
#[allow(clippy::cast_possible_truncation)]
pub fn build_turn_buckets(
    num_buckets: u16,
    flop_bucket_count: usize,
    turn_board_samples: &[Vec<[Card; 4]>],
) -> Result<Vec<Vec<u16>>, BucketError> {
    validate_buckets(num_buckets, turn_board_samples.len() as u16)?;
    let hands: Vec<CanonicalHand> = all_hands().collect();
    let features = compute_all_turn_features(&hands, flop_bucket_count, turn_board_samples);
    Ok(cluster_global(&features, num_buckets, turn_board_samples.len()))
}

/// Build river bucket assignments from turn bucket IDs and river transition boards.
///
/// # Errors
///
/// Returns `BucketError::ZeroBuckets` if `num_buckets == 0`, or
/// `BucketError::ZeroTextures` if `river_board_samples` is empty.
#[allow(clippy::cast_possible_truncation)]
pub fn build_river_buckets(
    num_buckets: u16,
    turn_bucket_count: usize,
    river_board_samples: &[Vec<[Card; 5]>],
) -> Result<Vec<Vec<u16>>, BucketError> {
    validate_buckets(num_buckets, river_board_samples.len() as u16)?;
    let hands: Vec<CanonicalHand> = all_hands().collect();
    let features = compute_all_river_features(&hands, turn_bucket_count, river_board_samples);
    Ok(cluster_global(&features, num_buckets, river_board_samples.len()))
}

/// Build river equity table from a set of representative river boards per bucket pair.
///
/// `river_boards[bucket_a][bucket_b]` supplies sample boards for that matchup.
#[must_use]
pub fn build_bucket_equity(
    num_river_buckets: usize,
    river_boards: &[Vec<Vec<[Card; 5]>>],
) -> BucketEquity {
    let equity = (0..num_river_buckets)
        .map(|a| build_equity_row(a, num_river_buckets, river_boards))
        .collect();
    BucketEquity {
        equity,
        num_buckets: num_river_buckets,
    }
}

/// Compute per-bucket average EHS from features and bucket assignments.
///
/// Returns `centroids[bucket_id]` → average EHS (first component of feature vector),
/// averaged across all textures and all hands assigned to that bucket.
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn bucket_ehs_centroids(
    features: &[Vec<EhsFeatures>],
    assignments: &[Vec<u16>],
    num_buckets: usize,
) -> Vec<f64> {
    let num_textures = features.first().map_or(0, Vec::len);
    let mut sums = vec![0.0f64; num_buckets];
    let mut counts = vec![0u32; num_buckets];

    for (hand_idx, hand_feats) in features.iter().enumerate() {
        for (tex_id, feat) in hand_feats.iter().enumerate() {
            if feat[0].is_nan() {
                continue; // skip blocked hands
            }
            let bucket = assignments[hand_idx][tex_id] as usize;
            sums[bucket] += feat[0]; // EHS component
            counts[bucket] += 1;
        }
    }
    let _ = num_textures; // used implicitly via iteration

    sums.iter()
        .zip(&counts)
        .map(|(&s, &c)| if c > 0 { s / f64::from(c) } else { 0.5 })
        .collect()
}

/// Build a bucket equity table from per-bucket EHS centroids.
///
/// `equity[a][b] = ehs_a / (ehs_a + ehs_b)` — a simple normalized comparison
/// of bucket strengths derived from their average EHS values.
#[must_use]
pub fn build_bucket_equity_from_centroids(centroids: &[f64]) -> BucketEquity {
    let n = centroids.len();
    let equity = (0..n)
        .map(|a| {
            (0..n)
                .map(|b| {
                    let sum = centroids[a] + centroids[b];
                    if sum > 0.0 {
                        (centroids[a] / sum) as f32
                    } else {
                        0.5f32
                    }
                })
                .collect()
        })
        .collect();
    BucketEquity {
        equity,
        num_buckets: n,
    }
}

/// Build bucket equity from the preflop 169×169 equity table.
///
/// For each bucket pair (a, b), averages the pairwise equities of all
/// canonical hands assigned to each bucket, weighted by card-removal counts
/// and the number of textures each hand appears in that bucket.
///
/// Much more accurate than the centroid formula because it uses actual
/// Monte Carlo showdown equities with proper card-removal weighting.
#[must_use]
pub fn build_bucket_equity_from_equity_table(
    assignments: &[Vec<u16>],
    equity_table: &EquityTable,
    num_buckets: usize,
) -> BucketEquity {
    let hands: Vec<CanonicalHand> = all_hands().collect();
    let hand_bucket_counts = count_hands_per_bucket(&hands, assignments, num_buckets);
    let equity = compute_pairwise_bucket_equity(&hand_bucket_counts, equity_table, num_buckets);
    BucketEquity {
        equity,
        num_buckets,
    }
}

/// For each bucket, count how many texture slots each canonical hand occupies.
///
/// Returns `counts[bucket][hand_idx]` — the number of textures where hand maps to that bucket.
fn count_hands_per_bucket(
    hands: &[CanonicalHand],
    assignments: &[Vec<u16>],
    num_buckets: usize,
) -> Vec<Vec<(usize, u32)>> {
    let mut raw_counts = vec![vec![0u32; hands.len()]; num_buckets];
    for (hand_idx, tex_assignments) in assignments.iter().enumerate() {
        for &bucket in tex_assignments {
            raw_counts[bucket as usize][hand_idx] += 1;
        }
    }
    // Compress to sparse: only hands with count > 0
    raw_counts
        .into_iter()
        .map(|counts| {
            counts
                .into_iter()
                .enumerate()
                .filter(|&(_, c)| c > 0)
                .collect()
        })
        .collect()
}

/// Compute bucket equity matrix from hand-per-bucket counts and the equity table.
#[allow(clippy::cast_precision_loss)]
fn compute_pairwise_bucket_equity(
    hand_bucket_counts: &[Vec<(usize, u32)>],
    equity_table: &EquityTable,
    num_buckets: usize,
) -> Vec<Vec<f32>> {
    (0..num_buckets)
        .into_par_iter()
        .map(|a| {
            (0..num_buckets)
                .map(|b| bucket_pair_equity(&hand_bucket_counts[a], &hand_bucket_counts[b], equity_table))
                .collect()
        })
        .collect()
}

/// Weighted average equity for one bucket pair.
#[allow(clippy::cast_precision_loss)]
fn bucket_pair_equity(
    hands_a: &[(usize, u32)],
    hands_b: &[(usize, u32)],
    equity_table: &EquityTable,
) -> f32 {
    let mut total_eq = 0.0f64;
    let mut total_weight = 0.0f64;
    for &(i, count_a) in hands_a {
        for &(j, count_b) in hands_b {
            let w = equity_table.weight(i, j) * f64::from(count_a) * f64::from(count_b);
            total_eq += w * equity_table.equity(i, j);
            total_weight += w;
        }
    }
    if total_weight > 0.0 {
        (total_eq / total_weight) as f32
    } else {
        0.5
    }
}

// ---------------------------------------------------------------------------
// Feature computation helpers
// ---------------------------------------------------------------------------

/// Compute EHS feature vectors for all canonical hands across all flop textures.
///
/// Returns `features[hand_idx][texture_id]` → EHS feature vector.
pub fn compute_all_flop_features(
    hands: &[CanonicalHand],
    board_samples: &[Vec<[Card; 3]>],
    on_hand_done: &(impl Fn(usize) + Sync + Send),
) -> Vec<Vec<EhsFeatures>> {
    let done = AtomicUsize::new(0);
    hands
        .par_iter()
        .map(|hand| {
            let feats: Vec<EhsFeatures> = board_samples
                .iter()
                .map(|boards| avg_features_for_hand(*hand, boards, |b| b.to_vec()))
                .collect();
            let count = done.fetch_add(1, Ordering::Relaxed) + 1;
            on_hand_done(count);
            feats
        })
        .collect()
}

fn compute_all_turn_features(
    hands: &[CanonicalHand],
    _flop_bucket_count: usize,
    turn_samples: &[Vec<[Card; 4]>],
) -> Vec<Vec<EhsFeatures>> {
    hands
        .par_iter()
        .map(|hand| {
            turn_samples
                .iter()
                .map(|boards| avg_features_for_hand(*hand, boards, |b| b.to_vec()))
                .collect()
        })
        .collect()
}

fn compute_all_river_features(
    hands: &[CanonicalHand],
    _turn_bucket_count: usize,
    river_samples: &[Vec<[Card; 5]>],
) -> Vec<Vec<EhsFeatures>> {
    hands
        .par_iter()
        .map(|hand| {
            river_samples
                .iter()
                .map(|boards| avg_features_for_hand(*hand, boards, |b| b.to_vec()))
                .collect()
        })
        .collect()
}

/// EHS features for a hand over representative boards using one combo per hand.
///
/// Uses the first non-conflicting combo rather than averaging over all combos.
/// This is ~8x faster with negligible impact on bucket quality, since all combos
/// of a canonical hand have the same equity (suits are interchangeable).
fn avg_features_for_hand<const N: usize>(
    hand: CanonicalHand,
    boards: &[[Card; N]],
    to_vec: impl Fn(&[Card; N]) -> Vec<Card>,
) -> EhsFeatures {
    let combos = hand.combos();
    let feats: Vec<EhsFeatures> = boards
        .iter()
        .filter_map(|b| {
            combos
                .iter()
                .find(|&&(c1, c2)| !board_conflicts([c1, c2], b))
                .map(|&(c1, c2)| ehs_features([c1, c2], &to_vec(b)))
        })
        .collect();
    average_features(&feats)
}

/// Check whether any board card conflicts with the hole cards.
fn board_conflicts<const N: usize>(hole: [Card; 2], board: &[Card; N]) -> bool {
    board.iter().any(|c| *c == hole[0] || *c == hole[1])
}

#[allow(clippy::cast_precision_loss)]
fn average_features(feats: &[EhsFeatures]) -> EhsFeatures {
    if feats.is_empty() {
        return [f64::NAN, f64::NAN, f64::NAN]; // sentinel: hand blocked on this board
    }
    // feats.len() is bounded by combo × texture count; precision acceptable
    let n = feats.len() as f64;
    let sum = feats.iter().fold([0.0f64; 3], |acc, f| {
        [acc[0] + f[0], acc[1] + f[1], acc[2] + f[2]]
    });
    [sum[0] / n, sum[1] / n, sum[2] / n]
}

// ---------------------------------------------------------------------------
// K-means clustering
// ---------------------------------------------------------------------------

/// Cluster `features[hand_idx][texture_id]` into buckets per texture.
///
/// Returns `assignments[hand_idx][texture_id]` → bucket ID.
/// Callers must validate inputs before calling (num_textures > 0, k > 0).
///
/// Hands with NaN features (board-conflict sentinel) are excluded from k-means,
/// then assigned to the nearest bucket centroid using their cross-texture average.
#[allow(clippy::cast_precision_loss)]
pub fn cluster_per_texture(
    features: &[Vec<EhsFeatures>],
    k: u16,
    num_textures: usize,
) -> Vec<Vec<u16>> {
    let num_hands = features.len();

    // Precompute each hand's average feature vector across non-blocked textures
    let hand_averages: Vec<EhsFeatures> = (0..num_hands)
        .map(|h| {
            let valid: Vec<&EhsFeatures> = features[h]
                .iter()
                .filter(|f| !f[0].is_nan())
                .collect();
            if valid.is_empty() {
                return [0.5, 0.0, 0.0]; // truly all-blocked (shouldn't happen in practice)
            }
            let n = valid.len() as f64;
            [
                valid.iter().map(|f| f[0]).sum::<f64>() / n,
                valid.iter().map(|f| f[1]).sum::<f64>() / n,
                valid.iter().map(|f| f[2]).sum::<f64>() / n,
            ]
        })
        .collect();

    let by_texture: Vec<Vec<u16>> = (0..num_textures)
        .into_par_iter()
        .map(|tex_id| {
            // Separate valid vs blocked hands
            let mut valid_indices = Vec::new();
            let mut blocked_indices = Vec::new();
            let mut valid_points = Vec::new();
            for (h, hand_feats) in features.iter().enumerate() {
                if hand_feats[tex_id][0].is_nan() {
                    blocked_indices.push(h);
                } else {
                    valid_indices.push(h);
                    valid_points.push(hand_feats[tex_id]);
                }
            }

            let valid_assignments = kmeans(&valid_points, k as usize, 100);

            // Build full assignment vector
            let mut full = vec![0u16; num_hands];
            if blocked_indices.is_empty() {
                // Fast path: no blocked hands, just map valid assignments back
                for (i, &h) in valid_indices.iter().enumerate() {
                    full[h] = valid_assignments[i];
                }
            } else {
                // Compute per-bucket centroids for assigning blocked hands
                let centroids = recompute_centroids(&valid_points, &valid_assignments, k as usize);
                for (i, &h) in valid_indices.iter().enumerate() {
                    full[h] = valid_assignments[i];
                }
                // Assign blocked hands to nearest centroid by their cross-texture average
                for &h in &blocked_indices {
                    full[h] = nearest_centroid(&hand_averages[h], &centroids);
                }
            }
            full
        })
        .collect();
    // Transpose: assignments[texture_id][hand_idx] → assignments[hand_idx][texture_id]
    let mut assignments = transpose(&by_texture, num_hands, num_textures);

    // Relabel bucket IDs so bucket 0 = highest average EHS across all textures.
    // Per-texture k-means produces arbitrary IDs; this makes them globally consistent.
    relabel_by_centroid_ehs(&mut assignments, features, k as usize);

    assignments
}

/// Cluster features globally: pool all `(hand, texture)` feature vectors into one
/// k-means run so bucket IDs are consistent across textures.
///
/// Returns `assignments[hand_idx][texture_id]` → bucket ID.
///
/// Unlike `cluster_per_texture` which runs independent k-means per texture,
/// this pools all non-NaN feature vectors, runs k-means once, and maps
/// assignments back. Blocked (NaN) hands are assigned to the nearest global
/// centroid using their cross-texture average.
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn cluster_global(
    features: &[Vec<EhsFeatures>],
    k: u16,
    num_textures: usize,
) -> Vec<Vec<u16>> {
    let num_hands = features.len();

    // Precompute each hand's average feature vector across non-blocked textures
    let hand_averages: Vec<EhsFeatures> = (0..num_hands)
        .map(|h| {
            let valid: Vec<&EhsFeatures> = features[h]
                .iter()
                .filter(|f| !f[0].is_nan())
                .collect();
            if valid.is_empty() {
                return [0.5, 0.0, 0.0];
            }
            let n = valid.len() as f64;
            [
                valid.iter().map(|f| f[0]).sum::<f64>() / n,
                valid.iter().map(|f| f[1]).sum::<f64>() / n,
                valid.iter().map(|f| f[2]).sum::<f64>() / n,
            ]
        })
        .collect();

    // Pool all non-NaN (hand, texture) feature vectors into one flat list.
    // Track the original (hand, texture) index for each pooled point.
    let mut pooled_points: Vec<EhsFeatures> = Vec::new();
    let mut pooled_origins: Vec<(usize, usize)> = Vec::new();
    let mut blocked: Vec<(usize, usize)> = Vec::new();

    for (h, hand_feats) in features.iter().enumerate() {
        for (t, feat) in hand_feats.iter().enumerate() {
            if feat[0].is_nan() {
                blocked.push((h, t));
            } else {
                pooled_points.push(*feat);
                pooled_origins.push((h, t));
            }
        }
    }

    // Run k-means once on all pooled points
    let pooled_assignments = kmeans(&pooled_points, k as usize, 100);

    // Compute global centroids for assigning blocked hands
    let centroids = recompute_centroids(&pooled_points, &pooled_assignments, k as usize);

    // Map assignments back to [hand][texture] shape
    let mut assignments = vec![vec![0u16; num_textures]; num_hands];
    for (i, &(h, t)) in pooled_origins.iter().enumerate() {
        assignments[h][t] = pooled_assignments[i];
    }
    for &(h, t) in &blocked {
        assignments[h][t] = nearest_centroid(&hand_averages[h], &centroids);
    }

    // Relabel so bucket 0 = highest average EHS
    relabel_by_centroid_ehs(&mut assignments, features, k as usize);

    assignments
}

/// Relabel bucket IDs so bucket 0 has the highest average EHS, bucket 1 next, etc.
///
/// Per-texture k-means produces arbitrary cluster IDs. This sorts them by average
/// EHS (descending) so that bucket IDs are globally consistent across textures.
#[allow(clippy::cast_precision_loss)]
fn relabel_by_centroid_ehs(
    assignments: &mut [Vec<u16>],
    features: &[Vec<EhsFeatures>],
    k: usize,
) {
    // Compute per-bucket average EHS across all (hand, texture) pairs
    let mut sums = vec![0.0f64; k];
    let mut counts = vec![0u32; k];
    for (hand_idx, hand_feats) in features.iter().enumerate() {
        for (tex_id, feat) in hand_feats.iter().enumerate() {
            if feat[0].is_nan() {
                continue;
            }
            let bucket = assignments[hand_idx][tex_id] as usize;
            sums[bucket] += feat[0];
            counts[bucket] += 1;
        }
    }

    let mut avg_ehs: Vec<(usize, f64)> = sums
        .iter()
        .zip(&counts)
        .enumerate()
        .map(|(i, (&s, &c))| {
            let avg = if c > 0 { s / f64::from(c) } else { 0.0 };
            (i, avg)
        })
        .collect();

    // Sort descending by EHS so rank 0 = strongest
    avg_ehs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    // Build remap: remap[old_id] = new_id
    let mut remap = vec![0u16; k];
    #[allow(clippy::cast_possible_truncation)]
    for (new_id, &(old_id, _)) in avg_ehs.iter().enumerate() {
        remap[old_id] = new_id as u16;
    }

    // Check if already in order (common after convergence)
    if remap.iter().enumerate().all(|(i, &r)| r as usize == i) {
        return;
    }

    // Apply remap in-place
    for hand_assignments in assignments.iter_mut() {
        for bucket in hand_assignments.iter_mut() {
            *bucket = remap[*bucket as usize];
        }
    }
}

/// Transpose a 2D vec from `[texture][hand]` to `[hand][texture]`.
fn transpose(by_texture: &[Vec<u16>], num_hands: usize, num_textures: usize) -> Vec<Vec<u16>> {
    (0..num_hands)
        .map(|h| (0..num_textures).map(|t| by_texture[t][h]).collect())
        .collect()
}

/// K-means clustering returning per-point cluster assignments.
///
/// Initialises centroids via k-means++ seeding then iterates until stable or `max_iter`.
#[must_use]
pub fn kmeans(points: &[EhsFeatures], k: usize, max_iter: usize) -> Vec<u16> {
    if points.is_empty() || k == 0 {
        return vec![];
    }
    let k = k.min(points.len());
    let mut centroids = kmeans_pp_init(points, k);
    let mut assignments = assign_clusters(points, &centroids);

    for _ in 0..max_iter {
        let new_centroids = recompute_centroids(points, &assignments, k);
        let new_assignments = assign_clusters(points, &new_centroids);
        if new_assignments == assignments {
            break;
        }
        assignments = new_assignments;
        centroids = new_centroids;
    }
    drop(centroids);
    assignments
}

/// K-means++ initialisation: picks first centroid uniformly, then
/// subsequent centroids proportional to squared distance from nearest centroid.
fn kmeans_pp_init(points: &[EhsFeatures], k: usize) -> Vec<EhsFeatures> {
    let mut centroids = vec![points[0]];
    let mut seed: u64 = 0xDEAD_BEEF_CAFE_1234;

    for _ in 1..k {
        let weights = compute_sq_distances(points, &centroids);
        let total: f64 = weights.iter().sum();
        seed = splitmix64(seed);
        // Normalise seed to [0,1]: both casts lose precision but that's fine for sampling
        #[allow(clippy::cast_precision_loss)]
        let threshold = (seed as f64 / u64::MAX as f64) * total;
        let idx = pick_weighted_index(&weights, threshold);
        centroids.push(points[idx]);
    }
    centroids
}

/// For each point, compute its squared distance to the nearest centroid.
fn compute_sq_distances(points: &[EhsFeatures], centroids: &[EhsFeatures]) -> Vec<f64> {
    points
        .iter()
        .map(|p| {
            centroids
                .iter()
                .map(|c| sq_dist(p, c))
                .fold(f64::MAX, f64::min)
        })
        .collect()
}

/// Pick index where cumulative weight first exceeds `threshold`.
fn pick_weighted_index(weights: &[f64], threshold: f64) -> usize {
    let mut cumulative = 0.0;
    for (i, &w) in weights.iter().enumerate() {
        cumulative += w;
        if cumulative >= threshold {
            return i;
        }
    }
    weights.len() - 1
}

/// Assign each point to its nearest centroid.
fn assign_clusters(points: &[EhsFeatures], centroids: &[EhsFeatures]) -> Vec<u16> {
    points
        .iter()
        .map(|p| nearest_centroid(p, centroids))
        .collect()
}

/// Return index of nearest centroid to point `p`.
#[allow(clippy::cast_possible_truncation)]
fn nearest_centroid(p: &EhsFeatures, centroids: &[EhsFeatures]) -> u16 {
    // k ≤ 65535 by type constraint on num_buckets, so cast is safe
    centroids
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| sq_dist(p, a).partial_cmp(&sq_dist(p, b)).unwrap_or(std::cmp::Ordering::Equal))
        .map_or(0, |(i, _)| i as u16)
}

/// Recompute centroids as the mean of assigned points.
fn recompute_centroids(points: &[EhsFeatures], assignments: &[u16], k: usize) -> Vec<EhsFeatures> {
    let (sums, counts) = accumulate_cluster_sums(points, assignments, k);
    (0..k)
        .map(|i| centroid_from_sum(&sums[i], counts[i]))
        .collect()
}

/// Accumulate per-cluster sums and counts.
fn accumulate_cluster_sums(
    points: &[EhsFeatures],
    assignments: &[u16],
    k: usize,
) -> (Vec<[f64; 3]>, Vec<usize>) {
    let mut sums = vec![[0.0f64; 3]; k];
    let mut counts = vec![0usize; k];
    for (p, &a) in points.iter().zip(assignments.iter()) {
        let i = a as usize;
        sums[i][0] += p[0];
        sums[i][1] += p[1];
        sums[i][2] += p[2];
        counts[i] += 1;
    }
    (sums, counts)
}

/// Compute centroid from accumulated sum.
#[allow(clippy::cast_precision_loss)]
fn centroid_from_sum(sum: &[f64; 3], count: usize) -> EhsFeatures {
    if count == 0 {
        return [0.5, 0.0, 0.0];
    }
    // count ≤ num_hands (169); precision loss negligible
    let n = count as f64;
    [sum[0] / n, sum[1] / n, sum[2] / n]
}

/// Squared Euclidean distance in 3D feature space.
fn sq_dist(a: &EhsFeatures, b: &EhsFeatures) -> f64 {
    (0..3).map(|i| (a[i] - b[i]).powi(2)).sum()
}

// ---------------------------------------------------------------------------
// Equity table helpers
// ---------------------------------------------------------------------------

fn build_equity_row(
    bucket_a: usize,
    num_buckets: usize,
    river_boards: &[Vec<Vec<[Card; 5]>>],
) -> Vec<f32> {
    (0..num_buckets)
        .map(|b| sample_bucket_equity(bucket_a, b, river_boards))
        .collect()
}

/// Average equity of `bucket_a` vs `bucket_b` over sampled river boards.
fn sample_bucket_equity(
    bucket_a: usize,
    bucket_b: usize,
    river_boards: &[Vec<Vec<[Card; 5]>>],
) -> f32 {
    let boards = river_boards.get(bucket_a).and_then(|row| row.get(bucket_b));

    match boards {
        None => 0.5,
        Some(boards) if boards.is_empty() => 0.5,
        Some(boards) => {
            let sum: f64 = boards.iter().map(|b| average_river_equity(b)).sum();
            // boards.len() ≤ num_river_boards config; precision loss negligible
            #[allow(clippy::cast_precision_loss)]
            let len = boards.len() as f64;
            #[allow(clippy::cast_possible_truncation)]
            let avg = (sum / len) as f32;
            avg
        }
    }
}

/// Average equity across all non-conflicting hand combos on a single river board.
fn average_river_equity(board: &[Card; 5]) -> f64 {
    let hands: Vec<CanonicalHand> = all_hands().collect();
    let samples: Vec<f64> = hands
        .iter()
        .flat_map(|h| {
            h.combos()
                .into_iter()
                .filter(|&(c1, c2)| !board_conflicts([c1, c2], board))
                .map(|(c1, c2)| ehs_features([c1, c2], board)[0])
        })
        .collect();
    if samples.is_empty() {
        return 0.5;
    }
    #[allow(clippy::cast_precision_loss)]
    let n = samples.len() as f64;
    samples.iter().sum::<f64>() / n
}

// ---------------------------------------------------------------------------
// Shared RNG
// ---------------------------------------------------------------------------

fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

// ---------------------------------------------------------------------------
// Validation
// ---------------------------------------------------------------------------

/// Validate that bucket and texture counts are non-zero.
///
/// # Errors
///
/// Returns `BucketError::ZeroBuckets` or `BucketError::ZeroTextures`.
pub fn validate_buckets(num_buckets: u16, num_textures: u16) -> Result<(), BucketError> {
    if num_buckets == 0 {
        return Err(BucketError::ZeroBuckets(num_buckets));
    }
    if num_textures == 0 {
        return Err(BucketError::ZeroTextures(num_textures));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Diagnostics
// ---------------------------------------------------------------------------

/// Print bucket diagnostics after k-means clustering.
///
/// Shows three sections:
/// 1. Bucket size statistics (avg, min, max, stdev, empty%)
/// 2. Sample hand bucket assignments for texture 0
/// 3. Per-bucket average EHS for texture 0 (top/bottom 5)
#[allow(clippy::cast_precision_loss)]
pub fn log_bucket_diagnostics(
    hands: &[CanonicalHand],
    features: &[Vec<EhsFeatures>],
    assignments: &[Vec<u16>],
    num_buckets: u16,
    prototype_flops: &[[Card; 3]],
) {
    let num_textures = prototype_flops.len();
    let stats = bucket_size_stats(assignments, num_buckets, num_textures);
    let flop_label = format_flop(prototype_flops.first());

    eprintln!("Hand bucket diagnostics ({num_buckets} buckets x {num_textures} textures):");
    eprintln!(
        "  Bucket sizes: avg {:.1}, min {}, max {}, stdev {:.1}, empty {:.1}%",
        stats.avg, stats.min, stats.max, stats.stdev, stats.empty_pct
    );

    eprintln!("  Sample hands (texture 0: {flop_label}):");
    let sample_line = format_sample_hands(hands, assignments);
    eprintln!("    {sample_line}");

    eprintln!("  Bucket EHS spread (texture 0, top/bottom 5):");
    let (top, bottom) = ehs_spread_for_texture(features, assignments, num_buckets as usize, 0);
    let top_str = format_ehs_entries(&top);
    let bottom_str = format_ehs_entries(&bottom);
    eprintln!("    {top_str}");
    eprintln!("    {bottom_str}");
}

struct BucketSizeStats {
    avg: f64,
    min: usize,
    max: usize,
    stdev: f64,
    empty_pct: f64,
}

/// Compute bucket size statistics across all textures.
#[allow(clippy::cast_precision_loss)]
fn bucket_size_stats(
    assignments: &[Vec<u16>],
    num_buckets: u16,
    num_textures: usize,
) -> BucketSizeStats {
    let total_slots = num_buckets as usize * num_textures;
    let mut counts = vec![0usize; num_buckets as usize];

    for hand_assignments in assignments {
        for &bucket in hand_assignments {
            counts[bucket as usize] += 1;
        }
    }

    let min = counts.iter().copied().min().unwrap_or(0);
    let max = counts.iter().copied().max().unwrap_or(0);
    let avg = if counts.is_empty() {
        0.0
    } else {
        counts.iter().sum::<usize>() as f64 / counts.len() as f64
    };
    let variance = if counts.is_empty() {
        0.0
    } else {
        counts.iter().map(|&c| (c as f64 - avg).powi(2)).sum::<f64>() / counts.len() as f64
    };
    let empty_count = counts.iter().filter(|&&c| c == 0).count();
    let empty_pct = if total_slots == 0 {
        0.0
    } else {
        empty_count as f64 / counts.len() as f64 * 100.0
    };

    BucketSizeStats {
        avg,
        min,
        max,
        stdev: variance.sqrt(),
        empty_pct,
    }
}

/// Format a flop as "Kh 7d 2c".
fn format_flop(flop: Option<&[Card; 3]>) -> String {
    match flop {
        Some(cards) => cards.iter().map(|c| format_card(*c)).collect::<Vec<_>>().join(" "),
        None => "none".to_string(),
    }
}

/// Format a single card as e.g. "Kh".
fn format_card(card: Card) -> String {
    let rank = match card.value {
        Value::Ace => 'A',
        Value::King => 'K',
        Value::Queen => 'Q',
        Value::Jack => 'J',
        Value::Ten => 'T',
        Value::Nine => '9',
        Value::Eight => '8',
        Value::Seven => '7',
        Value::Six => '6',
        Value::Five => '5',
        Value::Four => '4',
        Value::Three => '3',
        Value::Two => '2',
    };
    let suit = match card.suit {
        Suit::Spade => 's',
        Suit::Heart => 'h',
        Suit::Diamond => 'd',
        Suit::Club => 'c',
    };
    format!("{rank}{suit}")
}

/// Format sample hands with their bucket assignments for texture 0.
fn format_sample_hands(hands: &[CanonicalHand], assignments: &[Vec<u16>]) -> String {
    const SAMPLE_NAMES: &[&str] = &[
        "AA", "AKs", "AKo", "KQs", "JTs", "T9s", "77", "55", "A5s", "72o",
    ];

    SAMPLE_NAMES
        .iter()
        .filter_map(|name| {
            let target = CanonicalHand::parse(name).ok()?;
            let idx = hands.iter().position(|h| *h == target)?;
            let bucket = assignments.get(idx)?.first()?;
            Some(format!("{name}->b{bucket}"))
        })
        .collect::<Vec<_>>()
        .join("  ")
}

/// Compute per-bucket average EHS for texture 0, return top N and bottom N.
#[allow(clippy::cast_precision_loss)]
fn ehs_spread_for_texture(
    features: &[Vec<EhsFeatures>],
    assignments: &[Vec<u16>],
    num_buckets: usize,
    texture_id: usize,
) -> (Vec<EhsEntry>, Vec<EhsEntry>) {
    let mut sums = vec![0.0f64; num_buckets];
    let mut counts = vec![0u32; num_buckets];

    for (hand_idx, hand_feats) in features.iter().enumerate() {
        if let (Some(feat), Some(tex_assignments)) =
            (hand_feats.get(texture_id), assignments.get(hand_idx))
            && let Some(&bucket) = tex_assignments.get(texture_id)
            && !feat[0].is_nan()
        {
            sums[bucket as usize] += feat[0];
            counts[bucket as usize] += 1;
        }
    }

    let mut entries: Vec<EhsEntry> = sums
        .iter()
        .zip(&counts)
        .enumerate()
        .filter(|&(_, (_, &c))| c > 0)
        .map(|(i, (&s, &c))| EhsEntry {
            bucket: i,
            ehs: s / f64::from(c),
            hand_count: c,
        })
        .collect();

    entries.sort_by(|a, b| b.ehs.partial_cmp(&a.ehs).unwrap_or(std::cmp::Ordering::Equal));

    let n = 5;
    let top: Vec<EhsEntry> = entries.iter().take(n).copied().collect();
    let bottom: Vec<EhsEntry> = entries.iter().rev().take(n).rev().copied().collect();
    (top, bottom)
}

#[derive(Clone, Copy)]
struct EhsEntry {
    bucket: usize,
    ehs: f64,
    hand_count: u32,
}

/// Format EHS entries as "b3: EHS 0.921 (4 hands)  b5: EHS 0.873 (3 hands)".
fn format_ehs_entries(entries: &[EhsEntry]) -> String {
    entries
        .iter()
        .map(|e| {
            let label = if e.hand_count == 1 { "hand" } else { "hands" };
            format!("b{}: EHS {:.3} ({} {label})", e.bucket, e.ehs, e.hand_count)
        })
        .collect::<Vec<_>>()
        .join("    ")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::poker::{Card, Suit, Value};
    use test_macros::timed_test;

    fn card(v: Value, s: Suit) -> Card {
        Card::new(v, s)
    }

    fn sample_flop() -> [Card; 3] {
        [
            card(Value::Two, Suit::Diamond),
            card(Value::Seven, Suit::Club),
            card(Value::King, Suit::Heart),
        ]
    }

    fn sample_turn() -> [Card; 4] {
        [
            card(Value::Two, Suit::Diamond),
            card(Value::Seven, Suit::Club),
            card(Value::King, Suit::Heart),
            card(Value::Five, Suit::Spade),
        ]
    }

    fn sample_river() -> [Card; 5] {
        [
            card(Value::Two, Suit::Diamond),
            card(Value::Seven, Suit::Club),
            card(Value::King, Suit::Heart),
            card(Value::Five, Suit::Spade),
            card(Value::Nine, Suit::Diamond),
        ]
    }

    #[timed_test]
    fn kmeans_single_cluster_returns_all_zeros() {
        let points = vec![[0.2, 0.1, 0.0], [0.8, 0.0, 0.1], [0.5, 0.05, 0.05]];
        let assignments = kmeans(&points, 1, 10);
        assert_eq!(assignments.len(), 3);
        assert!(assignments.iter().all(|&a| a == 0));
    }

    #[timed_test]
    fn kmeans_k_equals_n_each_point_unique() {
        let points = vec![[0.1, 0.0, 0.0], [0.5, 0.0, 0.0], [0.9, 0.0, 0.0]];
        let assignments = kmeans(&points, 3, 50);
        assert_eq!(assignments.len(), 3);
        // All assignments should be distinct cluster IDs
        let mut seen = std::collections::HashSet::new();
        for a in &assignments {
            seen.insert(*a);
        }
        assert_eq!(seen.len(), 3);
    }

    #[timed_test]
    fn kmeans_separates_clearly_distinct_clusters() {
        // Two clearly separated groups
        let mut points = Vec::new();
        for _ in 0..5 {
            points.push([0.1, 0.0, 0.0]);
        }
        for _ in 0..5 {
            points.push([0.9, 0.0, 0.0]);
        }
        let assignments = kmeans(&points, 2, 100);
        // First 5 should all share a cluster, last 5 another
        let first_cluster = assignments[0];
        assert!(assignments[..5].iter().all(|&a| a == first_cluster));
        let second_cluster = assignments[5];
        assert!(assignments[5..].iter().all(|&a| a == second_cluster));
        assert_ne!(first_cluster, second_cluster);
    }

    #[timed_test]
    fn kmeans_empty_returns_empty() {
        let assignments = kmeans(&[], 5, 10);
        assert!(assignments.is_empty());
    }

    #[timed_test]
    fn kmeans_zero_k_returns_empty() {
        let points = vec![[0.5, 0.0, 0.0]];
        let assignments = kmeans(&points, 0, 10);
        assert!(assignments.is_empty());
    }

    #[timed_test]
    fn sq_dist_zero_for_identical_points() {
        let p = [0.5, 0.2, 0.1];
        assert!((sq_dist(&p, &p) - 0.0).abs() < 1e-12);
    }

    #[timed_test]
    fn sq_dist_correct_value() {
        let a = [1.0, 0.0, 0.0];
        let b = [0.0, 0.0, 0.0];
        assert!((sq_dist(&a, &b) - 1.0).abs() < 1e-12);
    }

    #[timed_test]
    fn validate_buckets_zero_buckets_errors() {
        assert!(validate_buckets(0, 5).is_err());
    }

    #[timed_test]
    fn validate_buckets_zero_textures_errors() {
        assert!(validate_buckets(5, 0).is_err());
    }

    #[timed_test]
    fn validate_buckets_valid_passes() {
        assert!(validate_buckets(10, 5).is_ok());
    }

    #[timed_test]
    fn board_conflicts_detects_overlap() {
        let hole = [
            card(Value::Ace, Suit::Spade),
            card(Value::King, Suit::Heart),
        ];
        let board = [
            card(Value::Ace, Suit::Spade),
            card(Value::Two, Suit::Club),
            card(Value::Three, Suit::Diamond),
        ];
        assert!(board_conflicts(hole, &board));
    }

    #[timed_test]
    fn board_conflicts_no_overlap_false() {
        let hole = [
            card(Value::Ace, Suit::Spade),
            card(Value::King, Suit::Heart),
        ];
        let board = [
            card(Value::Two, Suit::Diamond),
            card(Value::Three, Suit::Club),
            card(Value::Four, Suit::Heart),
        ];
        assert!(!board_conflicts(hole, &board));
    }

    #[timed_test(300)]
    #[ignore = "slow"]
    fn build_flop_buckets_returns_correct_shape() {
        let flop = sample_flop();
        let board_samples = vec![vec![flop]];
        let result = build_flop_buckets(2, &board_samples, |_| {});
        assert!(result.is_ok());
        let mapping = result.unwrap();
        assert_eq!(mapping.len(), NUM_HANDS, "one row per canonical hand");
        assert_eq!(mapping[0].len(), 1, "one column per texture");
    }

    #[timed_test]
    fn build_flop_buckets_zero_buckets_errors() {
        let board_samples = vec![vec![sample_flop()]];
        assert!(build_flop_buckets(0, &board_samples, |_| {}).is_err());
    }

    #[timed_test]
    fn build_flop_buckets_zero_textures_errors() {
        assert!(build_flop_buckets(2, &[], |_| {}).is_err());
    }

    #[timed_test(300)]
    #[ignore = "slow"]
    fn build_turn_buckets_returns_correct_shape() {
        let turn = sample_turn();
        let samples = vec![vec![turn]];
        let result = build_turn_buckets(2, 2, &samples);
        assert!(result.is_ok());
        let mapping = result.unwrap();
        assert_eq!(mapping.len(), NUM_HANDS);
        assert_eq!(mapping[0].len(), 1);
    }

    #[timed_test(300)]
    #[ignore = "slow"]
    fn build_river_buckets_returns_correct_shape() {
        let river = sample_river();
        let samples = vec![vec![river]];
        let result = build_river_buckets(2, 2, &samples);
        assert!(result.is_ok());
        let mapping = result.unwrap();
        assert_eq!(mapping.len(), NUM_HANDS);
        assert_eq!(mapping[0].len(), 1);
    }

    #[timed_test]
    fn bucket_equity_get_fallback_for_out_of_bounds() {
        let eq = BucketEquity {
            equity: vec![vec![0.7_f32]],
            num_buckets: 1,
        };
        assert!((eq.get(0, 0) - 0.7).abs() < 1e-6);
        assert!(
            (eq.get(99, 99) - 0.5).abs() < 1e-6,
            "out-of-bounds returns 0.5"
        );
    }

    #[timed_test]
    fn transpose_correct() {
        // 2 textures, 3 hands
        let by_texture = vec![vec![0u16, 1, 0], vec![1, 0, 1]];
        let result = transpose(&by_texture, 3, 2);
        assert_eq!(result.len(), 3);
        assert_eq!(result[0], vec![0, 1]); // hand 0: texture0=0, texture1=1
        assert_eq!(result[1], vec![1, 0]); // hand 1: texture0=1, texture1=0
        assert_eq!(result[2], vec![0, 1]); // hand 2: texture0=0, texture1=1
    }

    #[timed_test]
    fn bucket_ehs_centroids_computes_averages() {
        // 3 hands, 1 texture, 2 buckets
        let features = vec![
            vec![[0.8, 0.0, 0.0]], // hand 0 → bucket 0
            vec![[0.6, 0.0, 0.0]], // hand 1 → bucket 0
            vec![[0.3, 0.0, 0.0]], // hand 2 → bucket 1
        ];
        let assignments = vec![vec![0u16], vec![0], vec![1]];
        let centroids = bucket_ehs_centroids(&features, &assignments, 2);
        assert!((centroids[0] - 0.7).abs() < 1e-9, "bucket 0 avg = 0.7");
        assert!((centroids[1] - 0.3).abs() < 1e-9, "bucket 1 avg = 0.3");
    }

    #[timed_test]
    fn equity_from_centroids_strong_beats_weak() {
        let centroids = vec![0.8, 0.2];
        let eq = build_bucket_equity_from_centroids(&centroids);
        assert!(eq.get(0, 1) > 0.5, "strong bucket should beat weak");
        assert!(eq.get(1, 0) < 0.5, "weak bucket should lose to strong");
        let sum = eq.get(0, 1) + eq.get(1, 0);
        assert!((sum - 1.0).abs() < 1e-6, "equity should sum to 1");
    }

    #[timed_test]
    fn equity_from_centroids_equal_gives_half() {
        let centroids = vec![0.5, 0.5];
        let eq = build_bucket_equity_from_centroids(&centroids);
        assert!((eq.get(0, 1) - 0.5).abs() < 1e-6);
        assert!((eq.get(1, 0) - 0.5).abs() < 1e-6);
    }

    #[timed_test]
    fn equity_from_table_uses_pairwise_data() {
        // 2 buckets, 1 texture. Hand 0 → bucket 0, hand 1 → bucket 1.
        // With uniform equity table (all 0.5), bucket equity should be 0.5.
        let table = EquityTable::new_uniform();
        let assignments = vec![vec![0u16], vec![1u16]];
        let eq = build_bucket_equity_from_equity_table(&assignments, &table, 2);
        assert!(
            (eq.get(0, 1) - 0.5).abs() < 1e-6,
            "uniform equity → bucket equity 0.5, got {}",
            eq.get(0, 1)
        );
    }

    #[timed_test]
    fn count_hands_per_bucket_correct() {
        let hands: Vec<CanonicalHand> = all_hands().collect();
        // 3 hands, 2 textures: hand0→[b0,b1], hand1→[b0,b0], hand2→[b1,b0]
        let assignments = vec![vec![0u16, 1], vec![0, 0], vec![1, 0]];
        let counts = count_hands_per_bucket(&hands, &assignments, 2);
        // bucket 0: hand0×1, hand1×2, hand2×1
        let b0: Vec<usize> = counts[0].iter().map(|&(h, _)| h).collect();
        assert!(b0.contains(&0));
        assert!(b0.contains(&1));
        assert!(b0.contains(&2));
        // hand1 appears twice in bucket 0
        let hand1_count = counts[0].iter().find(|&&(h, _)| h == 1).unwrap().1;
        assert_eq!(hand1_count, 2);
    }

    #[timed_test]
    fn cluster_per_texture_assigns_blocked_hands_by_cross_texture_average() {
        // 5 hands, 2 textures, 2 buckets.
        // Hand 2 is blocked (NaN) on texture 0 but has strong EHS on texture 1.
        // It should be assigned to the strong bucket on texture 0 (not weak).
        let features = vec![
            // Strong hands (EHS ~0.9)
            vec![[0.90, 0.0, 0.0], [0.92, 0.0, 0.0]],
            vec![[0.88, 0.0, 0.0], [0.91, 0.0, 0.0]],
            // Blocked on texture 0, strong on texture 1
            vec![[f64::NAN, f64::NAN, f64::NAN], [0.95, 0.0, 0.0]],
            // Weak hands (EHS ~0.2)
            vec![[0.20, 0.0, 0.0], [0.18, 0.0, 0.0]],
            vec![[0.22, 0.0, 0.0], [0.19, 0.0, 0.0]],
        ];
        let assignments = cluster_per_texture(&features, 2, 2);

        // Hand 2 should be in the same bucket as hands 0,1 on texture 0
        // (the strong cluster), not with hands 3,4 (the weak cluster).
        let strong_bucket_tex0 = assignments[0][0];
        let weak_bucket_tex0 = assignments[3][0];
        assert_ne!(strong_bucket_tex0, weak_bucket_tex0, "should have 2 distinct clusters");
        assert_eq!(
            assignments[2][0], strong_bucket_tex0,
            "blocked strong hand should be assigned to strong bucket, not weak"
        );
    }

    #[timed_test]
    fn cluster_per_texture_relabels_bucket_0_as_strongest() {
        // 6 hands across 2 textures, 3 buckets.
        // Strong group (EHS ~0.9), medium (~0.5), weak (~0.1).
        let features = vec![
            // Strong
            vec![[0.90, 0.0, 0.0], [0.92, 0.0, 0.0]],
            vec![[0.88, 0.0, 0.0], [0.91, 0.0, 0.0]],
            // Medium
            vec![[0.50, 0.0, 0.0], [0.52, 0.0, 0.0]],
            vec![[0.48, 0.0, 0.0], [0.51, 0.0, 0.0]],
            // Weak
            vec![[0.10, 0.0, 0.0], [0.12, 0.0, 0.0]],
            vec![[0.08, 0.0, 0.0], [0.11, 0.0, 0.0]],
        ];
        let assignments = cluster_per_texture(&features, 3, 2);

        // After relabeling: bucket 0 = strong, bucket 1 = medium, bucket 2 = weak
        assert_eq!(assignments[0][0], 0, "strong hand should be bucket 0");
        assert_eq!(assignments[1][0], 0, "strong hand should be bucket 0");
        assert_eq!(assignments[2][0], 1, "medium hand should be bucket 1");
        assert_eq!(assignments[3][0], 1, "medium hand should be bucket 1");
        assert_eq!(assignments[4][0], 2, "weak hand should be bucket 2");
        assert_eq!(assignments[5][0], 2, "weak hand should be bucket 2");

        // Same on texture 1
        assert_eq!(assignments[0][1], 0);
        assert_eq!(assignments[4][1], 2);
    }

    #[timed_test]
    fn bucket_ehs_centroids_skips_nan_features() {
        // 2 hands, 1 texture, 1 bucket. Hand 1 is NaN.
        let features = vec![
            vec![[0.8, 0.0, 0.0]],
            vec![[f64::NAN, f64::NAN, f64::NAN]],
        ];
        let assignments = vec![vec![0u16], vec![0u16]];
        let centroids = bucket_ehs_centroids(&features, &assignments, 1);
        assert!(
            (centroids[0] - 0.8).abs() < 1e-9,
            "NaN should be excluded from centroid: got {}",
            centroids[0]
        );
    }

    #[timed_test]
    fn cluster_global_assigns_high_ehs_to_strong_buckets_across_textures() {
        // 6 hands across 3 textures, 3 buckets.
        // Hand 0-1: strong EHS ~0.9 on all textures
        // Hand 2-3: medium EHS ~0.5 on all textures
        // Hand 4-5: weak EHS ~0.1 on all textures
        // With per-texture k-means, bucket IDs could differ per texture.
        // With global k-means, same-strength hands must get same bucket everywhere.
        let features = vec![
            // Strong
            vec![[0.90, 0.0, 0.0], [0.91, 0.0, 0.0], [0.89, 0.0, 0.0]],
            vec![[0.88, 0.0, 0.0], [0.92, 0.0, 0.0], [0.87, 0.0, 0.0]],
            // Medium
            vec![[0.50, 0.0, 0.0], [0.52, 0.0, 0.0], [0.48, 0.0, 0.0]],
            vec![[0.48, 0.0, 0.0], [0.51, 0.0, 0.0], [0.49, 0.0, 0.0]],
            // Weak
            vec![[0.10, 0.0, 0.0], [0.12, 0.0, 0.0], [0.11, 0.0, 0.0]],
            vec![[0.08, 0.0, 0.0], [0.09, 0.0, 0.0], [0.13, 0.0, 0.0]],
        ];
        let assignments = cluster_global(&features, 3, 3);

        assert_eq!(assignments.len(), 6, "one row per hand");
        assert_eq!(assignments[0].len(), 3, "one column per texture");

        // After relabeling: bucket 0 = strong, bucket 2 = weak.
        // Strong hands should have bucket 0 on ALL textures.
        assert_eq!(assignments[0][0], 0);
        assert_eq!(assignments[0][1], 0);
        assert_eq!(assignments[0][2], 0);
        assert_eq!(assignments[1][0], 0);

        // Medium hands: bucket 1 on all textures.
        assert_eq!(assignments[2][0], 1);
        assert_eq!(assignments[2][1], 1);
        assert_eq!(assignments[3][0], 1);

        // Weak hands: bucket 2 on all textures.
        assert_eq!(assignments[4][0], 2);
        assert_eq!(assignments[4][1], 2);
        assert_eq!(assignments[5][0], 2);
    }

    #[timed_test]
    fn cluster_global_assigns_blocked_hands_to_nearest_centroid() {
        // 5 hands, 2 textures, 2 buckets.
        // Hand 2 is blocked (NaN) on texture 0 but strong on texture 1.
        // Should be assigned to strong bucket on texture 0 via cross-texture average.
        let features = vec![
            vec![[0.90, 0.0, 0.0], [0.92, 0.0, 0.0]],
            vec![[0.88, 0.0, 0.0], [0.91, 0.0, 0.0]],
            vec![[f64::NAN, f64::NAN, f64::NAN], [0.95, 0.0, 0.0]],
            vec![[0.20, 0.0, 0.0], [0.18, 0.0, 0.0]],
            vec![[0.22, 0.0, 0.0], [0.19, 0.0, 0.0]],
        ];
        let assignments = cluster_global(&features, 2, 2);

        let strong_bucket = assignments[0][0];
        let weak_bucket = assignments[3][0];
        assert_ne!(strong_bucket, weak_bucket, "should have 2 distinct clusters");
        assert_eq!(
            assignments[2][0], strong_bucket,
            "blocked strong hand should be assigned to strong bucket"
        );
    }
}
