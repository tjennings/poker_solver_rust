//! Hand bucketing via EHS k-means clustering for postflop abstraction.
//!
//! Provides per-street independent clustering (`StreetBuckets`) and bucket equity
//! tables (`BucketEquity`, `StreetEquity`) for postflop abstraction.

use std::sync::atomic::{AtomicUsize, Ordering};

use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::hands::CanonicalHand;
use crate::poker::Card;
use crate::preflop::ehs::{EhsFeatures, ehs_features, equity_histogram, HISTOGRAM_BINS};

/// Feature type for histogram-based k-means clustering.
pub type HistogramFeatures = [f64; HISTOGRAM_BINS];

/// Number of canonical preflop hands.
pub const NUM_HANDS: usize = 169;

/// River equity table: `equity[bucket_a][bucket_b]` = average equity of `a` vs `b`.
#[derive(Serialize, Deserialize)]
pub struct BucketEquity {
    pub equity: Vec<Vec<f32>>,
    pub num_buckets: usize,
}

/// Per-street equity tables for postflop showdown evaluation.
///
/// Each street has its own bucket equity table dimensioned by that street's
/// bucket count. River showdowns use river equity, flop/turn all-in showdowns
/// use the corresponding street's equity.
#[derive(Serialize, Deserialize)]
pub struct StreetEquity {
    pub flop: BucketEquity,
    pub turn: BucketEquity,
    pub river: BucketEquity,
}

/// Progress phases for `build_street_buckets_independent`.
pub enum BuildProgress {
    /// Flop feature computation progress: `(hands_done, total_hands)`.
    FlopFeatures(usize, usize),
    FlopClustering,
    /// Turn feature computation progress: `(hands_done, total_hands)`.
    TurnFeatures(usize, usize),
    TurnClustering,
    /// River feature computation progress: `(hands_done, total_hands)`.
    RiverFeatures(usize, usize),
    RiverClustering,
}

/// Per-street bucket assignments with imperfect recall.
///
/// Each street is clustered independently. The player "forgets" which bucket
/// they were in on the previous street, spending the full bucket budget on
/// present-state resolution.
///
/// Flop situations are indexed as `hand_idx * num_flop_boards + flop_idx`.
/// Turn and river have their own board-count-dependent indexing.
#[derive(Serialize, Deserialize)]
pub struct StreetBuckets {
    /// `flop[hand_idx * num_flop_boards + flop_idx]` → flop bucket ID
    pub flop: Vec<u16>,
    pub num_flop_buckets: u16,
    /// Number of flop boards used (for 2D indexing into the flat flop vec).
    pub num_flop_boards: usize,
    /// `turn[situation_idx]` → turn bucket ID
    pub turn: Vec<u16>,
    pub num_turn_buckets: u16,
    /// `river[situation_idx]` → river bucket ID
    pub river: Vec<u16>,
    pub num_river_buckets: u16,
}

/// Result of `build_street_buckets_independent`, containing both the bucket
/// assignments and the intermediate features needed to compute bucket-pair equity.
pub struct BucketingResult {
    /// Per-street bucket assignments.
    pub buckets: StreetBuckets,
    /// Flat histogram CDF features from flop clustering, indexed by
    /// `hand_idx * num_flop_boards + flop_idx`.
    pub flop_histograms: Vec<HistogramFeatures>,
    /// Flat histogram CDF features from turn clustering, indexed by
    /// `hand_idx * num_turn_boards + turn_idx`.
    pub turn_histograms: Vec<HistogramFeatures>,
    /// Flat scalar equities from river computation, indexed by
    /// `hand_idx * num_river_boards + river_idx`.
    pub river_equities: Vec<f64>,
}

impl BucketingResult {
    /// Compute per-street bucket-pair equity tables from the intermediate features.
    ///
    /// For flop/turn, extracts average equity from each histogram CDF via
    /// `cdf_to_avg_equity` (NaN-safe), then groups by bucket to derive pair equity.
    /// For river, uses raw scalar equities directly.
    #[must_use]
    pub fn compute_street_equity(&self) -> StreetEquity {
        let flop_avg: Vec<f64> = self
            .flop_histograms
            .iter()
            .map(|cdf| cdf_to_avg_equity(cdf))
            .collect();
        let flop = compute_bucket_pair_equity(
            &self.buckets.flop,
            self.buckets.num_flop_buckets as usize,
            &flop_avg,
        );

        let turn_avg: Vec<f64> = self
            .turn_histograms
            .iter()
            .map(|cdf| cdf_to_avg_equity(cdf))
            .collect();
        let turn = compute_bucket_pair_equity(
            &self.buckets.turn,
            self.buckets.num_turn_buckets as usize,
            &turn_avg,
        );

        let river = compute_bucket_pair_equity(
            &self.buckets.river,
            self.buckets.num_river_buckets as usize,
            &self.river_equities,
        );

        StreetEquity { flop, turn, river }
    }
}

impl StreetBuckets {
    /// Look up the flop bucket for a given situation index.
    #[must_use]
    pub fn flop_bucket(&self, idx: usize) -> u16 {
        self.flop[idx]
    }

    /// Look up the flop bucket for a given `(hand_idx, flop_idx)` pair.
    #[must_use]
    pub fn flop_bucket_for_hand(&self, hand_idx: usize, flop_idx: usize) -> u16 {
        self.flop[hand_idx * self.num_flop_boards + flop_idx]
    }

    /// Look up the turn bucket for a given situation index.
    #[must_use]
    pub fn turn_bucket(&self, idx: usize) -> u16 {
        self.turn[idx]
    }

    /// Look up the river bucket for a given situation index.
    #[must_use]
    pub fn river_bucket(&self, idx: usize) -> u16 {
        self.river[idx]
    }
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

// ---------------------------------------------------------------------------
// Bucket-pair equity from histogram CDFs
// ---------------------------------------------------------------------------

/// Extract average equity from a histogram CDF.
///
/// A CDF `cdf[i]` = P(equity <= (i+1)/N). The average equity is:
///   `avg = (1/N) * sum_{i=0}^{N-1} (1 - cdf[i])`
///
/// This works because `1 - cdf[i]` = P(equity > (i+1)/N), and summing these
/// survival probabilities over equal-width bins gives the mean.
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn cdf_to_avg_equity(cdf: &HistogramFeatures) -> f64 {
    if cdf.iter().any(|c| c.is_nan()) {
        return f64::NAN;
    }
    let n = HISTOGRAM_BINS as f64;
    let sum: f64 = cdf.iter().map(|&c| 1.0 - c).sum();
    sum / n
}

/// Compute bucket-pair equity from per-situation average equities and bucket assignments.
///
/// Groups situations by bucket to get per-bucket centroid equity (average of
/// member equities), then derives pair equity as `c_a / (c_a + c_b)`.
///
/// Situations with NaN equity are skipped.
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn compute_bucket_pair_equity(
    assignments: &[u16],
    num_buckets: usize,
    avg_equities: &[f64],
) -> BucketEquity {
    debug_assert_eq!(assignments.len(), avg_equities.len());

    // Accumulate per-bucket centroid equity.
    let mut sums = vec![0.0f64; num_buckets];
    let mut counts = vec![0u32; num_buckets];

    for (&bucket, &eq) in assignments.iter().zip(avg_equities.iter()) {
        if eq.is_nan() {
            continue;
        }
        let b = bucket as usize;
        sums[b] += eq;
        counts[b] += 1;
    }

    let centroids: Vec<f64> = sums
        .iter()
        .zip(&counts)
        .map(|(&s, &c)| if c > 0 { s / f64::from(c) } else { 0.5 })
        .collect();

    // Derive pair equity: equity(a, b) = c_a / (c_a + c_b).
    let mut equity = vec![vec![0.5f32; num_buckets]; num_buckets];
    for a in 0..num_buckets {
        for b in 0..num_buckets {
            let ca = centroids[a];
            let cb = centroids[b];
            let denom = ca + cb;
            #[allow(clippy::cast_possible_truncation)]
            if denom > 0.0 {
                equity[a][b] = (ca / denom) as f32;
            }
            // If both centroids are 0.0, keep default 0.5.
        }
    }

    BucketEquity { equity, num_buckets }
}

// ---------------------------------------------------------------------------
// Independent per-street clustering pipeline (imperfect recall)
// ---------------------------------------------------------------------------

/// Build independent per-street bucket assignments from canonical boards.
///
/// Each street runs k-means independently on equity histogram CDFs (flop/turn)
/// or raw equity (river). No cross-street dependencies. Implements standard
/// imperfect-recall abstraction as used by Pluribus.
///
/// Returns a `BucketingResult` containing both the bucket assignments and the
/// intermediate features (histograms, equities) needed for computing bucket-pair equity.
#[must_use]
pub fn build_street_buckets_independent(
    hands: &[CanonicalHand],
    flops: &[[Card; 3]],
    num_flop_buckets: u16,
    num_turn_buckets: u16,
    num_river_buckets: u16,
    on_progress: &(impl Fn(BuildProgress) + Sync + Send),
) -> BucketingResult {
    // --- Flop ---
    let num_hands = hands.len();
    on_progress(BuildProgress::FlopFeatures(0, num_hands));
    let flop_features = compute_flop_histograms(hands, flops, &|done| {
        on_progress(BuildProgress::FlopFeatures(done, num_hands));
    });

    on_progress(BuildProgress::FlopClustering);
    let flop_assignments = cluster_histograms(&flop_features, num_flop_buckets);

    // --- Turn ---
    // Sample a subset of flops to keep turn enumeration feasible.
    let max_flop_sample = 50;
    let flop_sample: Vec<&[Card; 3]> = flops.iter().take(max_flop_sample).collect();

    // For each sampled flop, enumerate all 47 live turn cards.
    // Use a dummy hole [Card(0), Card(1)] that won't conflict — we just need
    // the 47 cards not on the flop. Since we only care about board cards,
    // use two cards that are definitely not on any flop.
    let turn_boards: Vec<[Card; 4]> = flop_sample
        .iter()
        .flat_map(|flop| {
            let flop_set: Vec<Card> = flop.to_vec();
            let live: Vec<Card> = all_cards_vec()
                .into_iter()
                .filter(|c| !flop_set.contains(c))
                .collect();
            live.into_iter().map(move |tc| [flop[0], flop[1], flop[2], tc])
        })
        .collect();

    on_progress(BuildProgress::TurnFeatures(0, num_hands));
    let turn_features = compute_turn_histograms(hands, &turn_boards, &|done| {
        on_progress(BuildProgress::TurnFeatures(done, num_hands));
    });

    on_progress(BuildProgress::TurnClustering);
    let turn_assignments = cluster_histograms(&turn_features, num_turn_buckets);

    // --- River ---
    // Sample a subset of turn boards, enumerate all 46 live river cards each.
    let max_turn_sample = 100;
    let turn_sample: Vec<&[Card; 4]> = turn_boards.iter().take(max_turn_sample).collect();

    let river_boards: Vec<[Card; 5]> = turn_sample
        .iter()
        .flat_map(|tb| {
            let board_set: Vec<Card> = tb.to_vec();
            let live: Vec<Card> = all_cards_vec()
                .into_iter()
                .filter(|c| !board_set.contains(c))
                .collect();
            live.into_iter()
                .map(move |rc| [tb[0], tb[1], tb[2], tb[3], rc])
        })
        .collect();

    on_progress(BuildProgress::RiverFeatures(0, num_hands));
    let river_equities = compute_river_equities(hands, &river_boards, &|done| {
        on_progress(BuildProgress::RiverFeatures(done, num_hands));
    });

    on_progress(BuildProgress::RiverClustering);
    let river_assignments = cluster_river_equities(&river_equities, num_river_buckets);

    BucketingResult {
        buckets: StreetBuckets {
            flop: flop_assignments,
            num_flop_buckets,
            num_flop_boards: flops.len(),
            turn: turn_assignments,
            num_turn_buckets,
            river: river_assignments,
            num_river_buckets,
        },
        flop_histograms: flop_features,
        turn_histograms: turn_features,
        river_equities,
    }
}

/// All 52 cards as a Vec (for board enumeration without a hole-card reference).
fn all_cards_vec() -> Vec<Card> {
    use crate::poker::{Suit, Value};
    const VALUES: [Value; 13] = [
        Value::Two, Value::Three, Value::Four, Value::Five, Value::Six,
        Value::Seven, Value::Eight, Value::Nine, Value::Ten,
        Value::Jack, Value::Queen, Value::King, Value::Ace,
    ];
    const SUITS: [Suit; 4] = [Suit::Spade, Suit::Heart, Suit::Diamond, Suit::Club];
    VALUES
        .into_iter()
        .flat_map(|v| SUITS.into_iter().map(move |s| Card::new(v, s)))
        .collect()
}

/// Cluster histogram features into k buckets using k-means with L2 on CDFs.
#[allow(clippy::cast_precision_loss)]
fn cluster_histograms(features: &[HistogramFeatures], k: u16) -> Vec<u16> {
    let valid_indices: Vec<usize> = features
        .iter()
        .enumerate()
        .filter(|(_, f)| !f[0].is_nan())
        .map(|(i, _)| i)
        .collect();
    let valid_points: Vec<HistogramFeatures> =
        valid_indices.iter().map(|&i| features[i]).collect();

    if valid_points.is_empty() {
        return vec![0; features.len()];
    }

    let valid_assignments = kmeans_generic(&valid_points, k as usize, 100);
    let centroids = recompute_centroids_generic(&valid_points, &valid_assignments, k as usize);

    let mut full = vec![0u16; features.len()];
    for (vi, &orig_idx) in valid_indices.iter().enumerate() {
        full[orig_idx] = valid_assignments[vi];
    }

    // Assign blocked (NaN) to nearest centroid using uniform CDF fallback
    let uniform_cdf: HistogramFeatures = {
        let mut cdf = [0.0f64; HISTOGRAM_BINS];
        for (i, v) in cdf.iter_mut().enumerate() {
            *v = (i + 1) as f64 / HISTOGRAM_BINS as f64;
        }
        cdf
    };
    for (i, feat) in features.iter().enumerate() {
        if feat[0].is_nan() {
            full[i] = nearest_centroid_generic(&uniform_cdf, &centroids);
        }
    }
    full
}

/// Cluster scalar river equities into buckets.
///
/// Wraps equity values as 1D k-means (using `[f64; 3]` with padding for compat).
fn cluster_river_equities(equities: &[f64], k: u16) -> Vec<u16> {
    let points: Vec<EhsFeatures> = equities
        .iter()
        .map(|&eq| {
            if eq.is_nan() {
                [f64::NAN, 0.0, 0.0]
            } else {
                [eq, 0.0, 0.0]
            }
        })
        .collect();

    let valid_indices: Vec<usize> = points
        .iter()
        .enumerate()
        .filter(|(_, f)| !f[0].is_nan())
        .map(|(i, _)| i)
        .collect();
    let valid_points: Vec<EhsFeatures> =
        valid_indices.iter().map(|&i| points[i]).collect();

    if valid_points.is_empty() {
        return vec![0; equities.len()];
    }

    let valid_assignments = kmeans(&valid_points, k as usize, 100);
    let centroids = recompute_centroids_generic(&valid_points, &valid_assignments, k as usize);

    let mut full = vec![0u16; equities.len()];
    for (vi, &orig_idx) in valid_indices.iter().enumerate() {
        full[orig_idx] = valid_assignments[vi];
    }
    for (i, &eq) in equities.iter().enumerate() {
        if eq.is_nan() {
            full[i] = nearest_centroid_generic(&[0.5, 0.0, 0.0], &centroids);
        }
    }
    full
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

// ---------------------------------------------------------------------------
// Histogram-based feature computation (for imperfect-recall clustering)
// ---------------------------------------------------------------------------

/// Compute equity histogram CDF features for all (hand, flop) situations.
///
/// Returns flat `Vec<HistogramFeatures>` indexed by `hand_idx * num_flops + flop_idx`.
/// Blocked hands (hole cards conflict with board) get NaN sentinel.
pub fn compute_flop_histograms(
    hands: &[CanonicalHand],
    flops: &[[Card; 3]],
    on_hand_done: &(impl Fn(usize) + Sync + Send),
) -> Vec<HistogramFeatures> {
    let done = AtomicUsize::new(0);
    let per_hand: Vec<Vec<HistogramFeatures>> = hands
        .par_iter()
        .map(|hand| {
            let combos = hand.combos();
            let feats: Vec<HistogramFeatures> = flops
                .iter()
                .map(|flop| {
                    // Find first non-conflicting combo
                    if let Some(&(c1, c2)) = combos.iter().find(|&&(c1, c2)| !board_conflicts([c1, c2], flop)) {
                        equity_histogram(&[c1, c2], flop.as_slice())
                    } else {
                        [f64::NAN; HISTOGRAM_BINS]
                    }
                })
                .collect();
            let count = done.fetch_add(1, Ordering::Relaxed) + 1;
            on_hand_done(count);
            feats
        })
        .collect();

    // Flatten to hand_idx * num_flops + flop_idx
    let num_flops = flops.len();
    let mut flat = Vec::with_capacity(hands.len() * num_flops);
    for hand_feats in &per_hand {
        flat.extend_from_slice(hand_feats);
    }
    flat
}

/// Compute equity histogram CDF features for all (hand, `turn_board`) situations.
///
/// Returns flat `Vec<HistogramFeatures>` indexed by `hand_idx * num_turns + turn_idx`.
/// Blocked hands (hole cards conflict with board) get NaN sentinel.
#[must_use]
pub fn compute_turn_histograms(
    hands: &[CanonicalHand],
    turn_boards: &[[Card; 4]],
    on_hand_done: &(impl Fn(usize) + Sync + Send),
) -> Vec<HistogramFeatures> {
    let done = AtomicUsize::new(0);
    let per_hand: Vec<Vec<HistogramFeatures>> = hands
        .par_iter()
        .map(|hand| {
            let combos = hand.combos();
            let feats: Vec<HistogramFeatures> = turn_boards
                .iter()
                .map(|board| {
                    if let Some(&(c1, c2)) = combos.iter().find(|&&(c1, c2)| !board_conflicts([c1, c2], board)) {
                        equity_histogram(&[c1, c2], board.as_slice())
                    } else {
                        [f64::NAN; HISTOGRAM_BINS]
                    }
                })
                .collect();
            let count = done.fetch_add(1, Ordering::Relaxed) + 1;
            on_hand_done(count);
            feats
        })
        .collect();

    let num_turns = turn_boards.len();
    let mut flat = Vec::with_capacity(hands.len() * num_turns);
    for hand_feats in &per_hand {
        flat.extend_from_slice(hand_feats);
    }
    flat
}

/// Compute scalar equity for all (hand, `river_board`) situations.
///
/// Returns flat `Vec<f64>` indexed by `hand_idx * num_rivers + river_idx`.
/// Blocked hands get NaN sentinel.
#[must_use]
pub fn compute_river_equities(
    hands: &[CanonicalHand],
    river_boards: &[[Card; 5]],
    on_hand_done: &(impl Fn(usize) + Sync + Send),
) -> Vec<f64> {
    let done = AtomicUsize::new(0);
    let per_hand: Vec<Vec<f64>> = hands
        .par_iter()
        .map(|hand| {
            let combos = hand.combos();
            let equities: Vec<f64> = river_boards
                .iter()
                .map(|board| {
                    if let Some(&(c1, c2)) = combos.iter().find(|&&(c1, c2)| !board_conflicts([c1, c2], board)) {
                        ehs_features([c1, c2], board.as_slice())[0]
                    } else {
                        f64::NAN
                    }
                })
                .collect();
            let count = done.fetch_add(1, Ordering::Relaxed) + 1;
            on_hand_done(count);
            equities
        })
        .collect();

    let num_rivers = river_boards.len();
    let mut flat = Vec::with_capacity(hands.len() * num_rivers);
    for hand_equities in &per_hand {
        flat.extend_from_slice(hand_equities);
    }
    flat
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

// ---------------------------------------------------------------------------
// Generic k-means trait and implementations
// ---------------------------------------------------------------------------

/// Trait for types that can be used as k-means feature vectors.
pub(crate) trait KmeansPoint: Copy + Send + Sync + PartialEq {
    /// Number of dimensions in this feature vector.
    fn dims(&self) -> usize;
    /// Get the value at dimension `i`.
    fn get(&self, i: usize) -> f64;
    /// Return the zero/origin point.
    fn zero() -> Self;
    /// Set the value at dimension `i`.
    fn set(&mut self, i: usize, val: f64);
}

impl KmeansPoint for [f64; 3] {
    fn dims(&self) -> usize { 3 }
    fn get(&self, i: usize) -> f64 { self[i] }
    fn zero() -> Self { [0.0; 3] }
    fn set(&mut self, i: usize, val: f64) { self[i] = val; }
}

impl KmeansPoint for [f64; 10] {
    fn dims(&self) -> usize { 10 }
    fn get(&self, i: usize) -> f64 { self[i] }
    fn zero() -> Self { [0.0; 10] }
    fn set(&mut self, i: usize, val: f64) { self[i] = val; }
}

// ---------------------------------------------------------------------------
// Generic k-means functions
// ---------------------------------------------------------------------------

/// Generic k-means clustering returning per-point cluster assignments.
///
/// Initialises centroids via k-means++ seeding then iterates until stable or `max_iter`.
#[must_use]
pub(crate) fn kmeans_generic<P: KmeansPoint>(points: &[P], k: usize, max_iter: usize) -> Vec<u16> {
    if points.is_empty() || k == 0 {
        return vec![];
    }
    let k = k.min(points.len());
    let mut centroids = kmeans_pp_init_generic(points, k);
    let mut assignments = assign_clusters_generic(points, &centroids);

    for _ in 0..max_iter {
        let new_centroids = recompute_centroids_generic(points, &assignments, k);
        let new_assignments = assign_clusters_generic(points, &new_centroids);
        if new_assignments == assignments {
            break;
        }
        assignments = new_assignments;
        centroids = new_centroids;
    }
    drop(centroids);
    assignments
}

/// Generic k-means++ initialisation: picks first centroid uniformly, then
/// subsequent centroids proportional to squared distance from nearest centroid.
fn kmeans_pp_init_generic<P: KmeansPoint>(points: &[P], k: usize) -> Vec<P> {
    let mut centroids = vec![points[0]];
    let mut seed: u64 = 0xDEAD_BEEF_CAFE_1234;

    for _ in 1..k {
        let weights = compute_sq_distances_generic(points, &centroids);
        let total: f64 = weights.iter().sum();
        seed = splitmix64(seed);
        #[allow(clippy::cast_precision_loss)]
        let threshold = (seed as f64 / u64::MAX as f64) * total;
        let idx = pick_weighted_index(&weights, threshold);
        centroids.push(points[idx]);
    }
    centroids
}

/// For each point, compute its squared distance to the nearest centroid (generic).
fn compute_sq_distances_generic<P: KmeansPoint>(points: &[P], centroids: &[P]) -> Vec<f64> {
    points
        .iter()
        .map(|p| {
            centroids
                .iter()
                .map(|c| sq_dist_generic(p, c))
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

/// Assign each point to its nearest centroid (generic).
pub(crate) fn assign_clusters_generic<P: KmeansPoint>(points: &[P], centroids: &[P]) -> Vec<u16> {
    points
        .iter()
        .map(|p| nearest_centroid_generic(p, centroids))
        .collect()
}

/// Return index of nearest centroid to point `p` (generic).
#[allow(clippy::cast_possible_truncation)]
pub(crate) fn nearest_centroid_generic<P: KmeansPoint>(p: &P, centroids: &[P]) -> u16 {
    centroids
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| {
            sq_dist_generic(p, a)
                .partial_cmp(&sq_dist_generic(p, b))
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .map_or(0, |(i, _)| i as u16)
}

/// Recompute centroids as the mean of assigned points (generic).
pub(crate) fn recompute_centroids_generic<P: KmeansPoint>(
    points: &[P],
    assignments: &[u16],
    k: usize,
) -> Vec<P> {
    let dims = if points.is_empty() {
        // Fall back: create a zero point to query dims
        P::zero().dims()
    } else {
        points[0].dims()
    };

    let mut sums: Vec<P> = vec![P::zero(); k];
    let mut counts = vec![0usize; k];

    for (p, &a) in points.iter().zip(assignments.iter()) {
        let ci = a as usize;
        for d in 0..dims {
            let cur = sums[ci].get(d);
            sums[ci].set(d, cur + p.get(d));
        }
        counts[ci] += 1;
    }

    (0..k)
        .map(|i| {
            if counts[i] == 0 {
                // Empty cluster: place centroid at midpoint of first dimension
                let mut mid = P::zero();
                mid.set(0, 0.5);
                mid
            } else {
                #[allow(clippy::cast_precision_loss)]
                let n = counts[i] as f64;
                let mut centroid = P::zero();
                for d in 0..dims {
                    centroid.set(d, sums[i].get(d) / n);
                }
                centroid
            }
        })
        .collect()
}

/// Squared Euclidean distance for generic feature vectors.
pub(crate) fn sq_dist_generic<P: KmeansPoint>(a: &P, b: &P) -> f64 {
    let dims = a.dims();
    (0..dims).map(|i| (a.get(i) - b.get(i)).powi(2)).sum()
}

// ---------------------------------------------------------------------------
// Concrete k-means functions (delegate to generics for backward compat)
// ---------------------------------------------------------------------------

/// K-means clustering returning per-point cluster assignments.
///
/// Initialises centroids via k-means++ seeding then iterates until stable or `max_iter`.
#[must_use]
pub(crate) fn kmeans(points: &[EhsFeatures], k: usize, max_iter: usize) -> Vec<u16> {
    kmeans_generic(points, k, max_iter)
}

/// Squared Euclidean distance in 3D feature space.
#[cfg(test)]
fn sq_dist(a: &EhsFeatures, b: &EhsFeatures) -> f64 {
    sq_dist_generic(a, b)
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hands::all_hands;
    use crate::poker::{Card, Suit, Value};
    use test_macros::timed_test;

    fn card(v: Value, s: Suit) -> Card {
        Card::new(v, s)
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
    fn street_buckets_lookup_returns_correct_bucket() {
        let sb = StreetBuckets {
            flop: vec![0, 1, 2, 0, 1],
            num_flop_buckets: 3,
            num_flop_boards: 5,
            turn: vec![1, 0, 2, 1],
            num_turn_buckets: 3,
            river: vec![0, 0, 1, 1, 2, 2],
            num_river_buckets: 3,
        };
        assert_eq!(sb.flop_bucket(0), 0);
        assert_eq!(sb.flop_bucket(2), 2);
        assert_eq!(sb.turn_bucket(1), 0);
        assert_eq!(sb.river_bucket(5), 2);
    }

    #[timed_test]
    fn kmeans_generic_works_with_10d_vectors() {
        let low: [f64; 10] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0];
        let high: [f64; 10] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.2, 0.3, 0.4, 1.0];
        let points = vec![low, low, low, high, high, high];
        let assignments = kmeans_generic(&points, 2, 100);
        assert_eq!(assignments.len(), 6);
        assert_eq!(assignments[0], assignments[1]);
        assert_eq!(assignments[1], assignments[2]);
        assert_eq!(assignments[3], assignments[4]);
        assert_eq!(assignments[4], assignments[5]);
        assert_ne!(assignments[0], assignments[3]);
    }

    #[timed_test]
    fn kmeans_existing_still_works_after_generics() {
        // Existing 3D test still works through the wrapper
        let points = vec![
            [0.1, 0.0, 0.0],
            [0.9, 0.0, 0.0],
            [0.15, 0.0, 0.0],
            [0.85, 0.0, 0.0],
        ];
        let assignments = kmeans(&points, 2, 100);
        assert_eq!(assignments[0], assignments[2]); // low cluster
        assert_eq!(assignments[1], assignments[3]); // high cluster
        assert_ne!(assignments[0], assignments[1]);
    }

    #[timed_test]
    fn sq_dist_generic_matches_concrete() {
        let a = [1.0, 0.5, 0.2];
        let b = [0.3, 0.1, 0.9];
        let concrete = sq_dist(&a, &b);
        let generic = sq_dist_generic(&a, &b);
        assert!((concrete - generic).abs() < 1e-12);
    }

    #[timed_test]
    fn sq_dist_generic_10d() {
        let a: [f64; 10] = [1.0; 10];
        let b: [f64; 10] = [0.0; 10];
        let d = sq_dist_generic(&a, &b);
        assert!((d - 10.0).abs() < 1e-12);
    }

    #[timed_test(300)]
    #[ignore = "slow"]
    fn compute_flop_histograms_produces_valid_cdfs() {
        use crate::preflop::ehs::{canonical_flops, HISTOGRAM_BINS};
        let hands: Vec<CanonicalHand> = all_hands().collect();
        let flops = canonical_flops();
        let small_flops: Vec<[Card; 3]> = flops.into_iter().take(2).collect();
        let histograms = compute_flop_histograms(&hands, &small_flops, &|_| {});

        assert_eq!(histograms.len(), hands.len() * small_flops.len());

        // Every non-NaN histogram should be a valid CDF
        for hist in &histograms {
            if hist[0].is_nan() {
                continue;
            }
            for i in 1..HISTOGRAM_BINS {
                assert!(
                    hist[i] >= hist[i - 1] - 1e-9,
                    "CDF not monotonic: {:?}",
                    hist
                );
            }
            assert!(
                (hist[HISTOGRAM_BINS - 1] - 1.0).abs() < 1e-6,
                "CDF must end at 1.0: {:?}",
                hist
            );
        }
    }

    #[timed_test]
    fn compute_river_equities_basic() {
        let hands: Vec<CanonicalHand> = all_hands().collect();
        let river = [sample_river()];
        let equities = compute_river_equities(&hands, &river, &|_| {});

        assert_eq!(equities.len(), hands.len());

        let mut valid_count = 0;
        for &eq in &equities {
            if eq.is_nan() {
                continue;
            }
            valid_count += 1;
            assert!(eq >= 0.0 && eq <= 1.0, "equity out of range: {eq}");
        }
        assert!(valid_count > 100, "too few valid equities: {valid_count}");
    }

    // -----------------------------------------------------------------------
    // cdf_to_avg_equity tests
    // -----------------------------------------------------------------------

    #[timed_test]
    fn cdf_to_avg_equity_nan_guard() {
        let cdf = [f64::NAN; HISTOGRAM_BINS];
        assert!(cdf_to_avg_equity(&cdf).is_nan(), "all-NaN CDF should return NaN");

        let mut partial = [0.0f64; HISTOGRAM_BINS];
        partial[3] = f64::NAN;
        assert!(cdf_to_avg_equity(&partial).is_nan(), "any NaN element should return NaN");
    }

    #[timed_test]
    fn cdf_to_avg_equity_uniform_returns_half() {
        // Uniform CDF: cdf[i] = (i+1) / N
        let mut cdf = [0.0f64; HISTOGRAM_BINS];
        for (i, v) in cdf.iter_mut().enumerate() {
            *v = (i + 1) as f64 / HISTOGRAM_BINS as f64;
        }
        let avg = cdf_to_avg_equity(&cdf);
        assert!(
            (avg - 0.45).abs() < 1e-9,
            "uniform CDF avg equity should be 0.45 (midpoint of discrete bins), got {avg}"
        );
    }

    #[timed_test]
    fn cdf_to_avg_equity_all_ones_returns_zero() {
        // CDF is all 1.0 — all mass at equity = 0. avg = 0.
        let cdf = [1.0f64; HISTOGRAM_BINS];
        let avg = cdf_to_avg_equity(&cdf);
        assert!(
            avg.abs() < 1e-9,
            "all-ones CDF should give avg equity 0, got {avg}"
        );
    }

    #[timed_test]
    fn cdf_to_avg_equity_step_at_last_bin_returns_high() {
        // CDF is 0.0 for all bins except last which is 1.0.
        // This means all mass is in the highest bin.
        let mut cdf = [0.0f64; HISTOGRAM_BINS];
        cdf[HISTOGRAM_BINS - 1] = 1.0;
        let avg = cdf_to_avg_equity(&cdf);
        // sum of (1 - cdf[i]) = (N-1)*1.0 + 0.0 = N-1
        // avg = (N-1)/N = 0.9
        let expected = (HISTOGRAM_BINS - 1) as f64 / HISTOGRAM_BINS as f64;
        assert!(
            (avg - expected).abs() < 1e-9,
            "step-at-last CDF should give avg {expected}, got {avg}"
        );
    }

    // -----------------------------------------------------------------------
    // compute_bucket_pair_equity tests
    // -----------------------------------------------------------------------

    #[timed_test]
    fn compute_bucket_pair_equity_basic_ordering() {
        // 4 situations, 2 buckets. Bucket 0 has high equity, bucket 1 has low equity.
        let assignments = vec![0u16, 0, 1, 1];
        let avg_equities = vec![0.8, 0.9, 0.2, 0.3];
        let eq = compute_bucket_pair_equity(&assignments, 2, &avg_equities);

        // Centroid 0 = 0.85, centroid 1 = 0.25
        // equity(0,1) = 0.85 / (0.85 + 0.25) = 0.85/1.1 ≈ 0.7727
        // equity(1,0) = 0.25 / (0.25 + 0.85) = 0.25/1.1 ≈ 0.2273
        assert!(
            eq.get(0, 1) > 0.7,
            "strong bucket vs weak should have high equity, got {}",
            eq.get(0, 1)
        );
        assert!(
            eq.get(1, 0) < 0.3,
            "weak bucket vs strong should have low equity, got {}",
            eq.get(1, 0)
        );
    }

    #[timed_test]
    fn compute_bucket_pair_equity_symmetric() {
        // equity(a, b) + equity(b, a) should equal 1.0
        let assignments = vec![0u16, 0, 1, 1, 2, 2];
        let avg_equities = vec![0.9, 0.8, 0.5, 0.5, 0.2, 0.1];
        let eq = compute_bucket_pair_equity(&assignments, 3, &avg_equities);

        for a in 0..3 {
            for b in 0..3 {
                let sum = eq.get(a, b) + eq.get(b, a);
                assert!(
                    (sum - 1.0).abs() < 1e-5,
                    "equity({a},{b}) + equity({b},{a}) should = 1.0, got {sum}"
                );
            }
        }
    }

    #[timed_test]
    fn compute_bucket_pair_equity_self_is_half() {
        let assignments = vec![0u16, 1, 2];
        let avg_equities = vec![0.8, 0.5, 0.2];
        let eq = compute_bucket_pair_equity(&assignments, 3, &avg_equities);

        for b in 0..3 {
            assert!(
                (eq.get(b, b) - 0.5).abs() < 1e-6,
                "self-equity should be 0.5, got {} for bucket {b}",
                eq.get(b, b)
            );
        }
    }

    #[timed_test]
    fn compute_bucket_pair_equity_skips_nan() {
        // One NaN situation should be excluded from centroid calculation.
        let assignments = vec![0u16, 0, 1];
        let avg_equities = vec![0.8, f64::NAN, 0.3];
        let eq = compute_bucket_pair_equity(&assignments, 2, &avg_equities);

        // Centroid 0 = 0.8 (NaN skipped), centroid 1 = 0.3
        // equity(0,1) = 0.8 / 1.1 ≈ 0.7273
        let expected = (0.8 / 1.1) as f32;
        assert!(
            (eq.get(0, 1) - expected).abs() < 1e-5,
            "NaN should be skipped, expected {expected}, got {}",
            eq.get(0, 1)
        );
    }

    #[timed_test]
    fn compute_bucket_pair_equity_empty_bucket_defaults_half() {
        // Bucket 1 has no assignments.
        let assignments = vec![0u16, 0];
        let avg_equities = vec![0.7, 0.8];
        let eq = compute_bucket_pair_equity(&assignments, 2, &avg_equities);

        // Centroid 0 = 0.75, centroid 1 = 0.5 (default)
        // equity(0,1) = 0.75 / 1.25 = 0.6
        assert!(
            (eq.get(0, 1) - 0.6).abs() < 1e-5,
            "empty bucket should default to 0.5 centroid, got equity {}",
            eq.get(0, 1)
        );
    }

    #[timed_test(300)]
    #[ignore = "slow"]
    fn build_street_buckets_independent_produces_valid_assignments() {
        use crate::preflop::ehs::canonical_flops;
        let hands: Vec<CanonicalHand> = all_hands().collect();
        let flops = canonical_flops();
        let small_flops: Vec<[Card; 3]> = flops.into_iter().take(3).collect();

        let result = build_street_buckets_independent(
            &hands, &small_flops, 5, 5, 5, &|_| {},
        );
        let buckets = &result.buckets;

        // All flop bucket IDs should be in range
        for &b in &buckets.flop {
            assert!(b < buckets.num_flop_buckets, "flop bucket {b} >= {}", buckets.num_flop_buckets);
        }
        // Flop situation count = hands x flops
        assert_eq!(buckets.flop.len(), hands.len() * small_flops.len());
        // Turn and river should also have valid assignments
        assert!(!buckets.turn.is_empty(), "turn buckets should not be empty");
        assert!(!buckets.river.is_empty(), "river buckets should not be empty");
        for &b in &buckets.turn {
            assert!(b < buckets.num_turn_buckets);
        }
        for &b in &buckets.river {
            assert!(b < buckets.num_river_buckets);
        }
    }
}
