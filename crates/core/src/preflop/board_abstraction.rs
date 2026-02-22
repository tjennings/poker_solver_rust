//! Board abstraction for the preflop postflop model.
//!
//! Clusters the 1,755 canonical flops into texture buckets, then defines
//! turn and river transition types for each bucket.  These abstractions let
//! the postflop model sample a small finite set of board progressions rather
//! than every possible runout.
//!
//! # Architecture
//!
//! ```text
//! all_flops() → feature vectors → k-means → FlopTexture buckets
//!                                               ↓
//!                                 turn card features → k-means → TurnTransition
//!                                               ↓
//!                               river card features → k-means → RiverTransition
//! ```

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::card_utils::value_rank;
use crate::flops::{CanonicalFlop, RankTexture, SuitTexture, all_flops};
use crate::poker::{Card, Suit, Value};

// ──────────────────────────────────────────────────────────────────────────────
// Error type
// ──────────────────────────────────────────────────────────────────────────────

/// Errors that can occur when building a `BoardAbstraction`.
#[derive(Debug, Error)]
pub enum BoardAbstractionError {
    #[error("num_flop_textures must be at least 1, got {0}")]
    InvalidFlopTextures(u16),
    #[error("num_turn_transitions must be at least 1, got {0}")]
    InvalidTurnTransitions(u16),
    #[error("num_river_transitions must be at least 1, got {0}")]
    InvalidRiverTransitions(u16),
}

// ──────────────────────────────────────────────────────────────────────────────
// Public types
// ──────────────────────────────────────────────────────────────────────────────

/// A cluster of suit/rank-similar flops.
///
/// All concrete flops assigned to this cluster share a strategy in the
/// postflop model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FlopTexture {
    /// Cluster index (0-based).
    pub id: u16,
    /// Fraction of raw flop combos represented by this cluster.
    pub weight: f64,
    /// Dominant suit texture: 0=rainbow, 1=two-tone, 2=monotone.
    pub flush_type: u8,
    /// Dominant connectivity: 0=disconnected, 1=one-gap, 2=two-gap, 3=connected.
    pub connectivity: u8,
    /// Rank of the highest card (2–14).
    pub high_card: u8,
    /// Dominant pairing: 0=unpaired, 1=paired, 2=trips.
    pub pairing: u8,
}

/// How a turn card changes the board relative to a flop texture cluster.
#[allow(clippy::struct_excessive_bools)]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TurnTransition {
    /// Cluster index (0-based within the parent flop texture).
    pub id: u16,
    /// Fraction of turn cards (out of 49) that land in this cluster.
    pub weight: f64,
    /// This turn card completes a flush (gives three-of-a-suit a fourth).
    pub completes_flush: bool,
    /// This turn card pairs one of the flop ranks.
    pub pairs_board: bool,
    /// Adding this turn card makes a straight possible within the board ranks.
    pub enables_straight: bool,
    /// This turn card is higher than every flop card.
    pub overcard: bool,
}

/// How a river card changes the board relative to a turn state.
#[allow(clippy::struct_excessive_bools)]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiverTransition {
    /// Cluster index (0-based within the parent turn state).
    pub id: u16,
    /// Fraction of river cards (out of 48) that land in this cluster.
    pub weight: f64,
    /// This river card completes a flush.
    pub completes_flush: bool,
    /// This river card pairs the board.
    pub pairs_board: bool,
    /// Adding this river card makes a straight possible within the board ranks.
    pub enables_straight: bool,
    /// This river card is higher than all previous board cards.
    pub overcard: bool,
}

/// Precomputed board abstraction: flop texture buckets with turn/river transitions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BoardAbstraction {
    /// All flop texture clusters.
    pub flop_textures: Vec<FlopTexture>,
    /// `turn_transitions[i]` is the set of turn transition clusters for flop texture `i`.
    pub turn_transitions: Vec<Vec<TurnTransition>>,
    /// `river_transitions[i][j]` is the river transitions for flop texture `i`, turn transition `j`.
    pub river_transitions: Vec<Vec<Vec<RiverTransition>>>,
    /// One representative 3-card board per flop texture, for hand bucketing.
    pub prototype_flops: Vec<[Card; 3]>,
}

// ──────────────────────────────────────────────────────────────────────────────
// Configuration
// ──────────────────────────────────────────────────────────────────────────────

/// Configuration for board abstraction granularity.
#[derive(Debug, Clone)]
pub struct BoardAbstractionConfig {
    /// Number of flop texture clusters.
    pub num_flop_textures: u16,
    /// Number of turn transition clusters per flop texture.
    pub num_turn_transitions: u16,
    /// Number of river transition clusters per turn state.
    pub num_river_transitions: u16,
    /// Maximum k-means iterations.
    pub kmeans_max_iter: usize,
}

impl Default for BoardAbstractionConfig {
    fn default() -> Self {
        Self {
            num_flop_textures: 200,
            num_turn_transitions: 10,
            num_river_transitions: 10,
            kmeans_max_iter: 50,
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Orchestration
// ──────────────────────────────────────────────────────────────────────────────

impl BoardAbstraction {
    /// Build the full board abstraction from a configuration.
    ///
    /// This is the public API for this module.  It enumerates canonical flops,
    /// clusters them, then builds turn and river transitions for each cluster.
    ///
    /// # Errors
    ///
    /// Returns an error if any configuration value is zero.
    pub fn build(config: &BoardAbstractionConfig) -> Result<Self, BoardAbstractionError> {
        validate_config(config)?;

        let flops = all_flops();
        let total_weight: u32 = flops.iter().map(|f| u32::from(f.weight())).sum();

        let features: Vec<Vec<f64>> = flops.iter().map(flop_feature_vector).collect();
        let weights: Vec<f64> = flops
            .iter()
            .map(|f| f64::from(f.weight()) / f64::from(total_weight))
            .collect();

        let assignments = weighted_kmeans(
            &features,
            &weights,
            config.num_flop_textures as usize,
            config.kmeans_max_iter,
        );

        let flop_textures =
            build_flop_textures(&flops, &assignments, &weights, config.num_flop_textures);
        let (prototype_flops, turn_transitions, river_transitions) =
            build_all_transitions(&flop_textures, config);

        Ok(Self {
            flop_textures,
            turn_transitions,
            river_transitions,
            prototype_flops,
        })
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Validation
// ──────────────────────────────────────────────────────────────────────────────

fn validate_config(config: &BoardAbstractionConfig) -> Result<(), BoardAbstractionError> {
    if config.num_flop_textures == 0 {
        return Err(BoardAbstractionError::InvalidFlopTextures(0));
    }
    if config.num_turn_transitions == 0 {
        return Err(BoardAbstractionError::InvalidTurnTransitions(0));
    }
    if config.num_river_transitions == 0 {
        return Err(BoardAbstractionError::InvalidRiverTransitions(0));
    }
    Ok(())
}

// ──────────────────────────────────────────────────────────────────────────────
// Flop feature extraction
// ──────────────────────────────────────────────────────────────────────────────

/// Feature vector for a canonical flop: `[flush_type, connectivity, high_card_norm, pairing]`.
///
/// All dimensions are normalised to [0, 1] so that k-means distances are balanced.
fn flop_feature_vector(flop: &CanonicalFlop) -> Vec<f64> {
    let flush = flush_type_score(flop.suit_texture()) / 2.0;
    let conn = connectivity_score(
        flop.connectedness().gap_high_mid,
        flop.connectedness().gap_mid_low,
    ) / 3.0;
    let high = (f64::from(
        flop.cards()
            .iter()
            .map(|c| value_rank(c.value))
            .max()
            .unwrap_or(2),
    ) - 2.0)
        / 12.0;
    let pair = pairing_score(flop.rank_texture()) / 2.0;
    vec![flush, conn, high, pair]
}

fn flush_type_score(suit: SuitTexture) -> f64 {
    match suit {
        SuitTexture::Rainbow => 0.0,
        SuitTexture::TwoTone => 1.0,
        SuitTexture::Monotone => 2.0,
    }
}

fn pairing_score(rank: RankTexture) -> f64 {
    match rank {
        RankTexture::Unpaired => 0.0,
        RankTexture::Paired => 1.0,
        RankTexture::Trips => 2.0,
    }
}

/// Map gap sizes to a connectivity score (0 = connected, 3 = very disconnected).
fn connectivity_score(gap_high_mid: u8, gap_mid_low: u8) -> f64 {
    let total_gap = u16::from(gap_high_mid) + u16::from(gap_mid_low);
    match total_gap {
        0 => 3.0,
        1 | 2 => 2.0,
        3 | 4 => 1.0,
        _ => 0.0,
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// K-means
// ──────────────────────────────────────────────────────────────────────────────

/// Assign each point to one of `k` clusters using weighted k-means.
///
/// Initial centroids are chosen as the `k` most-spread points (simple spread
/// initialisation without random sampling, ensuring determinism).
/// Returns a cluster assignment for each point.
///
/// # Panics
///
/// Panics if `points.len() != weights.len()`.
#[must_use]
pub fn weighted_kmeans(
    points: &[Vec<f64>],
    weights: &[f64],
    k: usize,
    max_iter: usize,
) -> Vec<usize> {
    assert_eq!(points.len(), weights.len());
    if points.is_empty() || k == 0 {
        return vec![];
    }
    let k = k.min(points.len());
    let mut centroids = init_centroids(points, k);
    let mut assignments = vec![0usize; points.len()];
    for _ in 0..max_iter {
        let new_assignments = assign_to_centroids(points, &centroids);
        if new_assignments == assignments {
            break;
        }
        assignments = new_assignments;
        centroids = recompute_centroids(points, weights, &assignments, k);
    }
    assignments
}

/// Initialise `k` centroids by deterministic spread: first centroid is point 0,
/// then each subsequent centroid is the point farthest from all existing centroids.
fn init_centroids(points: &[Vec<f64>], k: usize) -> Vec<Vec<f64>> {
    let mut centroids = vec![points[0].clone()];
    while centroids.len() < k {
        let next = points
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| {
                let da = min_dist_sq(a, &centroids);
                let db = min_dist_sq(b, &centroids);
                da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
            })
            .map_or(0, |(i, _)| i);
        centroids.push(points[next].clone());
    }
    centroids
}

fn min_dist_sq(point: &[f64], centroids: &[Vec<f64>]) -> f64 {
    centroids
        .iter()
        .map(|c| euclidean_dist_sq(point, c))
        .fold(f64::INFINITY, f64::min)
}

fn assign_to_centroids(points: &[Vec<f64>], centroids: &[Vec<f64>]) -> Vec<usize> {
    points
        .iter()
        .map(|p| nearest_centroid(p, centroids))
        .collect()
}

fn nearest_centroid(point: &[f64], centroids: &[Vec<f64>]) -> usize {
    centroids
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| {
            euclidean_dist_sq(point, a)
                .partial_cmp(&euclidean_dist_sq(point, b))
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .map_or(0, |(i, _)| i)
}

fn euclidean_dist_sq(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)).sum()
}

/// Round an `f64` to the nearest `u8`, clamping to [0, 255].
fn round_to_u8(v: f64) -> u8 {
    // Value is clamped to [0, 255] before cast — truncation and sign loss cannot occur.
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    {
        v.round().clamp(0.0, 255.0) as u8
    }
}

fn recompute_centroids(
    points: &[Vec<f64>],
    weights: &[f64],
    assignments: &[usize],
    k: usize,
) -> Vec<Vec<f64>> {
    let dim = points.first().map_or(0, Vec::len);
    let mut sums = vec![vec![0.0f64; dim]; k];
    let mut totals = vec![0.0f64; k];

    for (i, assignment) in assignments.iter().enumerate() {
        let w = weights[i];
        totals[*assignment] += w;
        for (d, val) in points[i].iter().enumerate() {
            sums[*assignment][d] += val * w;
        }
    }

    for (s, &t) in sums.iter_mut().zip(totals.iter()) {
        if t > 0.0 {
            for v in s.iter_mut() {
                *v /= t;
            }
        }
    }

    sums
}

// ──────────────────────────────────────────────────────────────────────────────
// Flop texture construction
// ──────────────────────────────────────────────────────────────────────────────

fn build_flop_textures(
    flops: &[CanonicalFlop],
    assignments: &[usize],
    weights: &[f64],
    k: u16,
) -> Vec<FlopTexture> {
    let k = k as usize;
    let mut cluster_weight = vec![0.0f64; k];
    let mut cluster_flush = vec![0.0f64; k];
    let mut cluster_conn = vec![0.0f64; k];
    let mut cluster_high = vec![0.0f64; k];
    let mut cluster_pair = vec![0.0f64; k];

    for (i, flop) in flops.iter().enumerate() {
        let c = assignments[i];
        let w = weights[i];
        cluster_weight[c] += w;
        cluster_flush[c] += flush_type_score(flop.suit_texture()) * w;
        cluster_conn[c] += connectivity_score(
            flop.connectedness().gap_high_mid,
            flop.connectedness().gap_mid_low,
        ) * w;
        cluster_high[c] += f64::from(
            flop.cards()
                .iter()
                .map(|ca| value_rank(ca.value))
                .max()
                .unwrap_or(2),
        ) * w;
        cluster_pair[c] += pairing_score(flop.rank_texture()) * w;
    }

    (0..k)
        .map(|c| {
            let w = cluster_weight[c];
            let safe_w = if w > 0.0 { w } else { 1.0 };
            FlopTexture {
                id: u16::try_from(c).unwrap_or(u16::MAX),
                weight: w,
                flush_type: round_to_u8(cluster_flush[c] / safe_w),
                connectivity: round_to_u8(cluster_conn[c] / safe_w),
                high_card: round_to_u8(cluster_high[c] / safe_w),
                pairing: round_to_u8(cluster_pair[c] / safe_w),
            }
        })
        .collect()
}

// ──────────────────────────────────────────────────────────────────────────────
// Turn / River transition features
// ──────────────────────────────────────────────────────────────────────────────

/// Features extracted when a new street card arrives on a board.
#[allow(clippy::struct_excessive_bools)]
#[derive(Debug, Clone, PartialEq)]
struct StreetFeatures {
    completes_flush: bool,
    pairs_board: bool,
    enables_straight: bool,
    overcard: bool,
}

impl StreetFeatures {
    fn to_vec(&self) -> Vec<f64> {
        vec![
            bool_to_f64(self.completes_flush),
            bool_to_f64(self.pairs_board),
            bool_to_f64(self.enables_straight),
            bool_to_f64(self.overcard),
        ]
    }
}

fn bool_to_f64(b: bool) -> f64 {
    if b { 1.0 } else { 0.0 }
}

/// Compute the change-features for a new `card` arriving on `board`.
fn street_features(card: Card, board: &[Card]) -> StreetFeatures {
    StreetFeatures {
        completes_flush: card_completes_flush(card, board),
        pairs_board: card_pairs_board(card, board),
        enables_straight: card_enables_straight(card, board),
        overcard: card_is_overcard(card, board),
    }
}

fn card_completes_flush(card: Card, board: &[Card]) -> bool {
    let suit_count = board.iter().filter(|c| c.suit == card.suit).count();
    suit_count >= 2
}

fn card_pairs_board(card: Card, board: &[Card]) -> bool {
    board.iter().any(|c| c.value == card.value)
}

fn card_is_overcard(card: Card, board: &[Card]) -> bool {
    board
        .iter()
        .all(|c| value_rank(card.value) > value_rank(c.value))
}

/// Check if adding `card` to `board` creates a straight possibility (all combined ranks fit within a 5-wide window).
fn card_enables_straight(card: Card, board: &[Card]) -> bool {
    let mut ranks: Vec<u8> = board.iter().map(|c| value_rank(c.value)).collect();
    ranks.push(value_rank(card.value));
    ranks.sort_unstable();
    ranks.dedup();
    can_make_straight(&ranks)
}

/// True if all the given distinct ranks fit within some 5-rank straight window.
///
/// This detects "straight potential": all board ranks lie within a span of 5
/// consecutive ranks (meaning a straight is completable with the right cards).
fn can_make_straight(ranks: &[u8]) -> bool {
    if ranks.is_empty() {
        return false;
    }
    // Standard windows: [low, low+4] for low in 2..=10
    for low in 2u8..=10 {
        let high = low + 4;
        if ranks.iter().all(|&r| r >= low && r <= high) {
            return true;
        }
    }
    // Wheel window: A-2-3-4-5 (ace plays low; ranks include 14 and 2-5)
    let wheel_low = 1u8; // treat ace as 1 for wheel
    let ranks_with_ace_low: Vec<u8> = ranks.iter().map(|&r| if r == 14 { 1 } else { r }).collect();
    ranks_with_ace_low.iter().all(|&r| r >= wheel_low && r <= 5)
}

// ──────────────────────────────────────────────────────────────────────────────
// Deck helpers
// ──────────────────────────────────────────────────────────────────────────────

/// A full 52-card deck.
fn full_deck() -> Vec<Card> {
    let mut deck = Vec::with_capacity(52);
    for value in Value::values() {
        for suit in Suit::suits() {
            deck.push(Card::new(value, suit));
        }
    }
    deck
}

/// Cards not appearing in `used`.
fn remaining_cards(used: &[Card]) -> Vec<Card> {
    full_deck()
        .into_iter()
        .filter(|c| !used.contains(c))
        .collect()
}

// ──────────────────────────────────────────────────────────────────────────────
// Turn / River transition builders
// ──────────────────────────────────────────────────────────────────────────────

type TurnTransitions = Vec<Vec<TurnTransition>>;
type RiverTransitions = Vec<Vec<Vec<RiverTransition>>>;

/// Build all transitions for every flop texture.  Returns `(turn_transitions,
/// river_transitions)` where the river transitions are indexed by
/// `[flop_texture_id][turn_cluster_id]`.
fn build_all_transitions(
    flop_textures: &[FlopTexture],
    config: &BoardAbstractionConfig,
) -> (Vec<[Card; 3]>, TurnTransitions, RiverTransitions) {
    // Use representative "prototype" flop boards for each texture.
    let prototype_flops = prototype_boards_for_textures(flop_textures);

    let turn_transitions: Vec<Vec<TurnTransition>> = prototype_flops
        .iter()
        .map(|board| {
            build_turn_transitions(
                board,
                config.num_turn_transitions as usize,
                config.kmeans_max_iter,
            )
        })
        .collect();

    let river_transitions: Vec<Vec<Vec<RiverTransition>>> = prototype_flops
        .iter()
        .zip(turn_transitions.iter())
        .map(|(flop_board, turn_txs)| {
            build_river_transitions_for_flop(flop_board, turn_txs, config)
        })
        .collect();

    (prototype_flops, turn_transitions, river_transitions)
}

/// Build turn transition clusters for a representative flop board.
#[allow(clippy::trivially_copy_pass_by_ref)]
fn build_turn_transitions(
    flop_board: &[Card; 3],
    k: usize,
    max_iter: usize,
) -> Vec<TurnTransition> {
    let remaining = remaining_cards(flop_board);
    #[allow(clippy::cast_precision_loss)]
    let total = remaining.len() as f64;
    let features: Vec<Vec<f64>> = remaining
        .iter()
        .map(|&c| street_features(c, flop_board).to_vec())
        .collect();
    let weights: Vec<f64> = vec![1.0 / total; remaining.len()];
    let assignments = weighted_kmeans(&features, &weights, k, max_iter);
    build_street_transitions_from_assignments::<TurnTransition>(
        &remaining,
        flop_board,
        &assignments,
        &weights,
        k,
    )
}

/// Build river transition clusters for each turn cluster on a given flop.
#[allow(clippy::trivially_copy_pass_by_ref)]
fn build_river_transitions_for_flop(
    flop_board: &[Card; 3],
    turn_txs: &[TurnTransition],
    config: &BoardAbstractionConfig,
) -> Vec<Vec<RiverTransition>> {
    // Use prototypical turn cards: one per turn cluster.
    // We represent each turn cluster by the "typical" board change (all booleans from TurnTransition).
    // For simplicity, we synthesise a representative 4-card board per turn transition.
    let representative_turn_cards = representative_turn_cards(flop_board, turn_txs);

    representative_turn_cards
        .iter()
        .map(|turn_card| {
            let four_card_board: Vec<Card> = flop_board
                .iter()
                .copied()
                .chain(std::iter::once(*turn_card))
                .collect();
            let four_card_arr: [Card; 4] = [
                four_card_board[0],
                four_card_board[1],
                four_card_board[2],
                four_card_board[3],
            ];
            build_river_transitions(
                &four_card_arr,
                config.num_river_transitions as usize,
                config.kmeans_max_iter,
            )
        })
        .collect()
}

/// Build river transition clusters for a representative 4-card board.
#[allow(clippy::trivially_copy_pass_by_ref)]
fn build_river_transitions(board: &[Card; 4], k: usize, max_iter: usize) -> Vec<RiverTransition> {
    let remaining = remaining_cards(board);
    #[allow(clippy::cast_precision_loss)]
    let total = remaining.len() as f64;
    let features: Vec<Vec<f64>> = remaining
        .iter()
        .map(|&c| street_features(c, board).to_vec())
        .collect();
    let weights: Vec<f64> = vec![1.0 / total; remaining.len()];
    let assignments = weighted_kmeans(&features, &weights, k, max_iter);
    build_street_transitions_from_assignments::<RiverTransition>(
        &remaining,
        board,
        &assignments,
        &weights,
        k,
    )
}

// ──────────────────────────────────────────────────────────────────────────────
// Transition construction helpers
// ──────────────────────────────────────────────────────────────────────────────

/// A trait implemented by both `TurnTransition` and `RiverTransition` so the
/// cluster-building logic can be shared.
trait FromCluster {
    fn from_cluster(id: u16, weight: f64, feat: &StreetFeatures) -> Self;
}

impl FromCluster for TurnTransition {
    fn from_cluster(id: u16, weight: f64, feat: &StreetFeatures) -> Self {
        Self {
            id,
            weight,
            completes_flush: feat.completes_flush,
            pairs_board: feat.pairs_board,
            enables_straight: feat.enables_straight,
            overcard: feat.overcard,
        }
    }
}

impl FromCluster for RiverTransition {
    fn from_cluster(id: u16, weight: f64, feat: &StreetFeatures) -> Self {
        Self {
            id,
            weight,
            completes_flush: feat.completes_flush,
            pairs_board: feat.pairs_board,
            enables_straight: feat.enables_straight,
            overcard: feat.overcard,
        }
    }
}

fn build_street_transitions_from_assignments<T: FromCluster>(
    cards: &[Card],
    board: &[Card],
    assignments: &[usize],
    weights: &[f64],
    k: usize,
) -> Vec<T> {
    let mut cluster_weight = vec![0.0f64; k];
    let mut cluster_feat: Vec<Option<StreetFeatures>> = vec![None; k];

    for (i, &c) in assignments.iter().enumerate() {
        cluster_weight[c] += weights[i];
        if cluster_feat[c].is_none() {
            cluster_feat[c] = Some(street_features(cards[i], board));
        }
    }

    let total: f64 = cluster_weight.iter().sum::<f64>().max(1e-12);
    (0..k)
        .filter(|&c| cluster_weight[c] > 0.0)
        .map(|c| {
            let feat = cluster_feat[c].clone().unwrap_or(StreetFeatures {
                completes_flush: false,
                pairs_board: false,
                enables_straight: false,
                overcard: false,
            });
            T::from_cluster(
                u16::try_from(c).unwrap_or(u16::MAX),
                cluster_weight[c] / total,
                &feat,
            )
        })
        .collect()
}

// ──────────────────────────────────────────────────────────────────────────────
// Prototype board selection
// ──────────────────────────────────────────────────────────────────────────────

/// For each flop texture cluster, select one representative 3-card board.
///
/// We use actual canonical flop cards so downstream deck arithmetic is correct.
fn prototype_boards_for_textures(flop_textures: &[FlopTexture]) -> Vec<[Card; 3]> {
    let flops = all_flops();
    let total_weight: u32 = flops.iter().map(|f| u32::from(f.weight())).sum();
    let weights: Vec<f64> = flops
        .iter()
        .map(|f| f64::from(f.weight()) / f64::from(total_weight))
        .collect();
    let features: Vec<Vec<f64>> = flops.iter().map(flop_feature_vector).collect();

    // Re-assign flops to textures using nearest-centroid.
    let centroids: Vec<Vec<f64>> = flop_textures
        .iter()
        .map(|t| {
            vec![
                f64::from(t.flush_type) / 2.0,
                f64::from(t.connectivity) / 3.0,
                (f64::from(t.high_card) - 2.0) / 12.0,
                f64::from(t.pairing) / 2.0,
            ]
        })
        .collect();

    let assignments = assign_to_centroids(&features, &centroids);

    // Pick the flop in each cluster with the largest weight.
    flop_textures
        .iter()
        .map(|t| {
            let best = flops
                .iter()
                .zip(assignments.iter())
                .zip(weights.iter())
                .filter(|((_, a), _)| **a == t.id as usize)
                .max_by(|(_, wa), (_, wb)| wa.partial_cmp(wb).unwrap_or(std::cmp::Ordering::Equal))
                .map(|((f, _), _)| *f.cards());
            best.unwrap_or(*flops[0].cards())
        })
        .collect()
}

/// For each turn transition cluster, pick a representative turn card.
///
/// We choose the actual remaining card whose features best match the cluster
/// centroid (majority vote on boolean features).
#[allow(clippy::trivially_copy_pass_by_ref)]
pub fn representative_turn_cards(flop_board: &[Card; 3], turn_txs: &[TurnTransition]) -> Vec<Card> {
    let remaining = remaining_cards(flop_board);
    turn_txs
        .iter()
        .map(|tx| {
            remaining
                .iter()
                .copied()
                .find(|&c| {
                    let f = street_features(c, flop_board);
                    f.completes_flush == tx.completes_flush
                        && f.pairs_board == tx.pairs_board
                        && f.overcard == tx.overcard
                })
                .unwrap_or(remaining[0])
        })
        .collect()
}

/// For each river transition cluster, pick a representative river card.
///
/// Analogous to `representative_turn_cards` — chooses the actual remaining card
/// whose boolean features best match the cluster.
#[allow(clippy::trivially_copy_pass_by_ref)]
pub fn representative_river_cards(
    four_card_board: &[Card; 4],
    river_txs: &[RiverTransition],
) -> Vec<Card> {
    let remaining = remaining_cards(four_card_board);
    river_txs
        .iter()
        .map(|tx| {
            remaining
                .iter()
                .copied()
                .find(|&c| {
                    let f = street_features(c, four_card_board);
                    f.completes_flush == tx.completes_flush
                        && f.pairs_board == tx.pairs_board
                        && f.overcard == tx.overcard
                })
                .unwrap_or(remaining[0])
        })
        .collect()
}

// ──────────────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::poker::{Card, Suit, Value};

    // ── helpers ──────────────────────────────────────────────────────────────

    fn card(v: Value, s: Suit) -> Card {
        Card::new(v, s)
    }

    fn aks_flop() -> [Card; 3] {
        [
            card(Value::Ace, Suit::Spade),
            card(Value::King, Suit::Heart),
            card(Value::Seven, Suit::Diamond),
        ]
    }

    // ── flop feature extraction ───────────────────────────────────────────────

    #[test]
    fn flush_type_score_rainbow_is_zero() {
        assert_eq!(flush_type_score(SuitTexture::Rainbow), 0.0);
    }

    #[test]
    fn flush_type_score_monotone_is_two() {
        assert_eq!(flush_type_score(SuitTexture::Monotone), 2.0);
    }

    #[test]
    fn pairing_score_unpaired_is_zero() {
        assert_eq!(pairing_score(RankTexture::Unpaired), 0.0);
    }

    #[test]
    fn connectivity_score_fully_connected_is_three() {
        assert_eq!(connectivity_score(0, 0), 3.0);
    }

    #[test]
    fn connectivity_score_disconnected_is_zero() {
        assert_eq!(connectivity_score(5, 5), 0.0);
    }

    #[test]
    fn flop_feature_vector_has_four_dimensions() {
        let flops = all_flops();
        let vec = flop_feature_vector(&flops[0]);
        assert_eq!(vec.len(), 4);
    }

    #[test]
    fn flop_feature_vector_all_in_unit_range() {
        let flops = all_flops();
        for flop in &flops {
            for &v in flop_feature_vector(flop).iter() {
                assert!(
                    (0.0..=1.0).contains(&v),
                    "feature out of range: {v} for flop {:?}",
                    flop
                );
            }
        }
    }

    // ── street features ───────────────────────────────────────────────────────

    #[test]
    fn overcard_detected_on_aks_flop() {
        let flop = aks_flop();
        // Two comes on turn — not an overcard
        let two = card(Value::Two, Suit::Club);
        assert!(!card_is_overcard(two, &flop));
    }

    #[test]
    fn no_overcard_on_aks_flop_for_ace() {
        // Ace is already on the flop; the board has an ace so another card can't be higher
        let flop = aks_flop();
        let king = card(Value::King, Suit::Club);
        assert!(!card_is_overcard(king, &flop));
    }

    #[test]
    fn flush_completion_detected_on_two_tone_flop() {
        let flop = [
            card(Value::Ace, Suit::Spade),
            card(Value::King, Suit::Spade),
            card(Value::Seven, Suit::Heart),
        ];
        let spade_two = card(Value::Two, Suit::Spade);
        assert!(card_completes_flush(spade_two, &flop));
    }

    #[test]
    fn flush_not_completed_by_offsuit_card() {
        let flop = [
            card(Value::Ace, Suit::Spade),
            card(Value::King, Suit::Spade),
            card(Value::Seven, Suit::Heart),
        ];
        let diamond_two = card(Value::Two, Suit::Diamond);
        assert!(!card_completes_flush(diamond_two, &flop));
    }

    #[test]
    fn pairs_board_detected() {
        let flop = aks_flop();
        let pair_ace = card(Value::Ace, Suit::Club);
        assert!(card_pairs_board(pair_ace, &flop));
    }

    #[test]
    fn no_pair_for_new_rank() {
        let flop = aks_flop();
        let two = card(Value::Two, Suit::Club);
        assert!(!card_pairs_board(two, &flop));
    }

    #[test]
    fn straight_completion_with_jt9_flop_and_queen() {
        let flop = [
            card(Value::Jack, Suit::Spade),
            card(Value::Ten, Suit::Heart),
            card(Value::Nine, Suit::Diamond),
        ];
        let queen = card(Value::Queen, Suit::Club);
        assert!(card_enables_straight(queen, &flop));
    }

    #[test]
    fn no_straight_on_disconnected_board() {
        let flop = [
            card(Value::Ace, Suit::Spade),
            card(Value::Seven, Suit::Heart),
            card(Value::Two, Suit::Diamond),
        ];
        let three = card(Value::Three, Suit::Club);
        // A-7-2-3: no 5-card straight possible yet
        assert!(!card_enables_straight(three, &flop));
    }

    // ── k-means ───────────────────────────────────────────────────────────────

    #[test]
    fn kmeans_produces_k_or_fewer_assignments() {
        let points = vec![
            vec![0.0, 0.0],
            vec![0.1, 0.1],
            vec![1.0, 1.0],
            vec![0.9, 0.9],
            vec![0.5, 0.5],
        ];
        let weights = vec![0.2; 5];
        let assignments = weighted_kmeans(&points, &weights, 2, 20);
        assert_eq!(assignments.len(), 5);
        assert!(assignments.iter().all(|&a| a < 2));
    }

    #[test]
    fn kmeans_is_deterministic() {
        let points: Vec<Vec<f64>> = (0..20).map(|i| vec![i as f64, (20 - i) as f64]).collect();
        let weights = vec![1.0 / 20.0; 20];
        let a1 = weighted_kmeans(&points, &weights, 4, 20);
        let a2 = weighted_kmeans(&points, &weights, 4, 20);
        assert_eq!(a1, a2);
    }

    #[test]
    fn kmeans_separates_two_clear_clusters() {
        let mut points: Vec<Vec<f64>> = (0..10).map(|_| vec![0.0, 0.0]).collect();
        points.extend((0..10).map(|_| vec![100.0, 100.0]));
        let weights = vec![1.0 / 20.0; 20];
        let assignments = weighted_kmeans(&points, &weights, 2, 50);
        // All first 10 should be in one cluster, last 10 in another
        let first_cluster = assignments[0];
        assert!(assignments[..10].iter().all(|&a| a == first_cluster));
        let second_cluster = assignments[10];
        assert_ne!(first_cluster, second_cluster);
        assert!(assignments[10..].iter().all(|&a| a == second_cluster));
    }

    // ── full build ────────────────────────────────────────────────────────────

    #[test]
    fn build_with_invalid_zero_flop_textures_returns_error() {
        let config = BoardAbstractionConfig {
            num_flop_textures: 0,
            ..Default::default()
        };
        assert!(matches!(
            BoardAbstraction::build(&config),
            Err(BoardAbstractionError::InvalidFlopTextures(0))
        ));
    }

    #[test]
    fn build_produces_correct_flop_texture_count() {
        let config = BoardAbstractionConfig {
            num_flop_textures: 5,
            num_turn_transitions: 3,
            num_river_transitions: 3,
            kmeans_max_iter: 10,
        };
        let abs = BoardAbstraction::build(&config).unwrap();
        assert_eq!(abs.flop_textures.len(), 5);
    }

    #[test]
    fn flop_texture_weights_sum_to_approximately_one() {
        let config = BoardAbstractionConfig {
            num_flop_textures: 5,
            num_turn_transitions: 3,
            num_river_transitions: 3,
            kmeans_max_iter: 10,
        };
        let abs = BoardAbstraction::build(&config).unwrap();
        let total: f64 = abs.flop_textures.iter().map(|t| t.weight).sum();
        assert!(
            (total - 1.0).abs() < 1e-6,
            "flop texture weights should sum to 1.0, got {total}"
        );
    }

    #[test]
    fn build_produces_turn_transitions_for_each_flop_texture() {
        let config = BoardAbstractionConfig {
            num_flop_textures: 4,
            num_turn_transitions: 3,
            num_river_transitions: 3,
            kmeans_max_iter: 10,
        };
        let abs = BoardAbstraction::build(&config).unwrap();
        assert_eq!(abs.turn_transitions.len(), abs.flop_textures.len());
        for txs in &abs.turn_transitions {
            assert!(
                !txs.is_empty(),
                "every flop texture should have turn transitions"
            );
        }
    }

    #[test]
    fn turn_transition_weights_sum_to_one_per_flop_texture() {
        let config = BoardAbstractionConfig {
            num_flop_textures: 4,
            num_turn_transitions: 3,
            num_river_transitions: 3,
            kmeans_max_iter: 10,
        };
        let abs = BoardAbstraction::build(&config).unwrap();
        for (i, txs) in abs.turn_transitions.iter().enumerate() {
            let total: f64 = txs.iter().map(|t| t.weight).sum();
            assert!(
                (total - 1.0).abs() < 1e-6,
                "turn transitions for flop texture {i} should sum to 1.0, got {total}"
            );
        }
    }

    #[test]
    fn build_produces_river_transitions_nested_correctly() {
        let config = BoardAbstractionConfig {
            num_flop_textures: 3,
            num_turn_transitions: 2,
            num_river_transitions: 2,
            kmeans_max_iter: 10,
        };
        let abs = BoardAbstraction::build(&config).unwrap();
        assert_eq!(abs.river_transitions.len(), abs.flop_textures.len());
        for (fi, river_for_flop) in abs.river_transitions.iter().enumerate() {
            assert_eq!(
                river_for_flop.len(),
                abs.turn_transitions[fi].len(),
                "river transitions for flop {fi} should have one entry per turn transition"
            );
        }
    }

    #[test]
    fn river_transition_weights_sum_to_one() {
        let config = BoardAbstractionConfig {
            num_flop_textures: 3,
            num_turn_transitions: 2,
            num_river_transitions: 2,
            kmeans_max_iter: 10,
        };
        let abs = BoardAbstraction::build(&config).unwrap();
        for (fi, river_for_flop) in abs.river_transitions.iter().enumerate() {
            for (ti, txs) in river_for_flop.iter().enumerate() {
                let total: f64 = txs.iter().map(|r| r.weight).sum();
                assert!(
                    (total - 1.0).abs() < 1e-6,
                    "river transitions for flop {fi}, turn {ti} should sum to 1.0, got {total}"
                );
            }
        }
    }

    // ── deck helpers ──────────────────────────────────────────────────────────

    #[test]
    fn full_deck_has_52_cards() {
        assert_eq!(full_deck().len(), 52);
    }

    #[test]
    fn remaining_cards_excludes_flop() {
        let flop = aks_flop();
        let rem = remaining_cards(&flop);
        assert_eq!(rem.len(), 49);
        for c in &flop {
            assert!(
                !rem.contains(c),
                "flop card should not appear in remaining: {c:?}"
            );
        }
    }

    #[test]
    fn remaining_cards_excludes_four_card_board() {
        let board = [
            card(Value::Ace, Suit::Spade),
            card(Value::King, Suit::Heart),
            card(Value::Seven, Suit::Diamond),
            card(Value::Two, Suit::Club),
        ];
        let rem = remaining_cards(&board);
        assert_eq!(rem.len(), 48);
    }
}
