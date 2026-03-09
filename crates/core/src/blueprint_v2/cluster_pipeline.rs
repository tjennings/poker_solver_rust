//! Card abstraction clustering pipeline.
//!
//! **Preflop:** samples random 5-card boards, computes average showdown equity
//! for each of the 1326 hole-card combos across all non-conflicting boards,
//! and clusters the average equities into K buckets using 1-D k-means.
//!
//! **River:** samples random 5-card boards, computes showdown equity for every
//! valid hole-card combo on each board, and clusters the equity values into K
//! buckets using 1-D k-means.
//!
//! **Turn:** samples random 4-card (flop+turn) boards, enumerates all possible
//! river cards for each valid combo, builds a histogram over river equity bins,
//! and clusters these histograms using k-means with Earth Mover's Distance.
//!
//! **Flop:** samples random 3-card flop boards, enumerates all possible turn
//! cards for each valid combo, computes equity on the resulting 4-card board,
//! maps to turn buckets via uniform binning, builds a histogram over turn
//! buckets, and clusters with k-means EMD.
//!
//! Each street produces a [`BucketFile`] mapping `(board, combo)` to a bucket
//! index.

use std::collections::HashMap;
use std::path::Path;

use rand::prelude::*;
use rand::rngs::StdRng;
use rayon::prelude::*;

use crate::abstraction::isomorphism::CanonicalBoard;
use crate::flops::all_flops;
use crate::poker::{Card, Suit, Value, ALL_SUITS, ALL_VALUES};
use crate::showdown_equity::{compute_equity, rank_hand};

use super::bucket_file::{BucketFile, BucketFileHeader, PackedBoard};
use super::clustering::{kmeans_1d, kmeans_1d_weighted, kmeans_emd_weighted_u8, kmeans_emd_with_progress};
use super::config::ClusteringConfig;
use super::Street;

/// Number of sample 5-card boards for river clustering when no explicit count
/// is provided.
const DEFAULT_NUM_BOARDS: usize = 10_000;

/// Number of sample 4-card boards for turn clustering.
const DEFAULT_TURN_BOARDS: usize = 5_000;

/// Number of sample 3-card boards for flop clustering (fewer than turn because
/// each board requires enumerating all ~47 turn cards per combo).
const DEFAULT_FLOP_BOARDS: usize = 2_000;

/// Number of sample 5-card boards for preflop equity estimation.
const DEFAULT_PREFLOP_BOARDS: usize = 2_000;

/// Total number of 2-card combos from a 52-card deck: C(52,2) = 1326.
const TOTAL_COMBOS: u16 = 1326;

/// Cluster all river information situations by equity.
///
/// Samples `num_boards` random 5-card boards from the 52-card deck, computes
/// showdown equity for every valid hole-card combo on each board, then clusters
/// all equity values into `bucket_count` buckets using 1-D k-means.
///
/// Returns a `BucketFile` with:
/// - `board_count` = `num_boards`
/// - `combos_per_board` = 1326 (all 2-card combos from 52 cards)
/// - Entries where the combo conflicts with the board get bucket 0 (sentinel)
///
/// The `progress` callback receives values in `[0.0, 1.0]` as boards are
/// processed.
#[allow(dead_code)]
pub fn cluster_river(
    bucket_count: u16,
    kmeans_iterations: u32,
    seed: u64,
    progress: impl Fn(f64) + Sync,
) -> BucketFile {
    cluster_river_with_boards(
        bucket_count,
        kmeans_iterations,
        seed,
        DEFAULT_NUM_BOARDS,
        progress,
    )
}

/// Like [`cluster_river`] but with an explicit board sample count.
#[allow(dead_code)]
pub fn cluster_river_with_boards(
    bucket_count: u16,
    kmeans_iterations: u32,
    seed: u64,
    num_boards: usize,
    progress: impl Fn(f64) + Sync,
) -> BucketFile {
    let deck = build_deck();
    let combos = enumerate_combos(&deck);
    let boards = sample_boards(&deck, num_boards, seed);

    // --- Compute equity for each (board, combo) pair in parallel. -----------
    // Each board produces a Vec<Option<f64>>: Some(equity) for valid combos,
    // None for combos blocked by the board.
    let board_equities: Vec<Vec<Option<f64>>> = boards
        .par_iter()
        .enumerate()
        .map(|(i, board)| {
            let eq = compute_board_equities(*board, &combos);
            // Report progress (approximate — may interleave but that's fine).
            #[allow(clippy::cast_precision_loss)]
            progress((i + 1) as f64 / num_boards as f64);
            eq
        })
        .collect();

    // --- Collect all valid equity values for k-means. -----------------------
    let all_equities: Vec<f64> = board_equities
        .iter()
        .flat_map(|eqs| eqs.iter().filter_map(|&e| e))
        .collect();

    // --- Run 1-D k-means to find bucket boundaries. -------------------------
    let cluster_labels = kmeans_1d(&all_equities, bucket_count as usize, kmeans_iterations);

    // Build a mapping from the flat valid-equity index back to the per-combo
    // bucket assignment in the full (board * 1326) layout.
    let total = num_boards * TOTAL_COMBOS as usize;
    let mut buckets = vec![0_u16; total];

    let mut flat_idx = 0_usize;
    for (board_idx, eqs) in board_equities.iter().enumerate() {
        for (combo_idx, eq) in eqs.iter().enumerate() {
            if eq.is_some() {
                buckets[board_idx * TOTAL_COMBOS as usize + combo_idx] = cluster_labels[flat_idx];
                flat_idx += 1;
            }
            // Blocked combos keep bucket 0 (sentinel).
        }
    }

    #[allow(clippy::cast_possible_truncation)]
    let header = BucketFileHeader {
        street: Street::River,
        bucket_count,
        board_count: num_boards as u32,
        combos_per_board: TOTAL_COMBOS,
        version: 1,
    };

    BucketFile { header, boards: Vec::new(), buckets }
}

// ---------------------------------------------------------------------------
// Turn clustering
// ---------------------------------------------------------------------------

/// Cluster turn information situations using potential-aware features.
///
/// For each sampled 4-card (flop+turn) board and each valid hole-card combo:
/// 1. Enumerate all possible river cards (52 - 4 board - 2 hole = 46).
/// 2. For each river card, compute showdown equity on the resulting 5-card
///    board and map the equity to a river bucket via uniform binning.
/// 3. Build a histogram (probability distribution) over river buckets.
/// 4. Cluster all histograms with k-means using Earth Mover's Distance.
///
/// The `river_buckets` file defines how many river equity bins to use (its
/// `bucket_count` determines the histogram dimensionality).
///
/// Returns a `BucketFile` with `street = Turn` and the given `bucket_count`.
/// Blocked combos (overlapping with the board) receive bucket 0 as a sentinel.
#[allow(dead_code)]
pub fn cluster_turn(
    river_buckets: &BucketFile,
    bucket_count: u16,
    kmeans_iterations: u32,
    seed: u64,
    progress: impl Fn(f64) + Sync,
) -> BucketFile {
    cluster_turn_with_boards(
        river_buckets,
        bucket_count,
        kmeans_iterations,
        seed,
        DEFAULT_TURN_BOARDS,
        progress,
    )
}

/// Like [`cluster_turn`] but with an explicit board sample count.
#[allow(dead_code)]
pub fn cluster_turn_with_boards(
    river_buckets: &BucketFile,
    bucket_count: u16,
    kmeans_iterations: u32,
    seed: u64,
    num_boards: usize,
    progress: impl Fn(f64) + Sync,
) -> BucketFile {
    let deck = build_deck();
    let combos = enumerate_combos(&deck);
    let boards = sample_turn_boards(&deck, num_boards, seed);
    let num_river_buckets = river_buckets.header.bucket_count;

    // For each board, compute a histogram feature vector for every combo.
    // `None` means the combo is blocked by the board.
    let board_features: Vec<Vec<Option<Vec<f64>>>> = boards
        .par_iter()
        .enumerate()
        .map(|(board_idx, board)| {
            let features: Vec<Option<Vec<f64>>> = combos
                .iter()
                .map(|combo| {
                    if cards_overlap(*combo, board) {
                        return None;
                    }
                    Some(build_next_street_histogram(*combo, board, &deck, num_river_buckets))
                })
                .collect();

            #[allow(clippy::cast_precision_loss)]
            progress((board_idx + 1) as f64 / num_boards as f64 * 0.8);

            features
        })
        .collect();

    // Collect all valid feature vectors for k-means, tracking their position.
    let mut all_features: Vec<Vec<f64>> = Vec::new();
    let mut feature_positions: Vec<(usize, usize)> = Vec::new();

    for (board_idx, board_feats) in board_features.iter().enumerate() {
        for (combo_idx, feat) in board_feats.iter().enumerate() {
            if let Some(histogram) = feat {
                all_features.push(histogram.clone());
                feature_positions.push((board_idx, combo_idx));
            }
        }
    }

    // Cluster with k-means EMD (parallel assignment step + progress).
    let cluster_labels = kmeans_emd_with_progress(
        &all_features,
        bucket_count as usize,
        kmeans_iterations,
        seed,
        |iter, max_iter| {
            #[allow(clippy::cast_precision_loss)]
            progress(0.8 + 0.2 * f64::from(iter) / f64::from(max_iter));
        },
    );

    // Map cluster labels back to the flat (board * 1326) bucket array.
    let total = num_boards * TOTAL_COMBOS as usize;
    let mut buckets = vec![0_u16; total];

    for (flat_idx, &(board_idx, combo_idx)) in feature_positions.iter().enumerate() {
        buckets[board_idx * TOTAL_COMBOS as usize + combo_idx] = cluster_labels[flat_idx];
    }

    #[allow(clippy::cast_possible_truncation)]
    let header = BucketFileHeader {
        street: Street::Turn,
        bucket_count,
        board_count: num_boards as u32,
        combos_per_board: TOTAL_COMBOS,
        version: 1,
    };

    BucketFile { header, boards: Vec::new(), buckets }
}

/// Build a probability distribution over equity bins for the given combo
/// on a partial board.
///
/// Enumerates every card not in the board or combo as a potential next-street
/// card, computes equity on the extended board, maps equity to a bin, and
/// normalises the histogram.
fn build_next_street_histogram(
    combo: [Card; 2],
    board: &[Card],
    deck: &[Card],
    num_buckets: u16,
) -> Vec<f64> {
    let mut histogram = vec![0.0_f64; num_buckets as usize];
    let mut count = 0_u32;
    let mut extended = Vec::with_capacity(board.len() + 1);
    extended.extend_from_slice(board);
    extended.push(Card::new(crate::poker::Value::Two, crate::poker::Suit::Club)); // placeholder

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
        histogram[bucket as usize] += 1.0;
        count += 1;
    }

    // Normalise to a probability distribution.
    if count > 0 {
        #[allow(clippy::cast_precision_loss)]
        let inv = 1.0 / f64::from(count);
        for h in &mut histogram {
            *h *= inv;
        }
    }

    histogram
}

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
        // Max per-bin count is 47 (flop: 52 - 3 board - 2 hole), fits in u8.
        debug_assert!(histogram[bucket as usize] < 255);
        histogram[bucket as usize] += 1;
    }

    histogram
}

/// Map an equity value in [0, 1] to a bucket index via uniform binning.
///
/// `equity_to_bucket(0.0, K) = 0` and `equity_to_bucket(1.0, K) = K - 1`.
fn equity_to_bucket(equity: f64, num_buckets: u16) -> u16 {
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let bucket = (equity * f64::from(num_buckets)) as u16;
    bucket.min(num_buckets - 1)
}

// ---------------------------------------------------------------------------
// Flop clustering
// ---------------------------------------------------------------------------

/// Cluster flop information situations using potential-aware features.
///
/// For each sampled 3-card flop and each valid hole-card combo:
/// 1. Enumerate all possible turn cards (52 - 3 board - 2 hole = 47).
/// 2. For each turn card, compute equity on the resulting 4-card board and
///    map the equity to a turn bucket via uniform binning.
/// 3. Build a histogram (probability distribution) over turn buckets.
/// 4. Cluster all histograms with k-means using Earth Mover's Distance.
///
/// The `turn_buckets` file defines the histogram dimensionality (its
/// `bucket_count` determines the number of bins).
///
/// Returns a `BucketFile` with `street = Flop` and the given `bucket_count`.
/// Blocked combos (overlapping with the board) receive bucket 0 as a sentinel.
#[allow(dead_code)]
pub fn cluster_flop(
    turn_buckets: &BucketFile,
    bucket_count: u16,
    kmeans_iterations: u32,
    seed: u64,
    progress: impl Fn(f64) + Sync,
) -> BucketFile {
    cluster_flop_with_boards(
        turn_buckets,
        bucket_count,
        kmeans_iterations,
        seed,
        DEFAULT_FLOP_BOARDS,
        progress,
    )
}

/// Like [`cluster_flop`] but with an explicit board sample count.
#[allow(dead_code)]
pub fn cluster_flop_with_boards(
    turn_buckets: &BucketFile,
    bucket_count: u16,
    kmeans_iterations: u32,
    seed: u64,
    num_boards: usize,
    progress: impl Fn(f64) + Sync,
) -> BucketFile {
    let deck = build_deck();
    let combos = enumerate_combos(&deck);
    let boards = sample_flop_boards(&deck, num_boards, seed);
    let num_turn_buckets = turn_buckets.header.bucket_count;

    // For each board, compute a histogram feature vector for every combo.
    // `None` means the combo is blocked by the board.
    let board_features: Vec<Vec<Option<Vec<f64>>>> = boards
        .par_iter()
        .enumerate()
        .map(|(board_idx, board)| {
            let features: Vec<Option<Vec<f64>>> = combos
                .iter()
                .map(|combo| {
                    if cards_overlap(*combo, board) {
                        return None;
                    }
                    Some(build_next_street_histogram(*combo, board, &deck, num_turn_buckets))
                })
                .collect();

            #[allow(clippy::cast_precision_loss)]
            progress((board_idx + 1) as f64 / num_boards as f64 * 0.8);

            features
        })
        .collect();

    // Collect all valid feature vectors for k-means, tracking their position.
    let mut all_features: Vec<Vec<f64>> = Vec::new();
    let mut feature_positions: Vec<(usize, usize)> = Vec::new();

    for (board_idx, board_feats) in board_features.iter().enumerate() {
        for (combo_idx, feat) in board_feats.iter().enumerate() {
            if let Some(histogram) = feat {
                all_features.push(histogram.clone());
                feature_positions.push((board_idx, combo_idx));
            }
        }
    }

    // Cluster with k-means EMD (parallel assignment step + progress).
    let cluster_labels = kmeans_emd_with_progress(
        &all_features,
        bucket_count as usize,
        kmeans_iterations,
        seed,
        |iter, max_iter| {
            #[allow(clippy::cast_precision_loss)]
            progress(0.8 + 0.2 * f64::from(iter) / f64::from(max_iter));
        },
    );

    // Map cluster labels back to the flat (board * 1326) bucket array.
    let total = num_boards * TOTAL_COMBOS as usize;
    let mut buckets = vec![0_u16; total];

    for (flat_idx, &(board_idx, combo_idx)) in feature_positions.iter().enumerate() {
        buckets[board_idx * TOTAL_COMBOS as usize + combo_idx] = cluster_labels[flat_idx];
    }

    #[allow(clippy::cast_possible_truncation)]
    let header = BucketFileHeader {
        street: Street::Flop,
        bucket_count,
        board_count: num_boards as u32,
        combos_per_board: TOTAL_COMBOS,
        version: 1,
    };

    BucketFile { header, boards: Vec::new(), buckets }
}

// ---------------------------------------------------------------------------
// Canonical clustering (exhaustive enumeration + weighted k-means)
// ---------------------------------------------------------------------------

/// Cluster river information situations using exhaustive canonical board
/// enumeration and weighted 1-D k-means.
///
/// Instead of sampling random boards, enumerates all canonical 5-card rivers
/// (via suit isomorphism), computes showdown equity for every valid combo on
/// each board, and clusters using [`kmeans_1d_weighted`] where each sample's
/// weight reflects the number of raw boards that canonical board represents.
///
/// Returns a `BucketFile` with `street = River`, `board_count` equal to the
/// number of canonical rivers, and a populated `boards` field.
pub fn cluster_river_canonical(
    bucket_count: u16,
    kmeans_iterations: u32,
    _seed: u64,
    progress: impl Fn(f64) + Sync,
) -> BucketFile {
    let deck = build_deck();
    let combos = enumerate_combos(&deck);
    let boards = enumerate_canonical_rivers();
    let num_boards = boards.len();

    // Compute equity for each (board, combo) pair in parallel.
    let board_equities: Vec<Vec<Option<f64>>> = boards
        .par_iter()
        .enumerate()
        .map(|(i, wb)| {
            let eq = compute_board_equities(wb.cards, &combos);
            #[allow(clippy::cast_precision_loss)]
            progress((i + 1) as f64 / num_boards as f64 * 0.8);
            eq
        })
        .collect();

    // Collect valid equities with per-sample weights and positions.
    let mut all_equities: Vec<f64> = Vec::new();
    let mut all_weights: Vec<f64> = Vec::new();
    let mut positions: Vec<(usize, usize)> = Vec::new();

    for (board_idx, (eqs, wb)) in board_equities.iter().zip(boards.iter()).enumerate() {
        for (combo_idx, eq) in eqs.iter().enumerate() {
            if let Some(e) = eq {
                all_equities.push(*e);
                all_weights.push(f64::from(wb.weight));
                positions.push((board_idx, combo_idx));
            }
        }
    }

    let cluster_labels = kmeans_1d_weighted(
        &all_equities,
        &all_weights,
        bucket_count as usize,
        kmeans_iterations,
    );

    let total = num_boards * TOTAL_COMBOS as usize;
    let mut buckets = vec![0_u16; total];
    for (flat_idx, &(board_idx, combo_idx)) in positions.iter().enumerate() {
        buckets[board_idx * TOTAL_COMBOS as usize + combo_idx] = cluster_labels[flat_idx];
    }

    let packed_boards: Vec<PackedBoard> = boards
        .iter()
        .map(|wb| canonical_key(&wb.cards))
        .collect();

    #[allow(clippy::cast_possible_truncation)]
    BucketFile {
        header: BucketFileHeader {
            street: Street::River,
            bucket_count,
            board_count: num_boards as u32,
            combos_per_board: TOTAL_COMBOS,
            version: 2,
        },
        boards: packed_boards,
        buckets,
    }
}

/// Cluster turn information situations using exhaustive canonical board
/// enumeration and weighted EMD k-means.
///
/// Enumerates all canonical 4-card (flop+turn) boards, computes a histogram
/// feature vector over river buckets for every valid combo, and clusters using
/// [`kmeans_emd_weighted_u8`] with weights reflecting combinatorial multiplicity.
pub fn cluster_turn_canonical(
    river_buckets: &BucketFile,
    bucket_count: u16,
    kmeans_iterations: u32,
    seed: u64,
    progress: impl Fn(f64) + Sync,
) -> BucketFile {
    let deck = build_deck();
    let combos = enumerate_combos(&deck);
    let boards = enumerate_canonical_turns();
    let num_boards = boards.len();
    let num_river_buckets = river_buckets.header.bucket_count;

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

    let total = num_boards * TOTAL_COMBOS as usize;
    let mut buckets = vec![0_u16; total];
    for (flat_idx, &(board_idx, combo_idx)) in feature_positions.iter().enumerate() {
        buckets[board_idx * TOTAL_COMBOS as usize + combo_idx] = cluster_labels[flat_idx];
    }

    let packed_boards: Vec<PackedBoard> = boards
        .iter()
        .map(|wb| canonical_key(&wb.cards))
        .collect();

    #[allow(clippy::cast_possible_truncation)]
    BucketFile {
        header: BucketFileHeader {
            street: Street::Turn,
            bucket_count,
            board_count: num_boards as u32,
            combos_per_board: TOTAL_COMBOS,
            version: 2,
        },
        boards: packed_boards,
        buckets,
    }
}

/// Cluster flop information situations using exhaustive canonical board
/// enumeration and weighted EMD k-means.
///
/// Enumerates all 1,755 canonical 3-card flops, computes a histogram feature
/// vector over turn buckets for every valid combo, and clusters using
/// [`kmeans_emd_weighted_u8`] with weights reflecting combinatorial multiplicity.
pub fn cluster_flop_canonical(
    turn_buckets: &BucketFile,
    bucket_count: u16,
    kmeans_iterations: u32,
    seed: u64,
    progress: impl Fn(f64) + Sync,
) -> BucketFile {
    let deck = build_deck();
    let combos = enumerate_combos(&deck);
    let boards = enumerate_canonical_flops();
    let num_boards = boards.len();
    let num_turn_buckets = turn_buckets.header.bucket_count;

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

    let total = num_boards * TOTAL_COMBOS as usize;
    let mut buckets = vec![0_u16; total];
    for (flat_idx, &(board_idx, combo_idx)) in feature_positions.iter().enumerate() {
        buckets[board_idx * TOTAL_COMBOS as usize + combo_idx] = cluster_labels[flat_idx];
    }

    let packed_boards: Vec<PackedBoard> = boards
        .iter()
        .map(|wb| canonical_key(&wb.cards))
        .collect();

    #[allow(clippy::cast_possible_truncation)]
    BucketFile {
        header: BucketFileHeader {
            street: Street::Flop,
            bucket_count,
            board_count: num_boards as u32,
            combos_per_board: TOTAL_COMBOS,
            version: 2,
        },
        boards: packed_boards,
        buckets,
    }
}

// ---------------------------------------------------------------------------
// Preflop clustering
// ---------------------------------------------------------------------------

/// Cluster preflop hole-card combos by average showdown equity.
///
/// Samples random 5-card boards and computes the mean equity of each of the
/// 1326 two-card combos across all non-conflicting boards. The resulting
/// average equities are clustered into `bucket_count` groups using 1-D k-means.
///
/// Returns a `BucketFile` with `board_count = 1` and `combos_per_board = 1326`.
/// Every combo receives a valid bucket (no sentinels needed since there is no
/// board to conflict with preflop).
///
/// The `progress` callback receives values in `[0.0, 1.0]` as boards are
/// processed.
pub fn cluster_preflop(
    bucket_count: u16,
    kmeans_iterations: u32,
    seed: u64,
    progress: impl Fn(f64) + Sync,
) -> BucketFile {
    cluster_preflop_with_boards(
        bucket_count,
        kmeans_iterations,
        seed,
        DEFAULT_PREFLOP_BOARDS,
        progress,
    )
}

/// Like [`cluster_preflop`] but with an explicit board sample count.
pub fn cluster_preflop_with_boards(
    bucket_count: u16,
    kmeans_iterations: u32,
    seed: u64,
    num_boards: usize,
    progress: impl Fn(f64) + Sync,
) -> BucketFile {
    let deck = build_deck();
    let combos = enumerate_combos(&deck);
    let boards = sample_boards(&deck, num_boards, seed);

    // Compute average equity for each combo across all sampled boards.
    let avg_equities = compute_combo_avg_equities(&combos, &boards, &progress);

    // Cluster the average equities with 1-D k-means.
    let cluster_labels = kmeans_1d(&avg_equities, bucket_count as usize, kmeans_iterations);

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

/// Compute the average showdown equity for each combo across a set of 5-card
/// boards.
///
/// For each combo, sums equity over all boards that don't conflict with the
/// combo's cards and divides by the count.
fn compute_combo_avg_equities(
    combos: &[[Card; 2]],
    boards: &[[Card; 5]],
    progress: &(impl Fn(f64) + Sync),
) -> Vec<f64> {
    combos
        .par_iter()
        .enumerate()
        .map(|(combo_idx, &combo)| {
            let mut sum = 0.0_f64;
            let mut count = 0_u32;

            for board in boards {
                if cards_overlap(combo, board) {
                    continue;
                }
                sum += compute_equity(combo, board);
                count += 1;
            }

            #[allow(clippy::cast_precision_loss)]
            progress((combo_idx + 1) as f64 / combos.len() as f64);

            debug_assert!(count > 0, "combo matched zero boards");
            sum / f64::from(count)
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Full pipeline orchestrator
// ---------------------------------------------------------------------------

/// Run the full bottom-up clustering pipeline: river -> turn -> flop -> preflop.
///
/// Each street's bucket file is saved to `output_dir` as it completes, so
/// partial results are available even if a later stage fails or is interrupted.
///
/// The `progress` callback receives the street name (`"river"`, `"turn"`,
/// `"flop"`, `"preflop"`) and a value in `[0.0, 1.0]` for that street.
///
/// # Errors
///
/// Returns an error if any bucket file cannot be saved (I/O failure).
pub fn run_clustering_pipeline(
    config: &ClusteringConfig,
    output_dir: &Path,
    progress: impl Fn(&str, f64) + Sync,
) -> Result<(), Box<dyn std::error::Error>> {
    // 1. River (sampling-based equity clustering)
    progress("river", 0.0);
    let river = cluster_river(
        config.river.buckets,
        config.kmeans_iterations,
        config.seed,
        |p| progress("river", p),
    );
    river.save(&output_dir.join("river.buckets"))?;

    // 2. Turn (sampling-based, potential-aware EMD, depends on river)
    progress("turn", 0.0);
    let turn = cluster_turn(
        &river,
        config.turn.buckets,
        config.kmeans_iterations,
        config.seed,
        |p| progress("turn", p),
    );
    turn.save(&output_dir.join("turn.buckets"))?;

    // 3. Flop (sampling-based, potential-aware EMD, depends on turn)
    progress("flop", 0.0);
    let flop = cluster_flop(
        &turn,
        config.flop.buckets,
        config.kmeans_iterations,
        config.seed,
        |p| progress("flop", p),
    );
    flop.save(&output_dir.join("flop.buckets"))?;

    // 4. Preflop (independent, but run last by convention)
    progress("preflop", 0.0);
    let preflop = cluster_preflop(
        config.preflop.buckets,
        config.kmeans_iterations,
        config.seed,
        |p| progress("preflop", p),
    );
    preflop.save(&output_dir.join("preflop.buckets"))?;

    Ok(())
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Build the standard 52-card deck in a deterministic order.
pub(crate) fn build_deck() -> Vec<Card> {
    let mut deck = Vec::with_capacity(52);
    for &v in &ALL_VALUES {
        for &s in &ALL_SUITS {
            deck.push(Card::new(v, s));
        }
    }
    deck
}

// ---------------------------------------------------------------------------
// Canonical board enumeration
// ---------------------------------------------------------------------------

/// A board with its combinatorial weight (number of raw boards it represents).
#[derive(Debug, Clone)]
pub struct WeightedBoard<const N: usize> {
    pub cards: [Card; N],
    pub weight: u32,
}

/// Pack canonical cards into a deterministic key by sorting first.
///
/// `CanonicalBoard::from_cards` preserves input order, so the same canonical
/// board reached via different input orderings would produce different
/// `PackedBoard` values. Sorting by `(value_rank desc, suit asc)` before
/// packing ensures a unique key regardless of input order.
pub(crate) fn canonical_key(cards: &[Card]) -> PackedBoard {
    let mut sorted: Vec<Card> = cards.to_vec();
    sorted.sort_by(|a, b| {
        use crate::card_utils::value_rank;
        value_rank(b.value)
            .cmp(&value_rank(a.value))
            .then(a.suit.cmp(&b.suit))
    });
    PackedBoard::from_cards(&sorted)
}

/// Enumerate all 1,755 canonical flops with combinatorial weights.
///
/// Wraps [`all_flops()`] into `WeightedBoard<3>` form. Total weight equals
/// C(52,3) = 22,100.
#[must_use]
pub fn enumerate_canonical_flops() -> Vec<WeightedBoard<3>> {
    let flops = all_flops();
    let mut result: Vec<WeightedBoard<3>> = flops
        .into_iter()
        .map(|f| WeightedBoard {
            cards: *f.cards(),
            weight: u32::from(f.weight()),
        })
        .collect();
    result.sort_by_key(|wb| canonical_key(&wb.cards).0);
    result
}

/// Enumerate all canonical 4-card (flop + turn) boards with weights.
///
/// For each canonical flop, appends each of the 49 remaining cards as a turn,
/// canonicalizes the resulting 4-card board, and deduplicates. Weight equals the
/// sum of constituent flop weights. Total weight = C(52,3) x 49 = 1,082,900.
///
/// # Panics
/// Panics if canonical flop enumeration produces invalid card data.
#[must_use]
pub fn enumerate_canonical_turns() -> Vec<WeightedBoard<4>> {
    let flops = enumerate_canonical_flops();
    let deck = build_deck();

    let map: HashMap<PackedBoard, (u32, [Card; 4])> = flops
        .par_iter()
        .fold(
            HashMap::new,
            |mut map: HashMap<PackedBoard, (u32, [Card; 4])>, flop| {
                for &card in &deck {
                    if flop.cards.contains(&card) {
                        continue;
                    }
                    let board_vec = vec![
                        flop.cards[0],
                        flop.cards[1],
                        flop.cards[2],
                        card,
                    ];
                    // INVARIANT: 4-card boards are always valid for canonicalization
                    let canonical = CanonicalBoard::from_cards(&board_vec)
                        .expect("4-card board is always valid");
                    let key = canonical_key(&canonical.cards);
                    let sorted = key.to_cards(4);
                    map.entry(key)
                        .and_modify(|(w, _)| *w += flop.weight)
                        .or_insert_with(|| {
                            let cards: [Card; 4] =
                                [sorted[0], sorted[1], sorted[2], sorted[3]];
                            (flop.weight, cards)
                        });
                }
                map
            },
        )
        .reduce(HashMap::new, |mut a, b| {
            for (key, (weight, cards)) in b {
                a.entry(key)
                    .and_modify(|(w, _)| *w += weight)
                    .or_insert((weight, cards));
            }
            a
        });

    let mut result: Vec<WeightedBoard<4>> = map
        .into_iter()
        .map(|(_, (weight, cards))| WeightedBoard { cards, weight })
        .collect();
    result.sort_by_key(|wb| canonical_key(&wb.cards).0);
    result
}

/// Enumerate all canonical 5-card river boards with weights.
///
/// For each canonical turn, appends each of the 48 remaining cards as a river,
/// canonicalizes the resulting 5-card board, and deduplicates. Weight equals the
/// sum of constituent turn weights. Total weight = C(52,3) x 49 x 48.
///
/// # Panics
/// Panics if canonical turn enumeration produces invalid card data.
#[must_use]
pub fn enumerate_canonical_rivers() -> Vec<WeightedBoard<5>> {
    let turns = enumerate_canonical_turns();
    let deck = build_deck();

    let map: HashMap<PackedBoard, (u32, [Card; 5])> = turns
        .par_iter()
        .fold(
            HashMap::new,
            |mut map: HashMap<PackedBoard, (u32, [Card; 5])>, turn| {
                for &card in &deck {
                    if turn.cards.contains(&card) {
                        continue;
                    }
                    let board_vec = vec![
                        turn.cards[0],
                        turn.cards[1],
                        turn.cards[2],
                        turn.cards[3],
                        card,
                    ];
                    // INVARIANT: 5-card boards are always valid for canonicalization
                    let canonical = CanonicalBoard::from_cards(&board_vec)
                        .expect("5-card board is always valid");
                    let key = canonical_key(&canonical.cards);
                    let sorted = key.to_cards(5);
                    map.entry(key)
                        .and_modify(|(w, _)| *w += turn.weight)
                        .or_insert_with(|| {
                            let cards: [Card; 5] =
                                [sorted[0], sorted[1], sorted[2], sorted[3], sorted[4]];
                            (turn.weight, cards)
                        });
                }
                map
            },
        )
        .reduce(HashMap::new, |mut a, b| {
            for (key, (weight, cards)) in b {
                a.entry(key)
                    .and_modify(|(w, _)| *w += weight)
                    .or_insert((weight, cards));
            }
            a
        });

    let mut result: Vec<WeightedBoard<5>> = map
        .into_iter()
        .map(|(_, (weight, cards))| WeightedBoard { cards, weight })
        .collect();
    result.sort_by_key(|wb| canonical_key(&wb.cards).0);
    result
}

/// Enumerate all C(52,2) = 1326 two-card combos from the deck.
///
/// The combos are stored as `(Card, Card)` with `c0 < c1` by deck index.
/// The ordering is deterministic and matches the canonical combo index used
/// throughout the bucket file.
pub(crate) fn enumerate_combos(deck: &[Card]) -> Vec<[Card; 2]> {
    let mut combos = Vec::with_capacity(TOTAL_COMBOS as usize);
    for i in 0..deck.len() {
        for j in (i + 1)..deck.len() {
            combos.push([deck[i], deck[j]]);
        }
    }
    debug_assert_eq!(combos.len(), TOTAL_COMBOS as usize);
    combos
}

/// Map a card to its position in the canonical deck ordering.
///
/// The deck is ordered by value (Two..Ace) x suit (Spade, Heart, Diamond, Club),
/// matching `build_deck()`.
#[must_use]
pub fn card_to_deck_index(card: Card) -> usize {
    let value_idx = match card.value {
        Value::Two => 0,
        Value::Three => 1,
        Value::Four => 2,
        Value::Five => 3,
        Value::Six => 4,
        Value::Seven => 5,
        Value::Eight => 6,
        Value::Nine => 7,
        Value::Ten => 8,
        Value::Jack => 9,
        Value::Queen => 10,
        Value::King => 11,
        Value::Ace => 12,
    };
    let suit_idx = match card.suit {
        Suit::Spade => 0,
        Suit::Heart => 1,
        Suit::Diamond => 2,
        Suit::Club => 3,
    };
    value_idx * 4 + suit_idx
}

/// Map a two-card hole-card combo to its index in the canonical
/// `enumerate_combos()` ordering (0..1325).
///
/// The two cards can be in any order — they are sorted by deck index
/// internally to match the enumeration `for i in 0..52 { for j in i+1..52 }`.
#[must_use]
pub fn combo_index(c0: Card, c1: Card) -> u16 {
    let mut i = card_to_deck_index(c0);
    let mut j = card_to_deck_index(c1);
    if i > j {
        std::mem::swap(&mut i, &mut j);
    }
    // Triangular number: combos before row i = i*(2*52 - i - 1)/2
    // Offset within row = j - i - 1
    #[allow(clippy::cast_possible_truncation)]
    let idx = (i * (2 * 52 - i - 1) / 2 + (j - i - 1)) as u16;
    idx
}

/// Sample `count` random boards of `card_count` cards from the deck.
///
/// Each board is drawn via partial Fisher-Yates. Boards are sampled
/// independently (duplicates possible but astronomically unlikely).
#[allow(dead_code)]
fn sample_n_card_boards(
    deck: &[Card],
    card_count: usize,
    count: usize,
    seed: u64,
) -> Vec<Vec<Card>> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut boards = Vec::with_capacity(count);
    let indices: Vec<usize> = (0..deck.len()).collect();

    for _ in 0..count {
        let mut pool = indices.clone();
        for k in 0..card_count {
            let j = rng.random_range(k..pool.len());
            pool.swap(k, j);
        }
        boards.push(pool[..card_count].iter().map(|&i| deck[i]).collect());
    }

    boards
}

/// Sample `n` random 5-card boards from the deck without replacement.
#[allow(dead_code)]
pub(crate) fn sample_boards(deck: &[Card], n: usize, seed: u64) -> Vec<[Card; 5]> {
    sample_n_card_boards(deck, 5, n, seed)
        .into_iter()
        .map(|v| [v[0], v[1], v[2], v[3], v[4]])
        .collect()
}

/// Compute equity for all 1326 combos on a given 5-card board.
///
/// Returns `None` for combos that share a card with the board (blocked).
///
/// Uses a two-phase approach: first evaluates hand rank for every valid combo
/// once, then sweeps pairwise comparisons using pre-computed ranks. This avoids
/// redundantly re-evaluating opponent ranks for each hero combo (~1000x fewer
/// `rank_hand` calls).
pub(crate) fn compute_board_equities(board: [Card; 5], combos: &[[Card; 2]]) -> Vec<Option<f64>> {
    use std::cmp::Ordering;

    // Build a bitmask for the board cards for fast overlap checks.
    let mut board_mask = 0u64;
    for &c in &board {
        board_mask |= 1u64 << card_to_deck_index(c);
    }

    // Phase 1: evaluate hand rank for every non-blocked combo (once each).
    // Also store per-combo bitmask for fast hero-vs-opponent overlap detection.
    let mut ranks: Vec<u32> = Vec::with_capacity(combos.len());
    let mut combo_masks: Vec<u64> = Vec::with_capacity(combos.len());
    let mut blocked: Vec<bool> = Vec::with_capacity(combos.len());

    for &combo in combos {
        let mask = (1u64 << card_to_deck_index(combo[0]))
            | (1u64 << card_to_deck_index(combo[1]));
        combo_masks.push(mask);
        if mask & board_mask != 0 {
            blocked.push(true);
            ranks.push(0);
        } else {
            blocked.push(false);
            ranks.push(crate::showdown_equity::rank_to_ordinal(rank_hand(combo, &board)));
        }
    }

    // Phase 2: for each valid hero combo, sweep over all opponent combos using
    // pre-computed ranks. Skip opponents that overlap with the board (blocked)
    // or with the hero's cards (bitmask check).
    combos
        .iter()
        .enumerate()
        .map(|(i, _)| {
            if blocked[i] {
                return None;
            }
            let our_rank = ranks[i];
            let hero_mask = combo_masks[i];
            let mut wins = 0u32;
            let mut ties = 0u32;
            let mut total = 0u32;

            for j in 0..combos.len() {
                if blocked[j] || combo_masks[j] & hero_mask != 0 {
                    continue;
                }
                total += 1;
                match our_rank.cmp(&ranks[j]) {
                    Ordering::Greater => wins += 1,
                    Ordering::Equal => ties += 1,
                    Ordering::Less => {}
                }
            }

            if total == 0 {
                return Some(0.5);
            }
            Some((f64::from(wins) + f64::from(ties) * 0.5) / f64::from(total))
        })
        .collect()
}

/// Check whether any card in `combo` appears in a board of any size.
fn cards_overlap(combo: [Card; 2], board: &[Card]) -> bool {
    board.iter().any(|b| *b == combo[0] || *b == combo[1])
}

/// Sample `n` random 4-card boards from the deck without replacement.
#[allow(dead_code)]
fn sample_turn_boards(deck: &[Card], n: usize, seed: u64) -> Vec<[Card; 4]> {
    sample_n_card_boards(deck, 4, n, seed)
        .into_iter()
        .map(|v| [v[0], v[1], v[2], v[3]])
        .collect()
}

/// Sample `n` random 3-card flop boards from the deck without replacement.
#[allow(dead_code)]
fn sample_flop_boards(deck: &[Card], n: usize, seed: u64) -> Vec<[Card; 3]> {
    sample_n_card_boards(deck, 3, n, seed)
        .into_iter()
        .map(|v| [v[0], v[1], v[2]])
        .collect()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_enumerate_combos_count() {
        let deck = build_deck();
        assert_eq!(deck.len(), 52);
        let combos = enumerate_combos(&deck);
        assert_eq!(combos.len(), 1326);
    }

    #[test]
    fn test_enumerate_combos_no_duplicates() {
        let deck = build_deck();
        let combos = enumerate_combos(&deck);
        for combo in &combos {
            assert_ne!(combo[0], combo[1], "combo has duplicate card");
        }
    }

    #[test]
    fn test_sample_boards_count() {
        let deck = build_deck();
        let boards = sample_boards(&deck, 100, 42);
        assert_eq!(boards.len(), 100);
    }

    #[test]
    fn test_sample_boards_no_internal_duplicates() {
        let deck = build_deck();
        let boards = sample_boards(&deck, 50, 99);
        for board in &boards {
            for i in 0..5 {
                for j in (i + 1)..5 {
                    assert_ne!(board[i], board[j], "board has duplicate card");
                }
            }
        }
    }

    #[test]
    fn test_sample_boards_deterministic() {
        let deck = build_deck();
        let b1 = sample_boards(&deck, 20, 123);
        let b2 = sample_boards(&deck, 20, 123);
        assert_eq!(b1, b2);
    }

    #[test]
    fn test_cards_overlap_true() {
        let deck = build_deck();
        let combo = [deck[0], deck[1]];
        let board = [deck[0], deck[5], deck[10], deck[15], deck[20]];
        assert!(cards_overlap(combo, &board));
    }

    #[test]
    fn test_cards_overlap_false() {
        let deck = build_deck();
        let combo = [deck[0], deck[1]];
        let board = [deck[2], deck[5], deck[10], deck[15], deck[20]];
        assert!(!cards_overlap(combo, &board));
    }

    #[test]
    fn test_compute_board_equities_blocked_combos() {
        let deck = build_deck();
        let combos = enumerate_combos(&deck);
        let board = [deck[0], deck[1], deck[2], deck[3], deck[4]];
        let equities = compute_board_equities(board, &combos);

        assert_eq!(equities.len(), 1326);

        // Count blocked vs valid.
        let blocked = equities.iter().filter(|e| e.is_none()).count();
        let valid = equities.iter().filter(|e| e.is_some()).count();

        // With 5 board cards removed, valid combos = C(47,2) = 1081
        assert_eq!(valid, 1081);
        assert_eq!(blocked, 1326 - 1081);

        // All valid equities should be in [0, 1].
        for eq in equities.iter().flatten() {
            assert!(
                (0.0..=1.0).contains(eq),
                "equity out of range: {eq}"
            );
        }
    }

    #[test]
    #[ignore] // slow: equity enumeration in debug mode
    fn test_cluster_river_basic() {
        // Use few boards for speed in tests.
        let result = cluster_river_with_boards(10, 50, 42, 20, |_| {});
        assert_eq!(result.header.street, Street::River);
        assert_eq!(result.header.bucket_count, 10);
        assert_eq!(result.header.board_count, 20);
        assert_eq!(result.header.combos_per_board, 1326);
        assert_eq!(result.buckets.len(), 20 * 1326);
        // All bucket IDs should be in [0, 10).
        for &b in &result.buckets {
            assert!(b < 10, "bucket {b} out of range");
        }
    }

    #[test]
    #[ignore] // slow: equity enumeration in debug mode
    fn test_cluster_river_deterministic() {
        let r1 = cluster_river_with_boards(5, 30, 123, 10, |_| {});
        let r2 = cluster_river_with_boards(5, 30, 123, 10, |_| {});
        assert_eq!(r1.buckets, r2.buckets);
    }

    #[test]
    #[ignore] // slow: equity enumeration in debug mode
    fn test_cluster_river_bucket_distribution() {
        // Verify that buckets are not all the same value (equity varies).
        let result = cluster_river_with_boards(5, 50, 42, 30, |_| {});
        let mut seen = std::collections::HashSet::new();
        for &b in &result.buckets {
            seen.insert(b);
        }
        // With 30 boards and 5 buckets, we should see at least 2 distinct
        // buckets (bucket 0 for blocked combos, plus at least one real one).
        assert!(
            seen.len() >= 2,
            "expected at least 2 distinct buckets, got {seen:?}"
        );
    }

    // -----------------------------------------------------------------------
    // Turn clustering tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_sample_turn_boards_count() {
        let deck = build_deck();
        let boards = sample_turn_boards(&deck, 50, 42);
        assert_eq!(boards.len(), 50);
    }

    #[test]
    fn test_sample_turn_boards_no_duplicates() {
        let deck = build_deck();
        let boards = sample_turn_boards(&deck, 50, 42);
        for board in &boards {
            for i in 0..4 {
                for j in (i + 1)..4 {
                    assert_ne!(board[i], board[j], "board has duplicate card");
                }
            }
        }
    }

    #[test]
    fn test_sample_turn_boards_deterministic() {
        let deck = build_deck();
        let b1 = sample_turn_boards(&deck, 20, 123);
        let b2 = sample_turn_boards(&deck, 20, 123);
        assert_eq!(b1, b2);
    }

    #[test]
    fn test_cards_overlap_4_true() {
        let deck = build_deck();
        let combo = [deck[0], deck[1]];
        let board = [deck[0], deck[5], deck[10], deck[15]];
        assert!(cards_overlap(combo, &board));
    }

    #[test]
    fn test_cards_overlap_4_false() {
        let deck = build_deck();
        let combo = [deck[0], deck[1]];
        let board = [deck[2], deck[5], deck[10], deck[15]];
        assert!(!cards_overlap(combo, &board));
    }

    #[test]
    fn test_equity_to_bucket_bounds() {
        // equity=0.0 should map to bucket 0
        assert_eq!(equity_to_bucket(0.0, 10), 0);
        // equity=1.0 should map to the last bucket (9 for K=10)
        assert_eq!(equity_to_bucket(1.0, 10), 9);
        // equity=0.5 with 10 buckets → bucket 5
        assert_eq!(equity_to_bucket(0.5, 10), 5);
        // equity just below 1.0
        assert_eq!(equity_to_bucket(0.99, 10), 9);
    }

    #[test]
    fn test_equity_to_bucket_single_bucket() {
        // With 1 bucket, everything maps to 0.
        assert_eq!(equity_to_bucket(0.0, 1), 0);
        assert_eq!(equity_to_bucket(0.5, 1), 0);
        assert_eq!(equity_to_bucket(1.0, 1), 0);
    }

    #[test]
    fn test_build_next_street_histogram_river_sums_to_one() {
        let deck = build_deck();
        // combo: first two cards, board: next four cards (no overlap)
        let combo = [deck[0], deck[1]];
        let board = [deck[2], deck[3], deck[4], deck[5]];
        let histogram = build_next_street_histogram(combo, &board, &deck, 10);

        assert_eq!(histogram.len(), 10);

        let sum: f64 = histogram.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-10,
            "histogram should sum to 1.0, got {sum}"
        );

        // Every entry should be non-negative.
        for &h in &histogram {
            assert!(h >= 0.0);
        }
    }

    #[test]
    fn test_build_next_street_histogram_river_card_count() {
        // With 4 board cards + 2 hole cards = 6 used, there are 46 river cards.
        let deck = build_deck();
        let combo = [deck[0], deck[1]];
        let board = [deck[2], deck[3], deck[4], deck[5]];
        let histogram = build_next_street_histogram(combo, &board, &deck, 5);

        // The histogram entries times 46 should give integer counts.
        let total_rivers = 46.0;
        for &h in &histogram {
            let count = h * total_rivers;
            assert!(
                (count - count.round()).abs() < 1e-8,
                "expected integer count, got {count}"
            );
        }
    }

    #[test]
    #[ignore] // slow: equity enumeration in debug mode
    fn test_cluster_turn_basic() {
        // First cluster river with small params.
        let river = cluster_river_with_boards(5, 30, 42, 10, |_| {});
        // Then cluster turn using river buckets.
        let turn = cluster_turn_with_boards(&river, 5, 30, 42, 10, |_| {});

        assert_eq!(turn.header.street, Street::Turn);
        assert_eq!(turn.header.bucket_count, 5);
        assert_eq!(turn.header.board_count, 10);
        assert_eq!(turn.header.combos_per_board, 1326);
        assert_eq!(turn.buckets.len(), 10 * 1326);

        for &b in &turn.buckets {
            assert!(b < 5, "bucket {b} out of range");
        }
    }

    #[test]
    #[ignore] // slow: equity enumeration in debug mode
    fn test_cluster_turn_deterministic() {
        let river = cluster_river_with_boards(5, 30, 42, 10, |_| {});
        let t1 = cluster_turn_with_boards(&river, 3, 20, 123, 8, |_| {});
        let t2 = cluster_turn_with_boards(&river, 3, 20, 123, 8, |_| {});
        assert_eq!(t1.buckets, t2.buckets);
    }

    #[test]
    #[ignore] // slow: equity enumeration in debug mode
    fn test_cluster_turn_bucket_distribution() {
        let river = cluster_river_with_boards(5, 30, 42, 10, |_| {});
        let turn = cluster_turn_with_boards(&river, 4, 30, 42, 15, |_| {});
        let mut seen = std::collections::HashSet::new();
        for &b in &turn.buckets {
            seen.insert(b);
        }
        // Should see at least 2 distinct buckets.
        assert!(
            seen.len() >= 2,
            "expected at least 2 distinct turn buckets, got {seen:?}"
        );
    }

    // -----------------------------------------------------------------------
    // Flop clustering tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_sample_flop_boards_count() {
        let deck = build_deck();
        let boards = sample_flop_boards(&deck, 50, 42);
        assert_eq!(boards.len(), 50);
    }

    #[test]
    fn test_sample_flop_boards_no_duplicates() {
        let deck = build_deck();
        let boards = sample_flop_boards(&deck, 50, 42);
        for board in &boards {
            for i in 0..3 {
                for j in (i + 1)..3 {
                    assert_ne!(board[i], board[j], "board has duplicate card");
                }
            }
        }
    }

    #[test]
    fn test_sample_flop_boards_deterministic() {
        let deck = build_deck();
        let b1 = sample_flop_boards(&deck, 20, 123);
        let b2 = sample_flop_boards(&deck, 20, 123);
        assert_eq!(b1, b2);
    }

    #[test]
    fn test_cards_overlap_3_true() {
        let deck = build_deck();
        let combo = [deck[0], deck[1]];
        let board = [deck[0], deck[5], deck[10]];
        assert!(cards_overlap(combo, &board));
    }

    #[test]
    fn test_cards_overlap_3_false() {
        let deck = build_deck();
        let combo = [deck[0], deck[1]];
        let board = [deck[2], deck[5], deck[10]];
        assert!(!cards_overlap(combo, &board));
    }

    #[test]
    fn test_build_next_street_histogram_turn_sums_to_one() {
        let deck = build_deck();
        let combo = [deck[0], deck[1]];
        let board = [deck[10], deck[20], deck[30]];
        let hist = build_next_street_histogram(combo, &board, &deck, 5);

        assert_eq!(hist.len(), 5);

        let sum: f64 = hist.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-10,
            "histogram should sum to 1.0, got {sum}"
        );

        for &h in &hist {
            assert!(h >= 0.0);
        }
    }

    #[test]
    fn test_build_next_street_histogram_turn_card_count() {
        // With 3 board cards + 2 hole cards = 5 used, there are 47 turn cards.
        let deck = build_deck();
        let combo = [deck[0], deck[1]];
        let board = [deck[10], deck[20], deck[30]];
        let histogram = build_next_street_histogram(combo, &board, &deck, 5);

        let total_turns = 47.0;
        for &h in &histogram {
            let count = h * total_turns;
            assert!(
                (count - count.round()).abs() < 1e-8,
                "expected integer count, got {count}"
            );
        }
    }

    #[test]
    #[ignore] // slow: equity enumeration in debug mode
    fn test_cluster_flop_basic() {
        // Build dependencies: river -> turn -> flop.
        let river = cluster_river_with_boards(5, 30, 42, 10, |_| {});
        let turn = cluster_turn_with_boards(&river, 5, 20, 42, 10, |_| {});
        let flop = cluster_flop_with_boards(&turn, 3, 20, 42, 5, |_| {});

        assert_eq!(flop.header.street, Street::Flop);
        assert_eq!(flop.header.bucket_count, 3);
        assert_eq!(flop.header.board_count, 5);
        assert_eq!(flop.header.combos_per_board, 1326);
        assert_eq!(flop.buckets.len(), 5 * 1326);

        for &b in &flop.buckets {
            assert!(b < 3, "bucket {b} out of range");
        }
    }

    #[test]
    #[ignore] // slow: equity enumeration in debug mode
    fn test_cluster_flop_deterministic() {
        let river = cluster_river_with_boards(5, 30, 42, 10, |_| {});
        let turn = cluster_turn_with_boards(&river, 5, 20, 42, 10, |_| {});
        let f1 = cluster_flop_with_boards(&turn, 3, 20, 123, 5, |_| {});
        let f2 = cluster_flop_with_boards(&turn, 3, 20, 123, 5, |_| {});
        assert_eq!(f1.buckets, f2.buckets);
    }

    #[test]
    #[ignore] // slow: equity enumeration in debug mode
    fn test_cluster_flop_bucket_distribution() {
        let river = cluster_river_with_boards(5, 30, 42, 10, |_| {});
        let turn = cluster_turn_with_boards(&river, 5, 20, 42, 10, |_| {});
        let flop = cluster_flop_with_boards(&turn, 4, 20, 42, 10, |_| {});
        let mut seen = std::collections::HashSet::new();
        for &b in &flop.buckets {
            seen.insert(b);
        }
        assert!(
            seen.len() >= 2,
            "expected at least 2 distinct flop buckets, got {seen:?}"
        );
    }

    // -----------------------------------------------------------------------
    // Preflop clustering tests
    // -----------------------------------------------------------------------

    #[test]
    #[ignore] // slow: equity enumeration in debug mode
    fn test_preflop_cluster_basic() {
        let preflop = cluster_preflop_with_boards(10, 20, 42, 50, |_| {});
        assert_eq!(preflop.header.street, Street::Preflop);
        assert_eq!(preflop.header.bucket_count, 10);
        assert_eq!(preflop.header.board_count, 1);
        assert_eq!(preflop.header.combos_per_board, 1326);
        assert_eq!(preflop.buckets.len(), 1326);
        for &b in &preflop.buckets {
            assert!(b < 10, "bucket {b} out of range");
        }
    }

    #[test]
    #[ignore] // slow: equity enumeration in debug mode
    fn test_preflop_cluster_deterministic() {
        let a = cluster_preflop_with_boards(10, 20, 42, 50, |_| {});
        let b = cluster_preflop_with_boards(10, 20, 42, 50, |_| {});
        assert_eq!(a.buckets, b.buckets);
    }

    #[test]
    #[ignore] // slow: equity enumeration in debug mode
    fn test_preflop_cluster_bucket_distribution() {
        let preflop = cluster_preflop_with_boards(169, 50, 42, 80, |_| {});
        let unique: std::collections::HashSet<u16> =
            preflop.buckets.iter().copied().collect();
        assert!(
            unique.len() > 100,
            "expected many unique buckets, got {}",
            unique.len()
        );
    }

    #[test]
    #[ignore] // slow: equity enumeration in debug mode
    fn test_preflop_avg_equities_range() {
        let deck = build_deck();
        let combos = enumerate_combos(&deck);
        let boards = sample_boards(&deck, 30, 42);
        let equities = compute_combo_avg_equities(&combos, &boards, &|_| {});
        assert_eq!(equities.len(), 1326);
        for &eq in &equities {
            assert!(
                (0.0..=1.0).contains(&eq),
                "average equity out of range: {eq}"
            );
        }
    }

    // -----------------------------------------------------------------------
    // combo_index tests
    // -----------------------------------------------------------------------

    #[test]
    fn combo_index_first_and_last() {
        let deck = build_deck();
        assert_eq!(combo_index(deck[0], deck[1]), 0);
        assert_eq!(combo_index(deck[50], deck[51]), 1325);
    }

    #[test]
    fn combo_index_order_independent() {
        let c0 = Card::new(Value::Ace, Suit::Spade);
        let c1 = Card::new(Value::King, Suit::Heart);
        assert_eq!(combo_index(c0, c1), combo_index(c1, c0));
    }

    #[test]
    fn combo_index_matches_enumeration() {
        let deck = build_deck();
        let combos = enumerate_combos(&deck);
        for (expected_idx, combo) in combos.iter().enumerate() {
            let actual = combo_index(combo[0], combo[1]);
            assert_eq!(
                actual, expected_idx as u16,
                "Mismatch at combo {:?}: expected {expected_idx}, got {actual}",
                combo
            );
        }
    }

    // -----------------------------------------------------------------------
    // Canonical board enumeration tests
    // -----------------------------------------------------------------------

    #[test]
    fn canonical_flops_count() {
        let flops = enumerate_canonical_flops();
        assert_eq!(flops.len(), 1755);
        let total_weight: u32 = flops.iter().map(|f| f.weight).sum();
        assert_eq!(total_weight, 22100); // C(52,3)
    }

    #[test]
    fn canonical_turns_reasonable_count() {
        let turns = enumerate_canonical_turns();
        assert!(turns.len() > 10_000, "too few turns: {}", turns.len());
        assert!(turns.len() < 25_000, "too many turns: {}", turns.len());
        let total_weight: u32 = turns.iter().map(|t| t.weight).sum();
        assert_eq!(total_weight, 22100 * 49);
    }

    #[test]
    #[ignore]
    fn canonical_rivers_reasonable_count() {
        let rivers = enumerate_canonical_rivers();
        assert!(rivers.len() > 500_000, "too few rivers: {}", rivers.len());
        assert!(rivers.len() < 1_000_000, "too many rivers: {}", rivers.len());
        let total_weight: u64 = rivers.iter().map(|r| u64::from(r.weight)).sum();
        assert_eq!(total_weight, 22100 * 49 * 48);
    }

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
}
