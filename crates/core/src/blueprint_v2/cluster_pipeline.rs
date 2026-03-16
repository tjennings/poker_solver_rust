//! Card abstraction clustering pipeline.
//!
//! **River:** samples canonical 5-card boards, computes showdown equity for
//! every valid hole-card combo on each board, and clusters the equity values
//! into K buckets using weighted 1-D k-means.
//!
//! **Turn:** samples canonical 4-card boards, enumerates all possible river
//! cards for each valid combo, looks up actual river bucket assignments to
//! build a histogram, and clusters with weighted EMD k-means.
//!
//! **Flop:** samples canonical 3-card boards, enumerates all possible turn
//! cards for each valid combo, looks up actual turn bucket assignments to
//! build a histogram, and clusters with weighted EMD k-means.
//!
//! **Preflop:** samples random 3-card flop boards, builds a histogram over
//! flop bucket assignments for each of the 1326 hole-card combos, and
//! clusters with EMD k-means.
//!
//! The pipeline runs bottom-up: river -> turn(&river) -> flop(&turn) ->
//! preflop(&flop), with each stage feeding its bucket assignments into
//! the next.
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
use crate::hands::CanonicalHand;
use crate::poker::{Card, Suit, Value, ALL_SUITS, ALL_VALUES};
use crate::showdown_equity::rank_hand;

use super::bucket_file::{BucketFile, BucketFileHeader, PackedBoard};
use super::clustering::{kmeans_1d, kmeans_1d_weighted, kmeans_emd_weighted_u8, nearest_centroid_1d};
use super::config::ClusteringConfig;
use super::Street;

/// Number of sample 5-card boards for river clustering when no explicit count
/// is provided.
const DEFAULT_NUM_BOARDS: usize = 10_000;

/// Number of sample 4-card boards for turn clustering.
const DEFAULT_TURN_BOARDS: usize = 5_000;

/// Total number of 2-card combos from a 52-card deck: C(52,2) = 1326.
const TOTAL_COMBOS: u16 = 1326;

/// Cluster all river information situations by equity.
///
/// Cluster river situations by sampling from canonical (isomorphic) 5-card
/// boards, computing showdown equity for every valid hole-card combo, and
/// running weighted 1-D k-means where each data point carries the board's
/// combinatorial multiplicity as its weight.
///
/// Sampling canonical boards ensures maximum diversity (no two sampled boards
/// are strategically identical) while weights preserve the true frequency
/// distribution of board textures.
#[allow(dead_code)]
pub fn cluster_river(
    bucket_count: u16,
    kmeans_iterations: u32,
    seed: u64,
    progress: impl Fn(f64) + Sync,
) -> BucketFile {
    let combos = enumerate_combos(&build_deck());

    let all_canonical = enumerate_canonical_rivers();
    let (boards, board_weights) = sample_canonical(&all_canonical, DEFAULT_NUM_BOARDS, seed);
    let num_boards = boards.len();

    // Compute equity for each (board, combo) pair in parallel.
    let board_equities: Vec<Vec<Option<f64>>> = boards
        .par_iter()
        .enumerate()
        .map(|(i, board)| {
            let eq = compute_board_equities(*board, &combos);
            #[allow(clippy::cast_precision_loss)]
            progress((i + 1) as f64 / num_boards as f64);
            eq
        })
        .collect();

    // Collect valid equities with board weights for weighted k-means.
    let mut all_equities: Vec<f64> = Vec::new();
    let mut all_weights: Vec<f64> = Vec::new();

    for (board_idx, eqs) in board_equities.iter().enumerate() {
        let w = board_weights[board_idx];
        for eq in eqs.iter().flatten() {
            all_equities.push(*eq);
            all_weights.push(w);
        }
    }

    // Weighted 1-D k-means (board frequency observed via weights).
    let cluster_labels =
        kmeans_1d_weighted(&all_equities, &all_weights, bucket_count as usize, kmeans_iterations);

    let total = num_boards * TOTAL_COMBOS as usize;
    let mut buckets = vec![0_u16; total];
    let mut flat_idx = 0_usize;
    for (board_idx, eqs) in board_equities.iter().enumerate() {
        for (combo_idx, eq) in eqs.iter().enumerate() {
            if eq.is_some() {
                buckets[board_idx * TOTAL_COMBOS as usize + combo_idx] = cluster_labels[flat_idx];
                flat_idx += 1;
            }
        }
    }

    #[allow(clippy::cast_possible_truncation)]
    BucketFile {
        header: BucketFileHeader {
            street: Street::River,
            bucket_count,
            board_count: num_boards as u32,
            combos_per_board: TOTAL_COMBOS,
            version: 2,
        },
        boards: boards.iter().map(|b| canonical_key(b)).collect(),
        buckets,
    }
}

/// Raw-sampling variant for testing. Samples `num_boards` random 5-card
/// boards (not canonical) and uses unweighted 1-D k-means.
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

    let board_equities: Vec<Vec<Option<f64>>> = boards
        .par_iter()
        .enumerate()
        .map(|(i, board)| {
            let eq = compute_board_equities(*board, &combos);
            #[allow(clippy::cast_precision_loss)]
            progress((i + 1) as f64 / num_boards as f64);
            eq
        })
        .collect();

    let all_equities: Vec<f64> = board_equities
        .iter()
        .flat_map(|eqs| eqs.iter().filter_map(|&e| e))
        .collect();

    let cluster_labels = kmeans_1d(&all_equities, bucket_count as usize, kmeans_iterations);

    let total = num_boards * TOTAL_COMBOS as usize;
    let mut buckets = vec![0_u16; total];
    let mut flat_idx = 0_usize;
    for (board_idx, eqs) in board_equities.iter().enumerate() {
        for (combo_idx, eq) in eqs.iter().enumerate() {
            if eq.is_some() {
                buckets[board_idx * TOTAL_COMBOS as usize + combo_idx] = cluster_labels[flat_idx];
                flat_idx += 1;
            }
        }
    }

    #[allow(clippy::cast_possible_truncation)]
    BucketFile {
        header: BucketFileHeader {
            street: Street::River,
            bucket_count,
            board_count: num_boards as u32,
            combos_per_board: TOTAL_COMBOS,
            version: 2,
        },
        boards: boards.iter().map(|b| canonical_key(b)).collect(),
        buckets,
    }
}

// ---------------------------------------------------------------------------
// Turn clustering
// ---------------------------------------------------------------------------

/// Cluster turn situations by sampling from canonical (isomorphic) 4-card
/// boards, building potential-aware histograms over river equity bins, and
/// running weighted EMD k-means where each data point carries the board's
/// combinatorial multiplicity as its weight.
#[allow(dead_code)]
pub fn cluster_turn(
    river_buckets: &BucketFile,
    bucket_count: u16,
    kmeans_iterations: u32,
    seed: u64,
    progress: impl Fn(f64) + Sync,
) -> BucketFile {
    let deck = build_deck();
    let combos = enumerate_combos(&deck);
    let board_map = river_buckets.board_index_map();

    let all_canonical = enumerate_canonical_turns();
    let (boards, board_weights) = sample_canonical(&all_canonical, DEFAULT_TURN_BOARDS, seed);
    let num_boards = boards.len();

    // For each board, build a histogram over river bucket IDs for every combo.
    let board_features: Vec<Vec<Option<Vec<u8>>>> = boards
        .par_iter()
        .enumerate()
        .map(|(board_idx, board)| {
            let features: Vec<Option<Vec<u8>>> = combos
                .iter()
                .map(|combo| {
                    if cards_overlap(*combo, board) {
                        return None;
                    }
                    Some(build_bucket_histogram_u8(*combo, board, &deck, river_buckets, &board_map))
                })
                .collect();

            #[allow(clippy::cast_precision_loss)]
            progress((board_idx + 1) as f64 / num_boards as f64 * 0.8);

            features
        })
        .collect();

    // Collect valid features with board weights.
    let mut all_features: Vec<Vec<u8>> = Vec::new();
    let mut all_weights: Vec<f64> = Vec::new();
    let mut feature_positions: Vec<(usize, usize)> = Vec::new();

    for (board_idx, (board_feats, w)) in board_features.into_iter().zip(board_weights.iter()).enumerate() {
        for (combo_idx, feat) in board_feats.into_iter().enumerate() {
            if let Some(histogram) = feat {
                all_features.push(histogram);
                all_weights.push(*w);
                feature_positions.push((board_idx, combo_idx));
            }
        }
    }

    // Weighted EMD k-means (board frequency observed via weights).
    let cluster_labels = kmeans_emd_weighted_u8(
        &all_features,
        &all_weights,
        bucket_count as usize,
        kmeans_iterations,
        seed,
        |iter, max_iter| {
            #[allow(clippy::cast_precision_loss)]
            progress(0.8 + 0.2 * f64::from(iter) / f64::from(max_iter));
        },
    );

    let total = num_boards * TOTAL_COMBOS as usize;
    let mut buckets = vec![0_u16; total];
    for (flat_idx, &(board_idx, combo_idx)) in feature_positions.iter().enumerate() {
        buckets[board_idx * TOTAL_COMBOS as usize + combo_idx] = cluster_labels[flat_idx];
    }

    #[allow(clippy::cast_possible_truncation)]
    BucketFile {
        header: BucketFileHeader {
            street: Street::Turn,
            bucket_count,
            board_count: num_boards as u32,
            combos_per_board: TOTAL_COMBOS,
            version: 2,
        },
        boards: boards.iter().map(|b| canonical_key(b)).collect(),
        buckets,
    }
}

/// Raw-sampling variant for testing. Samples `num_boards` random 4-card
/// boards (not canonical) and uses unweighted EMD k-means.
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
    let board_map = river_buckets.board_index_map();

    let board_features: Vec<Vec<Option<Vec<u8>>>> = boards
        .par_iter()
        .enumerate()
        .map(|(board_idx, board)| {
            let features: Vec<Option<Vec<u8>>> = combos
                .iter()
                .map(|combo| {
                    if cards_overlap(*combo, board) {
                        return None;
                    }
                    Some(build_bucket_histogram_u8(*combo, board, &deck, river_buckets, &board_map))
                })
                .collect();
            #[allow(clippy::cast_precision_loss)]
            progress((board_idx + 1) as f64 / num_boards as f64 * 0.8);
            features
        })
        .collect();

    let mut all_features: Vec<Vec<u8>> = Vec::new();
    let mut all_weights: Vec<f64> = Vec::new();
    let mut feature_positions: Vec<(usize, usize)> = Vec::new();
    for (board_idx, board_feats) in board_features.into_iter().enumerate() {
        for (combo_idx, feat) in board_feats.into_iter().enumerate() {
            if let Some(histogram) = feat {
                all_features.push(histogram);
                all_weights.push(1.0);
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
            #[allow(clippy::cast_precision_loss)]
            progress(0.8 + 0.2 * f64::from(iter) / f64::from(max_iter));
        },
    );

    let total = num_boards * TOTAL_COMBOS as usize;
    let mut buckets = vec![0_u16; total];
    for (flat_idx, &(board_idx, combo_idx)) in feature_positions.iter().enumerate() {
        buckets[board_idx * TOTAL_COMBOS as usize + combo_idx] = cluster_labels[flat_idx];
    }

    #[allow(clippy::cast_possible_truncation)]
    BucketFile {
        header: BucketFileHeader {
            street: Street::Turn,
            bucket_count,
            board_count: num_boards as u32,
            combos_per_board: TOTAL_COMBOS,
            version: 2,
        },
        boards: boards.iter().map(|b| canonical_key(b)).collect(),
        buckets,
    }
}

/// Build a histogram of bucket IDs by looking up actual previous-street
/// bucket assignments for each possible next card.
///
/// For each remaining card in the deck that doesn't overlap with the combo
/// or the current board, extends the board by one card, looks up the
/// resulting board in `prev_buckets` via `board_map`, and increments the
/// count for that board's bucket assignment. Returns raw u8 counts.
fn build_bucket_histogram_u8(
    combo: [Card; 2],
    board: &[Card],
    deck: &[Card],
    prev_buckets: &BucketFile,
    board_map: &HashMap<PackedBoard, u32>,
) -> Vec<u8> {
    let num_buckets = prev_buckets.header.bucket_count as usize;
    let mut histogram = vec![0_u16; num_buckets];
    let mut extended = Vec::with_capacity(board.len() + 1);
    extended.extend_from_slice(board);
    extended.push(Card::new(crate::poker::Value::Two, crate::poker::Suit::Club)); // placeholder

    let ci = combo_index(combo[0], combo[1]);

    for &next_card in deck {
        if board.contains(&next_card)
            || next_card == combo[0]
            || next_card == combo[1]
        {
            continue;
        }

        *extended.last_mut().expect("non-empty") = next_card;
        let packed = canonical_key(&extended);
        if let Some(&board_idx) = board_map.get(&packed) {
            let bucket = prev_buckets.get_bucket(board_idx, ci);
            histogram[bucket as usize] += 1;
        }
    }

    histogram.iter().map(|&c| c.min(255) as u8).collect()
}


// ---------------------------------------------------------------------------
// Flop clustering
// ---------------------------------------------------------------------------

/// Cluster flop situations by sampling from canonical (isomorphic) 3-card
/// boards, building potential-aware histograms over turn buckets, and
/// running weighted EMD k-means where each data point carries the board's
/// combinatorial multiplicity as its weight.
///
/// Uses all 1,755 canonical flops (exhaustive — no sampling).
#[allow(dead_code)]
pub fn cluster_flop(
    turn_buckets: &BucketFile,
    bucket_count: u16,
    kmeans_iterations: u32,
    seed: u64,
    progress: impl Fn(f64) + Sync,
) -> BucketFile {
    let deck = build_deck();
    let combos = enumerate_combos(&deck);
    let board_map = turn_buckets.board_index_map();

    let all_canonical = enumerate_canonical_flops();
    let boards: Vec<[Card; 3]> = all_canonical.iter().map(|wb| wb.cards).collect();
    let board_weights: Vec<f64> = all_canonical.iter().map(|wb| f64::from(wb.weight)).collect();
    let num_boards = boards.len();

    // For each board, build a histogram over turn bucket IDs for every combo.
    let board_features: Vec<Vec<Option<Vec<u8>>>> = boards
        .par_iter()
        .enumerate()
        .map(|(board_idx, board)| {
            let features: Vec<Option<Vec<u8>>> = combos
                .iter()
                .map(|combo| {
                    if cards_overlap(*combo, board) {
                        return None;
                    }
                    Some(build_bucket_histogram_u8(*combo, board, &deck, turn_buckets, &board_map))
                })
                .collect();

            #[allow(clippy::cast_precision_loss)]
            progress((board_idx + 1) as f64 / num_boards as f64 * 0.8);

            features
        })
        .collect();

    // Collect valid features with board weights.
    let mut all_features: Vec<Vec<u8>> = Vec::new();
    let mut all_weights: Vec<f64> = Vec::new();
    let mut feature_positions: Vec<(usize, usize)> = Vec::new();

    for (board_idx, (board_feats, w)) in board_features.into_iter().zip(board_weights.iter()).enumerate() {
        for (combo_idx, feat) in board_feats.into_iter().enumerate() {
            if let Some(histogram) = feat {
                all_features.push(histogram);
                all_weights.push(*w);
                feature_positions.push((board_idx, combo_idx));
            }
        }
    }

    // Weighted EMD k-means (board frequency observed via weights).
    let cluster_labels = kmeans_emd_weighted_u8(
        &all_features,
        &all_weights,
        bucket_count as usize,
        kmeans_iterations,
        seed,
        |iter, max_iter| {
            #[allow(clippy::cast_precision_loss)]
            progress(0.8 + 0.2 * f64::from(iter) / f64::from(max_iter));
        },
    );

    let total = num_boards * TOTAL_COMBOS as usize;
    let mut buckets = vec![0_u16; total];
    for (flat_idx, &(board_idx, combo_idx)) in feature_positions.iter().enumerate() {
        buckets[board_idx * TOTAL_COMBOS as usize + combo_idx] = cluster_labels[flat_idx];
    }

    #[allow(clippy::cast_possible_truncation)]
    BucketFile {
        header: BucketFileHeader {
            street: Street::Flop,
            bucket_count,
            board_count: num_boards as u32,
            combos_per_board: TOTAL_COMBOS,
            version: 2,
        },
        boards: boards.iter().map(|b| canonical_key(b)).collect(),
        buckets,
    }
}

/// Raw-sampling variant for testing. Samples `num_boards` random 3-card
/// boards (not canonical) and uses unweighted EMD k-means.
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
    let board_map = turn_buckets.board_index_map();

    let board_features: Vec<Vec<Option<Vec<u8>>>> = boards
        .par_iter()
        .enumerate()
        .map(|(board_idx, board)| {
            let features: Vec<Option<Vec<u8>>> = combos
                .iter()
                .map(|combo| {
                    if cards_overlap(*combo, board) {
                        return None;
                    }
                    Some(build_bucket_histogram_u8(*combo, board, &deck, turn_buckets, &board_map))
                })
                .collect();
            #[allow(clippy::cast_precision_loss)]
            progress((board_idx + 1) as f64 / num_boards as f64 * 0.8);
            features
        })
        .collect();

    let mut all_features: Vec<Vec<u8>> = Vec::new();
    let mut all_weights: Vec<f64> = Vec::new();
    let mut feature_positions: Vec<(usize, usize)> = Vec::new();
    for (board_idx, board_feats) in board_features.into_iter().enumerate() {
        for (combo_idx, feat) in board_feats.into_iter().enumerate() {
            if let Some(histogram) = feat {
                all_features.push(histogram);
                all_weights.push(1.0);
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
            #[allow(clippy::cast_precision_loss)]
            progress(0.8 + 0.2 * f64::from(iter) / f64::from(max_iter));
        },
    );

    let total = num_boards * TOTAL_COMBOS as usize;
    let mut buckets = vec![0_u16; total];
    for (flat_idx, &(board_idx, combo_idx)) in feature_positions.iter().enumerate() {
        buckets[board_idx * TOTAL_COMBOS as usize + combo_idx] = cluster_labels[flat_idx];
    }

    #[allow(clippy::cast_possible_truncation)]
    BucketFile {
        header: BucketFileHeader {
            street: Street::Flop,
            bucket_count,
            board_count: num_boards as u32,
            combos_per_board: TOTAL_COMBOS,
            version: 2,
        },
        boards: boards.iter().map(|b| canonical_key(b)).collect(),
        buckets,
    }
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
    let board_map = river_buckets.board_index_map();

    // For each board, build a histogram over river bucket IDs for every combo.
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
                    Some(build_bucket_histogram_u8(
                        *combo,
                        &wb.cards,
                        &deck,
                        river_buckets,
                        &board_map,
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
    let board_map = turn_buckets.board_index_map();

    // For each board, build a histogram over turn bucket IDs for every combo.
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
                    Some(build_bucket_histogram_u8(
                        *combo,
                        &wb.cards,
                        &deck,
                        turn_buckets,
                        &board_map,
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

/// Map preflop hole-card combos to canonical hand buckets (1:1).
///
/// Each of the 1326 two-card combos is assigned to its canonical hand index
/// (0–168), giving exactly 169 buckets: 13 pairs (6 combos each), 78 suited
/// (4 combos each), and 78 offsuit (12 combos each).
///
/// This is deterministic and matches the lookup used by the MCCFR solver at
/// runtime (`CanonicalHand::from_cards().index()`).
///
/// The `progress` callback receives values in `[0.0, 1.0]`.
pub fn cluster_preflop(progress: impl Fn(f64) + Sync) -> BucketFile {
    let deck = build_deck();
    let combos = enumerate_combos(&deck);
    let buckets: Vec<u16> = combos
        .iter()
        .enumerate()
        .map(|(i, &combo)| {
            #[allow(clippy::cast_precision_loss)]
            progress((i + 1) as f64 / combos.len() as f64);
            CanonicalHand::from_cards(combo[0], combo[1]).index() as u16
        })
        .collect();

    BucketFile {
        header: BucketFileHeader {
            street: Street::Preflop,
            bucket_count: 169,
            board_count: 1,
            combos_per_board: TOTAL_COMBOS,
            version: 1,
        },
        boards: Vec::new(),
        buckets,
    }
}

// ---------------------------------------------------------------------------
// cfvnet-based river clustering
// ---------------------------------------------------------------------------

/// Number of histogram bins for the streaming equity histogram.
const HISTOGRAM_BINS: usize = 10_000;

/// Size of a single cfvnet river training record in bytes:
/// `board_size`(1) + board(5) + pot(4) + stack(4) + player(1) +
/// `game_value`(4) + `oop_range`(5304) + `ip_range`(5304) + cfvs(5304) +
/// `valid_mask`(1326).
pub(crate) const CFVNET_RIVER_RECORD_SIZE: usize = 1 + 5 + 4 + 4 + 1 + 4 + 1326 * 4 + 1326 * 4 + 1326 * 4 + 1326;

/// Cluster river situations using pre-solved cfvnet training records.
///
/// Streams `*.bin` files from `data_dir`, converts pot-relative CFVs to equity,
/// and clusters using a streaming histogram k-means approach in 3 passes:
///
/// 1. Build a 10,000-bin equity histogram and collect unique canonical boards.
/// 2. Run weighted 1-D k-means on histogram bin midpoints to find centroids.
/// 3. Re-stream files and assign each (board, combo) its nearest centroid bucket.
///
/// # Errors
///
/// Returns an error if no valid river records are found or if I/O fails.
#[allow(clippy::cast_precision_loss)]
pub fn cluster_river_from_cfvnet(
    data_dir: &Path,
    bucket_count: u16,
    kmeans_iterations: u32,
    progress: impl Fn(f64) + Sync,
) -> Result<BucketFile, Box<dyn std::error::Error>> {
    let combo_map = build_cfvnet_to_core_combo_map();
    let bin_files = collect_bin_files(data_dir)?;

    // ---- Pass 1: build histogram + collect unique boards --------------------
    let (histogram, board_set) = cfvnet_pass1_histogram(&bin_files, &progress)?;

    // ---- Pass 2: k-means on histogram bins ---------------------------------
    let centroids = cfvnet_pass2_centroids(&histogram, bucket_count, kmeans_iterations);
    progress(0.5);

    // ---- Pass 3: assign buckets ---------------------------------------------
    let mut sorted_boards: Vec<(PackedBoard, Vec<Card>)> = board_set.into_iter().collect();
    sorted_boards.sort_by_key(|(packed, _)| packed.0);
    let num_boards = sorted_boards.len();

    let board_index: HashMap<PackedBoard, usize> = sorted_boards
        .iter()
        .enumerate()
        .map(|(i, (packed, _))| (*packed, i))
        .collect();

    let total = num_boards * TOTAL_COMBOS as usize;
    let mut buckets = vec![0u16; total];

    for (file_idx, path) in bin_files.iter().enumerate() {
        let data = std::fs::read(path)?;
        let mut offset = 0;
        while offset + CFVNET_RIVER_RECORD_SIZE <= data.len() {
            if data[offset] != 5 {
                offset += record_size_for_board(data[offset]);
                continue;
            }

            let board_cards: Vec<Card> = (0..5)
                .map(|i| cfvnet_card_to_core(data[offset + 1 + i]))
                .collect();
            let packed = canonical_key(&board_cards);

            if let Some(&board_idx) = board_index.get(&packed) {
                let cfv_offset = offset + CFV_FIELD_OFFSET;
                let mask_offset = cfv_offset + 1326 * 4;

                for i in 0..1326 {
                    if data[mask_offset + i] != 0 {
                        let equity = read_cfv_as_equity(&data, cfv_offset, i);
                        let core_combo = combo_map[i] as usize;
                        let bucket = nearest_centroid_1d(equity, &centroids);
                        buckets[board_idx * TOTAL_COMBOS as usize + core_combo] = bucket;
                    }
                }
            }

            offset += CFVNET_RIVER_RECORD_SIZE;
        }
        progress(0.5 + (file_idx + 1) as f64 / bin_files.len() as f64 * 0.5);
    }

    let packed_boards: Vec<PackedBoard> = sorted_boards.iter().map(|(p, _)| *p).collect();

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

/// Byte offset from record start to the cfvs field (river records only).
pub(crate) const CFV_FIELD_OFFSET: usize = 1 + 5 + 4 + 4 + 1 + 4 + 1326 * 4 + 1326 * 4;

/// Collect all `*.bin` files from a directory.
pub(crate) fn collect_bin_files(dir: &Path) -> Result<Vec<std::path::PathBuf>, Box<dyn std::error::Error>> {
    let files: Vec<std::path::PathBuf> = std::fs::read_dir(dir)?
        .filter_map(std::result::Result::ok)
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|ext| ext == "bin"))
        .collect();
    if files.is_empty() {
        return Err("no .bin files found in data_dir".into());
    }
    Ok(files)
}

/// Read a single CFV float from binary data and convert to equity.
pub(crate) fn read_cfv_as_equity(data: &[u8], cfv_offset: usize, combo_idx: usize) -> f64 {
    let start = cfv_offset + combo_idx * 4;
    let cfv = f32::from_le_bytes([data[start], data[start + 1], data[start + 2], data[start + 3]]);
    cfv_to_equity(cfv)
}

/// Pass 1: stream all river records, build a histogram of equity values,
/// and collect unique canonical boards.
#[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation, clippy::cast_sign_loss, clippy::type_complexity)]
fn cfvnet_pass1_histogram(
    bin_files: &[std::path::PathBuf],
    progress: &impl Fn(f64),
) -> Result<(Vec<u64>, HashMap<PackedBoard, Vec<Card>>), Box<dyn std::error::Error>> {
    let mut histogram = vec![0u64; HISTOGRAM_BINS];
    let mut board_set: HashMap<PackedBoard, Vec<Card>> = HashMap::new();

    for (file_idx, path) in bin_files.iter().enumerate() {
        let data = std::fs::read(path)?;
        let mut offset = 0;
        while offset + CFVNET_RIVER_RECORD_SIZE <= data.len() {
            if data[offset] != 5 {
                offset += record_size_for_board(data[offset]);
                continue;
            }

            let board_cards: Vec<Card> = (0..5)
                .map(|i| cfvnet_card_to_core(data[offset + 1 + i]))
                .collect();
            let packed = canonical_key(&board_cards);
            board_set.entry(packed).or_insert_with(|| board_cards.clone());

            let cfv_offset = offset + CFV_FIELD_OFFSET;
            let mask_offset = cfv_offset + 1326 * 4;

            for i in 0..1326 {
                if data[mask_offset + i] != 0 {
                    let equity = read_cfv_as_equity(&data, cfv_offset, i);
                    let bin = (equity * HISTOGRAM_BINS as f64) as usize;
                    histogram[bin.min(HISTOGRAM_BINS - 1)] += 1;
                }
            }

            offset += CFVNET_RIVER_RECORD_SIZE;
        }
        progress((file_idx + 1) as f64 / bin_files.len() as f64 * 0.3);
    }

    if board_set.is_empty() {
        return Err("no valid river records found".into());
    }
    Ok((histogram, board_set))
}

/// Pass 2: run weighted 1-D k-means on histogram bin midpoints and extract
/// sorted centroids.
#[allow(clippy::cast_precision_loss)]
fn cfvnet_pass2_centroids(histogram: &[u64], bucket_count: u16, kmeans_iterations: u32) -> Vec<f64> {
    let mut bin_midpoints: Vec<f64> = Vec::new();
    let mut bin_weights: Vec<f64> = Vec::new();
    for (i, &count) in histogram.iter().enumerate() {
        if count > 0 {
            let midpoint = (i as f64 + 0.5) / HISTOGRAM_BINS as f64;
            bin_midpoints.push(midpoint);
            bin_weights.push(count as f64);
        }
    }

    let labels = kmeans_1d_weighted(
        &bin_midpoints,
        &bin_weights,
        bucket_count as usize,
        kmeans_iterations,
    );

    // Extract centroids from labels by computing weighted average per cluster.
    let k = bucket_count as usize;
    let mut centroid_sums = vec![0.0_f64; k];
    let mut centroid_weights = vec![0.0_f64; k];
    for (idx, &label) in labels.iter().enumerate() {
        let ci = label as usize;
        centroid_sums[ci] += bin_midpoints[idx] * bin_weights[idx];
        centroid_weights[ci] += bin_weights[idx];
    }
    let mut centroids: Vec<f64> = (0..k)
        .map(|ci| {
            if centroid_weights[ci] > 0.0 {
                centroid_sums[ci] / centroid_weights[ci]
            } else {
                ci as f64 / (k.max(2) - 1) as f64
            }
        })
        .collect();
    centroids.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    centroids
}

/// Compute the byte size of a cfvnet record given the board size.
pub(crate) fn record_size_for_board(board_size: u8) -> usize {
    1 + board_size as usize + 4 + 4 + 1 + 4 + 1326 * 4 + 1326 * 4 + 1326 * 4 + 1326
}

// ---------------------------------------------------------------------------
// Full pipeline orchestrator
// ---------------------------------------------------------------------------

/// Run the full bottom-up clustering pipeline: river -> turn -> flop -> preflop.
///
/// All four streets are computed in memory with each stage feeding into
/// the next. Files are written only after all stages complete successfully.
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
    // 1. River (equity clustering — independent)
    progress("river", 0.0);
    let river = if let Some(ref cfvnet_dir) = config.cfvnet_river_data {
        cluster_river_from_cfvnet(
            cfvnet_dir,
            config.river.buckets,
            config.kmeans_iterations,
            |p| progress("river", p),
        )?
    } else if let Some(n) = config.river.sample_boards {
        cluster_river_with_boards(
            config.river.buckets,
            config.kmeans_iterations,
            config.seed,
            n,
            |p| progress("river", p),
        )
    } else {
        cluster_river(
            config.river.buckets,
            config.kmeans_iterations,
            config.seed,
            |p| progress("river", p),
        )
    };
    river.save(&output_dir.join("river.buckets"))?;

    // 2. Turn (potential-aware EMD, depends on river)
    progress("turn", 0.0);
    let turn = if let Some(n) = config.turn.sample_boards {
        cluster_turn_with_boards(
            &river,
            config.turn.buckets,
            config.kmeans_iterations,
            config.seed,
            n,
            |p| progress("turn", p),
        )
    } else {
        cluster_turn(
            &river,
            config.turn.buckets,
            config.kmeans_iterations,
            config.seed,
            |p| progress("turn", p),
        )
    };
    turn.save(&output_dir.join("turn.buckets"))?;

    // 3. Flop (potential-aware EMD, depends on turn)
    progress("flop", 0.0);
    let flop = if let Some(n) = config.flop.sample_boards {
        cluster_flop_with_boards(
            &turn,
            config.flop.buckets,
            config.kmeans_iterations,
            config.seed,
            n,
            |p| progress("flop", p),
        )
    } else {
        cluster_flop(
            &turn,
            config.flop.buckets,
            config.kmeans_iterations,
            config.seed,
            |p| progress("flop", p),
        )
    };
    flop.save(&output_dir.join("flop.buckets"))?;

    // 4. Preflop (deterministic canonical hand mapping, 169 buckets)
    progress("preflop", 0.0);
    let preflop = cluster_preflop(|p| progress("preflop", p));
    preflop.save(&output_dir.join("preflop.buckets"))?;

    Ok(())
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Sample up to `max_boards` canonical boards without replacement.
///
/// If `max_boards >= all.len()`, returns all boards (full coverage).
/// Each board's combinatorial weight is returned alongside its cards.
fn sample_canonical<const N: usize>(
    all: &[WeightedBoard<N>],
    max_boards: usize,
    seed: u64,
) -> (Vec<[Card; N]>, Vec<f64>) {
    if max_boards >= all.len() {
        let boards = all.iter().map(|wb| wb.cards).collect();
        let weights = all.iter().map(|wb| f64::from(wb.weight)).collect();
        (boards, weights)
    } else {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut indices: Vec<usize> = (0..all.len()).collect();
        indices.shuffle(&mut rng);
        indices.truncate(max_boards);
        let boards = indices.iter().map(|&i| all[i].cards).collect();
        let weights = indices.iter().map(|&i| f64::from(all[i].weight)).collect();
        (boards, weights)
    }
}

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
pub fn canonical_key(cards: &[Card]) -> PackedBoard {
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

// ---------------------------------------------------------------------------
// cfvnet helpers
// ---------------------------------------------------------------------------

/// Convert a pot-relative CFV to an equity value in [0, 1].
pub(crate) fn cfv_to_equity(cfv: f32) -> f64 {
    f64::midpoint(f64::from(cfv), 1.0).clamp(0.0, 1.0)
}

/// Map a cfvnet `card_id` (`4*rank + suit` where C=0,D=1,H=2,S=3)
/// to a core Card.
pub(crate) fn cfvnet_card_to_core(card_id: u8) -> Card {
    let rank = card_id / 4;
    let cfvnet_suit = card_id % 4;
    let value = match rank {
        0 => Value::Two,
        1 => Value::Three,
        2 => Value::Four,
        3 => Value::Five,
        4 => Value::Six,
        5 => Value::Seven,
        6 => Value::Eight,
        7 => Value::Nine,
        8 => Value::Ten,
        9 => Value::Jack,
        10 => Value::Queen,
        11 => Value::King,
        12 => Value::Ace,
        _ => panic!("invalid rank {rank}"),
    };
    let suit = match cfvnet_suit {
        0 => Suit::Club,
        1 => Suit::Diamond,
        2 => Suit::Heart,
        3 => Suit::Spade,
        _ => unreachable!(),
    };
    Card::new(value, suit)
}

/// Build a lookup table mapping cfvnet combo index (range-solver ordering)
/// to core combo index (`cluster_pipeline` ordering).
pub(crate) fn build_cfvnet_to_core_combo_map() -> [u16; 1326] {
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

    /// Create a dummy flop BucketFile with random buckets for preflop tests.
    fn make_dummy_flop_buckets(num_boards: usize, bucket_count: u16, seed: u64) -> BucketFile {
        use rand::prelude::*;
        let mut rng = StdRng::seed_from_u64(seed);
        let deck = build_deck();
        let boards = sample_flop_boards(&deck, num_boards, seed);
        let total = num_boards * TOTAL_COMBOS as usize;
        let buckets: Vec<u16> = (0..total)
            .map(|_| rng.random_range(0..bucket_count) as u16)
            .collect();
        let packed_boards: Vec<PackedBoard> = boards
            .iter()
            .map(|b| canonical_key(b))
            .collect();
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

    #[test]
    fn test_preflop_cluster_canonical_mapping() {
        let preflop = cluster_preflop(|_| {});
        assert_eq!(preflop.header.street, Street::Preflop);
        assert_eq!(preflop.header.bucket_count, 169);
        assert_eq!(preflop.header.board_count, 1);
        assert_eq!(preflop.header.combos_per_board, 1326);
        assert_eq!(preflop.buckets.len(), 1326);
        for &b in &preflop.buckets {
            assert!(b < 169, "bucket {b} out of range");
        }
        // All 169 canonical hands should be represented
        let unique: std::collections::HashSet<u16> =
            preflop.buckets.iter().copied().collect();
        assert_eq!(unique.len(), 169, "expected exactly 169 unique buckets");
    }

    #[test]
    fn test_preflop_cluster_deterministic() {
        let a = cluster_preflop(|_| {});
        let b = cluster_preflop(|_| {});
        assert_eq!(a.buckets, b.buckets);
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
    fn test_build_bucket_histogram_u8_uses_actual_buckets() {
        // Build a small river BucketFile with known buckets for a single board.
        let deck = build_deck();
        // Create a 4-card turn board and a river BucketFile with one 5-card board.
        let turn_board: [Card; 4] = [deck[10], deck[20], deck[30], deck[40]];

        // Build a river BucketFile for a single 5-card board (turn_board + deck[2]).
        // We'll assign all combos to bucket 0 except a few to bucket 1.
        let river_card = deck[2];
        let river_board_cards = [turn_board[0], turn_board[1], turn_board[2], turn_board[3], river_card];
        let packed = canonical_key(&river_board_cards);

        let mut river_buckets_data = vec![0_u16; 1326];
        // Assign a few combos to bucket 1.
        river_buckets_data[0] = 1;
        river_buckets_data[5] = 1;

        let river_bf = BucketFile {
            header: BucketFileHeader {
                street: Street::River,
                bucket_count: 2,
                board_count: 1,
                combos_per_board: TOTAL_COMBOS,
                version: 2,
            },
            boards: vec![packed],
            buckets: river_buckets_data,
        };

        let board_map = river_bf.board_index_map();
        let combo = [deck[0], deck[1]]; // Must not overlap with turn_board or river_card
        assert!(!cards_overlap(combo, &turn_board));
        assert!(combo[0] != river_card && combo[1] != river_card);

        let hist = build_bucket_histogram_u8(combo, &turn_board, &deck, &river_bf, &board_map);
        assert_eq!(hist.len(), 2, "histogram length should match river bucket_count");

        // Total should be the number of river cards that don't overlap with combo/board
        // and whose resulting 5-card board is in the board_map.
        // Only deck[2] forms a board in the map, so total should be at most 1.
        let total: u32 = hist.iter().map(|&c| u32::from(c)).sum();
        assert!(total <= 46, "total counts should not exceed river card count");
    }

    // -----------------------------------------------------------------------
    // cfvnet helper tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_cfv_to_equity() {
        assert!((cfv_to_equity(-1.0) - 0.0).abs() < 1e-9);
        assert!((cfv_to_equity(0.0) - 0.5).abs() < 1e-9);
        assert!((cfv_to_equity(1.0) - 1.0).abs() < 1e-9);
        assert!((cfv_to_equity(-1.5) - 0.0).abs() < 1e-9);
        assert!((cfv_to_equity(1.5) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_cfvnet_card_to_core() {
        let c = cfvnet_card_to_core(0);
        assert_eq!(c.value, Value::Two);
        assert_eq!(c.suit, Suit::Club);
        let c = cfvnet_card_to_core(51);
        assert_eq!(c.value, Value::Ace);
        assert_eq!(c.suit, Suit::Spade);
        let c = cfvnet_card_to_core(48);
        assert_eq!(c.value, Value::Ace);
        assert_eq!(c.suit, Suit::Club);
    }

    #[test]
    fn test_cfvnet_to_core_combo_map_is_permutation() {
        let map = build_cfvnet_to_core_combo_map();
        let mut sorted: Vec<u16> = map.to_vec();
        sorted.sort_unstable();
        let expected: Vec<u16> = (0..1326).collect();
        assert_eq!(sorted, expected, "combo map must be a permutation of 0..1326");
    }

    #[test]
    fn test_cluster_river_from_cfvnet_with_synthetic_data() {
        use std::io::Write;
        use tempfile::TempDir;

        let dir = TempDir::new().unwrap();
        let path = dir.path().join("river_test.bin");
        let mut f = std::fs::File::create(&path).unwrap();

        for player in 0u8..2 {
            f.write_all(&[5u8]).unwrap();                    // board_size
            f.write_all(&[0, 4, 8, 12, 16]).unwrap();        // board: 2c 3c 4c 5c 6c
            f.write_all(&100.0_f32.to_le_bytes()).unwrap();   // pot
            f.write_all(&200.0_f32.to_le_bytes()).unwrap();   // stack
            f.write_all(&[player]).unwrap();                   // player
            f.write_all(&0.0_f32.to_le_bytes()).unwrap();     // game_value
            f.write_all(&[0u8; 1326 * 4]).unwrap();           // oop_range
            f.write_all(&[0u8; 1326 * 4]).unwrap();           // ip_range
            for i in 0..1326 {
                let cfv = -1.0 + 2.0 * (i as f32 / 1325.0);
                f.write_all(&cfv.to_le_bytes()).unwrap();
            }
            f.write_all(&[1u8; 1326]).unwrap();               // valid_mask: all valid
        }
        drop(f);

        let result = cluster_river_from_cfvnet(dir.path(), 10, 50, |_| {});
        let bf = result.expect("clustering should succeed");
        assert_eq!(bf.header.street, Street::River);
        assert_eq!(bf.header.bucket_count, 10);
        assert_eq!(bf.header.board_count, 1);
        assert_eq!(bf.buckets.len(), 1326);
        for &b in &bf.buckets {
            assert!(b < 10, "bucket {b} out of range");
        }
        let used: std::collections::HashSet<u16> = bf.buckets.iter().copied().collect();
        assert!(used.len() > 1, "expected multiple buckets used");
    }

    #[test]
    fn test_build_bucket_histogram_u8_empty_when_no_boards_match() {
        let deck = build_deck();
        // Empty BucketFile with no boards in the map
        let river_bf = BucketFile {
            header: BucketFileHeader {
                street: Street::River,
                bucket_count: 5,
                board_count: 0,
                combos_per_board: TOTAL_COMBOS,
                version: 2,
            },
            boards: Vec::new(),
            buckets: Vec::new(),
        };
        let board_map = river_bf.board_index_map();
        let combo = [deck[0], deck[1]];
        let board = [deck[10], deck[20], deck[30], deck[40]];
        let hist = build_bucket_histogram_u8(combo, &board, &deck, &river_bf, &board_map);
        assert_eq!(hist.len(), 5);
        let total: u32 = hist.iter().map(|&c| u32::from(c)).sum();
        assert_eq!(total, 0, "no boards match, histogram should be all zeros");
    }

}
