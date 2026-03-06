//! Card abstraction clustering pipeline.
//!
//! **River:** samples random 5-card boards, computes showdown equity for every
//! valid hole-card combo on each board, and clusters the equity values into K
//! buckets using 1-D k-means.
//!
//! **Turn:** samples random 4-card (flop+turn) boards, enumerates all possible
//! river cards for each valid combo, builds a histogram over river equity bins,
//! and clusters these histograms using k-means with Earth Mover's Distance.
//!
//! Each street produces a [`BucketFile`] mapping `(board, combo)` to a bucket
//! index.

use rand::prelude::*;
use rand::rngs::StdRng;
use rayon::prelude::*;

use crate::poker::{Card, ALL_SUITS, ALL_VALUES};
use crate::showdown_equity::compute_equity;

use super::bucket_file::{BucketFile, BucketFileHeader};
use super::clustering::{kmeans_1d, kmeans_emd};
use super::Street;

/// Number of sample 5-card boards for river clustering when no explicit count
/// is provided.
const DEFAULT_NUM_BOARDS: usize = 1_000;

/// Number of sample 4-card boards for turn clustering.
const DEFAULT_TURN_BOARDS: usize = 500;

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
    };

    BucketFile { header, buckets }
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
                    if cards_overlap_4(*combo, *board) {
                        return None;
                    }
                    Some(build_river_histogram(*combo, *board, &deck, num_river_buckets))
                })
                .collect();

            #[allow(clippy::cast_precision_loss)]
            progress((board_idx + 1) as f64 / num_boards as f64);

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

    // Cluster with k-means EMD.
    let cluster_labels = kmeans_emd(
        &all_features,
        bucket_count as usize,
        kmeans_iterations,
        seed,
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
    };

    BucketFile { header, buckets }
}

/// Build a probability distribution over river equity bins for the given
/// combo on a 4-card turn board.
///
/// Enumerates every card not in the board or combo as a potential river card,
/// computes equity on the resulting 5-card board, maps equity to a bin, and
/// normalises the histogram.
fn build_river_histogram(
    combo: [Card; 2],
    board: [Card; 4],
    deck: &[Card],
    num_river_buckets: u16,
) -> Vec<f64> {
    let mut histogram = vec![0.0_f64; num_river_buckets as usize];
    let mut count = 0_u32;

    for &river_card in deck {
        if board.contains(&river_card)
            || river_card == combo[0]
            || river_card == combo[1]
        {
            continue;
        }

        let five_board = [board[0], board[1], board[2], board[3], river_card];
        let eq = compute_equity(combo, &five_board);
        let bucket = equity_to_river_bucket(eq, num_river_buckets);
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

/// Map an equity value in [0, 1] to a river bucket index via uniform binning.
///
/// `equity_to_river_bucket(0.0, K) = 0` and `equity_to_river_bucket(1.0, K) = K - 1`.
fn equity_to_river_bucket(equity: f64, num_river_buckets: u16) -> u16 {
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let bucket = (equity * f64::from(num_river_buckets)) as u16;
    bucket.min(num_river_buckets - 1)
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Build the standard 52-card deck in a deterministic order.
fn build_deck() -> Vec<Card> {
    let mut deck = Vec::with_capacity(52);
    for &v in &ALL_VALUES {
        for &s in &ALL_SUITS {
            deck.push(Card::new(v, s));
        }
    }
    deck
}

/// Enumerate all C(52,2) = 1326 two-card combos from the deck.
///
/// The combos are stored as `(Card, Card)` with `c0 < c1` by deck index.
/// The ordering is deterministic and matches the canonical combo index used
/// throughout the bucket file.
fn enumerate_combos(deck: &[Card]) -> Vec<[Card; 2]> {
    let mut combos = Vec::with_capacity(TOTAL_COMBOS as usize);
    for i in 0..deck.len() {
        for j in (i + 1)..deck.len() {
            combos.push([deck[i], deck[j]]);
        }
    }
    debug_assert_eq!(combos.len(), TOTAL_COMBOS as usize);
    combos
}

/// Sample `n` random 5-card boards from the deck without replacement.
///
/// Each board is a sorted 5-card subset. Boards are sampled independently
/// (duplicates are possible but astronomically unlikely for reasonable n).
fn sample_boards(deck: &[Card], n: usize, seed: u64) -> Vec<[Card; 5]> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut boards = Vec::with_capacity(n);
    let indices: Vec<usize> = (0..deck.len()).collect();

    for _ in 0..n {
        // Partial Fisher-Yates: pick 5 indices without replacement.
        let mut pool = indices.clone();
        for k in 0..5 {
            let j = rng.random_range(k..pool.len());
            pool.swap(k, j);
        }
        let board = [
            deck[pool[0]],
            deck[pool[1]],
            deck[pool[2]],
            deck[pool[3]],
            deck[pool[4]],
        ];
        boards.push(board);
    }

    boards
}

/// Compute equity for all 1326 combos on a given 5-card board.
///
/// Returns `None` for combos that share a card with the board (blocked).
fn compute_board_equities(board: [Card; 5], combos: &[[Card; 2]]) -> Vec<Option<f64>> {
    combos
        .iter()
        .map(|&combo| {
            if cards_overlap(combo, board) {
                None
            } else {
                Some(compute_equity(combo, &board))
            }
        })
        .collect()
}

/// Check whether any card in `combo` appears in a 5-card `board`.
fn cards_overlap(combo: [Card; 2], board: [Card; 5]) -> bool {
    board.iter().any(|b| *b == combo[0] || *b == combo[1])
}

/// Check whether any card in `combo` appears in a 4-card `board`.
fn cards_overlap_4(combo: [Card; 2], board: [Card; 4]) -> bool {
    board.iter().any(|b| *b == combo[0] || *b == combo[1])
}

/// Sample `n` random 4-card boards from the deck without replacement.
///
/// Each board is a 4-card subset drawn via partial Fisher-Yates. Boards are
/// sampled independently (duplicates possible but astronomically unlikely).
fn sample_turn_boards(deck: &[Card], n: usize, seed: u64) -> Vec<[Card; 4]> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut boards = Vec::with_capacity(n);
    let indices: Vec<usize> = (0..deck.len()).collect();

    for _ in 0..n {
        let mut pool = indices.clone();
        for k in 0..4 {
            let j = rng.random_range(k..pool.len());
            pool.swap(k, j);
        }
        boards.push([deck[pool[0]], deck[pool[1]], deck[pool[2]], deck[pool[3]]]);
    }

    boards
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
        assert!(cards_overlap(combo, board));
    }

    #[test]
    fn test_cards_overlap_false() {
        let deck = build_deck();
        let combo = [deck[0], deck[1]];
        let board = [deck[2], deck[5], deck[10], deck[15], deck[20]];
        assert!(!cards_overlap(combo, board));
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
    fn test_cluster_river_deterministic() {
        let r1 = cluster_river_with_boards(5, 30, 123, 10, |_| {});
        let r2 = cluster_river_with_boards(5, 30, 123, 10, |_| {});
        assert_eq!(r1.buckets, r2.buckets);
    }

    #[test]
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
        assert!(cards_overlap_4(combo, board));
    }

    #[test]
    fn test_cards_overlap_4_false() {
        let deck = build_deck();
        let combo = [deck[0], deck[1]];
        let board = [deck[2], deck[5], deck[10], deck[15]];
        assert!(!cards_overlap_4(combo, board));
    }

    #[test]
    fn test_equity_to_river_bucket_bounds() {
        // equity=0.0 should map to bucket 0
        assert_eq!(equity_to_river_bucket(0.0, 10), 0);
        // equity=1.0 should map to the last bucket (9 for K=10)
        assert_eq!(equity_to_river_bucket(1.0, 10), 9);
        // equity=0.5 with 10 buckets → bucket 5
        assert_eq!(equity_to_river_bucket(0.5, 10), 5);
        // equity just below 1.0
        assert_eq!(equity_to_river_bucket(0.99, 10), 9);
    }

    #[test]
    fn test_equity_to_river_bucket_single_bucket() {
        // With 1 bucket, everything maps to 0.
        assert_eq!(equity_to_river_bucket(0.0, 1), 0);
        assert_eq!(equity_to_river_bucket(0.5, 1), 0);
        assert_eq!(equity_to_river_bucket(1.0, 1), 0);
    }

    #[test]
    fn test_build_river_histogram_sums_to_one() {
        let deck = build_deck();
        // combo: first two cards, board: next four cards (no overlap)
        let combo = [deck[0], deck[1]];
        let board = [deck[2], deck[3], deck[4], deck[5]];
        let histogram = build_river_histogram(combo, board, &deck, 10);

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
    fn test_build_river_histogram_river_card_count() {
        // With 4 board cards + 2 hole cards = 6 used, there are 46 river cards.
        let deck = build_deck();
        let combo = [deck[0], deck[1]];
        let board = [deck[2], deck[3], deck[4], deck[5]];
        let histogram = build_river_histogram(combo, board, &deck, 5);

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
    fn test_cluster_turn_deterministic() {
        let river = cluster_river_with_boards(5, 30, 42, 10, |_| {});
        let t1 = cluster_turn_with_boards(&river, 3, 20, 123, 8, |_| {});
        let t2 = cluster_turn_with_boards(&river, 3, 20, 123, 8, |_| {});
        assert_eq!(t1.buckets, t2.buckets);
    }

    #[test]
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
}
