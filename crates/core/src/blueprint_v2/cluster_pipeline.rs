//! River clustering: bucket hole-card combos by showdown equity.
//!
//! Samples random 5-card boards, computes equity for every valid hole-card
//! combo on each board, then clusters the equity values into K buckets using
//! 1-D k-means. The result is a [`BucketFile`] mapping each (board, combo)
//! pair to a bucket index.

use rand::prelude::*;
use rand::rngs::StdRng;
use rayon::prelude::*;

use crate::poker::{Card, ALL_SUITS, ALL_VALUES};
use crate::showdown_equity::compute_equity;

use super::bucket_file::{BucketFile, BucketFileHeader};
use super::clustering::kmeans_1d;
use super::Street;

/// Number of sample boards to use when no explicit count is provided.
const DEFAULT_NUM_BOARDS: usize = 1_000;

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

/// Check whether any card in `combo` appears in `board`.
fn cards_overlap(combo: [Card; 2], board: [Card; 5]) -> bool {
    board.iter().any(|b| *b == combo[0] || *b == combo[1])
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
}
