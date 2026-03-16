//! Bucket equity table precomputation for the bucketed GPU solver.
//!
//! For each river board, we precompute a `num_buckets x num_buckets` equity
//! matrix where `E[i][j]` is the average payoff for bucket `i` against bucket
//! `j` across all valid combo matchups, normalized to [-1, 1].
//!
//! This replaces the per-combo showdown evaluation used in the concrete solver.

use std::collections::HashMap;

use poker_solver_core::blueprint_v2::bucket_file::{BucketFile, PackedBoard};
use poker_solver_core::blueprint_v2::cluster_pipeline::canonical_key;
use range_solver::card::{evaluate_hand_strength, index_to_card_pair};
use rs_poker::core::{Card, Suit, Value};

/// Convert a range-solver u8 card to an `rs_poker::core::Card`.
///
/// Range-solver encoding: `card_id = 4 * rank + suit`
///   - rank: 2 -> 0, 3 -> 1, ..., A -> 12
///   - suit: club -> 0, diamond -> 1, heart -> 2, spade -> 3
fn u8_to_rs_card(card_id: u8) -> Card {
    let rank = card_id / 4;
    let suit_idx = card_id % 4;
    let value = Value::from(rank);
    let suit = match suit_idx {
        0 => Suit::Club,
        1 => Suit::Diamond,
        2 => Suit::Heart,
        3 => Suit::Spade,
        _ => unreachable!(),
    };
    Card::new(value, suit)
}

/// Compute the bucket-vs-bucket equity table for a single river board.
///
/// Returns a flat `Vec<f32>` of size `num_buckets * num_buckets` where
/// `result[i * num_buckets + j]` is the average payoff for bucket `i` against
/// bucket `j`, normalized to [-1, 1]:
///   - +1 if bucket `i` always wins against bucket `j`
///   - -1 if bucket `i` always loses
///   -  0 if evenly split or same bucket
///
/// The matrix is anti-symmetric: `E[i][j] = -E[j][i]`.
///
/// # Arguments
/// - `board`: 5 river cards in range-solver u8 encoding
/// - `bucket_file`: loaded bucket file for the river street
/// - `board_idx`: pre-looked-up board index in the bucket file
/// - `num_buckets`: number of buckets (e.g. 500)
pub fn compute_bucket_equity_table(
    board: &[u8; 5],
    bucket_file: &BucketFile,
    board_idx: u32,
    num_buckets: usize,
) -> Vec<f32> {
    let mut equity = vec![0.0f64; num_buckets * num_buckets];
    let mut counts = vec![0u32; num_buckets * num_buckets];

    // Precompute hand strengths and bucket assignments for all 1326 combos
    let mut strengths = [0u16; 1326];
    let mut buckets = [0u16; 1326];
    let mut valid = [false; 1326];

    let board_5: [u8; 5] = *board;

    for i in 0..1326u16 {
        let (c1, c2) = index_to_card_pair(i as usize);
        // Skip if combo conflicts with board
        if board_5.contains(&c1) || board_5.contains(&c2) {
            continue;
        }
        let strength = evaluate_hand_strength(&board_5, (c1, c2));
        if strength == 0 {
            continue; // card conflict detected by evaluator
        }
        strengths[i as usize] = strength;
        buckets[i as usize] = bucket_file.get_bucket(board_idx, i);
        valid[i as usize] = true;
    }

    // Iterate over all valid combo pairs
    for i in 0..1326usize {
        if !valid[i] {
            continue;
        }
        let (c1_i, c2_i) = index_to_card_pair(i);
        let strength_i = strengths[i];
        let bucket_i = buckets[i] as usize;

        for j in (i + 1)..1326usize {
            if !valid[j] {
                continue;
            }
            let (c1_j, c2_j) = index_to_card_pair(j);

            // Card blocking between hands
            if c1_i == c1_j || c1_i == c2_j || c2_i == c1_j || c2_i == c2_j {
                continue;
            }

            let strength_j = strengths[j];
            let bucket_j = buckets[j] as usize;

            let idx_ij = bucket_i * num_buckets + bucket_j;
            let idx_ji = bucket_j * num_buckets + bucket_i;

            if strength_i > strength_j {
                equity[idx_ij] += 1.0;
                equity[idx_ji] -= 1.0;
            } else if strength_i < strength_j {
                equity[idx_ij] -= 1.0;
                equity[idx_ji] += 1.0;
            }
            // Equal strengths: contribute 0 (tie), but still count
            counts[idx_ij] += 1;
            counts[idx_ji] += 1;
        }
    }

    // Normalize: equity[i][j] = sum / count -> average payoff in [-1, 1]
    equity
        .iter()
        .zip(counts.iter())
        .map(|(&eq, &cnt)| {
            if cnt > 0 {
                (eq / cnt as f64) as f32
            } else {
                0.0
            }
        })
        .collect()
}

/// Cache for looking up board indices in a `BucketFile`.
///
/// Builds a `HashMap<PackedBoard, u32>` once from the bucket file's board
/// table, then provides fast lookups for arbitrary boards given as u8 cards.
pub struct BucketedBoardCache {
    board_map: HashMap<PackedBoard, u32>,
}

impl BucketedBoardCache {
    /// Build the cache from a bucket file.
    pub fn new(bucket_file: &BucketFile) -> Self {
        Self {
            board_map: bucket_file.board_index_map(),
        }
    }

    /// Look up the board index for a set of u8-encoded board cards.
    ///
    /// The cards are converted to `rs_poker::core::Card`, canonicalized, and
    /// packed to find the matching board index. Returns `None` if the board
    /// is not present in the bucket file.
    pub fn find_board_index(&self, board_cards: &[u8]) -> Option<u32> {
        // Convert u8 cards to rs_poker Card
        let cards: Vec<Card> = board_cards.iter().map(|&c| u8_to_rs_card(c)).collect();

        // Pack using canonical_key (sorts by value_rank desc, suit asc)
        let packed = canonical_key(&cards);
        self.board_map.get(&packed).copied()
    }

    /// Number of boards in the cache.
    pub fn len(&self) -> usize {
        self.board_map.len()
    }

    /// Whether the cache is empty.
    pub fn is_empty(&self) -> bool {
        self.board_map.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    /// Test equity table computation with a synthetic bucket file.
    ///
    /// Uses a low board (2c 3d 4h 5s 7c) with 2 buckets: bucket 0 gets the
    /// weaker half of combos (by strength), bucket 1 the stronger half.
    /// This ensures non-trivial cross-bucket equity.
    #[test]
    fn test_equity_table_antisymmetric() {
        let board_cards: [u8; 5] = [
            4 * 0 + 0,  // 2c
            4 * 1 + 1,  // 3d
            4 * 2 + 2,  // 4h
            4 * 3 + 3,  // 5s
            4 * 5 + 0,  // 7c
        ];

        // Convert to rs_poker cards for PackedBoard
        let rs_cards: Vec<Card> = board_cards.iter().map(|&c| u8_to_rs_card(c)).collect();
        let packed = canonical_key(&rs_cards);

        // Compute strengths for all combos and sort to assign buckets by strength
        let combos_per_board = 1326u16;
        let mut combo_strengths: Vec<(u16, u16)> = Vec::new(); // (combo_idx, strength)
        for i in 0..combos_per_board {
            let (c1, c2) = index_to_card_pair(i as usize);
            if board_cards.contains(&c1) || board_cards.contains(&c2) {
                combo_strengths.push((i, 0));
            } else {
                let s = evaluate_hand_strength(&board_cards, (c1, c2));
                combo_strengths.push((i, s));
            }
        }

        // Sort by strength to divide into 2 buckets
        let mut sorted_by_strength = combo_strengths.clone();
        sorted_by_strength.sort_by_key(|&(_, s)| s);

        let num_buckets: u16 = 2;
        let mut bucket_data = vec![0u16; combos_per_board as usize];
        let midpoint = sorted_by_strength.len() / 2;
        for (rank, &(combo_idx, _)) in sorted_by_strength.iter().enumerate() {
            bucket_data[combo_idx as usize] = if rank < midpoint { 0 } else { 1 };
        }

        let bf = BucketFile {
            header: poker_solver_core::blueprint_v2::bucket_file::BucketFileHeader {
                street: poker_solver_core::blueprint_v2::Street::River,
                bucket_count: num_buckets,
                board_count: 1,
                combos_per_board,
                version: 2,
            },
            boards: vec![packed],
            buckets: bucket_data,
        };

        let equity = compute_bucket_equity_table(
            &board_cards,
            &bf,
            0,
            num_buckets as usize,
        );

        let n = num_buckets as usize;
        assert_eq!(equity.len(), n * n);

        // Check anti-symmetry: E[i][j] ~= -E[j][i]
        for i in 0..n {
            for j in 0..n {
                let eij = equity[i * n + j];
                let eji = equity[j * n + i];
                assert!(
                    (eij + eji).abs() < 1e-6,
                    "E[{i}][{j}]={eij} + E[{j}][{i}]={eji} = {} (should be ~0)",
                    eij + eji,
                );
            }
        }

        // All values should be in [-1, 1]
        for i in 0..n {
            for j in 0..n {
                let v = equity[i * n + j];
                assert!(
                    (-1.0..=1.0).contains(&v),
                    "E[{i}][{j}]={v} out of range [-1,1]"
                );
            }
        }

        // Bucket 0 = weaker half, bucket 1 = stronger half.
        // So E[0][1] should be negative (bucket 0 loses to bucket 1)
        // and E[1][0] should be positive.
        assert!(
            equity[0 * n + 1] < -0.01,
            "E[0][1] should be negative (weak vs strong), got {}",
            equity[0 * n + 1],
        );
        assert!(
            equity[1 * n + 0] > 0.01,
            "E[1][0] should be positive (strong vs weak), got {}",
            equity[1 * n + 0],
        );

        // At least some non-zero off-diagonal values
        let nonzero_count = equity.iter().filter(|&&v| v.abs() > 1e-6).count();
        assert!(
            nonzero_count > 0,
            "equity table is all zeros, expected some non-trivial entries"
        );
    }

    /// Test the BucketedBoardCache lookup.
    #[test]
    fn test_board_cache_lookup() {
        // Build a bucket file with one known board
        let board_u8: [u8; 5] = [
            4 * 0 + 0,  // 2c
            4 * 1 + 1,  // 3d
            4 * 2 + 2,  // 4h
            4 * 3 + 3,  // 5s
            4 * 4 + 0,  // 6c
        ];

        let rs_cards: Vec<Card> = board_u8.iter().map(|&c| u8_to_rs_card(c)).collect();
        let packed = canonical_key(&rs_cards);

        let bf = BucketFile {
            header: poker_solver_core::blueprint_v2::bucket_file::BucketFileHeader {
                street: poker_solver_core::blueprint_v2::Street::River,
                bucket_count: 10,
                board_count: 1,
                combos_per_board: 1326,
                version: 2,
            },
            boards: vec![packed],
            buckets: vec![0u16; 1326],
        };

        let cache = BucketedBoardCache::new(&bf);
        assert_eq!(cache.len(), 1);

        // Should find the board
        let idx = cache.find_board_index(&board_u8);
        assert_eq!(idx, Some(0));

        // Different board should not be found
        let other_board: [u8; 5] = [
            4 * 12 + 3, // As
            4 * 11 + 3, // Ks
            4 * 10 + 3, // Qs
            4 * 9 + 3,  // Js
            4 * 8 + 3,  // Ts
        ];
        let other_idx = cache.find_board_index(&other_board);
        assert_eq!(other_idx, None);
    }

    /// Convert an rs_poker Card to a range-solver u8 card.
    fn rs_card_to_u8(card: &Card) -> u8 {
        use poker_solver_core::blueprint_v2::full_depth_solver::rs_poker_card_to_id;
        rs_poker_card_to_id(*card)
    }

    /// Test with actual bucket file if it exists (integration test).
    #[test]
    fn test_equity_table_from_real_bucket_file() {
        let path = Path::new("../../local_data/clusters_500bkt_v3/river.buckets");
        if !path.exists() {
            eprintln!("Skipping test: river.buckets not found at {}", path.display());
            return;
        }

        let bf = BucketFile::load(path).expect("Failed to load river.buckets");
        let num_buckets = bf.header.bucket_count as usize;
        assert_eq!(num_buckets, 500, "Expected 500 buckets");

        let cache = BucketedBoardCache::new(&bf);
        assert!(!cache.is_empty(), "board cache should not be empty");

        // Pick the first board from the bucket file
        let first_packed = bf.boards[0];
        let board_cards_rs = first_packed.to_cards(5);

        // Convert rs_poker cards back to u8 encoding
        let board_u8: Vec<u8> = board_cards_rs.iter().map(rs_card_to_u8).collect();

        // Verify the board can be found in the cache
        let board_idx = cache.find_board_index(&board_u8);
        assert!(board_idx.is_some(), "should find board in cache");
        let board_idx = board_idx.unwrap();

        // Compute equity table
        let board_arr: [u8; 5] = [board_u8[0], board_u8[1], board_u8[2], board_u8[3], board_u8[4]];
        let equity = compute_bucket_equity_table(&board_arr, &bf, board_idx, num_buckets);

        // Size check
        assert_eq!(equity.len(), num_buckets * num_buckets);

        // Anti-symmetry
        let mut max_asym = 0.0f32;
        for i in 0..num_buckets {
            for j in 0..num_buckets {
                let eij = equity[i * num_buckets + j];
                let eji = equity[j * num_buckets + i];
                let asym = (eij + eji).abs();
                if asym > max_asym {
                    max_asym = asym;
                }
            }
        }
        assert!(
            max_asym < 1e-5,
            "max anti-symmetry violation: {max_asym}"
        );

        // Values in [-1, 1]
        for &v in &equity {
            assert!(
                (-1.0 - 1e-6..=1.0 + 1e-6).contains(&v),
                "equity value {v} out of range"
            );
        }

        // Non-trivial
        let nonzero = equity.iter().filter(|&&v| v.abs() > 1e-6).count();
        assert!(
            nonzero > 0,
            "equity table all zeros for a real board"
        );

        eprintln!(
            "Equity table: {}x{}, non-zero entries: {}/{}, board_idx={}",
            num_buckets, num_buckets, nonzero,
            num_buckets * num_buckets, board_idx
        );
    }
}
