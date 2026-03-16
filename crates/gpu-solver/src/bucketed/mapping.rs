//! Combo-to-bucket mapping utilities.
//!
//! Functions for converting between 1326-combo-space and bucket-space
//! representations of ranges, strategies, and counterfactual values.

use poker_solver_core::blueprint_v2::bucket_file::BucketFile;

/// Convert 1326-dim range weights to `num_buckets`-dim bucket reach.
///
/// Each bucket's reach is the sum of the range weights of all combos
/// assigned to that bucket. Combos with bucket index >= `num_buckets`
/// are silently skipped.
pub fn range_to_bucket_reach(
    range: &[f32],
    bucket_file: &BucketFile,
    board_idx: u32,
    num_buckets: usize,
) -> Vec<f32> {
    debug_assert!(
        range.len() >= 1326,
        "range must have at least 1326 entries, got {}",
        range.len()
    );
    let mut reach = vec![0.0f32; num_buckets];
    for combo in 0..1326u16 {
        let bucket = bucket_file.get_bucket(board_idx, combo) as usize;
        if bucket < num_buckets {
            reach[bucket] += range[combo as usize];
        }
    }
    reach
}

/// Expand bucket-space strategy to 1326-combo strategy for display/comparison.
///
/// Each combo gets its bucket's strategy probability. The input
/// `bucket_strategy` is laid out as `[action][bucket]`, i.e.
/// `bucket_strategy[a * num_buckets + b]` is the probability of action `a`
/// for bucket `b`. The output is `[action][combo]`.
pub fn bucket_strategy_to_combos(
    bucket_strategy: &[f32],
    bucket_file: &BucketFile,
    board_idx: u32,
    num_buckets: usize,
    num_actions: usize,
) -> Vec<f32> {
    debug_assert_eq!(
        bucket_strategy.len(),
        num_actions * num_buckets,
        "bucket_strategy length mismatch: expected {} ({}*{}), got {}",
        num_actions * num_buckets,
        num_actions,
        num_buckets,
        bucket_strategy.len()
    );
    let mut combo_strat = vec![0.0f32; num_actions * 1326];
    for combo in 0..1326u16 {
        let bucket = bucket_file.get_bucket(board_idx, combo) as usize;
        if bucket < num_buckets {
            for a in 0..num_actions {
                combo_strat[a * 1326 + combo as usize] =
                    bucket_strategy[a * num_buckets + bucket];
            }
        }
    }
    combo_strat
}

/// Expand bucket-space CFVs to 1326-combo CFVs.
///
/// Each combo gets its bucket's CFV. Combos whose bucket index is out of
/// range get a CFV of 0.
pub fn bucket_cfvs_to_combos(
    bucket_cfvs: &[f32],
    bucket_file: &BucketFile,
    board_idx: u32,
    num_buckets: usize,
) -> Vec<f32> {
    debug_assert!(
        bucket_cfvs.len() >= num_buckets,
        "bucket_cfvs length {} < num_buckets {}",
        bucket_cfvs.len(),
        num_buckets
    );
    let mut combo_cfvs = vec![0.0f32; 1326];
    for combo in 0..1326u16 {
        let bucket = bucket_file.get_bucket(board_idx, combo) as usize;
        if bucket < num_buckets {
            combo_cfvs[combo as usize] = bucket_cfvs[bucket];
        }
    }
    combo_cfvs
}

#[cfg(test)]
mod tests {
    use super::*;
    use poker_solver_core::blueprint_v2::bucket_file::BucketFileHeader;
    use poker_solver_core::blueprint_v2::cluster_pipeline::canonical_key;
    use poker_solver_core::blueprint_v2::Street;
    use range_solver::card::index_to_card_pair;
    use crate::bucketed::equity::{u8_to_rs_card, BucketedBoardCache};
    use std::path::Path;

    /// Helper to build a minimal 2-bucket BucketFile for a given board.
    fn make_test_bucket_file(board: &[u8; 5], num_buckets: u16) -> (BucketFile, u32) {
        let rs_cards: Vec<_> = board.iter().map(|&c| u8_to_rs_card(c)).collect();
        let packed = canonical_key(&rs_cards);
        let mut bucket_data = vec![0u16; 1326];
        // Assign combos to buckets round-robin
        for combo in 0..1326u16 {
            bucket_data[combo as usize] = combo % num_buckets;
        }
        let bf = BucketFile {
            header: BucketFileHeader {
                street: Street::River,
                bucket_count: num_buckets,
                board_count: 1,
                combos_per_board: 1326,
                version: 2,
            },
            boards: vec![packed],
            buckets: bucket_data,
        };
        (bf, 0)
    }

    #[test]
    fn test_range_to_bucket_reach_uniform() {
        let board = [0u8, 5, 10, 15, 20];
        let (bf, board_idx) = make_test_bucket_file(&board, 10);

        // Uniform range
        let range = vec![1.0f32; 1326];
        let reach = range_to_bucket_reach(&range, &bf, board_idx, 10);
        assert_eq!(reach.len(), 10);

        // Total reach should be 1326 (all combos weight 1.0)
        let total: f32 = reach.iter().sum();
        assert!((total - 1326.0).abs() < 0.01, "Total reach: {}", total);

        // Each bucket should get ~132-133 combos
        for (i, &r) in reach.iter().enumerate() {
            assert!(
                r >= 132.0 && r <= 134.0,
                "Bucket {} reach: {} (expected ~132-133)",
                i,
                r,
            );
        }
    }

    #[test]
    fn test_range_to_bucket_reach_zeros() {
        let board = [0u8, 5, 10, 15, 20];
        let (bf, board_idx) = make_test_bucket_file(&board, 5);

        // Zero range
        let range = vec![0.0f32; 1326];
        let reach = range_to_bucket_reach(&range, &bf, board_idx, 5);
        assert!(reach.iter().all(|&r| r == 0.0));
    }

    #[test]
    fn test_bucket_strategy_to_combos_basic() {
        let board = [0u8, 5, 10, 15, 20];
        let (bf, board_idx) = make_test_bucket_file(&board, 4);
        let num_actions = 3;
        let num_buckets = 4;

        // Set bucket 0: action 0 = 1.0, action 1 = 0.0, action 2 = 0.0
        // Set bucket 1: action 0 = 0.0, action 1 = 1.0, action 2 = 0.0
        let mut bucket_strat = vec![0.0f32; num_actions * num_buckets];
        bucket_strat[0 * num_buckets + 0] = 1.0; // bucket 0: action 0
        bucket_strat[1 * num_buckets + 1] = 1.0; // bucket 1: action 1

        let combo_strat =
            bucket_strategy_to_combos(&bucket_strat, &bf, board_idx, num_buckets, num_actions);
        assert_eq!(combo_strat.len(), num_actions * 1326);

        // Every combo in bucket 0 should have action 0 = 1.0
        for combo in 0..1326u16 {
            let bucket = bf.get_bucket(board_idx, combo) as usize;
            if bucket == 0 {
                assert_eq!(combo_strat[0 * 1326 + combo as usize], 1.0);
                assert_eq!(combo_strat[1 * 1326 + combo as usize], 0.0);
            } else if bucket == 1 {
                assert_eq!(combo_strat[0 * 1326 + combo as usize], 0.0);
                assert_eq!(combo_strat[1 * 1326 + combo as usize], 1.0);
            }
        }
    }

    #[test]
    fn test_bucket_cfvs_to_combos_basic() {
        let board = [0u8, 5, 10, 15, 20];
        let (bf, board_idx) = make_test_bucket_file(&board, 3);

        let bucket_cfvs = vec![1.5f32, -0.5, 0.0];
        let combo_cfvs = bucket_cfvs_to_combos(&bucket_cfvs, &bf, board_idx, 3);
        assert_eq!(combo_cfvs.len(), 1326);

        // Each combo should get its bucket's CFV
        for combo in 0..1326u16 {
            let bucket = bf.get_bucket(board_idx, combo) as usize;
            assert_eq!(combo_cfvs[combo as usize], bucket_cfvs[bucket]);
        }
    }

    #[test]
    fn test_cfv_roundtrip_preserves_bucket_values() {
        let board = [0u8, 5, 10, 15, 20];
        let (bf, board_idx) = make_test_bucket_file(&board, 100);

        // Set distinct CFV per bucket
        let bucket_cfvs: Vec<f32> = (0..100).map(|i| i as f32 * 0.01 - 0.5).collect();
        let combo_cfvs = bucket_cfvs_to_combos(&bucket_cfvs, &bf, board_idx, 100);

        // Verify each combo maps back to the right bucket value
        for combo in 0..1326u16 {
            let bucket = bf.get_bucket(board_idx, combo) as usize;
            let expected = bucket_cfvs[bucket];
            let actual = combo_cfvs[combo as usize];
            assert!(
                (actual - expected).abs() < 1e-6,
                "combo {combo} bucket {bucket}: expected {expected}, got {actual}"
            );
        }
    }

    /// Integration test with real bucket file (skipped if file not found).
    #[test]
    fn test_range_to_bucket_reach_real_file() {
        let path = Path::new("../../local_data/clusters_500bkt_v3/river.buckets");
        if !path.exists() {
            eprintln!(
                "Skipping test: river.buckets not found at {}",
                path.display()
            );
            return;
        }

        let bf = BucketFile::load(path).expect("Failed to load river.buckets");
        let cache = BucketedBoardCache::new(&bf);
        let num_buckets = bf.header.bucket_count as usize;

        // Use the first board from the bucket file
        let first_packed = bf.boards[0];
        let board_cards_rs = first_packed.to_cards(5);
        let board_u8: Vec<u8> = board_cards_rs
            .iter()
            .map(|c| {
                use poker_solver_core::blueprint_v2::full_depth_solver::rs_poker_card_to_id;
                rs_poker_card_to_id(*c)
            })
            .collect();

        let board_idx = cache.find_board_index(&board_u8).unwrap();

        // Build a range with 1.0 for non-blocked combos
        let mut range = vec![0.0f32; 1326];
        for combo in 0..1326usize {
            let (c1, c2) = index_to_card_pair(combo);
            if !board_u8.contains(&c1) && !board_u8.contains(&c2) {
                range[combo] = 1.0;
            }
        }

        let reach = range_to_bucket_reach(&range, &bf, board_idx, num_buckets);
        assert_eq!(reach.len(), num_buckets);

        let total: f32 = reach.iter().sum();
        // ~1081 non-blocked combos for a 5-card board
        assert!(
            total > 1000.0 && total < 1200.0,
            "Total reach: {}",
            total
        );

        // At least some buckets should have nonzero reach
        let nonzero = reach.iter().filter(|&&r| r > 0.0).count();
        assert!(
            nonzero > 0,
            "all buckets have zero reach"
        );

        eprintln!(
            "Real file: {} buckets with nonzero reach out of {}, total reach: {}",
            nonzero, num_buckets, total
        );
    }

    /// Integration test: bucket strategy roundtrip with real file.
    #[test]
    fn test_bucket_strategy_roundtrip_real_file() {
        let path = Path::new("../../local_data/clusters_500bkt_v3/river.buckets");
        if !path.exists() {
            eprintln!(
                "Skipping test: river.buckets not found at {}",
                path.display()
            );
            return;
        }

        let bf = BucketFile::load(path).expect("Failed to load river.buckets");
        let cache = BucketedBoardCache::new(&bf);
        let num_buckets = bf.header.bucket_count as usize;

        let first_packed = bf.boards[0];
        let board_cards_rs = first_packed.to_cards(5);
        let board_u8: Vec<u8> = board_cards_rs
            .iter()
            .map(|c| {
                use poker_solver_core::blueprint_v2::full_depth_solver::rs_poker_card_to_id;
                rs_poker_card_to_id(*c)
            })
            .collect();

        let board_idx = cache.find_board_index(&board_u8).unwrap();

        let num_actions = 3;
        let mut bucket_strat = vec![0.0f32; num_actions * num_buckets];
        // Set bucket 0: check 100%
        bucket_strat[0 * num_buckets + 0] = 1.0;
        // Set bucket 1: bet 100%
        bucket_strat[1 * num_buckets + 1] = 1.0;

        let combo_strat =
            bucket_strategy_to_combos(&bucket_strat, &bf, board_idx, num_buckets, num_actions);
        assert_eq!(combo_strat.len(), num_actions * 1326);

        // Every combo in bucket 0 should have action 0 = 1.0
        for combo in 0..1326u16 {
            let bucket = bf.get_bucket(board_idx, combo) as usize;
            if bucket == 0 {
                assert_eq!(combo_strat[0 * 1326 + combo as usize], 1.0);
            }
        }
    }
}
