//! Bucketed situation sampling using cfvnet's high-quality sampler.
//!
//! Uses cfvnet's `sample_situation()` to generate stratified pot/SPR + R(S,p) ranges
//! on the CPU, then maps the 1326-dim ranges to bucket-space reach using a BucketFile.

use cfvnet::config::DatagenConfig;
use cfvnet::datagen::sampler::{sample_situation, Situation};
use poker_solver_core::blueprint_v2::bucket_file::BucketFile;
use rand::Rng;

use super::equity::BucketedBoardCache;

/// Configuration for bucketed situation sampling.
#[derive(Debug, Clone)]
pub struct BucketedSamplingConfig {
    pub pot_intervals: Vec<[i32; 2]>,
    pub spr_intervals: Option<Vec<[f64; 2]>>,
    pub initial_stack: i32,
}

/// A situation mapped to bucket space, ready for the GPU solver.
pub struct BucketedSituation {
    /// Raw board cards (5 elements for river).
    pub board: [u8; 5],
    /// Index of this board in the BucketFile.
    pub board_idx: u32,
    /// Pot size in chips.
    pub pot: i32,
    /// Effective stack in chips.
    pub effective_stack: i32,
    /// OOP reach probabilities in bucket space, length = `num_buckets`.
    pub oop_reach: Vec<f32>,
    /// IP reach probabilities in bucket space, length = `num_buckets`.
    pub ip_reach: Vec<f32>,
}

/// Sample `count` situations and map them to bucket space.
///
/// Uses cfvnet's stratified sampler for high-quality pot/SPR distributions
/// and R(S,p) ranges, then aggregates the per-combo ranges into bucket-space
/// reach vectors using the bucket file.
///
/// Situations whose board is not found in the bucket file are silently skipped.
/// For canonical river boards, this should be rare.
///
/// # Arguments
/// - `config`: pot/SPR intervals and initial stack
/// - `bucket_file`: loaded bucket file (e.g. river.buckets)
/// - `board_cache`: precomputed board-to-index mapping
/// - `num_buckets`: number of buckets (e.g. 500)
/// - `board_size`: number of board cards (5 for river, 4 for turn)
/// - `count`: number of situations to sample
/// - `rng`: random number generator
pub fn sample_bucketed_situations(
    config: &BucketedSamplingConfig,
    bucket_file: &BucketFile,
    board_cache: &BucketedBoardCache,
    num_buckets: usize,
    board_size: usize,
    count: usize,
    rng: &mut impl Rng,
) -> Vec<BucketedSituation> {
    let datagen_config = DatagenConfig {
        pot_intervals: config.pot_intervals.clone(),
        spr_intervals: config.spr_intervals.clone(),
        ..Default::default()
    };

    let mut situations = Vec::with_capacity(count);

    for _ in 0..count {
        let sit = sample_situation(&datagen_config, config.initial_stack, board_size, rng);

        // Find board index in the bucket file
        let board_idx = match board_cache.find_board_index(sit.board_cards()) {
            Some(idx) => idx,
            None => continue, // skip boards not in bucket file
        };

        // Map 1326-dim ranges to bucket-space reach
        let mut oop_reach = vec![0.0f32; num_buckets];
        let mut ip_reach = vec![0.0f32; num_buckets];

        for combo in 0..1326u16 {
            let bucket = bucket_file.get_bucket(board_idx, combo) as usize;
            if bucket < num_buckets {
                oop_reach[bucket] += sit.ranges[0][combo as usize];
                ip_reach[bucket] += sit.ranges[1][combo as usize];
            }
        }

        situations.push(BucketedSituation {
            board: sit.board,
            board_idx,
            pot: sit.pot,
            effective_stack: sit.effective_stack,
            oop_reach,
            ip_reach,
        });
    }

    situations
}

/// Convert a cfvnet `Situation` to a `BucketedSituation` given precomputed mappings.
///
/// This is useful when you already have a `Situation` from cfvnet and want to
/// convert it to bucket space without going through the full sampling pipeline.
///
/// Returns `None` if the board is not found in the bucket file.
pub fn situation_to_bucketed(
    sit: &Situation,
    bucket_file: &BucketFile,
    board_cache: &BucketedBoardCache,
    num_buckets: usize,
) -> Option<BucketedSituation> {
    let board_idx = board_cache.find_board_index(sit.board_cards())?;

    let mut oop_reach = vec![0.0f32; num_buckets];
    let mut ip_reach = vec![0.0f32; num_buckets];

    for combo in 0..1326u16 {
        let bucket = bucket_file.get_bucket(board_idx, combo) as usize;
        if bucket < num_buckets {
            oop_reach[bucket] += sit.ranges[0][combo as usize];
            ip_reach[bucket] += sit.ranges[1][combo as usize];
        }
    }

    Some(BucketedSituation {
        board: sit.board,
        board_idx,
        pot: sit.pot,
        effective_stack: sit.effective_stack,
        oop_reach,
        ip_reach,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bucketed::equity::BucketedBoardCache;
    use poker_solver_core::blueprint_v2::bucket_file::{BucketFile, BucketFileHeader};
    use poker_solver_core::blueprint_v2::cluster_pipeline::canonical_key;
    use poker_solver_core::blueprint_v2::Street;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;
    use std::path::Path;

    use crate::bucketed::equity::u8_to_rs_card;

    /// Integration test with real bucket file (skipped if file not found).
    #[test]
    fn test_sample_bucketed_situations() {
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
        let config = BucketedSamplingConfig {
            pot_intervals: vec![[4, 50], [50, 100], [100, 150], [150, 200]],
            spr_intervals: Some(vec![
                [0.0, 0.5],
                [0.5, 1.5],
                [1.5, 4.0],
                [4.0, 8.0],
                [8.0, 50.0],
            ]),
            initial_stack: 200,
        };

        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let sits = sample_bucketed_situations(&config, &bf, &cache, 500, 5, 100, &mut rng);

        // Some may be skipped if board not in bucket file, but most should succeed
        assert!(
            sits.len() >= 50,
            "Expected at least 50 situations, got {}",
            sits.len()
        );

        for sit in &sits {
            // Reach should be non-negative
            assert!(sit.oop_reach.iter().all(|&r| r >= 0.0));
            assert!(sit.ip_reach.iter().all(|&r| r >= 0.0));

            // R(S,p) ranges sum to ~1.0 (allowing some tolerance for bucketing)
            let oop_sum: f32 = sit.oop_reach.iter().sum();
            assert!(
                oop_sum > 0.5 && oop_sum < 1.5,
                "OOP reach sum: {}",
                oop_sum
            );

            let ip_sum: f32 = sit.ip_reach.iter().sum();
            assert!(
                ip_sum > 0.5 && ip_sum < 1.5,
                "IP reach sum: {}",
                ip_sum
            );

            // Pot should be in configured intervals
            assert!(
                sit.pot >= 4 && sit.pot <= 200,
                "Pot {} out of range",
                sit.pot
            );

            // Board idx should be valid
            assert!(sit.board_idx < bf.header.board_count);
        }

        eprintln!(
            "Sampled {} bucketed situations out of 100 attempts",
            sits.len()
        );
    }

    /// Test situation_to_bucketed with a synthetic bucket file.
    #[test]
    fn test_situation_to_bucketed_synthetic() {
        // Create a synthetic bucket file with a known board
        let board_u8 = [0u8, 5, 10, 15, 20];
        let rs_cards: Vec<_> = board_u8.iter().map(|&c| u8_to_rs_card(c)).collect();
        let packed = canonical_key(&rs_cards);

        let num_buckets = 10u16;
        let mut bucket_data = vec![0u16; 1326];
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

        let cache = BucketedBoardCache::new(&bf);

        // Create a Situation manually
        let mut ranges = [[0.0f32; 1326]; 2];
        // Simple: spread weight uniformly across non-blocked combos
        let mut valid_count = 0;
        for combo in 0..1326usize {
            let (c1, c2) = range_solver::card::index_to_card_pair(combo);
            if !board_u8.contains(&c1) && !board_u8.contains(&c2) {
                valid_count += 1;
            }
        }
        let weight = 1.0 / valid_count as f32;
        for combo in 0..1326usize {
            let (c1, c2) = range_solver::card::index_to_card_pair(combo);
            if !board_u8.contains(&c1) && !board_u8.contains(&c2) {
                ranges[0][combo] = weight;
                ranges[1][combo] = weight;
            }
        }

        let sit = Situation {
            board: board_u8,
            board_size: 5,
            pot: 100,
            effective_stack: 200,
            ranges,
        };

        let bucketed = situation_to_bucketed(&sit, &bf, &cache, num_buckets as usize);
        assert!(bucketed.is_some());

        let bs = bucketed.unwrap();
        assert_eq!(bs.pot, 100);
        assert_eq!(bs.effective_stack, 200);
        assert_eq!(bs.oop_reach.len(), num_buckets as usize);
        assert_eq!(bs.ip_reach.len(), num_buckets as usize);

        // Total reach should sum to ~1.0
        let oop_sum: f32 = bs.oop_reach.iter().sum();
        assert!(
            (oop_sum - 1.0).abs() < 0.01,
            "OOP reach sum: {}",
            oop_sum
        );

        // All buckets should have some reach (since we assigned round-robin)
        for (i, &r) in bs.oop_reach.iter().enumerate() {
            assert!(r > 0.0, "Bucket {} has zero OOP reach", i);
        }
    }

    /// Test that a board NOT in the bucket file returns None.
    #[test]
    fn test_situation_to_bucketed_unknown_board() {
        let board_u8 = [0u8, 5, 10, 15, 20];
        let rs_cards: Vec<_> = board_u8.iter().map(|&c| u8_to_rs_card(c)).collect();
        let packed = canonical_key(&rs_cards);

        let bf = BucketFile {
            header: BucketFileHeader {
                street: Street::River,
                bucket_count: 10,
                board_count: 1,
                combos_per_board: 1326,
                version: 2,
            },
            boards: vec![packed],
            buckets: vec![0u16; 1326],
        };

        let cache = BucketedBoardCache::new(&bf);

        // Use a different board than what's in the bucket file
        let different_board = [1u8, 6, 11, 16, 21];
        let sit = Situation {
            board: different_board,
            board_size: 5,
            pot: 100,
            effective_stack: 200,
            ranges: [[0.0f32; 1326]; 2],
        };

        let result = situation_to_bucketed(&sit, &bf, &cache, 10);
        assert!(result.is_none());
    }
}
