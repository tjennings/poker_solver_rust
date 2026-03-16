//! Bucket mapping layer for Supremus-style training data.
//!
//! After the cfvnet solver produces 1326-dim per-combo CFVs, this module maps
//! them to bucket-space for the Supremus-style bucketed network. Buckets are
//! defined by a [`BucketFile`] from the core clustering pipeline.

use poker_solver_core::blueprint_v2::bucket_file::BucketFile;

use super::range_gen::NUM_COMBOS;

/// Map 1326-dim per-combo CFVs to bucket-space by range-weighted averaging.
///
/// For each bucket `b`:
///
/// ```text
/// bucket_cfv[b] = sum(cfv[c] * range[c]) / sum(range[c])
/// ```
///
/// for all combos `c` assigned to bucket `b`. Buckets with zero total range
/// weight get a value of `0.0`.
pub fn cfvs_to_buckets(
    cfvs: &[f32],
    range: &[f32],
    bucket_file: &BucketFile,
    board_idx: u32,
    num_buckets: usize,
) -> Vec<f32> {
    debug_assert_eq!(cfvs.len(), NUM_COMBOS);
    debug_assert_eq!(range.len(), NUM_COMBOS);

    let mut bucket_sum = vec![0.0f64; num_buckets];
    let mut bucket_weight = vec![0.0f64; num_buckets];

    for combo in 0..NUM_COMBOS {
        #[allow(clippy::cast_possible_truncation)]
        let bucket = bucket_file.get_bucket(board_idx, combo as u16) as usize;
        if bucket < num_buckets {
            let w = f64::from(range[combo]);
            bucket_sum[bucket] += f64::from(cfvs[combo]) * w;
            bucket_weight[bucket] += w;
        }
    }

    bucket_sum
        .iter()
        .zip(bucket_weight.iter())
        .map(|(&s, &w)| if w > 0.0 { (s / w) as f32 } else { 0.0 })
        .collect()
}

/// Map 1326-dim range to bucket-space reach probabilities.
///
/// For each bucket `b`, sums the range weight of all combos assigned to that
/// bucket: `reach[b] = sum(range[c])` for all `c` in bucket `b`.
pub fn range_to_buckets(
    range: &[f32],
    bucket_file: &BucketFile,
    board_idx: u32,
    num_buckets: usize,
) -> Vec<f32> {
    debug_assert_eq!(range.len(), NUM_COMBOS);

    let mut reach = vec![0.0f32; num_buckets];
    for combo in 0..NUM_COMBOS {
        #[allow(clippy::cast_possible_truncation)]
        let bucket = bucket_file.get_bucket(board_idx, combo as u16) as usize;
        if bucket < num_buckets {
            reach[bucket] += range[combo];
        }
    }
    reach
}

/// Encode a solved river situation as a bucketed training record for the
/// Supremus-style network.
///
/// # Returns
///
/// `(input, target)` where:
///
/// - **input**: `[oop_reach(num_buckets) | ip_reach(num_buckets) | pot/initial_stack]`
///   — length `2 * num_buckets + 1`
/// - **target**: `[oop_cfvs(num_buckets) | ip_cfvs(num_buckets)]`
///   — length `2 * num_buckets`
pub fn encode_bucketed_record(
    oop_range: &[f32; NUM_COMBOS],
    ip_range: &[f32; NUM_COMBOS],
    oop_cfvs: &[f32; NUM_COMBOS],
    ip_cfvs: &[f32; NUM_COMBOS],
    bucket_file: &BucketFile,
    board_idx: u32,
    num_buckets: usize,
    pot: f32,
    initial_stack: f32,
) -> (Vec<f32>, Vec<f32>) {
    let oop_reach = range_to_buckets(oop_range, bucket_file, board_idx, num_buckets);
    let ip_reach = range_to_buckets(ip_range, bucket_file, board_idx, num_buckets);
    let oop_bucket_cfvs = cfvs_to_buckets(oop_cfvs, oop_range, bucket_file, board_idx, num_buckets);
    let ip_bucket_cfvs = cfvs_to_buckets(ip_cfvs, ip_range, bucket_file, board_idx, num_buckets);

    let mut input = Vec::with_capacity(2 * num_buckets + 1);
    input.extend_from_slice(&oop_reach);
    input.extend_from_slice(&ip_reach);
    input.push(pot / initial_stack);

    let mut target = Vec::with_capacity(2 * num_buckets);
    target.extend_from_slice(&oop_bucket_cfvs);
    target.extend_from_slice(&ip_bucket_cfvs);

    (input, target)
}

#[cfg(test)]
mod tests {
    use super::*;
    use poker_solver_core::blueprint_v2::bucket_file::{BucketFile, BucketFileHeader};
    use poker_solver_core::blueprint_v2::Street;

    /// Create a tiny bucket file mapping NUM_COMBOS combos across `num_buckets`
    /// buckets for a single board via round-robin assignment.
    fn make_test_bucket_file(num_buckets: u16) -> BucketFile {
        let combos_per_board = NUM_COMBOS as u16;
        let mut buckets = Vec::with_capacity(NUM_COMBOS);
        for i in 0..NUM_COMBOS {
            #[allow(clippy::cast_possible_truncation)]
            buckets.push((i % num_buckets as usize) as u16);
        }
        BucketFile {
            header: BucketFileHeader {
                street: Street::River,
                bucket_count: num_buckets,
                board_count: 1,
                combos_per_board,
                version: 2,
            },
            boards: vec![],
            buckets,
        }
    }

    #[test]
    fn cfvs_to_buckets_uniform_range() {
        let num_buckets = 10;
        let bf = make_test_bucket_file(num_buckets);

        // Uniform range
        let range = [1.0f32; NUM_COMBOS];

        // CFVs = combo index as f32
        let mut cfvs = [0.0f32; NUM_COMBOS];
        for (i, v) in cfvs.iter_mut().enumerate() {
            *v = i as f32;
        }

        let result = cfvs_to_buckets(&cfvs, &range, &bf, 0, num_buckets as usize);
        assert_eq!(result.len(), num_buckets as usize);

        // With round-robin assignment and uniform range, bucket b should contain
        // combos b, b+10, b+20, ..., and the average is their mean.
        for b in 0..num_buckets as usize {
            let combo_indices: Vec<usize> = (b..NUM_COMBOS).step_by(num_buckets as usize).collect();
            let expected: f64 =
                combo_indices.iter().map(|&i| i as f64).sum::<f64>() / combo_indices.len() as f64;
            assert!(
                (f64::from(result[b]) - expected).abs() < 1e-3,
                "bucket {b}: got {}, expected {expected}",
                result[b]
            );
        }
    }

    #[test]
    fn range_to_buckets_sums_correctly() {
        let num_buckets = 5;
        let bf = make_test_bucket_file(num_buckets);

        let mut range = [0.0f32; NUM_COMBOS];
        for (i, r) in range.iter_mut().enumerate() {
            *r = (i % 3) as f32; // 0, 1, 2, 0, 1, 2, ...
        }

        let reach = range_to_buckets(&range, &bf, 0, num_buckets as usize);
        assert_eq!(reach.len(), num_buckets as usize);

        // Verify total reach matches
        let total_range: f32 = range.iter().sum();
        let total_reach: f32 = reach.iter().sum();
        assert!(
            (total_range - total_reach).abs() < 1e-3,
            "total range {total_range} != total reach {total_reach}"
        );
    }

    #[test]
    fn encode_bucketed_record_dimensions() {
        let num_buckets = 8;
        let bf = make_test_bucket_file(num_buckets);

        let oop_range = [1.0f32; NUM_COMBOS];
        let ip_range = [1.0f32; NUM_COMBOS];
        let oop_cfvs = [0.5f32; NUM_COMBOS];
        let ip_cfvs = [-0.3f32; NUM_COMBOS];

        let (input, target) = encode_bucketed_record(
            &oop_range,
            &ip_range,
            &oop_cfvs,
            &ip_cfvs,
            &bf,
            0,
            num_buckets as usize,
            100.0,
            200.0,
        );

        // Input: 2 * num_buckets + 1
        assert_eq!(input.len(), 2 * num_buckets as usize + 1);
        // Target: 2 * num_buckets
        assert_eq!(target.len(), 2 * num_buckets as usize);
        // Last input element is pot / initial_stack = 0.5
        assert!((input[2 * num_buckets as usize] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn zero_range_bucket_gives_zero_cfv() {
        let num_buckets = 10;
        let bf = make_test_bucket_file(num_buckets);

        // Range with some zeros: only even combos have nonzero range
        let mut range = [0.0f32; NUM_COMBOS];
        for (i, r) in range.iter_mut().enumerate() {
            if i % 2 == 0 {
                *r = 1.0;
            }
        }

        let cfvs = [1.0f32; NUM_COMBOS];
        let result = cfvs_to_buckets(&cfvs, &range, &bf, 0, num_buckets as usize);

        // All buckets with any nonzero range should have cfv = 1.0
        for (b, &v) in result.iter().enumerate() {
            // Bucket b contains combos b, b+10, b+20, ...
            // Of those, only even ones have nonzero range
            let has_nonzero = (b..NUM_COMBOS)
                .step_by(num_buckets as usize)
                .any(|i| i % 2 == 0);
            if has_nonzero {
                assert!(
                    (v - 1.0).abs() < 1e-6,
                    "bucket {b} with nonzero range should have cfv=1.0, got {v}"
                );
            } else {
                assert!(
                    v.abs() < 1e-6,
                    "bucket {b} with zero range should have cfv=0.0, got {v}"
                );
            }
        }
    }
}
