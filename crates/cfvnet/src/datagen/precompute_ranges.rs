//! Precompute average turn entry ranges from a blueprint strategy, bucketed by SPR.
//!
//! The precomputed ranges file allows datagen to use realistic turn entry
//! ranges (from blueprint play-through) instead of random RSP ranges.

use std::path::Path;

use rand::Rng;
use rand_chacha::ChaCha8Rng;
use rand::SeedableRng;
use serde::{Deserialize, Serialize};

use super::blueprint_ranges::{BlueprintRangeGenerator, NUM_COMBOS};
use super::sampler::sample_board;

/// Precomputed turn entry ranges bucketed by SPR.
#[derive(Serialize, Deserialize, Debug)]
pub struct PrecomputedRanges {
    /// SPR bucket boundaries. N boundaries define N-1 buckets.
    /// Bucket i covers SPR in [boundaries[i], boundaries[i+1]).
    pub spr_boundaries: Vec<f64>,
    /// One entry per SPR bucket.
    pub buckets: Vec<SprBucketRanges>,
}

/// Ranges for a single SPR bucket.
#[derive(Serialize, Deserialize, Debug)]
pub struct SprBucketRanges {
    /// SPR range lower bound (inclusive).
    pub spr_low: f64,
    /// SPR range upper bound (exclusive).
    pub spr_high: f64,
    /// Average OOP range (1326 combos, reach-weighted).
    pub oop_range: Vec<f32>,
    /// Average IP range (1326 combos, reach-weighted).
    pub ip_range: Vec<f32>,
    /// Number of samples that contributed to this average.
    pub sample_count: u32,
}

impl PrecomputedRanges {
    /// Save to file using bincode.
    pub fn save(&self, path: &Path) -> Result<(), String> {
        let file = std::fs::File::create(path).map_err(|e| format!("create: {e}"))?;
        let writer = std::io::BufWriter::new(file);
        bincode::serialize_into(writer, self).map_err(|e| format!("serialize: {e}"))
    }

    /// Load from file.
    pub fn load(path: &Path) -> Result<Self, String> {
        let file = std::fs::File::open(path).map_err(|e| format!("open: {e}"))?;
        let reader = std::io::BufReader::new(file);
        bincode::deserialize_from(reader).map_err(|e| format!("deserialize: {e}"))
    }

    /// Look up ranges for a given SPR value.
    pub fn lookup(&self, spr: f64) -> Option<&SprBucketRanges> {
        for bucket in &self.buckets {
            if spr >= bucket.spr_low && spr < bucket.spr_high {
                return Some(bucket);
            }
        }
        // Clamp to last bucket if above max
        self.buckets.last()
    }
}

/// Parse comma-separated SPR boundary string into a sorted vec of f64.
pub fn parse_spr_boundaries(s: &str) -> Vec<f64> {
    s.split(',')
        .filter_map(|tok| tok.trim().parse::<f64>().ok())
        .collect()
}

/// Sample a (pot, effective_stack) pair targeting an SPR in [spr_low, spr_high).
///
/// `initial_stack` is the game's starting stack (e.g. 200).
/// Constrains the pot so that `spr * pot <= initial_stack - pot/2`,
/// ensuring the derived stack is feasible without clamping distortion.
pub fn sample_pot_stack_for_spr(
    spr_low: f64,
    spr_high: f64,
    initial_stack: i32,
    rng: &mut impl Rng,
) -> (i32, i32) {
    let spr = rng.gen_range(spr_low..spr_high.max(spr_low + 0.01));
    // pot * spr = stack, stack <= initial_stack - pot/2
    // => pot <= initial_stack / (spr + 0.5)
    let feasible_pot_max = (initial_stack as f64 / (spr + 0.5)).floor() as i32;
    let max_pot = feasible_pot_max.min(initial_stack * 2).min(400);
    if max_pot < 4 {
        // Extreme SPR for this stack; return minimum viable values
        return (4, (spr * 4.0).max(1.0) as i32);
    }
    let pot = rng.gen_range(4..=max_pot);
    let max_stack = initial_stack - pot / 2;
    let stack = ((spr * pot as f64) as i32).min(max_stack).max(1);
    (pot, stack)
}

/// Precompute average turn entry ranges for each SPR bucket by sampling
/// through the blueprint strategy.
///
/// `spr_boundaries` defines N boundaries -> N-1 buckets.
/// For each bucket, `samples_per_bucket` successful samples are collected.
pub fn precompute_turn_ranges(
    generator: &BlueprintRangeGenerator,
    spr_boundaries: &[f64],
    samples_per_bucket: usize,
    _initial_stack: i32,
) -> PrecomputedRanges {
    let mut buckets = Vec::new();

    for window in spr_boundaries.windows(2) {
        let spr_low = window[0];
        let spr_high = window[1];

        let mut oop_sum = vec![0.0f64; NUM_COMBOS];
        let mut ip_sum = vec![0.0f64; NUM_COMBOS];
        let mut count = 0u32;
        let mut rng = ChaCha8Rng::seed_from_u64((spr_low * 1000.0) as u64);

        let max_attempts = samples_per_bucket * 20;
        let mut attempts = 0;
        while (count as usize) < samples_per_bucket && attempts < max_attempts {
            attempts += 1;

            let board = sample_board(4, &mut rng);

            if let Some(sit) = generator.sample_turn_ranges(&board[..4], &mut rng) {
                for i in 0..NUM_COMBOS {
                    oop_sum[i] += sit.oop_range[i] as f64;
                    ip_sum[i] += sit.ip_range[i] as f64;
                }
                count += 1;
            }
        }

        // Average
        let oop_range: Vec<f32> = oop_sum
            .iter()
            .map(|&s| if count > 0 { (s / count as f64) as f32 } else { 0.0 })
            .collect();
        let ip_range: Vec<f32> = ip_sum
            .iter()
            .map(|&s| if count > 0 { (s / count as f64) as f32 } else { 0.0 })
            .collect();

        eprintln!(
            "[precompute] SPR [{spr_low:.1}, {spr_high:.1}): {count} samples from {attempts} attempts"
        );

        buckets.push(SprBucketRanges {
            spr_low,
            spr_high,
            oop_range,
            ip_range,
            sample_count: count,
        });
    }

    PrecomputedRanges {
        spr_boundaries: spr_boundaries.to_vec(),
        buckets,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- PrecomputedRanges save/load roundtrip ----

    #[test]
    fn save_load_roundtrip_preserves_data() {
        let ranges = PrecomputedRanges {
            spr_boundaries: vec![0.0, 1.0, 4.0, 10.0],
            buckets: vec![
                SprBucketRanges {
                    spr_low: 0.0,
                    spr_high: 1.0,
                    oop_range: vec![0.5; NUM_COMBOS],
                    ip_range: vec![0.3; NUM_COMBOS],
                    sample_count: 100,
                },
                SprBucketRanges {
                    spr_low: 1.0,
                    spr_high: 4.0,
                    oop_range: vec![0.7; NUM_COMBOS],
                    ip_range: vec![0.6; NUM_COMBOS],
                    sample_count: 200,
                },
                SprBucketRanges {
                    spr_low: 4.0,
                    spr_high: 10.0,
                    oop_range: vec![0.9; NUM_COMBOS],
                    ip_range: vec![0.8; NUM_COMBOS],
                    sample_count: 50,
                },
            ],
        };

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("ranges.bin");
        ranges.save(&path).unwrap();
        let loaded = PrecomputedRanges::load(&path).unwrap();

        assert_eq!(loaded.spr_boundaries, ranges.spr_boundaries);
        assert_eq!(loaded.buckets.len(), 3);
        for (orig, load) in ranges.buckets.iter().zip(&loaded.buckets) {
            assert_eq!(orig.spr_low, load.spr_low);
            assert_eq!(orig.spr_high, load.spr_high);
            assert_eq!(orig.sample_count, load.sample_count);
            assert_eq!(orig.oop_range.len(), NUM_COMBOS);
            assert_eq!(load.oop_range.len(), NUM_COMBOS);
            assert_eq!(orig.oop_range, load.oop_range);
            assert_eq!(orig.ip_range, load.ip_range);
        }
    }

    #[test]
    fn save_load_roundtrip_empty_buckets() {
        let ranges = PrecomputedRanges {
            spr_boundaries: vec![0.0, 50.0],
            buckets: vec![SprBucketRanges {
                spr_low: 0.0,
                spr_high: 50.0,
                oop_range: vec![0.0; NUM_COMBOS],
                ip_range: vec![0.0; NUM_COMBOS],
                sample_count: 0,
            }],
        };

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("empty.bin");
        ranges.save(&path).unwrap();
        let loaded = PrecomputedRanges::load(&path).unwrap();

        assert_eq!(loaded.buckets[0].sample_count, 0);
        assert!(loaded.buckets[0].oop_range.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn load_nonexistent_file_returns_error() {
        let result = PrecomputedRanges::load(std::path::Path::new("/tmp/nonexistent_xyz.bin"));
        assert!(result.is_err());
    }

    // ---- lookup tests ----

    #[test]
    fn lookup_finds_correct_bucket() {
        let ranges = PrecomputedRanges {
            spr_boundaries: vec![0.0, 1.0, 4.0, 10.0],
            buckets: vec![
                SprBucketRanges {
                    spr_low: 0.0,
                    spr_high: 1.0,
                    oop_range: vec![1.0; NUM_COMBOS],
                    ip_range: vec![0.0; NUM_COMBOS],
                    sample_count: 10,
                },
                SprBucketRanges {
                    spr_low: 1.0,
                    spr_high: 4.0,
                    oop_range: vec![2.0; NUM_COMBOS],
                    ip_range: vec![0.0; NUM_COMBOS],
                    sample_count: 20,
                },
                SprBucketRanges {
                    spr_low: 4.0,
                    spr_high: 10.0,
                    oop_range: vec![3.0; NUM_COMBOS],
                    ip_range: vec![0.0; NUM_COMBOS],
                    sample_count: 30,
                },
            ],
        };

        // SPR 0.5 -> first bucket
        let b = ranges.lookup(0.5).unwrap();
        assert_eq!(b.sample_count, 10);

        // SPR 2.0 -> second bucket
        let b = ranges.lookup(2.0).unwrap();
        assert_eq!(b.sample_count, 20);

        // SPR 7.0 -> third bucket
        let b = ranges.lookup(7.0).unwrap();
        assert_eq!(b.sample_count, 30);
    }

    #[test]
    fn lookup_at_boundary_returns_higher_bucket() {
        let ranges = PrecomputedRanges {
            spr_boundaries: vec![0.0, 1.0, 4.0],
            buckets: vec![
                SprBucketRanges {
                    spr_low: 0.0,
                    spr_high: 1.0,
                    oop_range: vec![0.0; NUM_COMBOS],
                    ip_range: vec![0.0; NUM_COMBOS],
                    sample_count: 10,
                },
                SprBucketRanges {
                    spr_low: 1.0,
                    spr_high: 4.0,
                    oop_range: vec![0.0; NUM_COMBOS],
                    ip_range: vec![0.0; NUM_COMBOS],
                    sample_count: 20,
                },
            ],
        };

        // Exactly at boundary 1.0 -> second bucket [1.0, 4.0)
        let b = ranges.lookup(1.0).unwrap();
        assert_eq!(b.sample_count, 20);
    }

    #[test]
    fn lookup_above_max_clamps_to_last_bucket() {
        let ranges = PrecomputedRanges {
            spr_boundaries: vec![0.0, 1.0, 4.0],
            buckets: vec![
                SprBucketRanges {
                    spr_low: 0.0,
                    spr_high: 1.0,
                    oop_range: vec![0.0; NUM_COMBOS],
                    ip_range: vec![0.0; NUM_COMBOS],
                    sample_count: 10,
                },
                SprBucketRanges {
                    spr_low: 1.0,
                    spr_high: 4.0,
                    oop_range: vec![0.0; NUM_COMBOS],
                    ip_range: vec![0.0; NUM_COMBOS],
                    sample_count: 20,
                },
            ],
        };

        // SPR 100 is above max -> clamps to last bucket
        let b = ranges.lookup(100.0).unwrap();
        assert_eq!(b.sample_count, 20);
    }

    #[test]
    fn lookup_below_min_clamps_to_last_bucket() {
        let ranges = PrecomputedRanges {
            spr_boundaries: vec![1.0, 4.0],
            buckets: vec![SprBucketRanges {
                spr_low: 1.0,
                spr_high: 4.0,
                oop_range: vec![0.0; NUM_COMBOS],
                ip_range: vec![0.0; NUM_COMBOS],
                sample_count: 10,
            }],
        };

        // SPR 0.5 is below the first bucket's spr_low — clamped to last
        let b = ranges.lookup(0.5);
        assert!(b.is_some());
        assert_eq!(b.unwrap().sample_count, 10);
    }

    // ---- sample_pot_stack_for_spr tests ----

    #[test]
    fn sample_pot_stack_produces_valid_spr() {
        use rand::SeedableRng;
        use rand_chacha::ChaCha8Rng;
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let initial_stack = 200;

        for _ in 0..100 {
            let (pot, stack) = sample_pot_stack_for_spr(1.0, 4.0, initial_stack, &mut rng);
            assert!(pot >= 4, "pot too small: {pot}");
            assert!(stack >= 1, "stack too small: {stack}");
            assert!(stack <= initial_stack, "stack exceeds initial: {stack}");
            let spr = stack as f64 / pot as f64;
            // Allow some tolerance since we're clamping
            assert!(
                spr >= 0.5 && spr < 5.0,
                "SPR {spr} out of expected range for [1.0, 4.0), pot={pot}, stack={stack}"
            );
        }
    }

    #[test]
    fn sample_pot_stack_low_spr_bucket() {
        use rand::SeedableRng;
        use rand_chacha::ChaCha8Rng;
        let mut rng = ChaCha8Rng::seed_from_u64(99);
        let initial_stack = 200;

        for _ in 0..100 {
            let (pot, stack) = sample_pot_stack_for_spr(0.0, 0.5, initial_stack, &mut rng);
            assert!(pot >= 4);
            assert!(stack >= 1);
            let spr = stack as f64 / pot as f64;
            assert!(
                spr < 1.0,
                "SPR {spr} too high for low-SPR bucket, pot={pot}, stack={stack}"
            );
        }
    }

    #[test]
    fn sample_pot_stack_high_spr_bucket() {
        use rand::SeedableRng;
        use rand_chacha::ChaCha8Rng;
        let mut rng = ChaCha8Rng::seed_from_u64(77);
        let initial_stack = 200;

        for _ in 0..100 {
            let (pot, stack) = sample_pot_stack_for_spr(8.0, 50.0, initial_stack, &mut rng);
            assert!(pot >= 4);
            assert!(stack >= 1);
            let spr = stack as f64 / pot as f64;
            assert!(
                spr >= 4.0,
                "SPR {spr} too low for high-SPR bucket, pot={pot}, stack={stack}"
            );
        }
    }

    // ---- parse_spr_boundaries tests ----

    #[test]
    fn parse_spr_boundaries_valid() {
        let result = parse_spr_boundaries("0,0.5,1.5,4,8,50");
        assert_eq!(result, vec![0.0, 0.5, 1.5, 4.0, 8.0, 50.0]);
    }

    #[test]
    fn parse_spr_boundaries_with_spaces() {
        let result = parse_spr_boundaries("0, 0.5, 1.5, 4, 8, 50");
        assert_eq!(result, vec![0.0, 0.5, 1.5, 4.0, 8.0, 50.0]);
    }

    #[test]
    fn parse_spr_boundaries_ignores_non_numeric() {
        let result = parse_spr_boundaries("0,abc,1.5,4");
        assert_eq!(result, vec![0.0, 1.5, 4.0]);
    }
}
