use crate::abstraction::Street;
use serde::{Deserialize, Serialize};

/// Bucket boundaries for each street
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BucketBoundaries {
    pub flop: Vec<f32>,
    pub turn: Vec<f32>,
    pub river: Vec<f32>,
}

impl BucketBoundaries {
    /// Create boundaries from sorted EHS2 samples (percentile-based)
    pub fn from_samples(
        flop_samples: &mut [f32],
        turn_samples: &mut [f32],
        river_samples: &mut [f32],
        flop_buckets: usize,
        turn_buckets: usize,
        river_buckets: usize,
    ) -> Self {
        Self {
            flop: compute_percentile_boundaries(flop_samples, flop_buckets),
            turn: compute_percentile_boundaries(turn_samples, turn_buckets),
            river: compute_percentile_boundaries(river_samples, river_buckets),
        }
    }
}

fn compute_percentile_boundaries(samples: &mut [f32], num_buckets: usize) -> Vec<f32> {
    if samples.is_empty() || num_buckets == 0 {
        return vec![];
    }

    samples.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let mut boundaries = Vec::with_capacity(num_buckets.saturating_sub(1));
    for i in 1..num_buckets {
        let idx = (i * samples.len()) / num_buckets;
        let idx = idx.min(samples.len() - 1);
        boundaries.push(samples[idx]);
    }

    boundaries
}

/// Assigns EHS2 values to bucket indices
#[derive(Debug, Clone)]
pub struct BucketAssigner {
    boundaries: BucketBoundaries,
}

impl BucketAssigner {
    #[must_use]
    pub fn new(boundaries: BucketBoundaries) -> Self {
        Self { boundaries }
    }

    /// Get bucket index for an EHS2 value on a given street
    /// O(log n) via binary search
    ///
    /// Note: Preflop is not supported for EHS2-based bucketing (uses canonical hands instead).
    /// Calling this with `Street::Preflop` will return bucket 0.
    #[must_use]
    #[allow(clippy::cast_possible_truncation)] // Bucket counts are always small (<= 65535)
    pub fn get_bucket(&self, street: Street, ehs2: f32) -> u16 {
        let boundaries = match street {
            Street::Preflop => return 0, // Preflop doesn't use EHS2 bucketing
            Street::Flop => &self.boundaries.flop,
            Street::Turn => &self.boundaries.turn,
            Street::River => &self.boundaries.river,
        };

        boundaries.partition_point(|&b| b < ehs2) as u16
    }

    #[must_use]
    pub fn num_buckets(&self, street: Street) -> usize {
        match street {
            Street::Preflop => 169, // 169 canonical preflop hands (no EHS2 bucketing)
            Street::Flop => self.boundaries.flop.len() + 1,
            Street::Turn => self.boundaries.turn.len() + 1,
            Street::River => self.boundaries.river.len() + 1,
        }
    }

    #[must_use]
    pub fn boundaries(&self) -> &BucketBoundaries {
        &self.boundaries
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn percentile_boundaries_uniform_distribution() {
        let mut samples: Vec<f32> = (0..100).map(|i| i as f32 / 100.0).collect();
        let boundaries = compute_percentile_boundaries(&mut samples, 10);

        // Should have 9 boundaries for 10 buckets
        assert_eq!(boundaries.len(), 9);

        // Boundaries should be roughly 0.1, 0.2, ..., 0.9
        for (i, &b) in boundaries.iter().enumerate() {
            let expected = (i + 1) as f32 / 10.0;
            assert!(
                (b - expected).abs() < 0.02,
                "Boundary {} expected ~{}, got {}",
                i,
                expected,
                b
            );
        }
    }

    #[test]
    fn bucket_assignment_correct() {
        let mut samples: Vec<f32> = (0..100).map(|i| i as f32 / 100.0).collect();
        let boundaries = BucketBoundaries {
            flop: compute_percentile_boundaries(&mut samples.clone(), 10),
            turn: compute_percentile_boundaries(&mut samples.clone(), 10),
            river: compute_percentile_boundaries(&mut samples, 10),
        };
        let assigner = BucketAssigner::new(boundaries);

        // EHS2 = 0.05 should be bucket 0
        assert_eq!(assigner.get_bucket(Street::Flop, 0.05), 0);

        // EHS2 = 0.15 should be bucket 1
        assert_eq!(assigner.get_bucket(Street::Flop, 0.15), 1);

        // EHS2 = 0.95 should be bucket 9
        assert_eq!(assigner.get_bucket(Street::Flop, 0.95), 9);
    }

    #[test]
    fn extreme_values_boundary_buckets() {
        let boundaries = BucketBoundaries {
            flop: vec![0.2, 0.4, 0.6, 0.8],
            turn: vec![0.2, 0.4, 0.6, 0.8],
            river: vec![0.2, 0.4, 0.6, 0.8],
        };
        let assigner = BucketAssigner::new(boundaries);

        // EHS2 = 0.0 should be bucket 0
        assert_eq!(assigner.get_bucket(Street::River, 0.0), 0);

        // EHS2 = 1.0 should be last bucket (4)
        assert_eq!(assigner.get_bucket(Street::River, 1.0), 4);
    }

    #[test]
    fn num_buckets_correct() {
        let boundaries = BucketBoundaries {
            flop: vec![0.25, 0.5, 0.75], // 4 buckets
            turn: vec![0.5],             // 2 buckets
            river: vec![],               // 1 bucket
        };
        let assigner = BucketAssigner::new(boundaries);

        assert_eq!(assigner.num_buckets(Street::Flop), 4);
        assert_eq!(assigner.num_buckets(Street::Turn), 2);
        assert_eq!(assigner.num_buckets(Street::River), 1);
    }
}
