// Belief updates — Bayesian reach probability updates

use rand::Rng;

/// Update reach probabilities after an action is taken.
///
/// For each combo i: reach[i] *= action_probs_per_bucket[combo_buckets[i]][action_taken]
///
/// Combos with reach == 0.0 are skipped (they're blocked).
pub fn update_reach(
    reach: &mut [f32],
    combo_buckets: &[u16],
    action_probs_per_bucket: &[Vec<f32>],
    action_taken: usize,
) {
    debug_assert_eq!(reach.len(), combo_buckets.len());
    for (r, &bucket) in reach.iter_mut().zip(combo_buckets.iter()) {
        if *r == 0.0 {
            continue;
        }
        *r *= action_probs_per_bucket[bucket as usize][action_taken];
    }
}

/// Sample an action from a probability distribution.
///
/// Returns the index of the sampled action.
/// Uses cumulative probability: find first action where cumsum >= random value.
pub fn sample_action(action_probs: &[f32], rng: &mut impl Rng) -> usize {
    debug_assert!(!action_probs.is_empty());
    let r: f64 = rng.random();
    let mut cumulative = 0.0;
    for (i, &prob) in action_probs.iter().enumerate() {
        cumulative += f64::from(prob);
        if r < cumulative {
            return i;
        }
    }
    // Fallback to last action (handles floating-point rounding)
    action_probs.len() - 1
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    #[test]
    fn test_update_reach_basic() {
        // 3 combos, 2 buckets, 2 actions
        // Combo 0,1 → bucket 0 (probs [0.7, 0.3])
        // Combo 2 → bucket 1 (probs [0.4, 0.6])
        let mut reach = vec![1.0_f32, 1.0, 1.0];
        let combo_buckets: Vec<u16> = vec![0, 0, 1];
        let action_probs_per_bucket = vec![
            vec![0.7_f32, 0.3],
            vec![0.4_f32, 0.6],
        ];

        // Take action 0
        update_reach(&mut reach, &combo_buckets, &action_probs_per_bucket, 0);

        assert!((reach[0] - 0.7).abs() < 1e-6, "combo 0: expected 0.7, got {}", reach[0]);
        assert!((reach[1] - 0.7).abs() < 1e-6, "combo 1: expected 0.7, got {}", reach[1]);
        assert!((reach[2] - 0.4).abs() < 1e-6, "combo 2: expected 0.4, got {}", reach[2]);
    }

    #[test]
    fn test_update_reach_preserves_zero() {
        // Combo with reach=0.0 stays 0.0 after update
        let mut reach = vec![1.0_f32, 0.0, 1.0];
        let combo_buckets: Vec<u16> = vec![0, 0, 1];
        let action_probs_per_bucket = vec![
            vec![0.7_f32, 0.3],
            vec![0.4_f32, 0.6],
        ];

        update_reach(&mut reach, &combo_buckets, &action_probs_per_bucket, 0);

        assert!((reach[0] - 0.7).abs() < 1e-6, "combo 0: expected 0.7, got {}", reach[0]);
        assert_eq!(reach[1], 0.0, "combo 1: expected 0.0 (blocked), got {}", reach[1]);
        assert!((reach[2] - 0.4).abs() < 1e-6, "combo 2: expected 0.4, got {}", reach[2]);
    }

    #[test]
    fn test_update_reach_sequential() {
        // Two sequential updates
        // Combo 0 (bucket 0, probs [0.8, 0.2]): action 0 then action 1 → 0.8 * 0.2 = 0.16
        // Combo 1 (bucket 1, probs [0.3, 0.7]): action 0 then action 1 → 0.3 * 0.7 = 0.21
        let mut reach = vec![1.0_f32, 1.0];
        let combo_buckets: Vec<u16> = vec![0, 1];
        let action_probs_per_bucket = vec![
            vec![0.8_f32, 0.2],
            vec![0.3_f32, 0.7],
        ];

        // First update: action 0
        update_reach(&mut reach, &combo_buckets, &action_probs_per_bucket, 0);
        assert!((reach[0] - 0.8).abs() < 1e-6, "after action 0, combo 0: expected 0.8, got {}", reach[0]);
        assert!((reach[1] - 0.3).abs() < 1e-6, "after action 0, combo 1: expected 0.3, got {}", reach[1]);

        // Second update: action 1
        update_reach(&mut reach, &combo_buckets, &action_probs_per_bucket, 1);
        assert!((reach[0] - 0.16).abs() < 1e-6, "after action 1, combo 0: expected 0.16, got {}", reach[0]);
        assert!((reach[1] - 0.21).abs() < 1e-5, "after action 1, combo 1: expected 0.21, got {}", reach[1]);
    }

    #[test]
    fn test_sample_action_distribution() {
        // Sample 10000 times from [0.7, 0.2, 0.1]
        // Verify each action's frequency is within ±5% of expected
        let action_probs = vec![0.7_f32, 0.2, 0.1];
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let n = 10_000;

        let mut counts = vec![0usize; 3];
        for _ in 0..n {
            let action = sample_action(&action_probs, &mut rng);
            counts[action] += 1;
        }

        let freq0 = counts[0] as f64 / n as f64;
        let freq1 = counts[1] as f64 / n as f64;
        let freq2 = counts[2] as f64 / n as f64;

        assert!(
            (freq0 - 0.7).abs() < 0.05,
            "action 0: expected ~0.7, got {freq0}"
        );
        assert!(
            (freq1 - 0.2).abs() < 0.05,
            "action 1: expected ~0.2, got {freq1}"
        );
        assert!(
            (freq2 - 0.1).abs() < 0.05,
            "action 2: expected ~0.1, got {freq2}"
        );
    }

    #[test]
    fn test_sample_action_deterministic() {
        // With a seeded RNG, verify deterministic sampling
        let action_probs = vec![0.5_f32, 0.3, 0.2];

        let mut rng1 = ChaCha8Rng::seed_from_u64(123);
        let mut rng2 = ChaCha8Rng::seed_from_u64(123);

        let samples1: Vec<usize> = (0..100).map(|_| sample_action(&action_probs, &mut rng1)).collect();
        let samples2: Vec<usize> = (0..100).map(|_| sample_action(&action_probs, &mut rng2)).collect();

        assert_eq!(samples1, samples2, "same seed must produce same sequence");
    }
}
