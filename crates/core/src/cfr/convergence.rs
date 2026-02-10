//! Convergence metrics for MCCFR training.
//!
//! Pure functions that compute quantitative indicators of training progress.
//! All functions take `&FxHashMap` references and return scalar metrics.

use rustc_hash::FxHashMap;

/// Mean L1 distance between two strategy maps.
///
/// For each info set present in both maps, computes the sum of absolute
/// differences between corresponding action probabilities, then averages
/// across all shared info sets. Returns 0.0 if no info sets are shared.
#[must_use]
#[allow(clippy::implicit_hasher)]
pub fn strategy_delta(
    prev: &FxHashMap<u64, Vec<f64>>,
    curr: &FxHashMap<u64, Vec<f64>>,
) -> f64 {
    let mut total_delta = 0.0;
    let mut count = 0u64;

    for (key, prev_probs) in prev {
        if let Some(curr_probs) = curr.get(key) {
            let l1: f64 = prev_probs
                .iter()
                .zip(curr_probs.iter())
                .map(|(a, b)| (a - b).abs())
                .sum();
            total_delta += l1;
            count += 1;
        }
    }

    if count == 0 {
        0.0
    } else {
        #[allow(clippy::cast_precision_loss)]
        { total_delta / count as f64 }
    }
}

/// Maximum absolute regret across all info sets and actions.
///
/// Returns 0.0 if the regret map is empty.
#[must_use]
#[allow(clippy::implicit_hasher)]
pub fn max_regret(regret_sum: &FxHashMap<u64, Vec<f64>>) -> f64 {
    regret_sum
        .values()
        .flat_map(|v| v.iter())
        .map(|r| r.abs())
        .fold(0.0_f64, f64::max)
}

/// Mean absolute regret across all info sets and actions.
///
/// Returns 0.0 if the regret map is empty.
#[must_use]
#[allow(clippy::implicit_hasher)]
pub fn avg_regret(regret_sum: &FxHashMap<u64, Vec<f64>>) -> f64 {
    let mut total = 0.0;
    let mut count = 0u64;

    for regrets in regret_sum.values() {
        for &r in regrets {
            total += r.abs();
            count += 1;
        }
    }

    if count == 0 {
        0.0
    } else {
        #[allow(clippy::cast_precision_loss)]
        { total / count as f64 }
    }
}

/// Mean Shannon entropy across all info sets.
///
/// For each info set, computes `-sum(p * ln(p))` for each action probability `p > 0`,
/// then averages across all info sets. Higher entropy means more uniform/uncertain
/// strategies; lower entropy means more polarized/decided strategies.
///
/// Returns 0.0 if the strategy map is empty.
#[must_use]
#[allow(clippy::implicit_hasher)]
pub fn strategy_entropy(strategies: &FxHashMap<u64, Vec<f64>>) -> f64 {
    let mut total_entropy = 0.0;
    let mut count = 0u64;

    for probs in strategies.values() {
        let entropy: f64 = probs
            .iter()
            .filter(|&&p| p > 0.0)
            .map(|&p| -p * p.ln())
            .sum();
        total_entropy += entropy;
        count += 1;
    }

    if count == 0 {
        0.0
    } else {
        #[allow(clippy::cast_precision_loss)]
        { total_entropy / count as f64 }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_map(entries: &[(u64, Vec<f64>)]) -> FxHashMap<u64, Vec<f64>> {
        entries.iter().cloned().collect()
    }

    #[test]
    fn delta_zero_for_identical_maps() {
        let map = make_map(&[(1, vec![0.5, 0.5]), (2, vec![0.3, 0.7])]);
        assert!((strategy_delta(&map, &map)).abs() < 1e-10);
    }

    #[test]
    fn delta_nonzero_for_changes() {
        let prev = make_map(&[(1, vec![0.5, 0.5])]);
        let curr = make_map(&[(1, vec![1.0, 0.0])]);
        // L1 distance = |0.5-1.0| + |0.5-0.0| = 1.0, one info set → delta = 1.0
        let delta = strategy_delta(&prev, &curr);
        assert!((delta - 1.0).abs() < 1e-10, "expected 1.0, got {delta}");
    }

    #[test]
    fn delta_ignores_non_overlapping_keys() {
        let prev = make_map(&[(1, vec![0.5, 0.5])]);
        let curr = make_map(&[(2, vec![1.0, 0.0])]);
        assert!((strategy_delta(&prev, &curr)).abs() < 1e-10);
    }

    #[test]
    fn delta_averages_across_info_sets() {
        let prev = make_map(&[
            (1, vec![0.5, 0.5]),
            (2, vec![0.5, 0.5]),
        ]);
        let curr = make_map(&[
            (1, vec![1.0, 0.0]), // L1 = 1.0
            (2, vec![0.5, 0.5]), // L1 = 0.0
        ]);
        let delta = strategy_delta(&prev, &curr);
        assert!((delta - 0.5).abs() < 1e-10, "expected 0.5, got {delta}");
    }

    #[test]
    fn max_regret_finds_maximum() {
        let regrets = make_map(&[(1, vec![0.5, 1.0, 0.3]), (2, vec![0.2, 0.8])]);
        assert!((max_regret(&regrets) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn max_regret_handles_negatives() {
        let regrets = make_map(&[(1, vec![-5.0, 2.0])]);
        // abs(-5.0) = 5.0 > abs(2.0) = 2.0
        assert!((max_regret(&regrets) - 5.0).abs() < 1e-10);
    }

    #[test]
    fn max_regret_empty_map() {
        let regrets: FxHashMap<u64, Vec<f64>> = FxHashMap::default();
        assert!((max_regret(&regrets)).abs() < 1e-10);
    }

    #[test]
    fn avg_regret_computation() {
        let regrets = make_map(&[(1, vec![1.0, 3.0])]);
        // mean of abs values: (1.0 + 3.0) / 2 = 2.0
        assert!((avg_regret(&regrets) - 2.0).abs() < 1e-10);
    }

    #[test]
    fn avg_regret_multiple_info_sets() {
        let regrets = make_map(&[(1, vec![2.0, 4.0]), (2, vec![0.0, 6.0])]);
        // (2 + 4 + 0 + 6) / 4 = 3.0
        assert!((avg_regret(&regrets) - 3.0).abs() < 1e-10);
    }

    #[test]
    fn avg_regret_empty_map() {
        let regrets: FxHashMap<u64, Vec<f64>> = FxHashMap::default();
        assert!((avg_regret(&regrets)).abs() < 1e-10);
    }

    #[test]
    fn entropy_zero_for_pure_strategy() {
        let strategies = make_map(&[(1, vec![1.0, 0.0])]);
        // Only p=1.0 contributes: -1.0 * ln(1.0) = 0
        assert!((strategy_entropy(&strategies)).abs() < 1e-10);
    }

    #[test]
    fn entropy_max_for_uniform() {
        let strategies = make_map(&[(1, vec![0.5, 0.5])]);
        let expected = -(0.5_f64 * 0.5_f64.ln()) * 2.0; // ln(2) ≈ 0.693
        let entropy = strategy_entropy(&strategies);
        assert!(
            (entropy - expected).abs() < 1e-10,
            "expected {expected}, got {entropy}"
        );
    }

    #[test]
    fn entropy_averages_across_info_sets() {
        let pure = vec![1.0, 0.0];
        let uniform = vec![0.5, 0.5];
        let strategies = make_map(&[(1, pure), (2, uniform)]);
        let expected = (0.0 + 2.0_f64.ln()) / 2.0;
        let entropy = strategy_entropy(&strategies);
        assert!(
            (entropy - expected).abs() < 1e-10,
            "expected {expected}, got {entropy}"
        );
    }

    #[test]
    fn entropy_empty_map() {
        let strategies: FxHashMap<u64, Vec<f64>> = FxHashMap::default();
        assert!((strategy_entropy(&strategies)).abs() < 1e-10);
    }
}
