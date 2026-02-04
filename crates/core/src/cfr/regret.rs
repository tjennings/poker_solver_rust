/// Converts regret values to a strategy using regret matching.
///
/// If all regrets are non-positive, returns a uniform distribution.
/// Otherwise, normalizes positive regrets to sum to 1.
#[must_use]
pub fn regret_match(regrets: &[f64]) -> Vec<f64> {
    let positive_sum: f64 = regrets.iter().filter(|&&r| r > 0.0).sum();

    if positive_sum > 0.0 {
        regrets
            .iter()
            .map(|&r| if r > 0.0 { r / positive_sum } else { 0.0 })
            .collect()
    } else {
        #[allow(clippy::cast_precision_loss)]
        let uniform = 1.0 / regrets.len() as f64;
        vec![uniform; regrets.len()]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn positive_regrets_normalized() {
        let regrets = vec![1.0, 2.0, 3.0];
        let strategy = regret_match(&regrets);

        assert_eq!(strategy.len(), 3);
        assert!((strategy[0] - 1.0 / 6.0).abs() < 1e-10);
        assert!((strategy[1] - 2.0 / 6.0).abs() < 1e-10);
        assert!((strategy[2] - 3.0 / 6.0).abs() < 1e-10);

        let sum: f64 = strategy.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10);
    }

    #[test]
    fn mixed_regrets_ignore_negative() {
        let regrets = vec![-1.0, 2.0, 4.0];
        let strategy = regret_match(&regrets);

        assert!((strategy[0] - 0.0).abs() < 1e-10);
        assert!((strategy[1] - 2.0 / 6.0).abs() < 1e-10);
        assert!((strategy[2] - 4.0 / 6.0).abs() < 1e-10);
    }

    #[test]
    fn all_non_positive_returns_uniform() {
        let regrets = vec![-1.0, -2.0, 0.0];
        let strategy = regret_match(&regrets);

        let expected = 1.0 / 3.0;
        for &p in &strategy {
            assert!((p - expected).abs() < 1e-10);
        }
    }

    #[test]
    fn all_zero_returns_uniform() {
        let regrets = vec![0.0, 0.0];
        let strategy = regret_match(&regrets);

        assert!((strategy[0] - 0.5).abs() < 1e-10);
        assert!((strategy[1] - 0.5).abs() < 1e-10);
    }

    #[test]
    fn single_action_returns_one() {
        let regrets = vec![5.0];
        let strategy = regret_match(&regrets);

        assert_eq!(strategy.len(), 1);
        assert!((strategy[0] - 1.0).abs() < 1e-10);
    }
}
