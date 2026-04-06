/// Prediction accuracy metrics for CFVnet evaluation.
pub struct PredictionMetrics {
    /// Mean absolute error (pot-relative).
    pub mae: f64,
    /// Maximum absolute error (pot-relative).
    pub max_error: f64,
    /// Mean absolute error in milli-big-blinds per hand.
    pub mbb_error: f64,
}

/// Mean absolute error over valid (unmasked) entries.
///
/// Returns 0.0 if no entries are valid.
pub fn mean_absolute_error(pred: &[f32], actual: &[f32], mask: &[bool]) -> f64 {
    debug_assert_eq!(pred.len(), actual.len());
    debug_assert_eq!(pred.len(), mask.len());

    let mut sum = 0.0_f64;
    let mut count = 0_u64;

    for i in 0..pred.len() {
        if mask[i] {
            sum += f64::from((pred[i] - actual[i]).abs());
            count += 1;
        }
    }

    if count == 0 {
        0.0
    } else {
        sum / count as f64
    }
}

/// Maximum absolute error over valid (unmasked) entries.
///
/// Returns 0.0 if no entries are valid.
pub fn max_absolute_error(pred: &[f32], actual: &[f32], mask: &[bool]) -> f64 {
    debug_assert_eq!(pred.len(), actual.len());
    debug_assert_eq!(pred.len(), mask.len());

    let mut max = 0.0_f64;

    for i in 0..pred.len() {
        if mask[i] {
            let err = f64::from((pred[i] - actual[i]).abs());
            if err > max {
                max = err;
            }
        }
    }

    max
}

/// Convert a pot-relative error to milli-big-blinds.
///
/// Formula: `error * pot / big_blind * 1000.0`
pub fn pot_relative_to_mbb(error: f64, pot: f64, big_blind: f64) -> f64 {
    error * pot / big_blind * 1000.0
}

/// Compute all prediction metrics for a single evaluation sample.
///
/// Uses `big_blind = 2.0` (standard HUNL: SB=1, BB=2).
pub fn compute_prediction_metrics(
    pred: &[f32],
    actual: &[f32],
    mask: &[bool],
    pot: f32,
) -> PredictionMetrics {
    let mae = mean_absolute_error(pred, actual, mask);
    let max_error = max_absolute_error(pred, actual, mask);
    let big_blind = 2.0;
    let mbb_error = pot_relative_to_mbb(mae, f64::from(pot), big_blind);

    PredictionMetrics {
        mae,
        max_error,
        mbb_error,
    }
}

/// Compute mean absolute error on normalized (stake-relative) values.
///
/// Only counts entries where mask is true. Returns 0.0 if no valid entries.
pub fn compute_normalized_mae(pred: &[f32], target: &[f32], mask: &[bool]) -> f64 {
    let mut sum = 0.0_f64;
    let mut count = 0u64;
    for i in 0..pred.len().min(target.len()) {
        if mask.get(i).copied().unwrap_or(false) {
            sum += (pred[i] as f64 - target[i] as f64).abs();
            count += 1;
        }
    }
    if count == 0 { 0.0 } else { sum / count as f64 }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mae_zero_on_perfect() {
        let values = vec![0.1_f32, 0.5, -0.3, 0.0];
        let mask = vec![true; 4];
        let mae = mean_absolute_error(&values, &values, &mask);
        assert!((mae).abs() < 1e-10, "expected ~0, got {mae}");
    }

    #[test]
    fn mae_ignores_masked() {
        let pred = vec![0.0_f32, 0.0, 0.0];
        let actual = vec![0.0_f32, 0.0, 1000.0];
        let mask = vec![true, true, false]; // third entry masked out
        let mae = mean_absolute_error(&pred, &actual, &mask);
        assert!((mae).abs() < 1e-10, "expected ~0, got {mae}");
    }

    #[test]
    fn max_error_correct() {
        let pred = vec![0.0_f32, 0.0, 0.0, 0.0];
        let actual = vec![0.1_f32, 0.5, 0.3, 0.05];
        let mask = vec![true; 4];
        let max = max_absolute_error(&pred, &actual, &mask);
        assert!((max - 0.5).abs() < 1e-6, "expected 0.5, got {max}");
    }

    #[test]
    fn mbb_conversion() {
        // 0.01 pot-relative error × pot 100 / BB 2 × 1000 = 500 mbb
        let mbb = pot_relative_to_mbb(0.01, 100.0, 2.0);
        assert!((mbb - 500.0).abs() < 1e-10, "expected 500, got {mbb}");
    }

    #[test]
    fn compute_metrics_with_perfect_returns_zero() {
        let n = 1326;
        let values: Vec<f32> = (0..n).map(|i| (i as f32) * 0.001 - 0.5).collect();
        let mask = vec![true; n];
        let pot = 100.0_f32;

        let metrics = compute_prediction_metrics(&values, &values, &mask, pot);

        assert!(metrics.mae.abs() < 1e-10, "mae should be ~0");
        assert!(metrics.max_error.abs() < 1e-10, "max_error should be ~0");
        assert!(metrics.mbb_error.abs() < 1e-10, "mbb_error should be ~0");
    }

    #[test]
    fn all_masked_returns_zero() {
        let pred = vec![1.0_f32; 10];
        let actual = vec![0.0_f32; 10];
        let mask = vec![false; 10];

        let metrics = compute_prediction_metrics(&pred, &actual, &mask, 100.0);

        assert!(metrics.mae.abs() < 1e-10);
        assert!(metrics.max_error.abs() < 1e-10);
        assert!(metrics.mbb_error.abs() < 1e-10);
    }

    #[test]
    fn normalized_mae_basic() {
        let pred = vec![0.1, 0.2, 0.0];
        let target = vec![0.12, 0.18, 0.0];
        let mask = vec![true, true, false];
        let mae = compute_normalized_mae(&pred, &target, &mask);
        assert!((mae - 0.02).abs() < 1e-6);
    }
}
