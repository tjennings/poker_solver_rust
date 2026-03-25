use crate::solver_trait::SolverMetrics;
use std::collections::BTreeMap;

/// Build `SolverMetrics` from strategy delta, pct_moving, and avg positive regret.
///
/// When `strategy_delta` is `Some`, both "strategy_delta" and "pct_moving" keys are
/// included. "avg_pos_regret" and "iterations" are always present.
pub fn build_mccfr_metrics(
    strategy_delta: Option<(f64, f64)>,
    avg_pos_regret: f64,
    iterations: u64,
) -> SolverMetrics {
    let mut values = BTreeMap::new();
    if let Some((delta, pct_moving)) = strategy_delta {
        values.insert("strategy_delta".into(), delta);
        values.insert("pct_moving".into(), pct_moving);
    }
    values.insert("avg_pos_regret".into(), avg_pos_regret);
    values.insert("iterations".into(), iterations as f64);
    SolverMetrics { values }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_mccfr_metrics_with_delta() {
        let metrics = build_mccfr_metrics(Some((0.05, 0.12)), 3.7, 50_000);

        assert_eq!(metrics.values.len(), 4);
        assert!((metrics.values["strategy_delta"] - 0.05).abs() < 1e-9);
        assert!((metrics.values["pct_moving"] - 0.12).abs() < 1e-9);
        assert!((metrics.values["avg_pos_regret"] - 3.7).abs() < 1e-9);
        assert!((metrics.values["iterations"] - 50_000.0).abs() < 1e-9);
    }

    #[test]
    fn test_build_mccfr_metrics_without_delta() {
        let metrics = build_mccfr_metrics(None, 1.5, 10_000);

        assert_eq!(metrics.values.len(), 2);
        assert!(!metrics.values.contains_key("strategy_delta"));
        assert!(!metrics.values.contains_key("pct_moving"));
        assert!((metrics.values["avg_pos_regret"] - 1.5).abs() < 1e-9);
        assert!((metrics.values["iterations"] - 10_000.0).abs() < 1e-9);
    }

    #[test]
    fn test_build_mccfr_metrics_zero_iterations() {
        let metrics = build_mccfr_metrics(None, 0.0, 0);

        assert_eq!(metrics.values.len(), 2);
        assert!((metrics.values["avg_pos_regret"]).abs() < 1e-9);
        assert!((metrics.values["iterations"]).abs() < 1e-9);
    }

    #[test]
    fn test_build_mccfr_metrics_large_iteration_count() {
        let metrics = build_mccfr_metrics(Some((0.001, 0.99)), 0.01, 10_000_000);

        assert!((metrics.values["iterations"] - 10_000_000.0).abs() < 1e-9);
    }
}
