use poker_solver_core::preflop::{CfrVariant, PreflopConfig, PreflopSolver};

#[test]
fn preflop_avg_positive_regret_positive_after_training() {
    let mut config = PreflopConfig::heads_up(10);
    config.raise_sizes = vec![vec![3.0]];
    config.raise_cap = 1;

    let mut solver = PreflopSolver::new(&config);
    solver.train(100);

    let apr = solver.avg_positive_regret();
    assert!(apr > 0.0, "avg positive regret should be positive after training, got {apr}");
    assert!(apr.is_finite(), "avg positive regret should be finite, got {apr}");
}

#[test]
fn preflop_avg_positive_regret_zero_before_training() {
    let config = PreflopConfig::heads_up(10);
    let solver = PreflopSolver::new(&config);
    assert!((solver.avg_positive_regret()).abs() < 1e-10);
}

/// Vanilla CFR: avg_positive_regret decreases over training (O(1/sqrt(T)) convergence).
/// Also verifies the metric history is plottable (finite, positive values).
#[test]
fn vanilla_avg_positive_regret_decreases_over_training() {
    let mut config = PreflopConfig::heads_up(10);
    config.raise_sizes = vec![vec![3.0]];
    config.raise_cap = 1;
    config.cfr_variant = CfrVariant::Vanilla;

    let mut solver = PreflopSolver::new(&config);

    // Collect regret at periodic checkpoints (simulating the training loop's print_every)
    let mut history: Vec<f64> = Vec::new();
    let chunk = 50;
    let total = 500;

    for _ in 0..(total / chunk) {
        solver.train(chunk);
        let apr = solver.avg_positive_regret();
        assert!(apr.is_finite(), "regret must be finite at iter {}, got {apr}", solver.iteration());
        assert!(apr >= 0.0, "regret must be non-negative at iter {}, got {apr}", solver.iteration());
        history.push(apr);
    }

    // Metric should be positive after training
    assert!(
        *history.last().unwrap() > 0.0,
        "final avg positive regret should be positive, got {}",
        history.last().unwrap()
    );

    // Metric should decrease: last value < first value
    assert!(
        history.last().unwrap() < history.first().unwrap(),
        "avg positive regret should decrease: first={:.6}, last={:.6}",
        history.first().unwrap(),
        history.last().unwrap()
    );

    // Verify the history has enough variation for a meaningful plot
    let y_max = history.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let y_min = history.iter().cloned().fold(f64::INFINITY, f64::min);
    assert!(
        y_max > y_min,
        "regret history should have variation for plotting: min={y_min}, max={y_max}"
    );
}

/// DCFR: avg_positive_regret also decreases (regression guard for existing behavior).
/// DCFR's instantaneous regret metric needs many iterations to show convergence
/// on this small tree — the metric is inherently noisy under asymmetric discounting.
///
/// Run with: `cargo test -p poker-solver-core --release --test preflop_avg_regret_test -- --ignored`
#[test]
#[ignore]
fn dcfr_avg_positive_regret_decreases_over_training() {
    let mut config = PreflopConfig::heads_up(10);
    config.raise_sizes = vec![vec![3.0]];
    config.raise_cap = 1;
    config.cfr_variant = CfrVariant::Dcfr;

    let mut solver = PreflopSolver::new(&config);

    solver.train(500);
    let early = solver.avg_positive_regret();

    solver.train(4500);
    let late = solver.avg_positive_regret();

    assert!(early > 0.0, "early regret should be positive, got {early}");
    assert!(
        late < early,
        "avg positive regret should decrease: early={early:.6}, late={late:.6}"
    );
}
