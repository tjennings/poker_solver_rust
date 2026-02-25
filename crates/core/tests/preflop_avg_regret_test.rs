use poker_solver_core::preflop::{PreflopConfig, PreflopSolver};

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
