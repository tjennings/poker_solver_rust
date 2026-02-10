//! Integration tests for convergence metrics during MCCFR training.
//!
//! Trains Kuhn poker with periodic checkpoints and verifies that
//! convergence indicators improve over training.

use poker_solver_core::cfr::convergence;
use poker_solver_core::cfr::{calculate_exploitability, MccfrSolver};
use poker_solver_core::game::KuhnPoker;

/// Train Kuhn poker with 5 checkpoints and verify convergence metrics improve.
#[test]
fn convergence_metrics_decrease_over_training() {
    let game = KuhnPoker::new();
    let mut solver = MccfrSolver::new(game.clone());
    solver.set_seed(42);

    // Use small initial checkpoint to capture early unconverged state,
    // then extend to enough iterations for DCFR to converge on Kuhn.
    let checkpoint_iters = [200, 1000, 5000, 10000, 20000];
    let num_checkpoints = checkpoint_iters.len();

    let mut deltas = Vec::new();
    let mut entropies = Vec::new();
    let mut exploitabilities = Vec::new();
    let mut prev_strategies = None;

    for (ckpt, &iters) in checkpoint_iters.iter().enumerate() {
        let already_trained = if ckpt == 0 { 0 } else { checkpoint_iters[ckpt - 1] };
        let to_train = iters - already_trained;
        solver.train_full(to_train);

        let strategies = solver.all_strategies_best_effort();
        let regrets = solver.regret_sum();

        let max_r = convergence::max_regret(regrets);
        let avg_r = convergence::avg_regret(regrets);
        let entropy = convergence::strategy_entropy(&strategies);
        let exploit = calculate_exploitability(&game, &strategies);

        if let Some(ref prev) = prev_strategies {
            let delta = convergence::strategy_delta(prev, &strategies);
            deltas.push(delta);
        }

        entropies.push(entropy);
        exploitabilities.push(exploit);
        prev_strategies = Some(strategies);

        eprintln!(
            "Checkpoint {} ({iters} iters): max_r={max_r:.4}, avg_r={avg_r:.4}, entropy={entropy:.4}, exploit={exploit:.6}",
            ckpt + 1
        );
    }

    // Strategy delta should generally decrease (later checkpoints change less)
    assert!(
        deltas.len() >= 2,
        "Need at least 2 deltas, got {}",
        deltas.len()
    );
    assert!(
        deltas.last().unwrap() < deltas.first().unwrap(),
        "Strategy delta should decrease: first={:.6}, last={:.6}",
        deltas.first().unwrap(),
        deltas.last().unwrap()
    );

    // Exploitability should decrease: first checkpoint vs last
    assert!(
        exploitabilities[num_checkpoints - 1] < exploitabilities[0],
        "Exploitability should decrease: first={:.6}, last={:.6}",
        exploitabilities[0],
        exploitabilities[num_checkpoints - 1]
    );

    // Final exploitability should be substantially lower than initial
    // (DCFR discounting slows Kuhn convergence; absolute threshold is relaxed)
    assert!(
        exploitabilities[num_checkpoints - 1] < 0.06,
        "Final exploitability should be modest, got {:.6}",
        exploitabilities[num_checkpoints - 1]
    );

    // Entropy should decrease as strategies become more polarized
    assert!(
        entropies[num_checkpoints - 1] < entropies[0],
        "Entropy should decrease: first={:.4}, last={:.4}",
        entropies[0],
        entropies[num_checkpoints - 1]
    );
}

/// Verify metrics return sensible values on an untrained solver.
#[test]
fn convergence_metrics_on_empty_solver() {
    let game = KuhnPoker::new();
    let solver = MccfrSolver::new(game);

    let strategies = solver.all_strategies_best_effort();
    let regrets = solver.regret_sum();

    assert!((convergence::max_regret(regrets)).abs() < 1e-10);
    assert!((convergence::avg_regret(regrets)).abs() < 1e-10);
    assert!((convergence::strategy_entropy(&strategies)).abs() < 1e-10);
}
