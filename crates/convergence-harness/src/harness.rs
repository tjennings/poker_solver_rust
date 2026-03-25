use std::time::Instant;

use crate::baseline::{Baseline, BaselineSummary, ConvergenceSample};
use crate::evaluator;
use crate::game::{build_flop_poker_game_with_config, FlopPokerConfig};
use crate::solver_trait::ConvergenceSolver;
use crate::solvers::exhaustive::ExhaustiveSolver;
use crate::solvers::mccfr::{compute_mccfr_exploitability, MccfrSolver};
use crate::strategy_matrix;

/// Determines how often to sample exploitability.
/// Dense early, sparse later.
fn should_sample(iteration: u64) -> bool {
    if iteration < 100 {
        true
    } else if iteration < 1000 {
        iteration.is_multiple_of(10)
    } else {
        iteration.is_multiple_of(100)
    }
}

/// Run the exhaustive solver with default config and produce a baseline.
pub fn generate_baseline(
    max_iterations: u32,
    target_exploitability: f32,
) -> Result<Baseline, Box<dyn std::error::Error>> {
    generate_baseline_with_config(
        &FlopPokerConfig::default(),
        max_iterations,
        target_exploitability,
    )
}

/// Run the exhaustive solver with a custom config and produce a baseline.
pub fn generate_baseline_with_config(
    config: &FlopPokerConfig,
    max_iterations: u32,
    target_exploitability: f32,
) -> Result<Baseline, Box<dyn std::error::Error>> {
    let game = build_flop_poker_game_with_config(config)?;
    let num_combos = game.private_cards(0).len();
    let mut solver = ExhaustiveSolver::new(game);

    let mut convergence_curve = Vec::new();
    let start = Instant::now();

    // Record initial exploitability
    let initial_expl = solver.exploitability();
    convergence_curve.push(ConvergenceSample {
        iteration: 0,
        exploitability: initial_expl as f64,
        elapsed_ms: 0,
    });
    println!("iteration: 0 (exploitability = {:.4e})", initial_expl);

    for _t in 0..max_iterations {
        solver.solve_step();
        let iter = solver.iterations();

        if should_sample(iter) || iter == max_iterations as u64 {
            let expl = solver.exploitability();
            let elapsed = start.elapsed().as_millis() as u64;

            convergence_curve.push(ConvergenceSample {
                iteration: iter,
                exploitability: expl as f64,
                elapsed_ms: elapsed,
            });

            println!(
                "iteration: {} / {} (exploitability = {:.4e}, elapsed = {:.1}s)",
                iter,
                max_iterations,
                expl,
                elapsed as f64 / 1000.0
            );

            if expl <= target_exploitability {
                println!("Target exploitability reached. Stopping.");
                break;
            }
        }
    }

    let total_time = start.elapsed().as_millis() as u64;
    let final_expl = solver.exploitability();

    println!("\nFinalizing solver (computing EVs and normalizing strategy)...");
    solver.finalize();

    let mut game = solver.into_game();
    game.back_to_root();
    strategy_matrix::print_strategy_matrix(&game, 0);

    println!("Extracting strategy and combo EVs...");
    let strategy = evaluator::extract_strategy(&mut game);
    let combo_evs = evaluator::extract_combo_evs(&mut game);

    let summary = BaselineSummary {
        solver_name: "Exhaustive DCFR".into(),
        total_iterations: convergence_curve.last().map_or(0, |s| s.iteration),
        final_exploitability: final_expl as f64,
        total_time_ms: total_time,
        num_info_sets: strategy.len(),
        num_combos_per_player: num_combos,
        game_description: format!(
            "Flop Poker: {}, {}bb effective, {}bb pot",
            config.flop, config.effective_stack, config.starting_pot
        ),
    };

    Ok(Baseline {
        summary,
        convergence_curve,
        strategy,
        combo_evs,
    })
}

/// Generate an exact DCFR baseline using the same all-in-only config that
/// `run_mccfr_solver` uses, so the comparison is apples-to-apples.
pub fn generate_mccfr_matching_baseline() -> Result<Baseline, Box<dyn std::error::Error>> {
    let config = FlopPokerConfig {
        effective_stack: 10,
        bet_sizes: "a".into(),
        raise_sizes: "a".into(),
        ..Default::default()
    };
    generate_baseline_with_config(&config, 1000, 0.001)
}

/// Run the MCCFR solver and produce a Baseline with convergence data.
///
/// Uses an all-in-only `FlopPokerConfig` because the MCCFR exploitability
/// computation requires tree correspondence between the blueprint and
/// range-solver trees, which currently only holds for all-in-only configs.
///
/// `max_iterations` is the total number of MCCFR iterations to run.
/// `checkpoints` is a sorted list of iteration counts at which to compute
/// exploitability and record convergence samples.
pub fn run_mccfr_solver(
    max_iterations: u64,
    checkpoints: &[u64],
) -> Result<Baseline, Box<dyn std::error::Error>> {
    // All-in-only config: required for tree correspondence in exploitability
    let config = FlopPokerConfig {
        effective_stack: 10,
        bet_sizes: "a".into(),
        raise_sizes: "a".into(),
        ..Default::default()
    };

    eprintln!(
        "NOTE: MCCFR exploitability currently only works with all-in-only configs \
         due to tree correspondence requirements."
    );

    let mut solver = MccfrSolver::new(config.clone());
    let mut convergence_curve = Vec::new();
    let start = Instant::now();

    // Determine which checkpoints are within our iteration budget
    let active_checkpoints: Vec<u64> = checkpoints
        .iter()
        .copied()
        .filter(|&cp| cp <= max_iterations)
        .collect();

    let mut checkpoint_idx = 0;

    while solver.iterations() < max_iterations {
        solver.solve_step();
        let current_iter = solver.iterations();

        // Check if we've reached or passed the next checkpoint
        while checkpoint_idx < active_checkpoints.len()
            && current_iter >= active_checkpoints[checkpoint_idx]
        {
            let elapsed = start.elapsed().as_millis() as u64;
            let expl = compute_mccfr_exploitability(&solver, &config)?;

            convergence_curve.push(ConvergenceSample {
                iteration: current_iter,
                exploitability: expl,
                elapsed_ms: elapsed,
            });

            eprintln!(
                "checkpoint: {} iterations, exploitability = {:.4e}, elapsed = {:.1}s",
                current_iter,
                expl,
                elapsed as f64 / 1000.0
            );

            checkpoint_idx += 1;
        }
    }

    let total_time = start.elapsed().as_millis() as u64;
    let final_expl = convergence_curve
        .last()
        .map_or(0.0, |s| s.exploitability);

    // Print strategy matrix
    let rs_config = FlopPokerConfig {
        add_allin_threshold: 100.0,
        force_allin_threshold: 0.0,
        ..config.clone()
    };
    let mut game = build_flop_poker_game_with_config(&rs_config)?;
    game.allocate_memory(false);

    // Lock the MCCFR strategy into the range-solver game for visualization
    let bp_flop_root =
        crate::solvers::mccfr::find_flop_root(solver.tree());
    let mut history: Vec<usize> = Vec::new();
    crate::solvers::mccfr::lock_strategy_recursive(
        &mut game,
        solver.tree(),
        solver.storage(),
        bp_flop_root,
        &mut history,
    );

    game.back_to_root();
    strategy_matrix::print_strategy_matrix(&game, 0);

    // Extract strategy from MCCFR solver
    let strategy = solver.average_strategy();
    let num_combos = game.private_cards(0).len();

    let summary = BaselineSummary {
        solver_name: solver.name().into(),
        total_iterations: solver.iterations(),
        final_exploitability: final_expl,
        total_time_ms: total_time,
        num_info_sets: strategy.len(),
        num_combos_per_player: num_combos,
        game_description: format!(
            "Flop Poker: {}, {}bb effective, {}bb pot (all-in only)",
            config.flop, config.effective_stack, config.starting_pot
        ),
    };

    Ok(Baseline {
        summary,
        convergence_curve,
        strategy,
        combo_evs: std::collections::BTreeMap::new(), // MCCFR doesn't produce combo EVs
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A minimal flop config for fast tests: very small stack, single bet size.
    /// This produces a small tree that solves and walks quickly.
    fn fast_test_config() -> FlopPokerConfig {
        FlopPokerConfig {
            flop: "QhJdTh".into(),
            starting_pot: 10,
            effective_stack: 10,
            bet_sizes: "a".into(),
            raise_sizes: "".into(),
            add_allin_threshold: 0.0,
            force_allin_threshold: 0.0,
        }
    }

    #[test]
    fn test_generate_baseline_produces_convergence_samples() {
        let baseline =
            generate_baseline_with_config(&fast_test_config(), 10, 100.0).unwrap();
        assert!(
            baseline.convergence_curve.len() >= 2,
            "Expected at least 2 convergence samples, got {}",
            baseline.convergence_curve.len()
        );
    }

    #[test]
    fn test_generate_baseline_has_positive_iterations() {
        let baseline =
            generate_baseline_with_config(&fast_test_config(), 10, 100.0).unwrap();
        assert!(
            baseline.summary.total_iterations > 0,
            "Expected positive iteration count"
        );
    }

    #[test]
    fn test_generate_baseline_has_nonempty_strategy() {
        let baseline =
            generate_baseline_with_config(&fast_test_config(), 10, 100.0).unwrap();
        assert!(
            !baseline.strategy.is_empty(),
            "Strategy map should not be empty after solving"
        );
    }

    #[test]
    fn test_generate_baseline_has_nonempty_combo_evs() {
        let baseline =
            generate_baseline_with_config(&fast_test_config(), 10, 100.0).unwrap();
        assert!(
            !baseline.combo_evs.is_empty(),
            "Combo EV map should not be empty after solving"
        );
    }

    #[test]
    fn test_generate_baseline_exploitability_is_positive() {
        let baseline =
            generate_baseline_with_config(&fast_test_config(), 10, 100.0).unwrap();
        assert!(
            baseline.summary.final_exploitability > 0.0,
            "Final exploitability should be positive"
        );
    }

    #[test]
    fn test_generate_baseline_first_sample_is_iteration_zero() {
        let baseline =
            generate_baseline_with_config(&fast_test_config(), 10, 100.0).unwrap();
        assert_eq!(baseline.convergence_curve[0].iteration, 0);
    }

    #[test]
    fn test_generate_baseline_with_config_uses_config() {
        let config = FlopPokerConfig {
            effective_stack: 10,
            ..fast_test_config()
        };
        let baseline = generate_baseline_with_config(&config, 10, 100.0).unwrap();
        assert!(baseline.summary.total_iterations > 0);
    }

    #[test]
    fn test_generate_baseline_stops_at_target_exploitability() {
        let baseline =
            generate_baseline_with_config(&fast_test_config(), 5, 100.0).unwrap();
        assert!(baseline.summary.total_iterations <= 5);
    }

    #[test]
    fn test_should_sample_every_iter_under_100() {
        for i in 0..100 {
            assert!(should_sample(i), "should_sample({}) should be true", i);
        }
    }

    #[test]
    fn test_should_sample_every_10_between_100_and_1000() {
        assert!(should_sample(100));
        assert!(!should_sample(101));
        assert!(should_sample(110));
        assert!(!should_sample(115));
        assert!(should_sample(990));
    }

    #[test]
    fn test_should_sample_every_100_above_1000() {
        assert!(should_sample(1000));
        assert!(!should_sample(1001));
        assert!(should_sample(1100));
        assert!(!should_sample(1050));
    }
}
