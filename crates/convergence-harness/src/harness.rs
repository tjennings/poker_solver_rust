use std::time::Instant;

use crate::baseline::{Baseline, BaselineSummary, ConvergenceSample};
use crate::evaluator;
use crate::game::{build_flop_poker_game_with_config, FlopPokerConfig};
use crate::solver_trait::ConvergenceSolver;
use crate::solvers::exhaustive::ExhaustiveSolver;
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
