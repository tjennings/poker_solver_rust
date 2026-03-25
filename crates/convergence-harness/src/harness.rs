use std::time::Instant;

use crate::baseline::{Baseline, BaselineSummary, ConvergenceSample};
use crate::config::ConvergenceConfig;
use crate::evaluator;
use crate::game::{build_flop_poker_game_with_config, FlopPokerConfig};
use crate::solver_trait::ConvergenceSolver;
use crate::solvers::exhaustive::ExhaustiveSolver;
use crate::solvers::mccfr::{
    compute_head_to_head_ev, flop_str_to_core_cards, MccfrSolver,
};
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

/// Run the exhaustive solver with a custom config and produce a baseline.
///
/// If `checkpoints` is `Some`, exploitability is sampled at those iterations only.
/// If `None`, uses the default dense-early/sparse-later schedule.
pub fn generate_baseline_with_config(
    config: &FlopPokerConfig,
    max_iterations: u32,
    target_exploitability: f32,
) -> Result<Baseline, Box<dyn std::error::Error>> {
    generate_baseline_with_config_and_checkpoints(config, max_iterations, target_exploitability, None)
}

/// Run the exhaustive solver with explicit checkpoint iterations.
pub fn generate_baseline_with_config_and_checkpoints(
    config: &FlopPokerConfig,
    max_iterations: u32,
    target_exploitability: f32,
    checkpoints: Option<&[u64]>,
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

    let mut checkpoint_idx = 0;

    for _t in 0..max_iterations {
        solver.solve_step();
        let iter = solver.iterations();

        let should_record = match checkpoints {
            Some(cps) => {
                let mut hit = false;
                while checkpoint_idx < cps.len() && iter >= cps[checkpoint_idx] {
                    hit = true;
                    checkpoint_idx += 1;
                }
                hit
            }
            None => should_sample(iter) || iter == max_iterations as u64,
        };

        if should_record {
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

/// Generate an exact baseline from a `ConvergenceConfig`, solving each flop
/// independently and combining the results into a single `Baseline`.
///
/// Each flop is solved with the exhaustive DCFR solver using the baseline
/// settings from the config. Node IDs are prefixed with the flop index
/// to ensure uniqueness across flops.
pub fn generate_baseline_from_config(
    cfg: &ConvergenceConfig,
) -> Result<Baseline, Box<dyn std::error::Error>> {
    let start = Instant::now();
    let mut combined_strategy = crate::solver_trait::StrategyMap::new();
    let mut combined_combo_evs = crate::solver_trait::ComboEvMap::new();
    let mut total_info_sets = 0;
    let mut total_num_combos = 0;
    let mut worst_exploitability = 0.0_f64;
    let mut combined_convergence = Vec::new();
    let checkpoints: Vec<u64> = cfg.mccfr.checkpoints.clone();

    for (flop_idx, flop_str) in cfg.game.flops.iter().enumerate() {
        println!("\n--- Solving flop {}/{}: {} ---", flop_idx + 1, cfg.game.flops.len(), flop_str);
        let flop_config = FlopPokerConfig::from_game_def(&cfg.game, flop_str);

        let baseline = generate_baseline_with_config_and_checkpoints(
            &flop_config,
            cfg.baseline.max_iterations,
            cfg.baseline.target_exploitability,
            Some(&checkpoints),
        )?;

        // Merge strategies with flop-prefixed node IDs
        let offset = (flop_idx as u64) << 48;
        for (nid, strat) in &baseline.strategy {
            combined_strategy.insert(offset | nid, strat.clone());
        }
        for (nid, evs) in &baseline.combo_evs {
            combined_combo_evs.insert(offset | nid, evs.clone());
        }

        total_info_sets += baseline.summary.num_info_sets;
        total_num_combos = baseline.summary.num_combos_per_player; // same across flops
        worst_exploitability = worst_exploitability.max(baseline.summary.final_exploitability);

        // Use convergence from the first flop as representative
        if flop_idx == 0 {
            combined_convergence = baseline.convergence_curve;
        }
    }

    let total_time = start.elapsed().as_millis() as u64;
    let flop_list: String = cfg.game.flops.join(", ");

    let summary = BaselineSummary {
        solver_name: "Exhaustive DCFR (multi-flop)".into(),
        total_iterations: combined_convergence.last().map_or(0, |s| s.iteration),
        final_exploitability: worst_exploitability,
        total_time_ms: total_time,
        num_info_sets: total_info_sets,
        num_combos_per_player: total_num_combos,
        game_description: format!(
            "Flop Poker: [{}], {}bb effective, {}bb pot",
            flop_list, cfg.game.effective_stack, cfg.game.starting_pot
        ),
    };

    Ok(Baseline {
        summary,
        convergence_curve: combined_convergence,
        strategy: combined_strategy,
        combo_evs: combined_combo_evs,
    })
}

/// Compute multi-flop head-to-head EV loss: average mbb/hand across all flops.
///
/// For each flop, builds a per-flop baseline, computes h2h EV, and averages.
fn compute_multi_flop_h2h(
    solver: &MccfrSolver,
    per_flop_baselines: &[Baseline],
    configs: &[FlopPokerConfig],
) -> Result<(f64, f64, f64), String> {
    let mut total_oop = 0.0;
    let mut total_ip = 0.0;
    let n = configs.len();

    for (flop_idx, (config, baseline)) in configs.iter().zip(per_flop_baselines.iter()).enumerate() {
        let (oop, ip, _avg) = compute_head_to_head_ev(solver, baseline, config, flop_idx)?;
        total_oop += oop;
        total_ip += ip;
    }

    let avg_oop = total_oop / n as f64;
    let avg_ip = total_ip / n as f64;
    let avg = (avg_oop + avg_ip) / 2.0;
    Ok((avg_oop, avg_ip, avg))
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
/// `baseline` is an optional exact baseline; when provided, head-to-head
/// mbb/hand loss is computed and printed at each checkpoint.
pub fn run_mccfr_solver(
    max_iterations: u64,
    checkpoints: &[u64],
    baseline: Option<&Baseline>,
    turn_buckets: u16,
    river_buckets: u16,
) -> Result<Baseline, Box<dyn std::error::Error>> {
    // All-in-only config: required for tree correspondence in exploitability
    let config = FlopPokerConfig {
        effective_stack: 10,
        bet_sizes: "a".into(),
        raise_sizes: "a".into(),
        ..Default::default()
    };

    let mut solver = MccfrSolver::new(config.clone(), turn_buckets, river_buckets);
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

            if let Some(bl) = baseline {
                match compute_head_to_head_ev(&solver, bl, &config, 0) {
                    Ok((oop, ip, avg)) => {
                        convergence_curve.push(ConvergenceSample {
                            iteration: current_iter,
                            exploitability: avg, // h2h mbb/hand as primary metric
                            elapsed_ms: elapsed,
                        });
                        eprintln!(
                            "checkpoint: {} iterations, h2h mbb/hand = {:.2} (OOP {:.2}, IP {:.2}), elapsed = {:.1}s",
                            current_iter, avg, oop, ip,
                            elapsed as f64 / 1000.0
                        );
                    }
                    Err(e) => {
                        convergence_curve.push(ConvergenceSample {
                            iteration: current_iter,
                            exploitability: 0.0,
                            elapsed_ms: elapsed,
                        });
                        eprintln!(
                            "checkpoint: {} iterations, h2h error: {}, elapsed = {:.1}s",
                            current_iter, e, elapsed as f64 / 1000.0
                        );
                    }
                }
            } else {
                convergence_curve.push(ConvergenceSample {
                    iteration: current_iter,
                    exploitability: 0.0,
                    elapsed_ms: elapsed,
                });
                eprintln!(
                    "checkpoint: {} iterations, elapsed = {:.1}s",
                    current_iter, elapsed as f64 / 1000.0
                );
            }

            checkpoint_idx += 1;
        }
    }

    let total_time = start.elapsed().as_millis() as u64;
    let final_h2h = convergence_curve
        .last()
        .map_or(0.0, |s| s.exploitability);

    // Print strategy matrix
    let mut game = build_flop_poker_game_with_config(&config)?;
    game.allocate_memory(false);

    // Lock the MCCFR strategy into the range-solver game for visualization
    let bp_flop_root =
        crate::solvers::mccfr::find_flop_root(solver.tree());
    let mut history: Vec<usize> = Vec::new();
    let mut board_cards = (None, None);
    crate::solvers::mccfr::lock_strategy_recursive(
        &mut game,
        solver.tree(),
        solver.storage(),
        Some(solver.per_flop_buckets_for(0)),
        bp_flop_root,
        &mut history,
        &mut board_cards,
    );

    game.back_to_root();
    strategy_matrix::print_strategy_matrix(&game, 0);

    // Extract strategy from MCCFR solver
    let strategy = solver.average_strategy();
    let num_combos = game.private_cards(0).len();

    let summary = BaselineSummary {
        solver_name: solver.name().into(),
        total_iterations: solver.iterations(),
        final_exploitability: final_h2h,
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

/// Run the MCCFR solver from a `ConvergenceConfig` with multi-flop support.
///
/// Clusters each flop, trains a single blueprint across all flops, and
/// computes head-to-head mbb/hand loss against per-flop baselines at
/// each checkpoint.
pub fn run_mccfr_solver_from_config(
    cfg: &ConvergenceConfig,
    baseline: Option<&Baseline>,
) -> Result<Baseline, Box<dyn std::error::Error>> {
    let flops: Vec<[poker_solver_core::poker::Card; 3]> = cfg
        .game
        .flops
        .iter()
        .map(|s| flop_str_to_core_cards(s))
        .collect();

    let configs: Vec<FlopPokerConfig> = cfg
        .game
        .flops
        .iter()
        .map(|s| FlopPokerConfig::from_game_def(&cfg.game, s))
        .collect();

    // Use the first flop's config to build the MCCFR solver (tree structure
    // is identical for all flops since bet sizes are the same).
    let mut solver = MccfrSolver::new_multi_flop(
        configs[0].clone(),
        &flops,
        cfg.mccfr.buckets.turn,
        cfg.mccfr.buckets.river,
    );

    let max_iterations = cfg.mccfr.iterations;
    let checkpoints = &cfg.mccfr.checkpoints;

    // Build per-flop baselines for h2h computation (if a combined baseline was provided).
    // We solve each flop independently to get per-flop baselines.
    let per_flop_baselines: Option<Vec<Baseline>> = if baseline.is_some() {
        let mut baselines = Vec::new();
        for (flop_idx, flop_config) in configs.iter().enumerate() {
            eprintln!("Building per-flop baseline for flop {} ({})...", flop_idx, cfg.game.flops[flop_idx]);
            let bl = generate_baseline_with_config_and_checkpoints(
                flop_config,
                cfg.baseline.max_iterations,
                cfg.baseline.target_exploitability,
                Some(checkpoints),
            )?;
            baselines.push(bl);
        }
        Some(baselines)
    } else {
        None
    };

    let mut convergence_curve = Vec::new();
    let start = Instant::now();

    let active_checkpoints: Vec<u64> = checkpoints
        .iter()
        .copied()
        .filter(|&cp| cp <= max_iterations)
        .collect();

    let mut checkpoint_idx = 0;

    while solver.iterations() < max_iterations {
        solver.solve_step();
        let current_iter = solver.iterations();

        while checkpoint_idx < active_checkpoints.len()
            && current_iter >= active_checkpoints[checkpoint_idx]
        {
            let elapsed = start.elapsed().as_millis() as u64;

            if let Some(ref baselines) = per_flop_baselines {
                match compute_multi_flop_h2h(&solver, baselines, &configs) {
                    Ok((oop, ip, avg)) => {
                        convergence_curve.push(ConvergenceSample {
                            iteration: current_iter,
                            exploitability: avg,
                            elapsed_ms: elapsed,
                        });
                        eprintln!(
                            "checkpoint: {} iterations, h2h mbb/hand = {:.2} (OOP {:.2}, IP {:.2}), elapsed = {:.1}s",
                            current_iter, avg, oop, ip,
                            elapsed as f64 / 1000.0
                        );
                    }
                    Err(e) => {
                        convergence_curve.push(ConvergenceSample {
                            iteration: current_iter,
                            exploitability: 0.0,
                            elapsed_ms: elapsed,
                        });
                        eprintln!(
                            "checkpoint: {} iterations, h2h error: {}, elapsed = {:.1}s",
                            current_iter, e, elapsed as f64 / 1000.0
                        );
                    }
                }
            } else {
                convergence_curve.push(ConvergenceSample {
                    iteration: current_iter,
                    exploitability: 0.0,
                    elapsed_ms: elapsed,
                });
                eprintln!(
                    "checkpoint: {} iterations, elapsed = {:.1}s",
                    current_iter, elapsed as f64 / 1000.0
                );
            }

            checkpoint_idx += 1;
        }
    }

    let total_time = start.elapsed().as_millis() as u64;
    let final_h2h = convergence_curve
        .last()
        .map_or(0.0, |s| s.exploitability);

    // Extract strategy
    let strategy = solver.average_strategy();

    // Use the first flop to get num_combos (same for all flops with identical stack/pot)
    let rs_game = build_flop_poker_game_with_config(&configs[0])?;
    let num_combos = rs_game.private_cards(0).len();

    let flop_list: String = cfg.game.flops.join(", ");
    let summary = BaselineSummary {
        solver_name: solver.name().into(),
        total_iterations: solver.iterations(),
        final_exploitability: final_h2h,
        total_time_ms: total_time,
        num_info_sets: strategy.len(),
        num_combos_per_player: num_combos,
        game_description: format!(
            "Flop Poker: [{}], {}bb effective, {}bb pot",
            flop_list, cfg.game.effective_stack, cfg.game.starting_pot
        ),
    };

    Ok(Baseline {
        summary,
        convergence_curve,
        strategy,
        combo_evs: std::collections::BTreeMap::new(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

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
