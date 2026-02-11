//! Micro-benchmark for MCCFR training hot-path.
//!
//! Runs a small training loop with hand_class abstraction and prints timing.
//!
//! Usage:
//!   cargo run --release --example bench_mccfr          # quick timing
//!   cargo run --release --example bench_mccfr -- long   # longer for profiling
//!   cargo flamegraph --release --example bench_mccfr    # flamegraph

use std::time::Instant;

use poker_solver_core::Game;
use poker_solver_core::cfr::{MccfrConfig, MccfrSolver};
use poker_solver_core::game::{AbstractionMode, HunlPostflop, PostflopConfig};

fn main() {
    let long_mode = std::env::args().any(|a| a == "long");

    let config = PostflopConfig {
        stack_depth: 100,
        bet_sizes: vec![0.33, 0.67, 1.0, 2.0, 3.0],
        max_raises_per_street: 3,
    };

    let (iterations, samples, deal_count) = if long_mode {
        (10u64, 500usize, 5000usize)
    } else {
        (3, 100, 1000)
    };

    println!("=== MCCFR Micro-Benchmark{} ===", if long_mode { " (long)" } else { "" });
    println!(
        "  Config: {} iters x {} samples, {} deals, stack_depth={}",
        iterations, samples, deal_count, config.stack_depth
    );
    println!("  Bet sizes: {:?}", config.bet_sizes);
    println!();

    // Phase 1: Deal generation
    let t0 = Instant::now();
    let game = HunlPostflop::new(config, Some(AbstractionMode::HandClassV2 { strength_bits: 0, equity_bits: 0 }), deal_count);
    let deal_time = t0.elapsed();
    println!("[deal generation]  {:?}", deal_time);

    // Phase 2: initial_states() — generates the deal pool
    let t1 = Instant::now();
    let _states = game.initial_states();
    let init_time = t1.elapsed();
    println!("[initial_states]   {:?}  ({} deals)", init_time, _states.len());

    // Phase 3: MCCFR solver creation
    let mccfr_config = MccfrConfig {
        samples_per_iteration: samples,
        ..MccfrConfig::default()
    };

    let t2 = Instant::now();
    let mut solver = MccfrSolver::with_config(game, &mccfr_config);
    solver.set_seed(42);
    let create_time = t2.elapsed();
    println!("[solver creation]  {:?}", create_time);

    // Phase 4: Training (the hot path)
    let t3 = Instant::now();
    solver.train(iterations, samples);
    let train_time = t3.elapsed();
    println!("[training]         {:?}", train_time);

    // Summary
    let strategies = solver.all_strategies();
    println!();
    println!("Info sets:  {}", strategies.len());
    println!(
        "Per iter:   {:?}",
        train_time / iterations as u32,
    );
    println!(
        "Per sample: {:?}",
        train_time / (iterations as u32 * samples as u32),
    );

    // Phase 5: Parallel training comparison
    println!();
    println!("=== Parallel MCCFR ===");

    let par_config = PostflopConfig {
        stack_depth: 100,
        bet_sizes: vec![0.33, 0.67, 1.0, 2.0, 3.0],
        max_raises_per_street: 3,
    };
    let par_game = HunlPostflop::new(par_config, Some(AbstractionMode::HandClassV2 { strength_bits: 0, equity_bits: 0 }), deal_count);
    let mut par_solver = MccfrSolver::with_config(par_game, &mccfr_config);
    par_solver.set_seed(42);

    let t4 = Instant::now();
    par_solver.train_parallel(iterations, samples);
    let par_time = t4.elapsed();
    println!("[parallel training] {:?}", par_time);

    let par_strategies = par_solver.all_strategies();
    println!("Info sets:  {}", par_strategies.len());
    println!(
        "Per iter:   {:?}",
        par_time / iterations as u32,
    );
    println!(
        "Per sample: {:?}",
        par_time / (iterations as u32 * samples as u32),
    );

    // Phase 6: Parallel + Pruning comparison
    println!();
    println!("=== Parallel + Pruning MCCFR ===");

    let prune_game_config = PostflopConfig {
        stack_depth: 100,
        bet_sizes: vec![0.33, 0.67, 1.0, 2.0, 3.0],
        max_raises_per_street: 3,
    };
    let prune_game =
        HunlPostflop::new(prune_game_config, Some(AbstractionMode::HandClassV2 { strength_bits: 0, equity_bits: 0 }), deal_count);
    let prune_mccfr_config = MccfrConfig {
        samples_per_iteration: samples,
        pruning: true,
        pruning_warmup: 0, // no warmup for benchmark (short run)
        pruning_probe_interval: 20,
        ..MccfrConfig::default()
    };
    let mut prune_solver = MccfrSolver::with_config(prune_game, &prune_mccfr_config);
    prune_solver.set_seed(42);

    let t5 = Instant::now();
    prune_solver.train_parallel(iterations, samples);
    let prune_time = t5.elapsed();
    println!("[parallel+pruning]  {:?}", prune_time);

    let prune_strategies = prune_solver.all_strategies();
    println!("Info sets:  {}", prune_strategies.len());
    let (pruned, total) = prune_solver.pruning_stats();
    if total > 0 {
        let skip_pct = 100.0 * pruned as f64 / total as f64;
        println!("Pruned:     {pruned}/{total} ({skip_pct:.1}% skip rate)");
    }

    println!();
    println!("=== Comparison ===");
    println!("Sequential:       {:?}", train_time);
    println!("Parallel:         {:?}", par_time);
    println!("Parallel+Pruning: {:?}", prune_time);
    let par_speedup = train_time.as_secs_f64() / par_time.as_secs_f64();
    let prune_speedup = train_time.as_secs_f64() / prune_time.as_secs_f64();
    println!("Parallel speedup:         {:.2}x", par_speedup);
    println!("Parallel+Pruning speedup: {:.2}x", prune_speedup);
    println!("Total time: {:?}", t0.elapsed());
}
