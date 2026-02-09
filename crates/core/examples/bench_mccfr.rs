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
        bet_sizes: vec![0.3, 0.5, 1.0, 1.5],
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
    let game = HunlPostflop::new(config, Some(AbstractionMode::HandClass), deal_count);
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
        use_cfr_plus: true,
        discount_iterations: Some(0),
        skip_first_iterations: Some(0),
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
    let total = t0.elapsed();
    println!();
    println!("Info sets:  {}", strategies.len());
    println!("Total time: {:?}", total);
    println!(
        "Per iter:   {:?}",
        train_time / iterations as u32,
    );
    println!(
        "Per sample: {:?}",
        train_time / (iterations as u32 * samples as u32),
    );
}
