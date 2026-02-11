//! Benchmark comparing all three CFR solver backends.
//!
//! Runs identical training configs on MCCFR, Sequence-Form, and GPU solvers,
//! then compares timing and strategy agreement.
//!
//! Usage:
//!   cargo run --release -p poker-solver-gpu-cfr --example bench_solvers
//!   cargo run --release -p poker-solver-gpu-cfr --example bench_solvers -- long

use std::time::Instant;

use poker_solver_core::cfr::convergence;
use poker_solver_core::cfr::game_tree::materialize_postflop;
use poker_solver_core::cfr::{DealInfo, MccfrConfig, MccfrSolver, SequenceCfrConfig, SequenceCfrSolver};
use poker_solver_core::game::{AbstractionMode, HunlPostflop, PostflopConfig};
use poker_solver_core::info_key::InfoKey;
use poker_solver_core::Game;
use poker_solver_gpu_cfr::{GpuCfrConfig, GpuCfrSolver};
use rustc_hash::FxHashMap;

fn build_deal_infos(game: &HunlPostflop) -> Vec<DealInfo> {
    let states = game.initial_states();
    states
        .iter()
        .map(|state| {
            let key_p1 = game.info_set_key(state);
            let hand_bits_p1 = InfoKey::from_raw(key_p1).hand_bits();

            let first_action = game.actions(state)[0];
            let next_state = game.next_state(state, first_action);
            let key_p2 = game.info_set_key(&next_state);
            let hand_bits_p2 = InfoKey::from_raw(key_p2).hand_bits();

            let p1_wins = match (&state.p1_cache.rank, &state.p2_cache.rank) {
                (Some(r1), Some(r2)) => {
                    use std::cmp::Ordering;
                    match r1.cmp(r2) {
                        Ordering::Greater => Some(true),
                        Ordering::Less => Some(false),
                        Ordering::Equal => None,
                    }
                }
                _ => None,
            };

            DealInfo {
                hand_bits_p1,
                hand_bits_p2,
                p1_wins,
            }
        })
        .collect()
}

struct BenchResult {
    name: String,
    setup_ms: f64,
    train_ms: f64,
    info_sets: usize,
    per_iter_ms: f64,
    strategies: FxHashMap<u64, Vec<f64>>,
}

fn bench_mccfr_sequential(
    config: &PostflopConfig,
    iterations: u64,
    samples: usize,
    deal_count: usize,
) -> BenchResult {
    let t0 = Instant::now();
    let game = HunlPostflop::new(config.clone(), Some(AbstractionMode::HandClass), deal_count);
    let mccfr_config = MccfrConfig {
        samples_per_iteration: samples,
        ..MccfrConfig::default()
    };
    let mut solver = MccfrSolver::with_config(game, &mccfr_config);
    solver.set_seed(42);
    let setup_ms = t0.elapsed().as_secs_f64() * 1000.0;

    let t1 = Instant::now();
    solver.train(iterations, samples);
    let train_ms = t1.elapsed().as_secs_f64() * 1000.0;

    let strategies = solver.all_strategies();
    BenchResult {
        name: "MCCFR (sequential)".to_string(),
        setup_ms,
        train_ms,
        info_sets: strategies.len(),
        per_iter_ms: train_ms / iterations as f64,
        strategies,
    }
}

fn bench_mccfr_parallel(
    config: &PostflopConfig,
    iterations: u64,
    samples: usize,
    deal_count: usize,
) -> BenchResult {
    let t0 = Instant::now();
    let game = HunlPostflop::new(config.clone(), Some(AbstractionMode::HandClass), deal_count);
    let mccfr_config = MccfrConfig {
        samples_per_iteration: samples,
        ..MccfrConfig::default()
    };
    let mut solver = MccfrSolver::with_config(game, &mccfr_config);
    solver.set_seed(42);
    let setup_ms = t0.elapsed().as_secs_f64() * 1000.0;

    let t1 = Instant::now();
    solver.train_parallel(iterations, samples);
    let train_ms = t1.elapsed().as_secs_f64() * 1000.0;

    let strategies = solver.all_strategies();
    BenchResult {
        name: format!("MCCFR (parallel, {} threads)", rayon::current_num_threads()),
        setup_ms,
        train_ms,
        info_sets: strategies.len(),
        per_iter_ms: train_ms / iterations as f64,
        strategies,
    }
}

fn bench_sequence(
    config: &PostflopConfig,
    iterations: u64,
    deal_count: usize,
) -> BenchResult {
    let t0 = Instant::now();
    let game = HunlPostflop::new(config.clone(), Some(AbstractionMode::HandClass), deal_count);
    let states = game.initial_states();
    let tree = materialize_postflop(&game, &states[0]);
    let deals = build_deal_infos(&game);
    let seq_config = SequenceCfrConfig {
        dcfr_alpha: 1.5,
        dcfr_beta: 0.5,
        dcfr_gamma: 2.0,
    };
    let mut solver = SequenceCfrSolver::new(tree, deals, seq_config);
    let setup_ms = t0.elapsed().as_secs_f64() * 1000.0;

    let t1 = Instant::now();
    solver.train(iterations);
    let train_ms = t1.elapsed().as_secs_f64() * 1000.0;

    let strategies = solver.all_strategies();
    BenchResult {
        name: "Sequence-form (CPU)".to_string(),
        setup_ms,
        train_ms,
        info_sets: strategies.len(),
        per_iter_ms: train_ms / iterations as f64,
        strategies,
    }
}

fn bench_gpu(
    config: &PostflopConfig,
    iterations: u64,
    deal_count: usize,
) -> Option<BenchResult> {
    let t0 = Instant::now();
    let game = HunlPostflop::new(config.clone(), Some(AbstractionMode::HandClass), deal_count);
    let states = game.initial_states();
    let tree = materialize_postflop(&game, &states[0]);
    let deals = build_deal_infos(&game);
    let gpu_config = GpuCfrConfig::default();

    let solver = match GpuCfrSolver::new(&tree, deals, gpu_config) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("GPU solver init failed: {e}");
            return None;
        }
    };
    let setup_ms = t0.elapsed().as_secs_f64() * 1000.0;

    let mut solver = solver;
    let t1 = Instant::now();
    solver.train(iterations);
    let train_ms = t1.elapsed().as_secs_f64() * 1000.0;

    let strategies = solver.all_strategies();
    Some(BenchResult {
        name: "GPU (wgpu)".to_string(),
        setup_ms,
        train_ms,
        info_sets: strategies.len(),
        per_iter_ms: train_ms / iterations as f64,
        strategies,
    })
}

fn print_result(result: &BenchResult) {
    println!(
        "  {:<30} setup={:>8.1}ms  train={:>8.1}ms  per_iter={:>8.2}ms  info_sets={}",
        result.name, result.setup_ms, result.train_ms, result.per_iter_ms, result.info_sets
    );
}

fn print_comparison(results: &[&BenchResult]) {
    if results.len() < 2 {
        return;
    }

    let baseline = &results[0];
    println!("\n  Speedup vs {}:", baseline.name);
    for r in &results[1..] {
        let speedup = baseline.train_ms / r.train_ms;
        println!("    {:<30} {:.2}x", r.name, speedup);
    }
}

fn print_strategy_agreement(a: &BenchResult, b: &BenchResult) {
    let delta = convergence::strategy_delta(&a.strategies, &b.strategies);
    let common_keys = a
        .strategies
        .keys()
        .filter(|k| b.strategies.contains_key(k))
        .count();
    println!(
        "  {} vs {}: delta={:.6}, common_keys={}",
        a.name, b.name, delta, common_keys
    );
}

fn main() {
    let long_mode = std::env::args().any(|a| a == "long");

    let config = PostflopConfig {
        stack_depth: 25,
        bet_sizes: vec![0.5, 1.0],
        max_raises_per_street: 2,
    };

    let (iterations, samples, deal_count) = if long_mode {
        (100u64, 500usize, 5000usize)
    } else {
        (20, 100, 1000)
    };

    println!(
        "=== Solver Benchmark{} ===",
        if long_mode { " (long)" } else { "" }
    );
    println!(
        "  Config: {} BB, bet_sizes={:?}, max_raises={}",
        config.stack_depth, config.bet_sizes, config.max_raises_per_street
    );
    println!(
        "  MCCFR: {} iters x {} samples, {} deals",
        iterations, samples, deal_count
    );
    println!(
        "  Sequence/GPU: {} iters (full traversal), {} deals",
        iterations, deal_count
    );
    println!();

    // Run benchmarks
    println!("--- MCCFR Sequential ---");
    let mccfr_seq = bench_mccfr_sequential(&config, iterations, samples, deal_count);
    print_result(&mccfr_seq);

    println!("\n--- MCCFR Parallel ---");
    let mccfr_par = bench_mccfr_parallel(&config, iterations, samples, deal_count);
    print_result(&mccfr_par);

    println!("\n--- Sequence-Form CPU ---");
    let seq = bench_sequence(&config, iterations, deal_count);
    print_result(&seq);

    println!("\n--- GPU ---");
    let gpu = bench_gpu(&config, iterations, deal_count);
    if let Some(ref gpu_result) = gpu {
        print_result(gpu_result);
    } else {
        println!("  (skipped, no GPU available)");
    }

    // Comparison
    println!("\n=== Timing Comparison ===");
    let mut all_results: Vec<&BenchResult> = vec![&mccfr_seq, &mccfr_par, &seq];
    if let Some(ref g) = gpu {
        all_results.push(g);
    }
    for r in &all_results {
        print_result(r);
    }
    print_comparison(&all_results);

    // Strategy agreement between full-traversal solvers
    println!("\n=== Strategy Agreement ===");
    if let Some(ref g) = gpu {
        print_strategy_agreement(&seq, g);
    }
    // MCCFR vs sequence (expect higher delta due to sampling)
    print_strategy_agreement(&mccfr_par, &seq);

    println!("\n=== Done ===");
}
