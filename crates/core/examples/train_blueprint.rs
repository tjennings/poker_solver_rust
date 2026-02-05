//! Train a small blueprint strategy for testing.
//!
//! Run with: cargo run -p poker-solver-core --example train_blueprint --features gpu

use std::sync::Arc;
use std::time::Instant;

use burn::backend::ndarray::NdArray;
use poker_solver_core::blueprint::{BlueprintStrategy, CacheConfig, SubgameConfig, SubgameSolver};
use poker_solver_core::cfr::GpuCfrSolver;
use poker_solver_core::game::{HunlPostflop, PostflopConfig};

type Backend = NdArray;

fn main() {
    println!("=== Blueprint Training Test ===\n");

    // Minimal config for quick testing
    let config = PostflopConfig {
        stack_depth: 10,              // 10 BB stacks (short stack = smaller tree)
        bet_sizes: vec![1.0],         // Just pot-sized bets
        samples_per_iteration: 5,     // Only 5 deals per iteration (keeps info set count small)
    };

    println!("Game config:");
    println!("  Stack depth: {} BB", config.stack_depth);
    println!("  Bet sizes: {:?} pot", config.bet_sizes);
    println!("  Samples/iteration: {}", config.samples_per_iteration);
    println!();

    // Create game
    let game = HunlPostflop::new(config, None);
    let device = Default::default();

    // Create solver
    println!("Creating GPU solver...");
    let start = Instant::now();
    let mut solver = GpuCfrSolver::<Backend>::new(&game, device);
    println!("  Created in {:?}", start.elapsed());
    println!();

    // Train
    let iterations = 500;
    println!("Training for {} iterations...", iterations);
    let start = Instant::now();

    solver.train_with_callback(iterations, |iter| {
        if iter % 200 == 0 {
            println!("  Iteration {}", iter);
        }
    });

    let train_time = start.elapsed();
    println!("  Completed in {:?}", train_time);
    println!("  Rate: {:.1} iter/sec", iterations as f64 / train_time.as_secs_f64());
    println!();

    // Extract blueprint
    println!("Extracting blueprint strategy...");
    let strategies = solver.all_strategies();
    let blueprint = BlueprintStrategy::from_strategies(strategies, solver.iterations());

    println!("  Info sets: {}", blueprint.len());
    println!("  Iterations trained: {}", blueprint.iterations_trained());
    println!();

    // Save blueprint
    let path = std::path::Path::new("test_blueprint.bin");
    println!("Saving blueprint to {:?}...", path);
    blueprint.save(path).expect("Failed to save blueprint");

    let file_size = std::fs::metadata(path).map(|m| m.len()).unwrap_or(0);
    println!("  File size: {} KB", file_size / 1024);
    println!();

    // Load and verify
    println!("Loading blueprint...");
    let loaded = BlueprintStrategy::load(path).expect("Failed to load blueprint");
    assert_eq!(loaded.len(), blueprint.len());
    assert_eq!(loaded.iterations_trained(), blueprint.iterations_trained());
    println!("  Verified: {} info sets, {} iterations", loaded.len(), loaded.iterations_trained());
    println!();

    // Create subgame solver
    println!("Creating subgame solver...");
    let subgame_solver = SubgameSolver::new(
        Arc::new(loaded),
        None,
        SubgameConfig::default(),
        CacheConfig::default(),
    ).expect("Failed to create subgame solver");
    println!("  Created successfully");
    println!();

    // Show some sample strategies
    println!("Sample strategies (first 5 info sets):");
    let strategies = solver.all_strategies();
    for (i, (key, probs)) in strategies.iter().take(5).enumerate() {
        let probs_str: Vec<String> = probs.iter().map(|p| format!("{:.2}", p)).collect();
        println!("  {}: {} -> [{}]", i + 1, key, probs_str.join(", "));
    }
    println!();

    // Cleanup
    std::fs::remove_file(path).ok();

    println!("=== Test Complete ===");
    drop(subgame_solver); // Silence unused warning
}
