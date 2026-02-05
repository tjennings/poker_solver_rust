//! Integration tests for the full blueprint system.
//!
//! Tests the complete workflow: train -> extract -> save -> load -> use with `SubgameSolver`.
//! Uses minimal config values to keep tests fast (under 1 second each).

#![cfg(feature = "gpu")]
#![allow(clippy::doc_markdown)]

use std::sync::Arc;

use poker_solver_core::blueprint::{BlueprintStrategy, CacheConfig, SubgameConfig, SubgameSolver};
use poker_solver_core::cfr::GpuCfrSolver;
use poker_solver_core::game::{HunlPostflop, PostflopConfig};

use burn::backend::ndarray::NdArray;

type TestBackend = NdArray;

/// Test the full blueprint pipeline: train -> extract strategies -> create blueprint -> use with SubgameSolver.
///
/// Verifies that:
/// - Training completes successfully
/// - Strategies can be extracted
/// - `BlueprintStrategy` can be created from extracted strategies
/// - `SubgameSolver` can be constructed with the blueprint
#[test]
fn full_blueprint_pipeline() {
    // Create minimal config for fast testing
    let config = PostflopConfig {
        stack_depth: 10,
        bet_sizes: vec![1.0],
        samples_per_iteration: 2,
    };

    // Create and train the game
    let game = HunlPostflop::new(config, None);
    let device = Default::default();
    let mut solver = GpuCfrSolver::<TestBackend>::new(&game, device);

    solver.train(10);

    // Extract strategies from solver
    let strategies = solver.all_strategies();

    // Create blueprint from strategies
    let blueprint = BlueprintStrategy::from_strategies(strategies, solver.iterations());

    // Verify blueprint has content
    assert!(
        blueprint.len() > 0,
        "Blueprint should contain at least one strategy, got {}",
        blueprint.len()
    );

    // Verify iteration count matches
    assert_eq!(
        blueprint.iterations_trained(),
        10,
        "Blueprint should record 10 iterations, got {}",
        blueprint.iterations_trained()
    );

    // Create SubgameSolver with the blueprint
    let arc_blueprint = Arc::new(blueprint);
    let subgame_config = SubgameConfig::default();
    let cache_config = CacheConfig::default();

    let result = SubgameSolver::new(arc_blueprint, None, subgame_config, cache_config);

    assert!(
        result.is_ok(),
        "SubgameSolver construction should succeed, got error: {:?}",
        result.err()
    );

    let solver = result.expect("should construct subgame solver");
    assert_eq!(
        solver.config().depth_limit,
        4,
        "SubgameSolver should have default depth_limit"
    );

    println!("Full blueprint pipeline test passed");
}

/// Test blueprint save and load functionality.
///
/// Verifies that:
/// - Blueprint can be saved to a file
/// - Blueprint can be loaded from a file
/// - Loaded blueprint matches the original
#[test]
fn blueprint_save_and_load() {
    // Create minimal config for fast testing
    let config = PostflopConfig {
        stack_depth: 10,
        bet_sizes: vec![1.0],
        samples_per_iteration: 2,
    };

    // Create and train the game
    let game = HunlPostflop::new(config, None);
    let device = Default::default();
    let mut solver = GpuCfrSolver::<TestBackend>::new(&game, device);

    solver.train(5);

    // Extract strategies and create blueprint
    let strategies = solver.all_strategies();
    let original = BlueprintStrategy::from_strategies(strategies, solver.iterations());

    // Create temp directory for save/load test
    let temp_dir = tempfile::tempdir().expect("should create temp directory");
    let blueprint_path = temp_dir.path().join("blueprint.bin");

    // Save the blueprint
    original
        .save(&blueprint_path)
        .expect("blueprint save should succeed");

    // Load the blueprint
    let loaded = BlueprintStrategy::load(&blueprint_path).expect("blueprint load should succeed");

    // Verify loaded matches original
    assert_eq!(
        loaded.iterations_trained(),
        original.iterations_trained(),
        "Loaded iterations_trained should match original: loaded={}, original={}",
        loaded.iterations_trained(),
        original.iterations_trained()
    );

    assert_eq!(
        loaded.len(),
        original.len(),
        "Loaded len should match original: loaded={}, original={}",
        loaded.len(),
        original.len()
    );

    println!(
        "Blueprint save/load test passed with {} strategies",
        loaded.len()
    );
}
