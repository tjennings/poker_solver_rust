//! Integration tests for SD-CFR with HUNL (Heads-Up No-Limit) postflop game.
//!
//! Verifies the end-to-end training pipeline: create a small HUNL game,
//! generate a deal pool, train SD-CFR for a few iterations, extract strategies
//! via `ExplicitPolicy`, and confirm they are valid probability distributions.

use candle_core::Device;

use poker_solver_core::game::{Game, HunlPostflop, Player, PostflopConfig};
use poker_solver_deep_cfr::config::SdCfrConfig;
use poker_solver_deep_cfr::eval::ExplicitPolicy;
use poker_solver_deep_cfr::hunl_encoder::HunlStateEncoder;
use poker_solver_deep_cfr::solver::{SdCfrSolver, TrainedSdCfr};
use poker_solver_deep_cfr::traverse::StateEncoder;
use poker_solver_deep_cfr::SdCfrError;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Small game config for fast integration tests.
fn small_game_config() -> PostflopConfig {
    PostflopConfig {
        stack_depth: 10,
        bet_sizes: vec![0.5, 1.0],
        max_raises_per_street: 2,
    }
}

/// SD-CFR config with small parameters for speed.
fn small_sdcfr_config(cfr_iterations: u32, checkpoint_interval: u32) -> SdCfrConfig {
    SdCfrConfig {
        cfr_iterations,
        traversals_per_iter: 50,
        advantage_memory_cap: 10_000,
        hidden_dim: 32,
        num_actions: 5, // fold, check, call, bet(0), bet(1)
        sgd_steps: 20,
        batch_size: 64,
        learning_rate: 0.001,
        grad_clip_norm: 1.0,
        seed: 42,
        checkpoint_interval,
        parallel_traversals: 1,
    }
}

/// Assert that a probability vector forms a valid distribution:
/// all entries non-negative and summing to approximately 1.0.
fn assert_valid_distribution(probs: &[f32], label: &str) {
    let sum: f32 = probs.iter().sum();
    assert!(
        (sum - 1.0).abs() < 1e-4,
        "{label}: probabilities should sum to ~1.0, got {sum} (probs={probs:?})"
    );
    for (i, &p) in probs.iter().enumerate() {
        assert!(
            p >= -1e-6,
            "{label}: action {i} has negative probability {p}"
        );
    }
}

// ---------------------------------------------------------------------------
// Test 1: Train and extract strategy
// ---------------------------------------------------------------------------

#[test]
fn sdcfr_hunl_train_and_extract_strategy() {
    // -- Setup --
    let game_config = small_game_config();
    let game = HunlPostflop::new(game_config.clone(), None, 100);
    let deal_pool = game.initial_states();
    let encoder = HunlStateEncoder::new(game_config.bet_sizes.clone());
    let sdcfr_config = small_sdcfr_config(5, 0);

    // -- Train --
    let mut solver = SdCfrSolver::new(game, encoder, sdcfr_config)
        .expect("SdCfrSolver::new should succeed with valid config");
    let trained = solver
        .train_with_deals(Some(&deal_pool), None)
        .expect("training should complete without error");

    // -- Verify model buffer sizes --
    assert_eq!(
        trained.model_buffers[0].len(),
        5,
        "P1 model buffer should have 5 entries (one per iteration)"
    );
    assert_eq!(
        trained.model_buffers[1].len(),
        5,
        "P2 model buffer should have 5 entries (one per iteration)"
    );

    // -- Build ExplicitPolicy for P1 --
    let device = Device::Cpu;
    let policy = ExplicitPolicy::from_buffer(
        &trained.model_buffers[0],
        5,  // num_actions
        32, // hidden_dim
        &device,
    )
    .expect("ExplicitPolicy::from_buffer should succeed");

    // -- Query strategy for a deal --
    let encoder2 = HunlStateEncoder::new(game_config.bet_sizes.clone());
    let game2 = HunlPostflop::new(game_config, None, 1);
    let states = game2.initial_states();
    assert!(
        !states.is_empty(),
        "game should produce at least one initial state"
    );

    let features = encoder2.encode(&states[0], Player::Player1);
    let probs = policy
        .strategy(&features)
        .expect("strategy query should succeed");

    // -- Verify valid probability distribution --
    assert_eq!(
        probs.len(),
        5,
        "strategy should have 5 entries (num_actions=5)"
    );
    assert_valid_distribution(&probs, "P1 root strategy");
}

// ---------------------------------------------------------------------------
// Test 2: Checkpoint callback fires at the correct iterations
// ---------------------------------------------------------------------------

#[test]
fn sdcfr_hunl_checkpoint_callback_fires() {
    // -- Setup with checkpoint_interval=2, 4 iterations --
    let game_config = small_game_config();
    let game = HunlPostflop::new(game_config.clone(), None, 50);
    let deal_pool = game.initial_states();
    let encoder = HunlStateEncoder::new(game_config.bet_sizes.clone());
    let sdcfr_config = small_sdcfr_config(4, 2);

    let mut solver = SdCfrSolver::new(game, encoder, sdcfr_config)
        .expect("SdCfrSolver::new should succeed");

    // -- Train with checkpoint callback --
    let mut call_count = 0u32;
    let mut seen_iterations: Vec<u32> = Vec::new();

    let trained = solver
        .train_with_deals(
            Some(&deal_pool),
            Some(&mut |iter: u32, snapshot: &TrainedSdCfr| -> Result<(), SdCfrError> {
                call_count += 1;
                seen_iterations.push(iter);

                // Verify snapshot has correct number of model entries at checkpoint time
                assert_eq!(
                    snapshot.model_buffers[0].len(),
                    iter as usize,
                    "snapshot at iteration {iter} should have {iter} P1 model entries"
                );
                assert_eq!(
                    snapshot.model_buffers[1].len(),
                    iter as usize,
                    "snapshot at iteration {iter} should have {iter} P2 model entries"
                );
                Ok(())
            }),
        )
        .expect("training with checkpoint callback should succeed");

    // -- Verify callback invocations --
    assert_eq!(
        call_count, 2,
        "checkpoint callback should fire exactly twice (at iterations 2 and 4)"
    );
    assert_eq!(
        seen_iterations,
        vec![2, 4],
        "callback should fire at iterations 2 and 4"
    );

    // -- Verify final model buffers --
    assert_eq!(trained.model_buffers[0].len(), 4);
    assert_eq!(trained.model_buffers[1].len(), 4);
}

// ---------------------------------------------------------------------------
// Test 3: Both players have valid strategies
// ---------------------------------------------------------------------------

#[test]
fn sdcfr_hunl_both_players_have_strategies() {
    // -- Setup --
    let game_config = small_game_config();
    let game = HunlPostflop::new(game_config.clone(), None, 50);
    let deal_pool = game.initial_states();
    let encoder = HunlStateEncoder::new(game_config.bet_sizes.clone());
    let sdcfr_config = small_sdcfr_config(3, 0);

    // -- Train --
    let mut solver = SdCfrSolver::new(game, encoder, sdcfr_config)
        .expect("SdCfrSolver::new should succeed");
    let trained = solver
        .train_with_deals(Some(&deal_pool), None)
        .expect("training should complete without error");

    // -- Build ExplicitPolicy for both players --
    let device = Device::Cpu;
    let p1_policy = ExplicitPolicy::from_buffer(
        &trained.model_buffers[0],
        5,
        32,
        &device,
    )
    .expect("P1 ExplicitPolicy should load");

    let p2_policy = ExplicitPolicy::from_buffer(
        &trained.model_buffers[1],
        5,
        32,
        &device,
    )
    .expect("P2 ExplicitPolicy should load");

    // -- Query strategies from both player perspectives --
    let encoder2 = HunlStateEncoder::new(game_config.bet_sizes.clone());
    let game2 = HunlPostflop::new(game_config, None, 1);
    let states = game2.initial_states();
    assert!(
        !states.is_empty(),
        "game should produce at least one initial state"
    );

    let p1_features = encoder2.encode(&states[0], Player::Player1);
    let p1_probs = p1_policy
        .strategy(&p1_features)
        .expect("P1 strategy query should succeed");
    assert_eq!(p1_probs.len(), 5);
    assert_valid_distribution(&p1_probs, "P1 strategy");

    let p2_features = encoder2.encode(&states[0], Player::Player2);
    let p2_probs = p2_policy
        .strategy(&p2_features)
        .expect("P2 strategy query should succeed");
    assert_eq!(p2_probs.len(), 5);
    assert_valid_distribution(&p2_probs, "P2 strategy");
}
