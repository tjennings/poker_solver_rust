//! Integration tests: SD-CFR on Kuhn Poker converges toward Nash equilibrium.
//!
//! These tests verify the full pipeline: traversal → training → evaluation.
//! They use a richer encoder that includes action history, enabling the network
//! to distinguish all 12 Kuhn info sets.

use candle_core::Device;
use rustc_hash::FxHashMap;

use poker_solver_core::cfr::calculate_exploitability;
use poker_solver_core::game::{Game, KuhnPoker, Player};
use poker_solver_core::info_key::InfoKey;

use poker_solver_deep_cfr::card_features::{BET_FEATURES, InfoSetFeatures};
use poker_solver_deep_cfr::config::SdCfrConfig;
use poker_solver_deep_cfr::eval::ExplicitPolicy;
use poker_solver_deep_cfr::solver::SdCfrSolver;
use poker_solver_deep_cfr::traverse::StateEncoder;

// ---------------------------------------------------------------------------
// Encoder: card + action history
// ---------------------------------------------------------------------------

/// Encodes Kuhn states with both card value and action history.
///
/// Unlike the unit-test encoder (card only), this gives the network enough
/// information to learn distinct strategies at each of the 12 info sets.
struct KuhnFullEncoder {
    game: KuhnPoker,
}

impl KuhnFullEncoder {
    fn new() -> Self {
        Self {
            game: KuhnPoker::new(),
        }
    }
}

impl StateEncoder<<KuhnPoker as Game>::State> for KuhnFullEncoder {
    fn encode(&self, state: &<KuhnPoker as Game>::State, _player: Player) -> InfoSetFeatures {
        let key = self.game.info_set_key(state);
        let info = InfoKey::from_raw(key);
        let card_value = info.hand_bits() as i8;

        let mut cards = [-1i8; 7];
        cards[0] = card_value;

        // Encode action codes from the info set key into bet features.
        // Raw codes give the network clear position signal (check=2, bet=4).
        let mut bets = [0.0f32; BET_FEATURES];
        let actions_bits = info.actions_bits();
        for (i, slot) in bets.iter_mut().enumerate().take(6) {
            let code = ((actions_bits >> (20 - i * 4)) & 0xF) as u8;
            if code == 0 {
                break;
            }
            *slot = f32::from(code);
        }

        InfoSetFeatures { cards, bets }
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// SD-CFR config tuned for Kuhn poker convergence in reasonable time.
///
/// Key tuning choices:
/// - Small hidden_dim (16): Kuhn has only 12 info sets, large nets overfit
/// - High SGD steps (500): Network trains from scratch each iteration
/// - Relaxed gradient clipping (10.0): Small game, aggressive clipping hinders learning
/// - Many iterations (300): CFR convergence requires iteration count
fn convergence_config(seed: u64) -> SdCfrConfig {
    SdCfrConfig {
        cfr_iterations: 300,
        traversals_per_iter: 300,
        advantage_memory_cap: 50_000,
        hidden_dim: 16,
        num_actions: 2,
        sgd_steps: 500,
        batch_size: 128,
        learning_rate: 0.003,
        grad_clip_norm: 10.0,
        seed,
    }
}

/// Walk the game tree and extract the weighted-average strategy for every info set.
///
/// Uses `ExplicitPolicy` (one per player) to compute the SD-CFR average strategy,
/// then packs results into the `FxHashMap<u64, Vec<f64>>` format expected by
/// `calculate_exploitability`.
fn extract_sdcfr_strategy(
    game: &KuhnPoker,
    encoder: &KuhnFullEncoder,
    p1_policy: &ExplicitPolicy,
    p2_policy: &ExplicitPolicy,
) -> FxHashMap<u64, Vec<f64>> {
    let mut strategy_map = FxHashMap::default();
    for state in &game.initial_states() {
        walk_tree(
            game,
            state,
            encoder,
            p1_policy,
            p2_policy,
            &mut strategy_map,
        );
    }
    strategy_map
}

/// Recursively visit every non-terminal game state, computing the average
/// strategy at each info set via the appropriate player's `ExplicitPolicy`.
fn walk_tree(
    game: &KuhnPoker,
    state: &<KuhnPoker as Game>::State,
    encoder: &KuhnFullEncoder,
    p1_policy: &ExplicitPolicy,
    p2_policy: &ExplicitPolicy,
    strategy_map: &mut FxHashMap<u64, Vec<f64>>,
) {
    if game.is_terminal(state) {
        return;
    }

    let player = game.player(state);
    let key = game.info_set_key(state);

    strategy_map.entry(key).or_insert_with(|| {
        let features = encoder.encode(state, player);
        let policy = match player {
            Player::Player1 => p1_policy,
            Player::Player2 => p2_policy,
        };
        let probs = policy.strategy(&features).unwrap();
        let actions = game.actions(state);
        probs
            .iter()
            .take(actions.len())
            .map(|&p| f64::from(p))
            .collect()
    });

    for &action in game.actions(state).as_slice() {
        let next = game.next_state(state, action);
        walk_tree(game, &next, encoder, p1_policy, p2_policy, strategy_map);
    }
}

/// Train SD-CFR on Kuhn Poker and return exploitability.
fn train_and_measure(seed: u64) -> (f64, FxHashMap<u64, Vec<f64>>) {
    let game = KuhnPoker::new();
    let encoder = KuhnFullEncoder::new();
    let config = convergence_config(seed);
    let mut solver = SdCfrSolver::new(game.clone(), encoder, config).unwrap();

    let trained = solver.train().unwrap();

    let device = Device::Cpu;
    let p1_policy = ExplicitPolicy::from_buffer(&trained.model_buffers[0], 2, 16, &device).unwrap();
    let p2_policy = ExplicitPolicy::from_buffer(&trained.model_buffers[1], 2, 16, &device).unwrap();

    let eval_encoder = KuhnFullEncoder::new();
    let strategy = extract_sdcfr_strategy(&game, &eval_encoder, &p1_policy, &p2_policy);
    let exploitability = calculate_exploitability(&game, &strategy);

    (exploitability, strategy)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

/// SD-CFR should converge toward Nash equilibrium on Kuhn Poker.
///
/// Neural function approximation adds inherent noise compared to tabular CFR.
/// The threshold is intentionally generous — SD-CFR is designed for large games
/// where tabular methods are infeasible; on Kuhn it's a proof-of-concept.
#[test]
fn sd_cfr_kuhn_converges() {
    let (exploitability, strategy) = train_and_measure(42);

    println!("SD-CFR Kuhn exploitability: {exploitability:.4}");
    println!("Strategy map ({} info sets):", strategy.len());
    for (key, strat) in &strategy {
        let info = InfoKey::from_raw(*key);
        println!(
            "  card={} actions_bits={:#08x} → [{:.3}, {:.3}]",
            info.hand_bits(),
            info.actions_bits(),
            strat[0],
            strat[1]
        );
    }

    // Neural SD-CFR won't match tabular precision on toy games.
    // Uniform play has exploitability ~0.5; getting below 0.25 shows
    // meaningful convergence toward Nash.
    assert!(
        exploitability < 0.25,
        "SD-CFR should converge on Kuhn Poker, got exploitability={exploitability:.4}"
    );

    // Verify all 12 info sets are present
    assert_eq!(
        strategy.len(),
        12,
        "Kuhn Poker has 12 info sets, got {}",
        strategy.len()
    );
}

/// Same seed produces consistent training structure.
///
/// NOTE: Candle's weight initialization uses a global thread-local RNG that
/// cannot be seeded from application code, so bit-exact determinism is not
/// achievable. Instead we verify structural consistency: same number of
/// model entries and similar exploitability.
#[test]
fn sd_cfr_kuhn_consistent_structure() {
    let game = KuhnPoker::new();
    let config = SdCfrConfig {
        cfr_iterations: 10,
        traversals_per_iter: 50,
        advantage_memory_cap: 10_000,
        hidden_dim: 16,
        num_actions: 2,
        sgd_steps: 20,
        batch_size: 64,
        learning_rate: 0.001,
        grad_clip_norm: 1.0,
        seed: 123,
    };

    let run = || {
        let encoder = KuhnFullEncoder::new();
        let mut solver = SdCfrSolver::new(game.clone(), encoder, config.clone()).unwrap();
        let trained = solver.train().unwrap();
        let p1_len = trained.model_buffers[0].len();
        let p2_len = trained.model_buffers[1].len();

        let device = Device::Cpu;
        let p1 = ExplicitPolicy::from_buffer(&trained.model_buffers[0], 2, 16, &device).unwrap();
        let p2 = ExplicitPolicy::from_buffer(&trained.model_buffers[1], 2, 16, &device).unwrap();

        let eval_encoder = KuhnFullEncoder::new();
        let strategy = extract_sdcfr_strategy(&game, &eval_encoder, &p1, &p2);
        let exploitability = calculate_exploitability(&game, &strategy);
        (p1_len, p2_len, strategy.len(), exploitability)
    };

    let (p1a, p2a, info_sets_a, e1) = run();
    let (p1b, p2b, info_sets_b, e2) = run();

    // Structural properties must be identical
    assert_eq!(p1a, p1b, "P1 model buffer size should be consistent");
    assert_eq!(p2a, p2b, "P2 model buffer size should be consistent");
    assert_eq!(
        p1a, config.cfr_iterations as usize,
        "Should have one model entry per iteration"
    );
    assert_eq!(info_sets_a, info_sets_b, "Same number of info sets");
    assert_eq!(info_sets_a, 12, "Kuhn Poker has 12 info sets");

    // Exploitability should be in a similar range (not bit-exact due to
    // candle's non-deterministic weight initialization)
    println!("Run 1 exploitability: {e1:.4}");
    println!("Run 2 exploitability: {e2:.4}");
    assert!(
        (e1 - e2).abs() < 0.15,
        "Exploitability should be similar between runs: {e1:.4} vs {e2:.4}"
    );
}

/// Training should produce a strategy significantly better than uniform play.
///
/// Neural SD-CFR convergence is not monotonic — exploitability can fluctuate
/// between checkpoints. Instead we verify that the final trained strategy
/// substantially outperforms the uniform (random) baseline.
#[test]
fn sd_cfr_kuhn_beats_uniform_baseline() {
    let game = KuhnPoker::new();

    // Compute uniform strategy exploitability (empty map → uniform fallback)
    let uniform_strategy: FxHashMap<u64, Vec<f64>> = FxHashMap::default();
    let uniform_exploit = calculate_exploitability(&game, &uniform_strategy);

    // Train SD-CFR
    let (trained_exploit, _) = train_and_measure(99);

    println!("Uniform exploitability: {uniform_exploit:.4}");
    println!("Trained exploitability: {trained_exploit:.4}");

    assert!(
        trained_exploit < uniform_exploit,
        "Trained strategy should beat uniform: trained={trained_exploit:.4}, uniform={uniform_exploit:.4}"
    );

    // Should achieve at least 30% reduction from uniform
    let reduction = 1.0 - trained_exploit / uniform_exploit;
    println!("Reduction: {:.1}%", reduction * 100.0);
    assert!(
        reduction > 0.3,
        "Should achieve >30% exploitability reduction, got {:.1}%",
        reduction * 100.0
    );
}

/// The trained strategy should exhibit known Nash equilibrium properties.
///
/// In Nash equilibrium for Kuhn Poker:
/// - King always calls when facing a bet (never folds)
/// - Jack always folds when facing a bet (never calls)
#[test]
fn sd_cfr_kuhn_qualitative_strategy() {
    let (exploitability, strategy) = train_and_measure(42);

    println!("Exploitability: {exploitability:.4}");

    // Kuhn action codes: check=2, bet=4
    let king_after_bet = InfoKey::new(2, 0, 0, &[4]).as_u64(); // K facing bet
    let king_after_check_bet = InfoKey::new(2, 0, 0, &[2, 4]).as_u64(); // K facing check-bet
    let jack_after_bet = InfoKey::new(0, 0, 0, &[4]).as_u64(); // J facing bet
    let jack_after_check_bet = InfoKey::new(0, 0, 0, &[2, 4]).as_u64(); // J facing check-bet

    // King should strongly prefer calling when facing a bet
    // Actions at bet-facing nodes: [Fold, Call]
    if let Some(s) = strategy.get(&king_after_bet) {
        println!("K facing bet: fold={:.3}, call={:.3}", s[0], s[1]);
        assert!(
            s[1] > 0.7,
            "King should usually call a bet, got call={:.3}",
            s[1]
        );
    }

    if let Some(s) = strategy.get(&king_after_check_bet) {
        println!("K facing check-bet: fold={:.3}, call={:.3}", s[0], s[1]);
        assert!(
            s[1] > 0.7,
            "King should usually call after check-bet, got call={:.3}",
            s[1]
        );
    }

    // Jack should strongly prefer folding when facing a bet
    if let Some(s) = strategy.get(&jack_after_bet) {
        println!("J facing bet: fold={:.3}, call={:.3}", s[0], s[1]);
        assert!(
            s[0] > 0.7,
            "Jack should usually fold facing a bet, got fold={:.3}",
            s[0]
        );
    }

    if let Some(s) = strategy.get(&jack_after_check_bet) {
        println!("J facing check-bet: fold={:.3}, call={:.3}", s[0], s[1]);
        assert!(
            s[0] > 0.7,
            "Jack should usually fold after check-bet, got fold={:.3}",
            s[0]
        );
    }
}
