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

/// SD-CFR config tuned for Kuhn poker convergence in under 10 seconds.
///
/// Key tuning choices:
/// - Small hidden_dim (16): Kuhn has only 12 info sets, large nets overfit
/// - Moderate SGD steps (100): Enough to fit the tiny network (~914 params)
/// - Higher learning rate (0.005): Faster convergence with fewer steps
/// - 50 iterations: Sufficient for CFR convergence on a 3-card game
fn convergence_config(seed: u64) -> SdCfrConfig {
    SdCfrConfig {
        cfr_iterations: 50,
        traversals_per_iter: 100,
        advantage_memory_cap: 50_000,
        hidden_dim: 16,
        num_actions: 2,
        sgd_steps: 100,
        batch_size: 64,
        learning_rate: 0.005,
        grad_clip_norm: 10.0,
        seed,
        checkpoint_interval: 0,
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
/// This single test trains once and verifies all key properties:
/// 1. All 12 info sets are discovered
/// 2. Exploitability is significantly better than uniform random play
/// 3. Qualitative Nash equilibrium properties hold (King calls, Jack folds)
///
/// Neural function approximation adds inherent noise compared to tabular CFR.
/// The thresholds are intentionally generous — SD-CFR is designed for large games
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

    // --- 1. All 12 info sets present ---
    assert_eq!(
        strategy.len(),
        12,
        "Kuhn Poker has 12 info sets, got {}",
        strategy.len()
    );

    // --- 2. Beats uniform baseline with meaningful convergence ---
    let game = KuhnPoker::new();
    let uniform_strategy: FxHashMap<u64, Vec<f64>> = FxHashMap::default();
    let uniform_exploit = calculate_exploitability(&game, &uniform_strategy);

    println!("Uniform exploitability: {uniform_exploit:.4}");

    assert!(
        exploitability < uniform_exploit,
        "Trained strategy should beat uniform: trained={exploitability:.4}, uniform={uniform_exploit:.4}"
    );

    let reduction = 1.0 - exploitability / uniform_exploit;
    println!("Reduction: {:.1}%", reduction * 100.0);
    assert!(
        reduction > 0.3,
        "Should achieve >30% exploitability reduction, got {:.1}%",
        reduction * 100.0
    );

    // --- 3. Qualitative Nash properties ---
    // In Nash equilibrium: King always calls facing a bet, Jack always folds.
    // Kuhn action codes: check=2, bet=4. Actions at bet-facing nodes: [Fold, Call].
    let king_after_bet = InfoKey::new(2, 0, 0, &[4]).as_u64();
    let king_after_check_bet = InfoKey::new(2, 0, 0, &[2, 4]).as_u64();
    let jack_after_bet = InfoKey::new(0, 0, 0, &[4]).as_u64();
    let jack_after_check_bet = InfoKey::new(0, 0, 0, &[2, 4]).as_u64();

    if let Some(s) = strategy.get(&king_after_bet) {
        println!("K facing bet: fold={:.3}, call={:.3}", s[0], s[1]);
        assert!(
            s[1] > 0.6,
            "King should usually call a bet, got call={:.3}",
            s[1]
        );
    }

    if let Some(s) = strategy.get(&king_after_check_bet) {
        println!("K facing check-bet: fold={:.3}, call={:.3}", s[0], s[1]);
        assert!(
            s[1] > 0.6,
            "King should usually call after check-bet, got call={:.3}",
            s[1]
        );
    }

    if let Some(s) = strategy.get(&jack_after_bet) {
        println!("J facing bet: fold={:.3}, call={:.3}", s[0], s[1]);
        assert!(
            s[0] > 0.6,
            "Jack should usually fold facing a bet, got fold={:.3}",
            s[0]
        );
    }

    if let Some(s) = strategy.get(&jack_after_check_bet) {
        println!("J facing check-bet: fold={:.3}, call={:.3}", s[0], s[1]);
        assert!(
            s[0] > 0.6,
            "Jack should usually fold after check-bet, got fold={:.3}",
            s[0]
        );
    }
}
