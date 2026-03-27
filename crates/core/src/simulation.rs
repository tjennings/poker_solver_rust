//! Simulation module for running agent-vs-agent poker competitions.
//!
//! Uses `rs_poker::arena` to play complete poker hands between two strategy
//! sources (trained bundles or rule-based agents) and report performance
//! in mbb/h (milli-big-blinds per hand).

use std::cell::RefCell;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use rs_poker::arena::action::AgentAction;
use rs_poker::arena::agent::{Agent, AgentGenerator};
use rs_poker::arena::competition::{HoldemCompetition, StandardSimulationIterator};
use rs_poker::arena::game_state::{GameState, Round};
use rs_poker::core::{Card, Hand, Suit, Value};

thread_local! {
    /// Shared action log for both players' actions within a single hand.
    ///
    /// The arena runs agents sequentially on one thread, so thread-local
    /// storage avoids the need for `Arc<Mutex<...>>`. Each entry is a
    /// `(Round, action_code)` pair where action_code is the 4-bit encoding
    /// from `encode_action`. Cleared at the start of each hand.
    static ACTION_LOG: RefCell<Vec<(Round, u8)>> = const { RefCell::new(Vec::new()) };
}

use crate::agent::{AgentConfig, FrequencyMap};
use crate::hand_class::classify;
use crate::hands::CanonicalHand;

/// Progress update emitted during a simulation run.
#[derive(Debug, Clone)]
pub struct SimProgress {
    pub hands_played: u64,
    pub total_hands: u64,
    pub p1_profit_bb: f64,
    pub current_mbbh: f64,
}

/// Final result of a completed simulation.
#[derive(Debug, Clone)]
pub struct SimResult {
    pub hands_played: u64,
    pub p1_profit_bb: f64,
    pub mbbh: f64,
    pub equity_curve: Vec<f64>,
    pub elapsed_ms: u64,
}

// ============================================================================
// Rule-based agent
// ============================================================================

/// An arena agent that uses `AgentConfig` rule-based logic.
struct RuleBasedAgent {
    config: Arc<AgentConfig>,
    rng: SmallRng,
}

impl RuleBasedAgent {
    fn new(config: Arc<AgentConfig>) -> Self {
        Self {
            config,
            rng: SmallRng::from_os_rng(),
        }
    }
}

impl Agent for RuleBasedAgent {
    fn act(&mut self, _id: u128, game_state: &GameState) -> AgentAction {
        let idx = game_state.to_act_idx();
        let hand = game_state.hands[idx];
        let hole_cards = extract_hole_cards(hand, &game_state.board);

        let freq = resolve_frequency(&self.config, game_state, hole_cards);
        let r: f32 = self.rng.random();
        sample_frequency(freq, game_state, r)
    }
}

/// Generator that produces fresh `RuleBasedAgent` instances per game.
pub struct RuleBasedAgentGenerator {
    config: Arc<AgentConfig>,
}

impl RuleBasedAgentGenerator {
    #[must_use]
    pub fn new(config: Arc<AgentConfig>) -> Self {
        Self { config }
    }
}

impl AgentGenerator for RuleBasedAgentGenerator {
    fn generate(&self, _game_state: &GameState) -> Box<dyn Agent> {
        Box::new(RuleBasedAgent::new(self.config.clone()))
    }
}

// ============================================================================
// Dealer-rotating game state generator
// ============================================================================

/// A game state iterator that rotates the dealer index each hand.
///
/// `CloneGameStateGenerator` from `rs_poker` always uses the same `dealer_idx`,
/// creating a positional bias. This generator advances the dealer each hand
/// so that both seats share the SB/BB positions equally.
struct RotatingDealerGenerator {
    base: GameState,
    num_players: usize,
    hand_number: usize,
}

impl RotatingDealerGenerator {
    fn new(base: GameState) -> Self {
        let num_players = base.stacks.len();
        Self {
            base,
            num_players,
            hand_number: 0,
        }
    }
}

impl Iterator for RotatingDealerGenerator {
    type Item = GameState;

    fn next(&mut self) -> Option<Self::Item> {
        let dealer_idx = self.hand_number % self.num_players;
        self.hand_number += 1;
        Some(GameState::new_starting(
            self.base.stacks.clone(),
            self.base.big_blind,
            self.base.small_blind,
            0.0,
            dealer_idx,
        ))
    }
}

// ============================================================================
// Action-logging wrappers
// ============================================================================

/// Wraps any `Agent` to record its actions in the thread-local `ACTION_LOG`.
///
/// After the inner agent returns an action, the wrapper converts it to its
/// 4-bit action code and appends it to the log. This ensures that *both*
/// players' actions are visible when agents build keys.
struct LoggingAgentWrapper {
    inner: Box<dyn Agent>,
    bet_sizes: Vec<f32>,
}

impl Agent for LoggingAgentWrapper {
    fn act(&mut self, id: u128, game_state: &GameState) -> AgentAction {
        let action = self.inner.act(id, game_state);
        let code = agent_action_to_code(&action, game_state, &self.bet_sizes);
        let round = game_state.round;
        ACTION_LOG.with(|log| {
            log.borrow_mut().push((round, code));
        });
        action
    }
}

/// Wraps any `AgentGenerator` to produce `LoggingAgentWrapper`-wrapped agents.
///
/// Clears the thread-local `ACTION_LOG` when generating a new agent (once per
/// hand), ensuring a fresh log for each hand.
struct LoggingAgentGeneratorWrapper {
    inner: Box<dyn AgentGenerator>,
    bet_sizes: Vec<f32>,
}

impl AgentGenerator for LoggingAgentGeneratorWrapper {
    fn generate(&self, game_state: &GameState) -> Box<dyn Agent> {
        ACTION_LOG.with(|log| log.borrow_mut().clear());
        let inner = self.inner.generate(game_state);
        Box::new(LoggingAgentWrapper {
            inner,
            bet_sizes: self.bet_sizes.clone(),
        })
    }
}

// ============================================================================
// Competition runner
// ============================================================================

/// Build a `HoldemCompetition` with logging-wrapped agent generators.
#[allow(clippy::cast_precision_loss)]
fn setup_competition(
    p1_gen: Box<dyn AgentGenerator>,
    p2_gen: Box<dyn AgentGenerator>,
    stack_depth: u32,
    bet_sizes: &[f32],
) -> HoldemCompetition<StandardSimulationIterator<RotatingDealerGenerator>> {
    // stack_depth is in chips (1 BB = 2 chips).
    let stacks = vec![stack_depth as f32; 2];
    let game_state = GameState::new_starting(stacks, 2.0, 1.0, 0.0, 0);

    let p1_wrapped: Box<dyn AgentGenerator> = Box::new(LoggingAgentGeneratorWrapper {
        inner: p1_gen,
        bet_sizes: bet_sizes.to_vec(),
    });
    let p2_wrapped: Box<dyn AgentGenerator> = Box::new(LoggingAgentGeneratorWrapper {
        inner: p2_gen,
        bet_sizes: bet_sizes.to_vec(),
    });

    let sim_gen = StandardSimulationIterator::new(
        vec![p1_wrapped, p2_wrapped],
        vec![],
        RotatingDealerGenerator::new(game_state),
    );
    HoldemCompetition::new(sim_gen)
}

/// Play hands in batches, reporting progress and collecting the equity curve.
#[allow(clippy::cast_precision_loss)]
fn play_hands(
    competition: &mut HoldemCompetition<StandardSimulationIterator<RotatingDealerGenerator>>,
    num_hands: u64,
    stop: &AtomicBool,
    mut on_progress: impl FnMut(&SimProgress),
) -> Result<(u64, Vec<f64>), String> {
    let batch_size = 100u64;
    let mut hands_played = 0u64;
    let mut equity_curve = Vec::new();

    while hands_played < num_hands && !stop.load(Ordering::Relaxed) {
        let batch = batch_size.min(num_hands - hands_played);
        #[allow(clippy::cast_possible_truncation)]
        let n = batch as usize;
        competition
            .run(n)
            .map_err(|e| format!("Simulation error: {e:?}"))?;
        hands_played += batch;

        let p1_profit_bb = f64::from(competition.total_change[0]);
        let mbbh = (p1_profit_bb / hands_played as f64) * 1000.0;
        equity_curve.push(mbbh);

        on_progress(&SimProgress {
            hands_played,
            total_hands: num_hands,
            p1_profit_bb,
            current_mbbh: mbbh,
        });
    }

    Ok((hands_played, equity_curve))
}

/// Run a simulation between two agent generators.
///
/// Both generators are wrapped with `LoggingAgentGeneratorWrapper` so that
/// every action is recorded in the thread-local `ACTION_LOG`. This allows
/// agents to see both players' actions when constructing info set keys.
///
/// `bet_sizes` is the list of pot-fraction bet sizes used to convert
/// `AgentAction::Bet(amount)` into key-string indices (e.g. `b0`, `r1`).
/// Pass the trained bundle's `bet_sizes` when a blueprint agent is playing,
/// or a default like `&[0.5, 1.0]` for simulations without blueprint agents.
///
/// # Errors
///
/// Returns an error string if the competition fails to run.
#[allow(clippy::cast_precision_loss)]
pub fn run_simulation(
    p1_gen: Box<dyn AgentGenerator>,
    p2_gen: Box<dyn AgentGenerator>,
    num_hands: u64,
    stack_depth: u32,
    stop: &AtomicBool,
    bet_sizes: &[f32],
    on_progress: impl FnMut(&SimProgress),
) -> Result<SimResult, String> {
    let mut competition = setup_competition(p1_gen, p2_gen, stack_depth, bet_sizes);
    let start = std::time::Instant::now();

    let (hands_played, equity_curve) = play_hands(&mut competition, num_hands, stop, on_progress)?;

    #[allow(clippy::cast_possible_truncation)]
    let elapsed_ms = start.elapsed().as_millis() as u64;
    let p1_profit_bb = f64::from(competition.total_change[0]);
    let mbbh = if hands_played > 0 {
        (p1_profit_bb / hands_played as f64) * 1000.0
    } else {
        0.0
    };

    Ok(SimResult {
        hands_played,
        p1_profit_bb,
        mbbh,
        equity_curve,
        elapsed_ms,
    })
}

// ============================================================================
// Helper functions
// ============================================================================

/// Extract the two hole cards from an arena Hand by subtracting board cards.
///
/// The arena adds community cards to each player's `Hand`, so we filter
/// out any card that also appears on the board.
///
/// # Panics
///
/// Panics in debug builds if fewer than 2 hole cards remain after filtering.
/// In release builds, defaults to the first available cards (should never happen
/// in a properly configured game).
fn extract_hole_cards(hand: Hand, board: &[Card]) -> [Card; 2] {
    let cards: Vec<Card> = hand.iter().filter(|c| !board.contains(c)).collect();
    debug_assert!(
        cards.len() >= 2,
        "extract_hole_cards: expected >= 2 cards after board filter, got {}",
        cards.len()
    );
    [
        cards
            .first()
            .copied()
            .unwrap_or(Card::new(Value::Two, Suit::Spade)),
        cards
            .get(1)
            .copied()
            .unwrap_or(Card::new(Value::Three, Suit::Spade)),
    ]
}

/// Resolve the frequency map for the current game state using rule-based logic.
fn resolve_frequency<'a>(
    config: &'a AgentConfig,
    game_state: &GameState,
    hole_cards: [Card; 2],
) -> &'a FrequencyMap {
    let board = &game_state.board;

    if board.is_empty() {
        // Preflop: use range-based lookup
        let position = if game_state.to_act_idx() == dealer_idx(game_state) {
            "btn"
        } else {
            "bb"
        };
        let canonical = CanonicalHand::from_cards(hole_cards[0], hole_cards[1]);
        config.preflop_frequency(position, &canonical)
    } else {
        // Postflop: classify hand and resolve
        let board_cards: Vec<crate::poker::Card> = board.clone();
        match classify(hole_cards, &board_cards) {
            Ok(classification) => config.resolve(&classification),
            Err(_) => &config.default,
        }
    }
}

/// Get the dealer index from the game state.
fn dealer_idx(game_state: &GameState) -> usize {
    game_state.dealer_idx
}

/// Sample an action from a `FrequencyMap` given the current game state.
///
/// When fold is not available (nothing to call), fold probability is
/// redistributed to call (check). When raise probability is zero,
/// the agent checks/calls instead of accidentally raising.
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
fn sample_frequency(freq: &FrequencyMap, game_state: &GameState, r: f32) -> AgentAction {
    let to_call =
        game_state.current_round_bet() - game_state.round_data.player_bet[game_state.to_act_idx()];
    let can_fold = to_call > 0.0;

    // Compute effective probabilities (redistribute fold when unavailable)
    let (eff_fold, eff_call, eff_raise) = if can_fold {
        (freq.fold, freq.call, freq.raise)
    } else {
        // Can't fold: add fold probability to call/check
        (0.0, freq.call + freq.fold, freq.raise)
    };

    if eff_fold > 0.0 && r < eff_fold {
        return AgentAction::Fold;
    }

    if r < eff_fold + eff_call {
        return AgentAction::Call;
    }

    // Guard: if raise frequency is effectively zero, just call/check
    if eff_raise <= 0.001 {
        return AgentAction::Call;
    }

    // Raise: use pot-sized bet as default.
    // Round the raise portion to match the game model's integer arithmetic
    // (resolve_bet_amount rounds `(pot * fraction).round() as u32`).
    let pot = game_state.total_pot;
    let current_bet = game_state.current_round_bet();
    let bet_amount = current_bet + pot.round(); // pot-sized raise, rounded

    if let Some(ref sizes) = freq.raise_sizes {
        let raise_r = (r - eff_fold - eff_call) / eff_raise.max(0.001);
        let mut cumulative = 0.0;
        for (key, &size_freq) in sizes {
            cumulative += size_freq;
            if raise_r < cumulative {
                if key == "allin" {
                    return AgentAction::AllIn;
                }
                if let Ok(fraction) = key.parse::<f32>() {
                    let raise_amount = (pot * fraction).round();
                    let size = current_bet + raise_amount;
                    return AgentAction::Bet(size);
                }
            }
        }
    }

    AgentAction::Bet(bet_amount)
}

/// Convert an `AgentAction` to a 4-bit action code.
///
/// Uses the same encoding as `encode_action` in `info_key`:
/// 1=fold, 2=check, 3=call, 4-8=bet idx 0-4, 9-13=raise idx 0-4,
/// 14=bet all-in, 15=raise all-in.
fn agent_action_to_code(action: &AgentAction, game_state: &GameState, bet_sizes: &[f32]) -> u8 {
    let to_call =
        game_state.current_round_bet() - game_state.round_data.player_bet[game_state.to_act_idx()];
    let is_bet = to_call <= 0.0;

    match action {
        AgentAction::Fold => 1, // fold
        AgentAction::Call => {
            if is_bet { 2 } else { 3 } // check or call
        }
        AgentAction::Bet(amount) => {
            let pot = game_state.total_pot;
            let current_bet = game_state.current_round_bet();
            let bet_portion = amount - current_bet;
            let fraction = if pot > 0.0 { bet_portion / pot } else { 1.0 };

            let closest_idx = bet_sizes
                .iter()
                .enumerate()
                .min_by(|(_, a), (_, b)| {
                    ((**a - fraction).abs())
                        .partial_cmp(&((**b - fraction).abs()))
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .map_or(0, |(i, _)| i);

            #[allow(clippy::cast_possible_truncation)]
            let idx = closest_idx.min(4) as u8;
            if is_bet { 4 + idx } else { 9 + idx } // bet(idx) or raise(idx)
        }
        AgentAction::AllIn => {
            if is_bet { 14 } else { 15 } // bet all-in or raise all-in
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use test_macros::timed_test;

    #[timed_test]
    fn extract_hole_cards_works() {
        let mut hand = Hand::default();
        hand.insert(Card::new(Value::Ace, Suit::Spade));
        hand.insert(Card::new(Value::King, Suit::Heart));
        let cards = extract_hole_cards(hand, &[]);
        // Hand iterates in a specific order, just verify we get 2 cards
        assert_ne!(cards[0], cards[1]);
    }

    #[timed_test]
    fn extract_hole_cards_filters_board() {
        let mut hand = Hand::default();
        hand.insert(Card::new(Value::Ace, Suit::Spade));
        hand.insert(Card::new(Value::King, Suit::Heart));
        // Simulate arena adding community cards to hand
        hand.insert(Card::new(Value::Two, Suit::Diamond));
        hand.insert(Card::new(Value::Three, Suit::Club));
        hand.insert(Card::new(Value::Four, Suit::Heart));
        let board = vec![
            Card::new(Value::Two, Suit::Diamond),
            Card::new(Value::Three, Suit::Club),
            Card::new(Value::Four, Suit::Heart),
        ];
        let cards = extract_hole_cards(hand, &board);
        assert_ne!(cards[0], cards[1]);
        // Should only return hole cards, not board cards
        assert!(
            !board.contains(&cards[0]) && !board.contains(&cards[1]),
            "Hole cards should not include board cards"
        );
    }

    #[timed_test]
    fn rule_based_agent_generator_creates_agents() {
        let toml = r"
[game]
stack_depth = 100
bet_sizes = [0.5, 1.0]

[default]
fold = 0.33
call = 0.34
raise = 0.33
";
        let config = Arc::new(AgentConfig::from_toml(toml).unwrap());
        let generator = RuleBasedAgentGenerator::new(config);
        let gs = GameState::new_starting(vec![100.0, 100.0], 2.0, 1.0, 0.0, 0);
        let _agent = generator.generate(&gs);
    }

    #[timed_test]
    fn run_simulation_builtin_calling_vs_folding() {
        // Use arena's built-in agents as a sanity check
        use rs_poker::arena::agent::{CallingAgentGenerator, FoldingAgentGenerator};

        let p1_gen: Box<dyn AgentGenerator> = Box::new(CallingAgentGenerator);
        let p2_gen: Box<dyn AgentGenerator> = Box::new(FoldingAgentGenerator);

        let stop = AtomicBool::new(false);
        let result = run_simulation(p1_gen, p2_gen, 2000, 200, &stop, &[0.5, 1.0], |_| {}).unwrap();

        assert_eq!(result.hands_played, 2000);
        // With dealer rotation:
        // When folder is SB (dealer): folds, caller wins 0.5 BB
        // When caller is SB (dealer): limps, folder can't fold (check), go to showdown (~0 EV)
        // Net: caller wins ~0.25 BB/hand = ~250 mbb/h
        assert!(
            result.mbbh > 100.0,
            "Calling agent should profit vs folding agent, got {} mbb/h",
            result.mbbh
        );
    }

    #[timed_test]
    fn run_simulation_rule_based_agents() {
        // Two rule-based agents with different strategies should complete without error
        let aggressive_toml = r"
[game]
stack_depth = 100
bet_sizes = [0.5, 1.0]

[default]
fold = 0.1
call = 0.3
raise = 0.6
";
        let passive_toml = r"
[game]
stack_depth = 100
bet_sizes = [0.5, 1.0]

[default]
fold = 0.3
call = 0.6
raise = 0.1
";

        let agg_config = Arc::new(AgentConfig::from_toml(aggressive_toml).unwrap());
        let passive_config = Arc::new(AgentConfig::from_toml(passive_toml).unwrap());

        let p1_gen: Box<dyn AgentGenerator> = Box::new(RuleBasedAgentGenerator::new(agg_config));
        let p2_gen: Box<dyn AgentGenerator> =
            Box::new(RuleBasedAgentGenerator::new(passive_config));

        let stop = AtomicBool::new(false);
        let result = run_simulation(p1_gen, p2_gen, 200, 200, &stop, &[0.5, 1.0], |_| {}).unwrap();

        assert_eq!(result.hands_played, 200);
        // Simulation completed successfully with two different agents
        assert!(result.mbbh.is_finite(), "mbb/h should be finite");
    }

    #[timed_test]
    fn run_simulation_cancellation() {
        let toml = r"
[game]
stack_depth = 100
bet_sizes = [0.5, 1.0]

[default]
fold = 0.33
call = 0.34
raise = 0.33
";
        let config = Arc::new(AgentConfig::from_toml(toml).unwrap());

        let p1_gen: Box<dyn AgentGenerator> =
            Box::new(RuleBasedAgentGenerator::new(config.clone()));
        let p2_gen: Box<dyn AgentGenerator> = Box::new(RuleBasedAgentGenerator::new(config));

        let stop = AtomicBool::new(true); // Start already stopped
        let result =
            run_simulation(p1_gen, p2_gen, 10000, 200, &stop, &[0.5, 1.0], |_| {}).unwrap();

        assert_eq!(result.hands_played, 0, "Should stop immediately");
    }

    #[timed_test]
    fn run_simulation_equity_curve_populated() {
        let toml = r"
[game]
stack_depth = 100
bet_sizes = [0.5, 1.0]

[default]
fold = 0.2
call = 0.4
raise = 0.4
";
        let config = Arc::new(AgentConfig::from_toml(toml).unwrap());

        let p1_gen: Box<dyn AgentGenerator> =
            Box::new(RuleBasedAgentGenerator::new(config.clone()));
        let p2_gen: Box<dyn AgentGenerator> = Box::new(RuleBasedAgentGenerator::new(config));

        let stop = AtomicBool::new(false);
        let mut progress_count = 0u32;
        let result = run_simulation(p1_gen, p2_gen, 200, 200, &stop, &[0.5, 1.0], |_| {
            progress_count += 1;
        })
        .unwrap();

        assert_eq!(result.hands_played, 200);
        assert!(
            !result.equity_curve.is_empty(),
            "Equity curve should have data"
        );
        assert!(progress_count > 0, "Should have received progress updates");
    }

    #[timed_test]
    fn swapping_players_flips_mbbh_sign() {
        use rs_poker::arena::agent::{CallingAgentGenerator, FoldingAgentGenerator};

        let stop = AtomicBool::new(false);

        // P1=caller, P2=folder
        let r1 = run_simulation(
            Box::new(CallingAgentGenerator),
            Box::new(FoldingAgentGenerator),
            2000,
            100,
            &stop,
            &[0.5, 1.0],
            |_| {},
        )
        .unwrap();

        // P1=folder, P2=caller
        let r2 = run_simulation(
            Box::new(FoldingAgentGenerator),
            Box::new(CallingAgentGenerator),
            2000,
            100,
            &stop,
            &[0.5, 1.0],
            |_| {},
        )
        .unwrap();

        // Signs should be opposite: caller profits, folder loses
        assert!(
            r1.mbbh > 0.0 && r2.mbbh < 0.0,
            "Swapping should flip sign: r1={}, r2={}",
            r1.mbbh,
            r2.mbbh
        );
    }
}
