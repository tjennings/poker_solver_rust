//! Simulation module for running agent-vs-agent poker competitions.
//!
//! Uses `rs_poker::arena` to play complete poker hands between two strategy
//! sources (trained bundles or rule-based agents) and report performance
//! in mbb/h (milli-big-blinds per hand).

use std::cell::RefCell;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

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
    /// `(Round, key_string)` pair. Cleared at the start of each hand.
    static ACTION_LOG: RefCell<Vec<(Round, String)>> = const { RefCell::new(Vec::new()) };
}

use crate::agent::{AgentConfig, FrequencyMap};
use crate::blueprint::{BlueprintStrategy, BundleConfig};
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
        let hand = &game_state.hands[idx];
        let hole_cards = extract_hole_cards(hand, &game_state.board);

        let freq = resolve_frequency(&self.config, game_state, &hole_cards);
        let r: f32 = self.rng.random();
        sample_frequency(freq, game_state, r)
    }
}

/// Generator that produces fresh `RuleBasedAgent` instances per game.
pub struct RuleBasedAgentGenerator {
    config: Arc<AgentConfig>,
}

impl RuleBasedAgentGenerator {
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
// Blueprint agent
// ============================================================================

/// An arena agent that looks up actions from a trained `BlueprintStrategy`.
///
/// Reads from the thread-local `ACTION_LOG` to reconstruct the full action
/// history (both players) for info set key construction. The
/// `LoggingAgentWrapper` is responsible for writing to the log after each
/// agent's action.
struct BlueprintAgent {
    blueprint: Arc<BlueprintStrategy>,
    bundle_config: BundleConfig,
    rng: SmallRng,
}

impl BlueprintAgent {
    fn new(blueprint: Arc<BlueprintStrategy>, bundle_config: BundleConfig) -> Self {
        Self {
            blueprint,
            bundle_config,
            rng: SmallRng::from_os_rng(),
        }
    }

    /// Build the info set key for the current game state.
    ///
    /// Reads the shared `ACTION_LOG` thread-local to include both players'
    /// actions on the current street. This matches the game model's
    /// `info_set_key_into()` which iterates over the full `state.history`.
    fn build_info_set_key(&self, game_state: &GameState) -> String {
        let idx = game_state.to_act_idx();
        let hand = &game_state.hands[idx];
        let board = &game_state.board;
        let hole_cards = extract_hole_cards(hand, board);
        let round = &game_state.round;

        let use_hand_class = self.bundle_config.abstraction_mode == "hand_class";

        // Bucket or hand string
        let bucket_or_hand = if board.is_empty() {
            canonical_hand_string(&hole_cards)
        } else if use_hand_class {
            let board_cards: Vec<crate::poker::Card> =
                board.iter().map(|c| to_core_card(*c)).collect();
            let hole = [to_core_card(hole_cards[0]), to_core_card(hole_cards[1])];
            match classify(hole, &board_cards) {
                Ok(classification) => classification.bits().to_string(),
                Err(_) => "?".to_string(),
            }
        } else {
            // EHS2 mode - not supported in simulation, fall back to hand string
            canonical_hand_string(&hole_cards)
        };

        let street_char = round_to_street_char(round);

        // Both pot and stacks are in internal units (1 BB = 2 units).
        // Round before truncating to u32 to avoid float drift shifting
        // values into the wrong bucket (e.g. 119.8 → 119 vs 120).
        // Divide by 20 for 10-BB-interval buckets, matching the game model.
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let pot_bucket = game_state.total_pot.round() as u32 / 20;
        let eff_stack = effective_stack_rounded(game_state);
        let stack_bucket = eff_stack / 20;

        // Read current-street actions from both players via the shared log
        let action_str = ACTION_LOG.with(|log| {
            log.borrow()
                .iter()
                .filter(|(r, _)| r == round)
                .map(|(_, s)| s.as_str())
                .collect::<Vec<_>>()
                .join("")
        });

        format!("{bucket_or_hand}|{street_char}|p{pot_bucket}s{stack_bucket}|{action_str}")
    }

    /// Look up strategy probabilities, trying nearby pot/stack buckets when
    /// the exact key is missing from the blueprint.
    ///
    /// Searches outward by Manhattan distance in (pot_bucket, stack_bucket)
    /// space. Falls back to a uniform distribution if no nearby key is found.
    fn lookup_nearest(&self, key: &str) -> Vec<f32> {
        // Try exact key first
        if let Some(probs) = self.blueprint.lookup(key) {
            return probs.to_vec();
        }

        // Parse key: "{bucket}|{street}|p{pot}s{stack}|{actions}"
        if let Some((prefix, pot_bucket, stack_bucket, suffix)) = parse_info_set_key(key) {
            // Search nearby pot/stack buckets by increasing Manhattan distance
            for distance in 1..=5i32 {
                for dp in -distance..=distance {
                    let ds_abs = distance - dp.abs();
                    for &ds in &[-ds_abs, ds_abs] {
                        let p = pot_bucket as i32 + dp;
                        let s = stack_bucket as i32 + ds;
                        if p < 0 || s < 0 {
                            continue;
                        }
                        let candidate = format!("{prefix}|p{}s{}|{suffix}", p, s);
                        if let Some(probs) = self.blueprint.lookup(&candidate) {
                            return probs.to_vec();
                        }
                    }
                }
            }
        }

        // Fallback: uniform over all actions
        let num_actions = 2 + self.bundle_config.game.bet_sizes.len() + 1; // fold/call + bets + all-in
        vec![1.0 / num_actions as f32; num_actions]
    }
}

impl Agent for BlueprintAgent {
    fn act(&mut self, _id: u128, game_state: &GameState) -> AgentAction {
        let key = self.build_info_set_key(game_state);

        let probs = self.lookup_nearest(&key);

        let r: f32 = self.rng.random();
        // Action recording is handled by LoggingAgentWrapper
        sample_blueprint_action(&probs, game_state, &self.bundle_config, r)
    }
}

/// Generator that produces fresh `BlueprintAgent` instances per game.
pub struct BlueprintAgentGenerator {
    blueprint: Arc<BlueprintStrategy>,
    bundle_config: BundleConfig,
}

impl BlueprintAgentGenerator {
    pub fn new(blueprint: Arc<BlueprintStrategy>, bundle_config: BundleConfig) -> Self {
        Self {
            blueprint,
            bundle_config,
        }
    }
}

impl AgentGenerator for BlueprintAgentGenerator {
    fn generate(&self, _game_state: &GameState) -> Box<dyn Agent> {
        Box::new(BlueprintAgent::new(
            self.blueprint.clone(),
            self.bundle_config.clone(),
        ))
    }
}

// ============================================================================
// Dealer-rotating game state generator
// ============================================================================

/// A game state iterator that rotates the dealer index each hand.
///
/// `CloneGameStateGenerator` from rs_poker always uses the same `dealer_idx`,
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
/// info-set-key string representation and appends it to the log. This ensures
/// that *both* players' actions are visible when `BlueprintAgent` builds keys.
struct LoggingAgentWrapper {
    inner: Box<dyn Agent>,
    bet_sizes: Vec<f32>,
}

impl Agent for LoggingAgentWrapper {
    fn act(&mut self, id: u128, game_state: &GameState) -> AgentAction {
        let action = self.inner.act(id, game_state);
        let key_str = agent_action_to_key_str(&action, game_state, &self.bet_sizes);
        let round = game_state.round;
        ACTION_LOG.with(|log| {
            log.borrow_mut().push((round, key_str));
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

/// Run a simulation between two agent generators.
///
/// Both generators are wrapped with `LoggingAgentGeneratorWrapper` so that
/// every action is recorded in the thread-local `ACTION_LOG`. This allows
/// `BlueprintAgent` to see both players' actions when constructing info set
/// keys.
///
/// `bet_sizes` is the list of pot-fraction bet sizes used to convert
/// `AgentAction::Bet(amount)` into key-string indices (e.g. `b0`, `r1`).
/// Pass the trained bundle's `bet_sizes` when a `BlueprintAgent` is playing,
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
    mut on_progress: impl FnMut(&SimProgress),
) -> Result<SimResult, String> {
    // stack_depth is in BB.  Internal units: 1 BB = 2 units, so
    // arena stacks = stack_depth * 2 (with big_blind=2.0, small_blind=1.0).
    let stacks = vec![(stack_depth * 2) as f32; 2];
    let game_state = GameState::new_starting(stacks, 2.0, 1.0, 0.0, 0);

    // Wrap both generators so every action is logged for position-aware keys
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
    let mut competition = HoldemCompetition::new(sim_gen);

    let batch_size = 100u64;
    let mut hands_played = 0u64;
    let mut equity_curve = Vec::new();
    let start = std::time::Instant::now();

    while hands_played < num_hands && !stop.load(Ordering::Relaxed) {
        let batch = batch_size.min(num_hands - hands_played);
        competition
            .run(batch as usize)
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
fn extract_hole_cards(hand: &Hand, board: &[Card]) -> [Card; 2] {
    let cards: Vec<Card> = hand
        .iter()
        .filter(|c| !board.contains(c))
        .collect();
    if cards.len() >= 2 {
        [cards[0], cards[1]]
    } else {
        // Fallback (should not happen in a properly configured game)
        [
            Card::new(Value::Two, Suit::Spade),
            Card::new(Value::Three, Suit::Spade),
        ]
    }
}

/// Convert an rs_poker Card to our core Card type (they're the same type).
fn to_core_card(card: Card) -> crate::poker::Card {
    card
}

/// Get the canonical hand string (e.g. "AKs", "QQ", "T9o") for preflop keys.
fn canonical_hand_string(hole: &[Card; 2]) -> String {
    let r1 = value_to_char(hole[0].value);
    let r2 = value_to_char(hole[1].value);

    let rank_order = |c: char| match c {
        'A' => 14u8,
        'K' => 13,
        'Q' => 12,
        'J' => 11,
        'T' => 10,
        _ => c.to_digit(10).unwrap_or(0) as u8,
    };

    let (high, low) = if rank_order(r1) >= rank_order(r2) {
        (r1, r2)
    } else {
        (r2, r1)
    };

    if high == low {
        format!("{high}{low}")
    } else if hole[0].suit == hole[1].suit {
        format!("{high}{low}s")
    } else {
        format!("{high}{low}o")
    }
}

fn value_to_char(v: Value) -> char {
    match v {
        Value::Two => '2',
        Value::Three => '3',
        Value::Four => '4',
        Value::Five => '5',
        Value::Six => '6',
        Value::Seven => '7',
        Value::Eight => '8',
        Value::Nine => '9',
        Value::Ten => 'T',
        Value::Jack => 'J',
        Value::Queen => 'Q',
        Value::King => 'K',
        Value::Ace => 'A',
    }
}

/// Parse an info set key into (prefix, pot_bucket, stack_bucket, action_suffix).
///
/// Key format: `"{bucket}|{street}|p{pot}s{stack}|{actions}"`
/// Returns `("{bucket}|{street}", pot, stack, "{actions}")`.
fn parse_info_set_key(key: &str) -> Option<(&str, u32, u32, &str)> {
    // Split at "|p" to find the pot/stack segment
    let pipe_positions: Vec<usize> = key.match_indices('|').map(|(i, _)| i).collect();
    if pipe_positions.len() < 3 {
        return None;
    }
    // prefix = everything before the second pipe (bucket|street)
    let prefix = &key[..pipe_positions[1]];
    // pot_stack segment is between pipe_positions[1]+1 and pipe_positions[2]
    let ps_segment = &key[pipe_positions[1] + 1..pipe_positions[2]];
    // actions is everything after pipe_positions[2]+1
    let actions = &key[pipe_positions[2] + 1..];

    // Parse "p{N}s{M}"
    let ps_segment = ps_segment.strip_prefix('p')?;
    let s_pos = ps_segment.find('s')?;
    let pot_bucket: u32 = ps_segment[..s_pos].parse().ok()?;
    let stack_bucket: u32 = ps_segment[s_pos + 1..].parse().ok()?;

    Some((prefix, pot_bucket, stack_bucket, actions))
}

/// Convert arena `Round` to street character for info set keys.
fn round_to_street_char(round: &Round) -> char {
    match round {
        Round::Preflop => 'P',
        Round::Flop => 'F',
        Round::Turn => 'T',
        Round::River => 'R',
        _ => 'P', // Starting, Ante, Deal rounds default to preflop
    }
}

/// Effective stack in internal units, rounded to nearest integer.
///
/// Rounds each stack before taking the min, so that float drift from
/// bet-amount rounding doesn't shift the bucket boundary.
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
fn effective_stack_rounded(game_state: &GameState) -> u32 {
    let s0 = game_state.stacks[0].round();
    let s1 = game_state.stacks[1].round();
    s0.min(s1) as u32
}

/// Resolve the frequency map for the current game state using rule-based logic.
fn resolve_frequency<'a>(
    config: &'a AgentConfig,
    game_state: &GameState,
    hole_cards: &[Card; 2],
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
        let board_cards: Vec<crate::poker::Card> =
            board.iter().map(|c| to_core_card(*c)).collect();
        let hole = [to_core_card(hole_cards[0]), to_core_card(hole_cards[1])];
        match classify(hole, &board_cards) {
            Ok(classification) => config.resolve(&classification),
            Err(_) => &config.default,
        }
    }
}

/// Get the dealer index from the game state.
fn dealer_idx(game_state: &GameState) -> usize {
    game_state.dealer_idx
}

/// Sample an action from a FrequencyMap given the current game state.
///
/// When fold is not available (nothing to call), fold probability is
/// redistributed to call (check). When raise probability is zero,
/// the agent checks/calls instead of accidentally raising.
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
fn sample_frequency(freq: &FrequencyMap, game_state: &GameState, r: f32) -> AgentAction {

    let to_call = game_state.current_round_bet()
        - game_state.round_data.player_bet[game_state.to_act_idx()];
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

/// Sample an action from blueprint probabilities.
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
fn sample_blueprint_action(
    probs: &[f32],
    game_state: &GameState,
    bundle_config: &BundleConfig,
    r: f32,
) -> AgentAction {

    let to_call = game_state.current_round_bet()
        - game_state.round_data.player_bet[game_state.to_act_idx()];
    let can_fold = to_call > 0.0;

    // Action order matches HunlPostflop::actions():
    // [fold?] [check/call] [bet/raise sizes...] [all-in]
    let mut idx = 0;
    let mut cumulative = 0.0;

    // Fold (if applicable)
    if can_fold && idx < probs.len() {
        cumulative += probs[idx];
        if r < cumulative {
            return AgentAction::Fold;
        }
        idx += 1;
    }

    // Check/Call
    if idx < probs.len() {
        cumulative += probs[idx];
        if r < cumulative {
            return AgentAction::Call;
        }
        idx += 1;
    }

    // Bet/raise sizes.
    // Round the raise portion to match the game model's integer arithmetic
    // (resolve_bet_amount rounds `(pot * fraction).round() as u32`).
    let pot = game_state.total_pot;
    let current_bet = game_state.current_round_bet();

    for &fraction in &bundle_config.game.bet_sizes {
        if idx < probs.len() {
            cumulative += probs[idx];
            if r < cumulative {
                let raise_amount = (pot * fraction).round();
                let size = current_bet + raise_amount;
                return AgentAction::Bet(size);
            }
            idx += 1;
        }
    }

    // All-in (last action)
    AgentAction::AllIn
}

/// Convert an `AgentAction` to its info set key string representation.
///
/// Uses the same encoding as the game model's `write_action_to_buf`:
/// `f`old, `x` (check), `c`all, `b{idx}`/`r{idx}` for sized bets, `bA`/`rA`
/// for all-in. The `bet_sizes` slice maps pot-fraction sizes to indices.
fn agent_action_to_key_str(
    action: &AgentAction,
    game_state: &GameState,
    bet_sizes: &[f32],
) -> String {
    let to_call = game_state.current_round_bet()
        - game_state.round_data.player_bet[game_state.to_act_idx()];
    let is_bet = to_call <= 0.0;

    match action {
        AgentAction::Fold => "f".to_string(),
        AgentAction::Call => {
            if is_bet { "x".to_string() } else { "c".to_string() }
        }
        AgentAction::Bet(amount) => {
            let pot = game_state.total_pot;
            let current_bet = game_state.current_round_bet();
            let bet_portion = amount - current_bet;
            let fraction = if pot > 0.0 {
                bet_portion / pot
            } else {
                1.0
            };

            let closest_idx = bet_sizes
                .iter()
                .enumerate()
                .min_by(|(_, a), (_, b)| {
                    ((**a - fraction).abs())
                        .partial_cmp(&((**b - fraction).abs()))
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .map(|(i, _)| i)
                .unwrap_or(0);

            let prefix = if is_bet { 'b' } else { 'r' };
            format!("{prefix}{closest_idx}")
        }
        AgentAction::AllIn => {
            let prefix = if is_bet { 'b' } else { 'r' };
            format!("{prefix}A")
        }
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use test_macros::timed_test;

    #[timed_test]
    fn canonical_hand_string_pair() {
        let hand = [
            Card::new(Value::Ace, Suit::Spade),
            Card::new(Value::Ace, Suit::Heart),
        ];
        assert_eq!(canonical_hand_string(&hand), "AA");
    }

    #[timed_test]
    fn canonical_hand_string_suited() {
        let hand = [
            Card::new(Value::Ace, Suit::Spade),
            Card::new(Value::King, Suit::Spade),
        ];
        assert_eq!(canonical_hand_string(&hand), "AKs");
    }

    #[timed_test]
    fn canonical_hand_string_offsuit() {
        let hand = [
            Card::new(Value::King, Suit::Heart),
            Card::new(Value::Ace, Suit::Spade),
        ];
        assert_eq!(canonical_hand_string(&hand), "AKo");
    }

    #[timed_test]
    fn canonical_hand_string_low_first() {
        let hand = [
            Card::new(Value::Two, Suit::Club),
            Card::new(Value::Seven, Suit::Diamond),
        ];
        assert_eq!(canonical_hand_string(&hand), "72o");
    }

    #[timed_test]
    fn extract_hole_cards_works() {
        let mut hand = Hand::default();
        hand.insert(Card::new(Value::Ace, Suit::Spade));
        hand.insert(Card::new(Value::King, Suit::Heart));
        let cards = extract_hole_cards(&hand, &[]);
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
        let cards = extract_hole_cards(&hand, &board);
        assert_ne!(cards[0], cards[1]);
        // Should only return hole cards, not board cards
        assert!(
            !board.contains(&cards[0]) && !board.contains(&cards[1]),
            "Hole cards should not include board cards"
        );
    }

    #[timed_test]
    fn round_to_street_char_preflop() {
        assert_eq!(round_to_street_char(&Round::Preflop), 'P');
        assert_eq!(round_to_street_char(&Round::Flop), 'F');
        assert_eq!(round_to_street_char(&Round::Turn), 'T');
        assert_eq!(round_to_street_char(&Round::River), 'R');
    }

    #[timed_test]
    fn value_to_char_all_values() {
        // Just verify all values produce valid chars
        let values = [
            (Value::Two, '2'),
            (Value::Three, '3'),
            (Value::Ace, 'A'),
            (Value::King, 'K'),
            (Value::Ten, 'T'),
        ];
        for (v, expected) in values {
            assert_eq!(value_to_char(v), expected);
        }
    }

    #[timed_test]
    fn rule_based_agent_generator_creates_agents() {
        let toml = r#"
[game]
stack_depth = 100
bet_sizes = [0.5, 1.0]

[default]
fold = 0.33
call = 0.34
raise = 0.33
"#;
        let config = Arc::new(AgentConfig::from_toml(toml).unwrap());
        let generator = RuleBasedAgentGenerator::new(config);
        let gs = GameState::new_starting(vec![100.0, 100.0], 2.0, 1.0, 0.0, 0);
        let _agent = generator.generate(&gs);
    }

    #[timed_test]
    fn blueprint_agent_generator_creates_agents() {
        let blueprint = Arc::new(BlueprintStrategy::new());
        let config = BundleConfig {
            game: crate::game::PostflopConfig {
                stack_depth: 100,
                bet_sizes: vec![0.5, 1.0],
                ..crate::game::PostflopConfig::default()
            },
            abstraction: None,
            abstraction_mode: "hand_class".to_string(),
        };
        let generator = BlueprintAgentGenerator::new(blueprint, config);
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
        let result = run_simulation(p1_gen, p2_gen, 2000, 100, &stop, &[0.5, 1.0], |_| {}).unwrap();

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
        let aggressive_toml = r#"
[game]
stack_depth = 100
bet_sizes = [0.5, 1.0]

[default]
fold = 0.1
call = 0.3
raise = 0.6
"#;
        let passive_toml = r#"
[game]
stack_depth = 100
bet_sizes = [0.5, 1.0]

[default]
fold = 0.3
call = 0.6
raise = 0.1
"#;

        let agg_config = Arc::new(AgentConfig::from_toml(aggressive_toml).unwrap());
        let passive_config = Arc::new(AgentConfig::from_toml(passive_toml).unwrap());

        let p1_gen: Box<dyn AgentGenerator> =
            Box::new(RuleBasedAgentGenerator::new(agg_config));
        let p2_gen: Box<dyn AgentGenerator> =
            Box::new(RuleBasedAgentGenerator::new(passive_config));

        let stop = AtomicBool::new(false);
        let result = run_simulation(p1_gen, p2_gen, 200, 100, &stop, &[0.5, 1.0], |_| {}).unwrap();

        assert_eq!(result.hands_played, 200);
        // Simulation completed successfully with two different agents
        assert!(result.mbbh.is_finite(), "mbb/h should be finite");
    }

    #[timed_test]
    fn run_simulation_cancellation() {
        let toml = r#"
[game]
stack_depth = 100
bet_sizes = [0.5, 1.0]

[default]
fold = 0.33
call = 0.34
raise = 0.33
"#;
        let config = Arc::new(AgentConfig::from_toml(toml).unwrap());

        let p1_gen: Box<dyn AgentGenerator> =
            Box::new(RuleBasedAgentGenerator::new(config.clone()));
        let p2_gen: Box<dyn AgentGenerator> =
            Box::new(RuleBasedAgentGenerator::new(config));

        let stop = AtomicBool::new(true); // Start already stopped
        let result = run_simulation(p1_gen, p2_gen, 10000, 100, &stop, &[0.5, 1.0], |_| {}).unwrap();

        assert_eq!(result.hands_played, 0, "Should stop immediately");
    }

    #[timed_test]
    fn run_simulation_equity_curve_populated() {
        let toml = r#"
[game]
stack_depth = 100
bet_sizes = [0.5, 1.0]

[default]
fold = 0.2
call = 0.4
raise = 0.4
"#;
        let config = Arc::new(AgentConfig::from_toml(toml).unwrap());

        let p1_gen: Box<dyn AgentGenerator> =
            Box::new(RuleBasedAgentGenerator::new(config.clone()));
        let p2_gen: Box<dyn AgentGenerator> =
            Box::new(RuleBasedAgentGenerator::new(config));

        let stop = AtomicBool::new(false);
        let mut progress_count = 0u32;
        let result = run_simulation(p1_gen, p2_gen, 200, 100, &stop, &[0.5, 1.0], |_| {
            progress_count += 1;
        })
        .unwrap();

        assert_eq!(result.hands_played, 200);
        assert!(!result.equity_curve.is_empty(), "Equity curve should have data");
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

    #[timed_test]
    fn parse_info_set_key_valid() {
        let key = "128|F|p11s2|b3r2r1";
        let (prefix, pot, stack, actions) = parse_info_set_key(key).unwrap();
        assert_eq!(prefix, "128|F");
        assert_eq!(pot, 11);
        assert_eq!(stack, 2);
        assert_eq!(actions, "b3r2r1");
    }

    #[timed_test]
    fn parse_info_set_key_preflop() {
        let key = "AKs|P|p0s9|";
        let (prefix, pot, stack, actions) = parse_info_set_key(key).unwrap();
        assert_eq!(prefix, "AKs|P");
        assert_eq!(pot, 0);
        assert_eq!(stack, 9);
        assert_eq!(actions, "");
    }

    #[timed_test]
    fn lookup_nearest_falls_back_to_uniform() {
        let blueprint = Arc::new(BlueprintStrategy::new());
        let config = BundleConfig {
            game: crate::game::PostflopConfig {
                stack_depth: 100,
                bet_sizes: vec![0.3, 0.5, 1.0, 1.5],
                ..crate::game::PostflopConfig::default()
            },
            abstraction: None,
            abstraction_mode: "hand_class".to_string(),
        };
        let agent = BlueprintAgent::new(blueprint, config);
        let probs = agent.lookup_nearest("128|F|p11s2|b3r2r1");
        // 4 bet sizes → fold + call + 4 bets + all-in = 7 actions
        assert_eq!(probs.len(), 7);
        let expected = 1.0 / 7.0;
        for &p in &probs {
            assert!((p - expected).abs() < 1e-5);
        }
    }

    #[timed_test]
    fn lookup_nearest_finds_nearby_bucket() {
        let mut blueprint = BlueprintStrategy::new();
        // Insert a key with pot=10, stack=3
        blueprint.insert("128|F|p10s3|b0".to_string(), vec![0.1, 0.2, 0.3, 0.15, 0.1, 0.05, 0.1]);
        let blueprint = Arc::new(blueprint);
        let config = BundleConfig {
            game: crate::game::PostflopConfig {
                stack_depth: 100,
                bet_sizes: vec![0.3, 0.5, 1.0, 1.5],
                ..crate::game::PostflopConfig::default()
            },
            abstraction: None,
            abstraction_mode: "hand_class".to_string(),
        };
        let agent = BlueprintAgent::new(blueprint, config);
        // Look up with pot=11, stack=2 — should find nearby pot=10, stack=3 (distance 2)
        let probs = agent.lookup_nearest("128|F|p11s2|b0");
        assert_eq!(probs, vec![0.1, 0.2, 0.3, 0.15, 0.1, 0.05, 0.1]);
    }
}
