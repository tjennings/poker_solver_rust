//! Heads-Up Limit Texas Hold'em game implementation.
//!
//! Models the complete game tree for heads-up limit Texas Hold'em,
//! implementing the [`Game`] trait for use with CFR solvers.
//!
//! ## Unit System
//! - SB = 1 internal unit, BB = 2 internal units
//! - Small bet (preflop/flop) = 2 units (1 BB)
//! - Big bet (turn/river) = 4 units (2 BB)
//!
//! ## Action Model
//! - `Bet(0)` / `Raise(0)` represent the single fixed-size bet for the street
//! - No bet sizing index needed since limit hold'em has only one size per street

use arrayvec::ArrayVec;
use rs_poker::core::Rank;
use serde::{Deserialize, Serialize};

use crate::abstraction::Street;
use crate::card_utils::hand_rank;
use crate::poker::Card;

use super::{Action, Actions, Game, Player};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for a Limit Hold'em game.
///
/// Controls stack depth, number of streets, and raise caps.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LimitHoldemConfig {
    /// Stack depth in big blinds (stacks = `stack_depth` * 2 in SB units)
    pub stack_depth: u32,
    /// Number of streets to play (2 = Flop HE, 4 = full HULH)
    pub num_streets: u8,
    /// Raise cap on preflop and flop
    #[serde(default = "default_max_raises_early")]
    pub max_raises_early: u8,
    /// Raise cap on turn and river
    #[serde(default = "default_max_raises_late")]
    pub max_raises_late: u8,
    /// Small bet size in SB units (preflop/flop)
    #[serde(default = "default_small_bet")]
    pub small_bet: u32,
    /// Big bet size in SB units (turn/river)
    #[serde(default = "default_big_bet")]
    pub big_bet: u32,
}

fn default_max_raises_early() -> u8 {
    3
}
fn default_max_raises_late() -> u8 {
    4
}
fn default_small_bet() -> u32 {
    2
}
fn default_big_bet() -> u32 {
    4
}

impl Default for LimitHoldemConfig {
    fn default() -> Self {
        Self {
            stack_depth: 20,
            num_streets: 4,
            max_raises_early: 3,
            max_raises_late: 4,
            small_bet: 2,
            big_bet: 4,
        }
    }
}

// ---------------------------------------------------------------------------
// Terminal type
// ---------------------------------------------------------------------------

/// Terminal state classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LimitTerminal {
    /// A player folded; the opponent wins the pot.
    Fold(Player),
    /// Showdown: hands are compared to determine the winner.
    Showdown,
}

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

/// State of a Limit Hold'em hand.
///
/// Tracks holdings, board, pot, stacks, and action history.
/// Internal units: SB = 1, BB = 2.
#[derive(Debug, Clone)]
pub struct LimitHoldemState {
    /// Player 1 (SB) hole cards
    pub p1_holding: [Card; 2],
    /// Player 2 (BB) hole cards
    pub p2_holding: [Card; 2],
    /// Pre-dealt 5-card board, revealed progressively
    pub full_board: [Card; 5],
    /// Number of board cards currently revealed (0, 3, 4, or 5)
    pub board_len: u8,
    /// Current street
    pub street: Street,
    /// Pot size in SB units
    pub pot: u32,
    /// Remaining stacks [SB, BB] in SB units
    pub stacks: [u32; 2],
    /// Amount the active player must put in to call
    pub to_call: u32,
    /// Player to act (None if terminal)
    pub to_act: Option<Player>,
    /// Raises so far on the current street
    pub street_raises: u8,
    /// Action history with street context
    pub history: ArrayVec<(Street, Action), 48>,
    /// Terminal classification (None if hand is ongoing)
    pub terminal: Option<LimitTerminal>,
    /// Cached 7-card hand rank for P1 (precomputed at deal time)
    pub p1_rank: Option<Rank>,
    /// Cached 7-card hand rank for P2 (precomputed at deal time)
    pub p2_rank: Option<Rank>,
}

impl LimitHoldemState {
    /// Create a new preflop state with blinds posted and a pre-dealt board.
    ///
    /// SB posts 1, BB posts 2. SB acts first preflop.
    #[must_use]
    pub fn new_preflop(
        p1_holding: [Card; 2],
        p2_holding: [Card; 2],
        full_board: [Card; 5],
        stack_depth_bb: u32,
    ) -> Self {
        let stack = stack_depth_bb * 2;
        Self {
            p1_holding,
            p2_holding,
            full_board,
            board_len: 0,
            street: Street::Preflop,
            pot: 3,
            stacks: [stack - 1, stack - 2],
            to_call: 1,
            to_act: Some(Player::Player1),
            street_raises: 0,
            history: ArrayVec::new(),
            terminal: None,
            p1_rank: None,
            p2_rank: None,
        }
    }

    /// The visible board cards (slice of `full_board` up to `board_len`).
    #[must_use]
    pub fn visible_board(&self) -> &[Card] {
        &self.full_board[..self.board_len as usize]
    }

    /// The active player, assuming the state is non-terminal.
    #[must_use]
    pub fn active_player(&self) -> Player {
        debug_assert!(
            self.to_act.is_some(),
            "active_player called on terminal state"
        );
        self.to_act.unwrap_or(Player::Player1)
    }

    /// Current player's remaining stack.
    #[must_use]
    pub fn current_stack(&self) -> u32 {
        self.stacks[player_index(self.active_player())]
    }

    /// Current player's holding.
    #[must_use]
    pub fn current_holding(&self) -> [Card; 2] {
        match self.to_act {
            Some(Player::Player2) => self.p2_holding,
            _ => self.p1_holding,
        }
    }
}

// ---------------------------------------------------------------------------
// Game engine
// ---------------------------------------------------------------------------

/// The Limit Hold'em game engine.
///
/// Implements the [`Game`] trait for CFR solving.
#[derive(Debug, Clone)]
pub struct LimitHoldem {
    /// Game configuration
    pub config: LimitHoldemConfig,
    /// Number of random deals to generate
    pub deal_count: usize,
    /// RNG seed for reproducible deal generation
    pub rng_seed: u64,
}

impl LimitHoldem {
    /// Create a new Limit Hold'em game with the given configuration.
    #[must_use]
    pub fn new(config: LimitHoldemConfig, deal_count: usize, rng_seed: u64) -> Self {
        Self {
            config,
            deal_count,
            rng_seed,
        }
    }
}

// ---------------------------------------------------------------------------
// Game trait implementation
// ---------------------------------------------------------------------------

impl Game for LimitHoldem {
    type State = LimitHoldemState;

    fn initial_states(&self) -> Vec<Self::State> {
        generate_deals(self.rng_seed, self.deal_count, self.config.stack_depth)
    }

    fn is_terminal(&self, state: &Self::State) -> bool {
        state.terminal.is_some()
    }

    fn player(&self, state: &Self::State) -> Player {
        state.active_player()
    }

    fn actions(&self, state: &Self::State) -> Actions {
        compute_actions(state, &self.config)
    }

    fn next_state(&self, state: &Self::State, action: Action) -> Self::State {
        apply_action(state, action, &self.config)
    }

    fn utility(&self, state: &Self::State, player: Player) -> f64 {
        compute_utility(state, player, self.config.stack_depth)
    }

    fn info_set_key(&self, state: &Self::State) -> u64 {
        compute_info_set_key(state)
    }
}

// ---------------------------------------------------------------------------
// Deal generation
// ---------------------------------------------------------------------------

/// Generate random deals with pre-dealt 5-card boards.
fn generate_deals(seed: u64, count: usize, stack_depth_bb: u32) -> Vec<LimitHoldemState> {
    use rand::SeedableRng;
    use rand::prelude::SliceRandom;
    use rand::rngs::StdRng;

    let deck = crate::poker::full_deck();
    let mut rng = StdRng::seed_from_u64(seed);

    (0..count)
        .map(|_| {
            let mut shuffled = deck.clone();
            shuffled.shuffle(&mut rng);
            let p1 = [shuffled[0], shuffled[1]];
            let p2 = [shuffled[2], shuffled[3]];
            let board = [
                shuffled[4],
                shuffled[5],
                shuffled[6],
                shuffled[7],
                shuffled[8],
            ];
            LimitHoldemState::new_preflop(p1, p2, board, stack_depth_bb)
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Action computation
// ---------------------------------------------------------------------------

/// Compute the available actions for the current state.
fn compute_actions(state: &LimitHoldemState, config: &LimitHoldemConfig) -> Actions {
    let mut actions = Actions::new();
    if state.terminal.is_some() {
        return actions;
    }

    let stack = state.current_stack();

    if state.to_call > 0 {
        actions.push(Action::Fold);
        actions.push(Action::Call);
        if can_raise(state, config, stack) {
            actions.push(Action::Raise(0));
        }
    } else {
        actions.push(Action::Check);
        if stack > 0 {
            actions.push(Action::Bet(0));
        }
    }

    actions
}

/// Whether a raise is allowed (under raise cap and player has chips beyond calling).
fn can_raise(state: &LimitHoldemState, config: &LimitHoldemConfig, stack: u32) -> bool {
    let raise_cap = raise_cap_for_street(state.street, config);
    state.street_raises < raise_cap && stack > state.to_call
}

/// The raise cap for the current street.
fn raise_cap_for_street(street: Street, config: &LimitHoldemConfig) -> u8 {
    match street {
        Street::Preflop | Street::Flop => config.max_raises_early,
        Street::Turn | Street::River => config.max_raises_late,
    }
}

/// The fixed bet/raise size for the current street in SB units.
fn bet_size_for_street(street: Street, config: &LimitHoldemConfig) -> u32 {
    match street {
        Street::Preflop | Street::Flop => config.small_bet,
        Street::Turn | Street::River => config.big_bet,
    }
}

// ---------------------------------------------------------------------------
// State transitions
// ---------------------------------------------------------------------------

/// Apply an action to produce a new state.
fn apply_action(
    state: &LimitHoldemState,
    action: Action,
    config: &LimitHoldemConfig,
) -> LimitHoldemState {
    let mut new = state.clone();
    new.history.push((state.street, action));

    match action {
        Action::Fold => apply_fold(state, &mut new),
        Action::Check => apply_check(state, &mut new, config),
        Action::Call => apply_call(state, &mut new, config),
        Action::Bet(0) => apply_bet(state, &mut new, config),
        Action::Raise(0) => apply_raise(state, &mut new, config),
        _ => unreachable!("invalid action for limit holdem: {action:?}"),
    }

    new
}

/// Apply a fold action.
fn apply_fold(state: &LimitHoldemState, new: &mut LimitHoldemState) {
    new.terminal = Some(LimitTerminal::Fold(state.active_player()));
    new.to_act = None;
}

/// Apply a check action.
fn apply_check(state: &LimitHoldemState, new: &mut LimitHoldemState, config: &LimitHoldemConfig) {
    if is_first_action_on_street(state) {
        new.to_act = Some(state.active_player().opponent());
    } else {
        advance_street(new, config);
    }
}

/// Whether no action has been taken on the current street yet.
fn is_first_action_on_street(state: &LimitHoldemState) -> bool {
    !state.history.iter().any(|(s, _)| *s == state.street)
}

/// Apply a call action.
fn apply_call(state: &LimitHoldemState, new: &mut LimitHoldemState, config: &LimitHoldemConfig) {
    let player = state.active_player();
    let call_amount = state.to_call.min(state.current_stack());
    let idx = player_index(player);

    new.stacks[idx] -= call_amount;
    new.pot += call_amount;
    new.to_call = 0;

    if both_all_in(new) {
        finalize_all_in(new);
        return;
    }

    // SB limps preflop: BB gets to act
    if state.street == Street::Preflop && is_first_action_on_street(state) {
        new.to_act = Some(player.opponent());
        return;
    }

    advance_street(new, config);
}

/// Apply a bet action (opening bet when no prior betting on this street).
fn apply_bet(state: &LimitHoldemState, new: &mut LimitHoldemState, config: &LimitHoldemConfig) {
    let player = state.active_player();
    let idx = player_index(player);
    let bet = bet_size_for_street(state.street, config).min(new.stacks[idx]);

    new.stacks[idx] -= bet;
    new.pot += bet;
    new.to_call = bet;
    new.street_raises += 1;
    new.to_act = Some(player.opponent());
}

/// Apply a raise action.
fn apply_raise(state: &LimitHoldemState, new: &mut LimitHoldemState, config: &LimitHoldemConfig) {
    let player = state.active_player();
    let idx = player_index(player);
    let raise_size = bet_size_for_street(state.street, config);
    let total_needed = state.to_call + raise_size;
    let actual = total_needed.min(new.stacks[idx]);

    new.stacks[idx] -= actual;
    new.pot += actual;
    // The opponent now owes the raise increment
    new.to_call = if actual >= total_needed {
        raise_size
    } else {
        // Partial raise (all-in): opponent owes whatever extra beyond call
        actual.saturating_sub(state.to_call)
    };
    new.street_raises += 1;
    new.to_act = Some(player.opponent());
}

/// Index into the stacks array for a player.
const fn player_index(player: Player) -> usize {
    match player {
        Player::Player1 => 0,
        Player::Player2 => 1,
    }
}

/// Whether both players have zero remaining stack.
fn both_all_in(state: &LimitHoldemState) -> bool {
    state.stacks[0] == 0 && state.stacks[1] == 0
}

// ---------------------------------------------------------------------------
// Street advancement
// ---------------------------------------------------------------------------

/// Advance to the next street, or to showdown if on the final street.
fn advance_street(state: &mut LimitHoldemState, config: &LimitHoldemConfig) {
    if is_final_street(state.street, config) {
        state.terminal = Some(LimitTerminal::Showdown);
        state.to_act = None;
        state.board_len = board_len_for_showdown(config);
        return;
    }

    let next = next_street(state.street);
    state.street = next;
    state.board_len = board_len_for_street(next);
    state.street_raises = 0;
    state.to_call = 0;
    state.to_act = Some(Player::Player2);
}

/// Whether the current street is the final street in this configuration.
fn is_final_street(street: Street, config: &LimitHoldemConfig) -> bool {
    match config.num_streets {
        1 => street == Street::Preflop,
        2 => street == Street::Flop,
        3 => street == Street::Turn,
        _ => street == Street::River,
    }
}

/// The next street after the given one.
fn next_street(street: Street) -> Street {
    match street {
        Street::Preflop => Street::Flop,
        Street::Flop => Street::Turn,
        Street::Turn | Street::River => Street::River,
    }
}

/// Number of visible board cards for a given street.
fn board_len_for_street(street: Street) -> u8 {
    match street {
        Street::Preflop => 0,
        Street::Flop => 3,
        Street::Turn => 4,
        Street::River => 5,
    }
}

/// Board length needed for showdown based on `num_streets` config.
fn board_len_for_showdown(config: &LimitHoldemConfig) -> u8 {
    match config.num_streets {
        1 => 0,
        2 => 3,
        3 => 4,
        _ => 5,
    }
}

/// Both players are all-in: reveal full board and go to showdown.
fn finalize_all_in(state: &mut LimitHoldemState) {
    state.board_len = 5;
    state.terminal = Some(LimitTerminal::Showdown);
    state.to_act = None;
}

// ---------------------------------------------------------------------------
// Utility calculation
// ---------------------------------------------------------------------------

/// Convert internal chip units to BB (1 BB = 2 internal units).
fn to_bb(chips: u32) -> f64 {
    f64::from(chips) / 2.0
}

/// Compute the utility for a given player at a terminal state.
///
/// Returns the payoff in BB. Positive = profit, negative = loss.
fn compute_utility(state: &LimitHoldemState, player: Player, stack_depth_bb: u32) -> f64 {
    let Some(terminal) = state.terminal else {
        return 0.0;
    };

    let starting_stack = stack_depth_bb * 2;
    let p1_invested = starting_stack - state.stacks[0];
    let p2_invested = starting_stack - state.stacks[1];

    let p1_ev = match terminal {
        LimitTerminal::Fold(folder) => {
            if folder == Player::Player1 {
                -to_bb(p1_invested)
            } else {
                to_bb(p2_invested)
            }
        }
        LimitTerminal::Showdown => showdown_payoff(state, p1_invested, p2_invested),
    };

    if player == Player::Player1 {
        p1_ev
    } else {
        -p1_ev
    }
}

/// P1's payoff at showdown.
fn showdown_payoff(state: &LimitHoldemState, p1_invested: u32, p2_invested: u32) -> f64 {
    use std::cmp::Ordering;

    let p1_rank = state
        .p1_rank
        .unwrap_or_else(|| hand_rank(state.p1_holding, state.visible_board()));
    let p2_rank = state
        .p2_rank
        .unwrap_or_else(|| hand_rank(state.p2_holding, state.visible_board()));

    let pot_bb = to_bb(p1_invested + p2_invested);

    match p1_rank.cmp(&p2_rank) {
        Ordering::Greater => pot_bb - to_bb(p1_invested),
        Ordering::Less => -to_bb(p1_invested),
        Ordering::Equal => pot_bb / 2.0 - to_bb(p1_invested),
    }
}

// ---------------------------------------------------------------------------
// Info set key
// ---------------------------------------------------------------------------

/// Compute the info set key for the current state.
fn compute_info_set_key(state: &LimitHoldemState) -> u64 {
    use crate::info_key::{InfoKey, canonical_hand_index, encode_action, spr_bucket};

    let holding = state.current_holding();
    let hand_bits = u32::from(canonical_hand_index(holding));

    let street_code = match state.street {
        Street::Preflop => 0,
        Street::Flop => 1,
        Street::Turn => 2,
        Street::River => 3,
    };

    let eff_stack = state.stacks[0].min(state.stacks[1]);
    let spr = spr_bucket(state.pot, eff_stack);

    let action_codes: ArrayVec<u8, 6> = state
        .history
        .iter()
        .filter(|(s, _)| *s == state.street)
        .take(6)
        .map(|(_, a)| encode_action(*a))
        .collect();

    InfoKey::new(hand_bits, street_code, spr, &action_codes).as_u64()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    #![allow(clippy::float_cmp)]
    use super::*;
    use crate::poker::{Suit, Value};
    use test_macros::timed_test;

    fn make_card(value: Value, suit: Suit) -> Card {
        Card::new(value, suit)
    }

    /// Standard holdings: P1 = AKs (spades), P2 = QJh (hearts)
    fn sample_holdings() -> ([Card; 2], [Card; 2]) {
        (
            [
                make_card(Value::Ace, Suit::Spade),
                make_card(Value::King, Suit::Spade),
            ],
            [
                make_card(Value::Queen, Suit::Heart),
                make_card(Value::Jack, Suit::Heart),
            ],
        )
    }

    /// Board giving P1 pair of aces (P1 wins).
    /// P1=AKs, P2=QJh. Board has no straights/flushes, no Q or J.
    fn p1_winning_board() -> [Card; 5] {
        [
            make_card(Value::Ace, Suit::Diamond),
            make_card(Value::Two, Suit::Club),
            make_card(Value::Seven, Suit::Heart),
            make_card(Value::Nine, Suit::Diamond),
            make_card(Value::Four, Suit::Club),
        ]
    }

    /// Board giving P2 pair of queens (P2 wins).
    /// P1=AKs, P2=QJh. Board pairs Q but not A or K.
    fn p2_winning_board() -> [Card; 5] {
        [
            make_card(Value::Queen, Suit::Club),
            make_card(Value::Two, Suit::Club),
            make_card(Value::Seven, Suit::Heart),
            make_card(Value::Nine, Suit::Diamond),
            make_card(Value::Four, Suit::Diamond),
        ]
    }

    /// Board where both players effectively tie.
    /// Use same holding for both players (different suits) to guarantee tie.
    /// We'll use special holdings for the tie test instead of this board.
    fn tie_holdings_and_board() -> ([Card; 2], [Card; 2], [Card; 5]) {
        // P1 = Ac Kc, P2 = Ad Kd (same ranks, different suits)
        // Board has no flush possibility for either suit
        let p1 = [
            make_card(Value::Ace, Suit::Club),
            make_card(Value::King, Suit::Club),
        ];
        let p2 = [
            make_card(Value::Ace, Suit::Diamond),
            make_card(Value::King, Suit::Diamond),
        ];
        let board = [
            make_card(Value::Two, Suit::Heart),
            make_card(Value::Seven, Suit::Spade),
            make_card(Value::Nine, Suit::Heart),
            make_card(Value::Four, Suit::Spade),
            make_card(Value::Six, Suit::Heart),
        ];
        (p1, p2, board)
    }

    fn make_state(board: [Card; 5]) -> LimitHoldemState {
        let (p1, p2) = sample_holdings();
        LimitHoldemState::new_preflop(p1, p2, board, 20)
    }

    fn default_game() -> LimitHoldem {
        LimitHoldem::new(LimitHoldemConfig::default(), 10, 42)
    }

    /// Advance through check-down on a single street.
    fn check_down(game: &LimitHoldem, state: LimitHoldemState) -> LimitHoldemState {
        let s = game.next_state(&state, Action::Check);
        game.next_state(&s, Action::Check)
    }

    /// Navigate from initial state to flop via SB call + BB check.
    fn to_flop(game: &LimitHoldem, state: &LimitHoldemState) -> LimitHoldemState {
        let s = game.next_state(state, Action::Call);
        game.next_state(&s, Action::Check)
    }

    /// Check down all 4 streets (preflop limp, then check-check on each).
    fn check_down_all(game: &LimitHoldem, state: &LimitHoldemState) -> LimitHoldemState {
        let s = game.next_state(state, Action::Call); // SB calls
        let s = game.next_state(&s, Action::Check); // BB checks => flop
        let s = check_down(game, s); // Flop => turn
        let s = check_down(game, s); // Turn => river
        check_down(game, s) // River => showdown
    }

    // -----------------------------------------------------------------------
    // 1. Config defaults
    // -----------------------------------------------------------------------

    #[timed_test]
    fn config_defaults_are_valid() {
        let config = LimitHoldemConfig::default();
        assert_eq!(config.stack_depth, 20);
        assert_eq!(config.num_streets, 4);
        assert_eq!(config.max_raises_early, 3);
        assert_eq!(config.max_raises_late, 4);
        assert_eq!(config.small_bet, 2);
        assert_eq!(config.big_bet, 4);
    }

    // -----------------------------------------------------------------------
    // 2. Initial states
    // -----------------------------------------------------------------------

    #[timed_test]
    fn initial_states_returns_correct_count() {
        let game = LimitHoldem::new(LimitHoldemConfig::default(), 50, 42);
        assert_eq!(game.initial_states().len(), 50);
    }

    #[timed_test]
    fn initial_states_have_nine_unique_cards() {
        let game = LimitHoldem::new(LimitHoldemConfig::default(), 10, 42);
        for state in game.initial_states() {
            let mut cards = Vec::with_capacity(9);
            cards.extend_from_slice(&state.p1_holding);
            cards.extend_from_slice(&state.p2_holding);
            cards.extend_from_slice(&state.full_board);
            let unique: std::collections::HashSet<_> = cards.iter().collect();
            assert_eq!(unique.len(), 9);
        }
    }

    #[timed_test]
    fn initial_states_are_not_terminal() {
        let game = default_game();
        for state in game.initial_states() {
            assert!(!game.is_terminal(&state));
        }
    }

    #[timed_test]
    fn initial_states_have_correct_pot_and_stacks() {
        let game = default_game();
        for state in game.initial_states() {
            assert_eq!(state.pot, 3);
            assert_eq!(state.stacks[0], 39);
            assert_eq!(state.stacks[1], 38);
        }
    }

    // -----------------------------------------------------------------------
    // 3. Preflop actions
    // -----------------------------------------------------------------------

    #[timed_test]
    fn preflop_sb_can_fold_call_raise() {
        let game = default_game();
        let state = make_state(p1_winning_board());

        assert_eq!(game.player(&state), Player::Player1);
        assert_eq!(
            game.actions(&state).as_slice(),
            &[Action::Fold, Action::Call, Action::Raise(0)]
        );
    }

    #[timed_test]
    fn preflop_bb_can_check_or_bet_after_sb_limps() {
        let game = default_game();
        let state = game.next_state(&make_state(p1_winning_board()), Action::Call);

        assert_eq!(game.player(&state), Player::Player2);
        assert_eq!(
            game.actions(&state).as_slice(),
            &[Action::Check, Action::Bet(0)]
        );
    }

    // -----------------------------------------------------------------------
    // 4. Raise cap enforcement
    // -----------------------------------------------------------------------

    #[timed_test]
    fn raise_cap_prevents_further_raises_preflop() {
        let game = default_game();
        let s = make_state(p1_winning_board());

        // 3 raises: SB, BB, SB
        let s = game.next_state(&s, Action::Raise(0));
        let s = game.next_state(&s, Action::Raise(0));
        let s = game.next_state(&s, Action::Raise(0));

        assert_eq!(
            game.actions(&s).as_slice(),
            &[Action::Fold, Action::Call],
            "After max_raises_early=3 raises, only fold/call allowed"
        );
    }

    // -----------------------------------------------------------------------
    // 5. Street transition
    // -----------------------------------------------------------------------

    #[timed_test]
    fn street_advances_to_flop_after_preflop_closes() {
        let game = default_game();
        let s = to_flop(&game, &make_state(p1_winning_board()));

        assert_eq!(s.street, Street::Flop);
        assert_eq!(s.board_len, 3);
        assert_eq!(s.street_raises, 0);
        assert_eq!(s.to_call, 0);
    }

    // -----------------------------------------------------------------------
    // 6. Postflop position
    // -----------------------------------------------------------------------

    #[timed_test]
    fn p2_acts_first_on_flop() {
        let game = default_game();
        let s = to_flop(&game, &make_state(p1_winning_board()));
        assert_eq!(game.player(&s), Player::Player2);
    }

    #[timed_test]
    fn p2_acts_first_on_turn() {
        let game = default_game();
        let s = to_flop(&game, &make_state(p1_winning_board()));
        let s = check_down(&game, s);

        assert_eq!(s.street, Street::Turn);
        assert_eq!(game.player(&s), Player::Player2);
    }

    // -----------------------------------------------------------------------
    // 7. Turn/river big bet
    // -----------------------------------------------------------------------

    #[timed_test]
    fn flop_uses_small_bet() {
        let game = default_game();
        let s = to_flop(&game, &make_state(p1_winning_board()));
        let pot_before = s.pot;
        let stack_before = s.stacks[1];

        let s = game.next_state(&s, Action::Bet(0));
        assert_eq!(
            stack_before - s.stacks[1],
            2,
            "Flop bet = small bet = 2 SB units"
        );
        assert_eq!(s.pot, pot_before + 2);
    }

    #[timed_test]
    fn turn_uses_big_bet() {
        let game = default_game();
        let s = to_flop(&game, &make_state(p1_winning_board()));
        let s = check_down(&game, s); // to turn

        assert_eq!(s.street, Street::Turn);
        let pot_before = s.pot;
        let stack_before = s.stacks[1];

        let s = game.next_state(&s, Action::Bet(0));
        assert_eq!(
            stack_before - s.stacks[1],
            4,
            "Turn bet = big bet = 4 SB units"
        );
        assert_eq!(s.pot, pot_before + 4);
    }

    // -----------------------------------------------------------------------
    // 8. Fold terminal
    // -----------------------------------------------------------------------

    #[timed_test]
    fn fold_ends_hand_opponent_wins_pot() {
        let game = default_game();
        let s = game.next_state(&make_state(p1_winning_board()), Action::Fold);

        assert!(game.is_terminal(&s));
        assert_eq!(s.terminal, Some(LimitTerminal::Fold(Player::Player1)));
        assert_eq!(game.utility(&s, Player::Player1), -0.5);
        assert_eq!(game.utility(&s, Player::Player2), 0.5);
    }

    // -----------------------------------------------------------------------
    // 9. Showdown terminal
    // -----------------------------------------------------------------------

    #[timed_test]
    fn showdown_after_river_check_down() {
        let game = default_game();
        let s = check_down_all(&game, &make_state(p1_winning_board()));

        assert!(game.is_terminal(&s));
        assert_eq!(s.terminal, Some(LimitTerminal::Showdown));
    }

    // -----------------------------------------------------------------------
    // 10. Utility calculation
    // -----------------------------------------------------------------------

    #[timed_test]
    fn utility_p1_wins_showdown() {
        let game = default_game();
        let s = check_down_all(&game, &make_state(p1_winning_board()));

        assert_eq!(game.utility(&s, Player::Player1), 1.0);
        assert_eq!(game.utility(&s, Player::Player2), -1.0);
    }

    #[timed_test]
    fn utility_p2_wins_showdown() {
        let game = default_game();
        let s = check_down_all(&game, &make_state(p2_winning_board()));

        assert_eq!(game.utility(&s, Player::Player1), -1.0);
        assert_eq!(game.utility(&s, Player::Player2), 1.0);
    }

    // -----------------------------------------------------------------------
    // 11. Tie pot split
    // -----------------------------------------------------------------------

    #[timed_test]
    fn tie_splits_pot_evenly() {
        let game = default_game();
        let (p1, p2, board) = tie_holdings_and_board();
        let state = LimitHoldemState::new_preflop(p1, p2, board, 20);
        let s = check_down_all(&game, &state);

        assert_eq!(game.utility(&s, Player::Player1), 0.0);
        assert_eq!(game.utility(&s, Player::Player2), 0.0);
    }

    // -----------------------------------------------------------------------
    // 12. All-in handling
    // -----------------------------------------------------------------------

    #[timed_test]
    fn all_in_with_shallow_stacks() {
        // 3 BB = 6 SB units. After blinds: SB=5, BB=4
        let config = LimitHoldemConfig {
            stack_depth: 3,
            ..Default::default()
        };
        let game = LimitHoldem::new(config, 1, 42);
        let (p1, p2) = sample_holdings();
        let s = LimitHoldemState::new_preflop(p1, p2, p1_winning_board(), 3);

        // SB raises: call 1 + raise 2 = 3 used
        let s = game.next_state(&s, Action::Raise(0));
        assert_eq!(s.stacks[0], 2);

        // BB raises: call 2 + raise 2 = 4 used => all-in
        let s = game.next_state(&s, Action::Raise(0));
        assert_eq!(s.stacks[1], 0);

        // SB calls with remaining 2 => both all-in => showdown
        let s = game.next_state(&s, Action::Call);
        assert_eq!(s.stacks[0], 0);
        assert!(game.is_terminal(&s));
        assert_eq!(s.terminal, Some(LimitTerminal::Showdown));
    }

    #[timed_test]
    fn bet_capped_to_stack() {
        let config = LimitHoldemConfig {
            stack_depth: 2,
            ..Default::default()
        };
        let game = LimitHoldem::new(config, 1, 42);
        let (p1, p2) = sample_holdings();
        let s = LimitHoldemState::new_preflop(p1, p2, p1_winning_board(), 2);

        // SB calls (1), BB checks => flop. P2 has 2 chips. Bet 2 = all-in.
        let s = game.next_state(&s, Action::Call);
        let s = game.next_state(&s, Action::Check);
        assert_eq!(s.stacks[1], 2);

        let s = game.next_state(&s, Action::Bet(0));
        assert_eq!(s.stacks[1], 0);
    }

    // -----------------------------------------------------------------------
    // 13. Flop HE variant
    // -----------------------------------------------------------------------

    #[timed_test]
    fn flop_he_ends_after_flop_betting() {
        let config = LimitHoldemConfig {
            num_streets: 2,
            ..Default::default()
        };
        let game = LimitHoldem::new(config, 1, 42);
        let s = to_flop(&game, &make_state(p1_winning_board()));
        let s = check_down(&game, s);

        assert!(game.is_terminal(&s));
        assert_eq!(s.terminal, Some(LimitTerminal::Showdown));
        assert_eq!(s.board_len, 3);
    }

    // -----------------------------------------------------------------------
    // 14. Info set key
    // -----------------------------------------------------------------------

    #[timed_test]
    fn info_set_key_different_holdings_different_keys() {
        let game = default_game();
        let (p1, p2) = sample_holdings();
        let board = p1_winning_board();

        let s1 = LimitHoldemState::new_preflop(p1, p2, board, 20);
        let alt_p1 = [
            make_card(Value::Two, Suit::Diamond),
            make_card(Value::Three, Suit::Diamond),
        ];
        let s2 = LimitHoldemState::new_preflop(alt_p1, p2, board, 20);

        assert_ne!(game.info_set_key(&s1), game.info_set_key(&s2));
    }

    #[timed_test]
    fn info_set_key_same_holding_same_actions_same_key() {
        let game = default_game();
        let (p1, _) = sample_holdings();
        let alt_p2 = [
            make_card(Value::Two, Suit::Heart),
            make_card(Value::Three, Suit::Heart),
        ];

        let s1 = LimitHoldemState::new_preflop(
            p1,
            [
                make_card(Value::Queen, Suit::Heart),
                make_card(Value::Jack, Suit::Heart),
            ],
            p1_winning_board(),
            20,
        );
        let s2 = LimitHoldemState::new_preflop(p1, alt_p2, p1_winning_board(), 20);

        assert_eq!(game.info_set_key(&s1), game.info_set_key(&s2));
    }

    #[timed_test]
    fn info_set_key_changes_after_action() {
        let game = default_game();
        let s = make_state(p1_winning_board());
        let key_before = game.info_set_key(&s);

        let s = game.next_state(&s, Action::Call);
        assert_ne!(key_before, game.info_set_key(&s));
    }

    // -----------------------------------------------------------------------
    // 15. Full hand playthrough
    // -----------------------------------------------------------------------

    #[timed_test]
    fn full_hand_preflop_through_river() {
        let game = default_game();
        let mut s = make_state(p1_winning_board());

        // Preflop
        assert_eq!(s.street, Street::Preflop);
        assert_eq!(s.board_len, 0);
        s = game.next_state(&s, Action::Raise(0));
        s = game.next_state(&s, Action::Call);

        // Flop
        assert_eq!(s.street, Street::Flop);
        assert_eq!(s.board_len, 3);
        assert_eq!(game.player(&s), Player::Player2);
        s = game.next_state(&s, Action::Bet(0));
        s = game.next_state(&s, Action::Call);

        // Turn
        assert_eq!(s.street, Street::Turn);
        assert_eq!(s.board_len, 4);
        s = game.next_state(&s, Action::Check);
        s = game.next_state(&s, Action::Bet(0));
        s = game.next_state(&s, Action::Call);

        // River
        assert_eq!(s.street, Street::River);
        assert_eq!(s.board_len, 5);
        s = game.next_state(&s, Action::Check);
        s = game.next_state(&s, Action::Check);

        assert!(game.is_terminal(&s));
        assert!(game.utility(&s, Player::Player1) > 0.0);
    }

    // -----------------------------------------------------------------------
    // Additional tests
    // -----------------------------------------------------------------------

    #[timed_test]
    fn preflop_raise_stacks_are_correct() {
        let game = default_game();
        let s = make_state(p1_winning_board());

        // SB raises: call 1 + raise 2 = 3
        let s = game.next_state(&s, Action::Raise(0));
        assert_eq!(s.stacks[0], 36);
        assert_eq!(s.pot, 6);
        assert_eq!(s.to_call, 2);

        // BB re-raises: call 2 + raise 2 = 4
        let s = game.next_state(&s, Action::Raise(0));
        assert_eq!(s.stacks[1], 34);
        assert_eq!(s.pot, 10);
        assert_eq!(s.to_call, 2);
    }

    #[timed_test]
    fn fold_on_flop_after_bet() {
        let game = default_game();
        let s = to_flop(&game, &make_state(p1_winning_board()));

        let s = game.next_state(&s, Action::Bet(0));
        let s = game.next_state(&s, Action::Fold);

        assert!(game.is_terminal(&s));
        assert_eq!(s.terminal, Some(LimitTerminal::Fold(Player::Player1)));
        assert_eq!(game.utility(&s, Player::Player1), -1.0);
        assert_eq!(game.utility(&s, Player::Player2), 1.0);
    }

    #[timed_test]
    fn seeded_deals_are_reproducible() {
        let g1 = LimitHoldem::new(LimitHoldemConfig::default(), 10, 12345);
        let g2 = LimitHoldem::new(LimitHoldemConfig::default(), 10, 12345);

        for (s1, s2) in g1.initial_states().iter().zip(g2.initial_states().iter()) {
            assert_eq!(s1.p1_holding, s2.p1_holding);
            assert_eq!(s1.p2_holding, s2.p2_holding);
            assert_eq!(s1.full_board, s2.full_board);
        }
    }

    #[timed_test]
    fn turn_raise_cap_is_independent_of_flop() {
        let game = default_game();
        let s = to_flop(&game, &make_state(p1_winning_board()));

        // Use all 3 flop raises then call
        let s = game.next_state(&s, Action::Bet(0));
        let s = game.next_state(&s, Action::Raise(0));
        let s = game.next_state(&s, Action::Raise(0));
        let s = game.next_state(&s, Action::Call);

        assert_eq!(s.street, Street::Turn);
        assert_eq!(s.street_raises, 0);
        assert!(game.actions(&s).contains(&Action::Bet(0)));
    }

    #[timed_test]
    fn river_raise_cap_is_4() {
        let game = default_game();
        let s = to_flop(&game, &make_state(p1_winning_board()));
        let s = check_down(&game, s); // flop => turn
        let s = check_down(&game, s); // turn => river

        assert_eq!(s.street, Street::River);

        // 4 raises on river
        let s = game.next_state(&s, Action::Bet(0)); // 1
        let s = game.next_state(&s, Action::Raise(0)); // 2
        let s = game.next_state(&s, Action::Raise(0)); // 3
        let s = game.next_state(&s, Action::Raise(0)); // 4

        // After 4 raises, only fold/call
        assert_eq!(game.actions(&s).as_slice(), &[Action::Fold, Action::Call]);
    }
}
