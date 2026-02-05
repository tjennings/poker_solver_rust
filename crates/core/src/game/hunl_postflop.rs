//! Heads-Up No-Limit Texas Hold'em Postflop game implementation.
//!
//! Models the complete postflop game tree for heads-up no-limit Texas Hold'em.
//! This is a more complex implementation than `HunlPreflop` that handles
//! flop, turn, and river streets with configurable bet sizing.

use std::sync::Arc;

use rand::prelude::*;

use crate::abstraction::{CardAbstraction, Street};
use crate::poker::{Card, FlatDeck, Value};

use super::{Action, Game, Player};

/// Configuration for the postflop game.
///
/// Controls stack depth, bet sizing options, and sampling parameters.
#[derive(Debug, Clone)]
pub struct PostflopConfig {
    /// Stack depth in big blinds
    pub stack_depth: u32,
    /// Available bet sizes as fractions of pot (e.g., 0.5 = half pot)
    pub bet_sizes: Vec<f32>,
    /// Number of samples per iteration for Monte Carlo methods
    pub samples_per_iteration: usize,
}

impl Default for PostflopConfig {
    fn default() -> Self {
        Self {
            stack_depth: 100,
            bet_sizes: vec![0.33, 0.5, 0.75, 1.0],
            samples_per_iteration: 1000,
        }
    }
}

/// Type of terminal state in the game.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TerminalType {
    /// Player folded, opponent wins the pot
    Fold(Player),
    /// Showdown - hands are compared to determine winner
    Showdown,
}

/// State of a HUNL Postflop hand.
///
/// Tracks the board, holdings, pot, stacks, and action history.
/// Chips are tracked in cents for precision (1 BB = 100 cents).
#[derive(Debug, Clone)]
pub struct PostflopState {
    /// Community cards (0-5 cards)
    pub board: Vec<Card>,
    /// Player 1's (SB) hole cards
    pub p1_holding: [Card; 2],
    /// Player 2's (BB) hole cards
    pub p2_holding: [Card; 2],
    /// Current street
    pub street: Street,
    /// Pot size in cents (1 BB = 100 cents)
    pub pot: u32,
    /// Remaining stacks in cents [SB stack, BB stack]
    pub stacks: [u32; 2],
    /// Amount to call in cents
    pub to_call: u32,
    /// Player to act (None if terminal)
    pub to_act: Option<Player>,
    /// Action history with street context
    pub history: Vec<(Street, super::Action)>,
    /// Terminal state type (None if not terminal)
    pub terminal: Option<TerminalType>,
    /// Number of bets/raises on the current street
    pub street_bets: u8,
}

impl PostflopState {
    /// Create a new preflop state with blinds posted.
    ///
    /// SB posts 50 cents (0.5 BB), BB posts 100 cents (1 BB).
    /// SB acts first preflop.
    ///
    /// # Arguments
    /// * `p1_holding` - Player 1 (SB) hole cards
    /// * `p2_holding` - Player 2 (BB) hole cards
    /// * `stack_depth_bb` - Stack depth in big blinds
    #[must_use]
    pub fn new_preflop(p1_holding: [Card; 2], p2_holding: [Card; 2], stack_depth_bb: u32) -> Self {
        let stack_cents = stack_depth_bb * 100;

        Self {
            board: Vec::new(),
            p1_holding,
            p2_holding,
            street: Street::Preflop,
            pot: 150,                                      // SB (50) + BB (100) = 150 cents
            stacks: [stack_cents - 50, stack_cents - 100], // SB posted 50, BB posted 100
            to_call: 50,                                   // SB needs to call 50 more to match BB
            to_act: Some(Player::Player1),                 // SB acts first preflop
            history: Vec::new(),
            terminal: None,
            street_bets: 1, // BB's post counts as first bet
        }
    }

    /// Get the current player's stack in cents.
    #[must_use]
    pub fn current_stack(&self) -> u32 {
        match self.to_act {
            Some(Player::Player1) => self.stacks[0],
            Some(Player::Player2) => self.stacks[1],
            None => 0,
        }
    }

    /// Get the opponent's stack in cents.
    #[must_use]
    pub fn opponent_stack(&self) -> u32 {
        match self.to_act {
            Some(Player::Player1) => self.stacks[1],
            Some(Player::Player2) => self.stacks[0],
            None => 0,
        }
    }

    /// Get the current player's holding.
    #[must_use]
    pub fn current_holding(&self) -> [Card; 2] {
        match self.to_act {
            Some(Player::Player2) => self.p2_holding,
            Some(Player::Player1) | None => self.p1_holding, // Default to P1 for terminal states
        }
    }
}

/// Full HUNL game including postflop streets.
///
/// Implements the [`Game`] trait for use with CFR solvers.
#[derive(Debug)]
pub struct HunlPostflop {
    config: PostflopConfig,
    abstraction: Option<Arc<CardAbstraction>>,
    rng_seed: u64,
}

impl HunlPostflop {
    /// Create a new HUNL postflop game.
    ///
    /// # Arguments
    /// * `config` - Game configuration (stack depth, bet sizes, etc.)
    /// * `abstraction` - Optional card abstraction for bucketing
    #[must_use]
    pub fn new(config: PostflopConfig, abstraction: Option<Arc<CardAbstraction>>) -> Self {
        Self {
            config,
            abstraction,
            rng_seed: 42,
        }
    }

    /// Set RNG seed for reproducible sampling.
    pub fn set_seed(&mut self, seed: u64) {
        self.rng_seed = seed;
    }

    /// Get the configuration.
    #[must_use]
    pub fn config(&self) -> &PostflopConfig {
        &self.config
    }

    /// Get bet sizes based on pot and remaining stack.
    ///
    /// Returns bet sizes as fractions of the pot, capped at the effective stack.
    fn get_bet_sizes(&self, pot: u32, stack: u32) -> Vec<u32> {
        let mut sizes = Vec::new();
        for &fraction in &self.config.bet_sizes {
            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
            let size = (f64::from(pot) * f64::from(fraction)).round() as u32;
            if size > 0 && size <= stack {
                sizes.push(size);
            }
        }
        // Always include all-in if not already present
        if !sizes.contains(&stack) && stack > 0 {
            sizes.push(stack);
        }
        sizes
    }

    /// Advance the game to the next street or showdown.
    #[allow(clippy::unused_self)]
    fn advance_street(&self, state: &mut PostflopState) {
        match state.street {
            Street::Preflop => {
                state.street = Street::Flop;
                // Note: Board cards would be dealt by MCCFR when sampling
                state.street_bets = 0;
                state.to_call = 0;
                // Postflop: P1 (SB) is out of position, acts first
                state.to_act = Some(Player::Player1);
            }
            Street::Flop => {
                state.street = Street::Turn;
                state.street_bets = 0;
                state.to_call = 0;
                state.to_act = Some(Player::Player1);
            }
            Street::Turn => {
                state.street = Street::River;
                state.street_bets = 0;
                state.to_call = 0;
                state.to_act = Some(Player::Player1);
            }
            Street::River => {
                // River checked through or called - showdown
                state.terminal = Some(TerminalType::Showdown);
                state.to_act = None;
            }
        }
    }
}

impl Game for HunlPostflop {
    type State = PostflopState;

    fn initial_states(&self) -> Vec<Self::State> {
        let mut rng = StdRng::seed_from_u64(self.rng_seed);
        let mut states = Vec::with_capacity(self.config.samples_per_iteration);
        let stack = self.config.stack_depth;

        for _ in 0..self.config.samples_per_iteration {
            let mut deck = FlatDeck::default();
            deck.shuffle(&mut rng);

            // Deal 4 cards: 2 for each player
            // Note: unwrap is safe here because FlatDeck::default() has 52 cards
            let p1 = [deck.deal().unwrap(), deck.deal().unwrap()];
            let p2 = [deck.deal().unwrap(), deck.deal().unwrap()];

            states.push(PostflopState::new_preflop(p1, p2, stack));
        }

        states
    }

    fn is_terminal(&self, state: &Self::State) -> bool {
        state.terminal.is_some()
    }

    fn player(&self, state: &Self::State) -> Player {
        state.to_act.unwrap_or(Player::Player1)
    }

    fn actions(&self, state: &Self::State) -> Vec<Action> {
        if state.terminal.is_some() {
            return vec![];
        }

        let mut actions = Vec::new();
        let stack = state.current_stack();
        let to_call = state.to_call;

        // Can fold if facing a bet
        if to_call > 0 {
            actions.push(Action::Fold);
        }

        // Check or call
        if to_call == 0 {
            actions.push(Action::Check);
        } else if stack >= to_call {
            actions.push(Action::Call);
        }

        // Raise/bet sizes
        let effective_stack = stack.saturating_sub(to_call);
        if effective_stack > 0 {
            let bet_sizes = self.get_bet_sizes(state.pot, effective_stack);
            for size in bet_sizes {
                if to_call == 0 {
                    actions.push(Action::Bet(size));
                } else {
                    actions.push(Action::Raise(to_call + size));
                }
            }
        }

        actions
    }

    fn next_state(&self, state: &Self::State, action: Action) -> Self::State {
        let mut new_state = state.clone();
        let is_p1 = state.to_act == Some(Player::Player1);
        let player_idx = usize::from(!is_p1);

        new_state.history.push((state.street, action));

        match action {
            Action::Fold => {
                let folder = state.to_act.unwrap_or(Player::Player1);
                new_state.terminal = Some(TerminalType::Fold(folder));
                new_state.to_act = None;
            }

            Action::Check => {
                // Special case: BB checking option preflop after SB limp
                if !is_p1 && state.street == Street::Preflop && state.to_call == 0 {
                    // BB checked option, advance to flop
                    self.advance_street(&mut new_state);
                } else {
                    // Check if opponent just checked on the SAME STREET (both players checked)
                    let last_action_on_street = state
                        .history
                        .iter()
                        .rev()
                        .find(|(s, _)| *s == state.street)
                        .map(|(_, a)| a);

                    if matches!(last_action_on_street, Some(Action::Check)) {
                        // Both checked on this street - advance street
                        self.advance_street(&mut new_state);
                    } else {
                        // First check on this street, switch to opponent
                        new_state.to_act = Some(state.to_act.unwrap_or(Player::Player1).opponent());
                    }
                }
            }

            Action::Call => {
                let call_amount = state.to_call.min(state.current_stack());
                new_state.stacks[player_idx] -= call_amount;
                new_state.pot += call_amount;
                new_state.to_call = 0;

                // Check if this is SB limping preflop (BB gets option to raise)
                if is_p1 && state.street == Street::Preflop && state.street_bets == 1 {
                    // SB limped, BB gets option
                    new_state.to_act = Some(Player::Player2);
                } else {
                    // Call closes action - advance street or showdown
                    self.advance_street(&mut new_state);
                }
            }

            Action::Bet(amount) | Action::Raise(amount) => {
                let actual_amount = amount.min(state.current_stack());
                new_state.stacks[player_idx] -= actual_amount;
                new_state.pot += actual_amount;
                new_state.to_call = actual_amount - state.to_call;
                new_state.street_bets += 1;
                new_state.to_act = Some(state.to_act.unwrap_or(Player::Player1).opponent());

                // Check if opponent is all-in
                if new_state.opponent_stack() == 0 {
                    new_state.terminal = Some(TerminalType::Showdown);
                    new_state.to_act = None;
                }
            }
        }

        new_state
    }

    fn utility(&self, state: &Self::State, player: Player) -> f64 {
        let Some(terminal) = state.terminal else {
            return 0.0;
        };

        let stack_cents = self.config.stack_depth * 100;
        let p1_invested = stack_cents - state.stacks[0];
        let p2_invested = stack_cents - state.stacks[1];

        match terminal {
            TerminalType::Fold(folder) => {
                if folder == Player::Player1 {
                    if player == Player::Player1 {
                        -f64::from(p1_invested) / 100.0
                    } else {
                        f64::from(p1_invested) / 100.0
                    }
                } else if player == Player::Player2 {
                    -f64::from(p2_invested) / 100.0
                } else {
                    f64::from(p2_invested) / 100.0
                }
            }
            TerminalType::Showdown => {
                // For now, use 50% equity (placeholder - Task 18 will add actual hand eval)
                let pot = f64::from(p1_invested + p2_invested) / 100.0;
                let p1_equity = 0.5;
                let p1_ev = p1_equity * pot - f64::from(p1_invested) / 100.0;

                if player == Player::Player1 {
                    p1_ev
                } else {
                    -p1_ev
                }
            }
        }
    }

    fn info_set_key(&self, state: &Self::State) -> String {
        let holding = state.current_holding();

        // Get bucket if abstraction available, otherwise use card chars
        let bucket = if let Some(ref abstraction) = self.abstraction {
            if state.board.is_empty() {
                // Preflop: no bucket, use hand string
                format!("{}{}", card_to_char(holding[0]), card_to_char(holding[1]))
            } else {
                abstraction
                    .get_bucket(&state.board, (holding[0], holding[1]))
                    .map_or_else(|_| "?".to_string(), |b| b.to_string())
            }
        } else {
            format!("{}{}", card_to_char(holding[0]), card_to_char(holding[1]))
        };

        let street_char = match state.street {
            Street::Preflop => 'P',
            Street::Flop => 'F',
            Street::Turn => 'T',
            Street::River => 'R',
        };

        let history_str: String = state
            .history
            .iter()
            .map(|(_, a)| action_to_char(*a))
            .collect();

        format!("{bucket}|{street_char}|{history_str}")
    }
}

/// Convert a card to a single character (rank only).
fn card_to_char(card: Card) -> char {
    match card.value {
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

/// Convert an action to a string for the info set key.
fn action_to_char(action: Action) -> String {
    match action {
        Action::Fold => "f".to_string(),
        Action::Check => "x".to_string(),
        Action::Call => "c".to_string(),
        Action::Bet(amt) => format!("b{amt}"),
        Action::Raise(amt) => format!("r{amt}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::poker::{Suit, Value};

    fn make_card(value: Value, suit: Suit) -> Card {
        Card::new(value, suit)
    }

    fn sample_holdings() -> ([Card; 2], [Card; 2]) {
        let p1 = [
            make_card(Value::Ace, Suit::Spade),
            make_card(Value::King, Suit::Spade),
        ];
        let p2 = [
            make_card(Value::Queen, Suit::Heart),
            make_card(Value::Jack, Suit::Heart),
        ];
        (p1, p2)
    }

    #[test]
    fn postflop_config_default_values() {
        let config = PostflopConfig::default();
        assert_eq!(config.stack_depth, 100);
        assert_eq!(config.bet_sizes, vec![0.33, 0.5, 0.75, 1.0]);
        assert_eq!(config.samples_per_iteration, 1000);
    }

    #[test]
    fn postflop_config_custom_values() {
        let config = PostflopConfig {
            stack_depth: 50,
            bet_sizes: vec![0.5, 1.0],
            samples_per_iteration: 500,
        };
        assert_eq!(config.stack_depth, 50);
        assert_eq!(config.bet_sizes.len(), 2);
        assert_eq!(config.samples_per_iteration, 500);
    }

    #[test]
    fn new_preflop_sets_correct_pot() {
        let (p1, p2) = sample_holdings();
        let state = PostflopState::new_preflop(p1, p2, 100);

        // Pot should be SB (50) + BB (100) = 150 cents
        assert_eq!(state.pot, 150);
    }

    #[test]
    fn new_preflop_sets_correct_stacks() {
        let (p1, p2) = sample_holdings();
        let state = PostflopState::new_preflop(p1, p2, 100);

        // SB posted 50 cents: 10000 - 50 = 9950
        // BB posted 100 cents: 10000 - 100 = 9900
        assert_eq!(state.stacks[0], 9950);
        assert_eq!(state.stacks[1], 9900);
    }

    #[test]
    fn new_preflop_sb_acts_first() {
        let (p1, p2) = sample_holdings();
        let state = PostflopState::new_preflop(p1, p2, 100);

        assert_eq!(state.to_act, Some(Player::Player1));
    }

    #[test]
    fn new_preflop_to_call_is_half_bb() {
        let (p1, p2) = sample_holdings();
        let state = PostflopState::new_preflop(p1, p2, 100);

        // SB needs to call 50 more cents to match BB's 100
        assert_eq!(state.to_call, 50);
    }

    #[test]
    fn new_preflop_street_is_preflop() {
        let (p1, p2) = sample_holdings();
        let state = PostflopState::new_preflop(p1, p2, 100);

        assert_eq!(state.street, Street::Preflop);
    }

    #[test]
    fn new_preflop_board_is_empty() {
        let (p1, p2) = sample_holdings();
        let state = PostflopState::new_preflop(p1, p2, 100);

        assert!(state.board.is_empty());
    }

    #[test]
    fn new_preflop_is_not_terminal() {
        let (p1, p2) = sample_holdings();
        let state = PostflopState::new_preflop(p1, p2, 100);

        assert!(state.terminal.is_none());
    }

    #[test]
    fn new_preflop_history_is_empty() {
        let (p1, p2) = sample_holdings();
        let state = PostflopState::new_preflop(p1, p2, 100);

        assert!(state.history.is_empty());
    }

    #[test]
    fn new_preflop_street_bets_is_one() {
        let (p1, p2) = sample_holdings();
        let state = PostflopState::new_preflop(p1, p2, 100);

        // BB's post counts as first bet
        assert_eq!(state.street_bets, 1);
    }

    #[test]
    fn current_stack_returns_sb_stack_when_sb_to_act() {
        let (p1, p2) = sample_holdings();
        let state = PostflopState::new_preflop(p1, p2, 100);

        // SB acts first, should return SB's stack
        assert_eq!(state.current_stack(), 9950);
    }

    #[test]
    fn current_stack_returns_bb_stack_when_bb_to_act() {
        let (p1, p2) = sample_holdings();
        let mut state = PostflopState::new_preflop(p1, p2, 100);
        state.to_act = Some(Player::Player2);

        assert_eq!(state.current_stack(), 9900);
    }

    #[test]
    fn opponent_stack_returns_bb_stack_when_sb_to_act() {
        let (p1, p2) = sample_holdings();
        let state = PostflopState::new_preflop(p1, p2, 100);

        // SB acts first, opponent is BB
        assert_eq!(state.opponent_stack(), 9900);
    }

    #[test]
    fn opponent_stack_returns_sb_stack_when_bb_to_act() {
        let (p1, p2) = sample_holdings();
        let mut state = PostflopState::new_preflop(p1, p2, 100);
        state.to_act = Some(Player::Player2);

        assert_eq!(state.opponent_stack(), 9950);
    }

    #[test]
    fn current_holding_returns_p1_holding_when_sb_to_act() {
        let (p1, p2) = sample_holdings();
        let state = PostflopState::new_preflop(p1, p2, 100);

        assert_eq!(state.current_holding(), p1);
    }

    #[test]
    fn current_holding_returns_p2_holding_when_bb_to_act() {
        let (p1, p2) = sample_holdings();
        let mut state = PostflopState::new_preflop(p1, p2, 100);
        state.to_act = Some(Player::Player2);

        assert_eq!(state.current_holding(), p2);
    }

    #[test]
    fn terminal_type_fold_equality() {
        let fold_p1 = TerminalType::Fold(Player::Player1);
        let fold_p1_again = TerminalType::Fold(Player::Player1);
        let fold_p2 = TerminalType::Fold(Player::Player2);

        assert_eq!(fold_p1, fold_p1_again);
        assert_ne!(fold_p1, fold_p2);
    }

    #[test]
    fn terminal_type_showdown_equality() {
        let showdown1 = TerminalType::Showdown;
        let showdown2 = TerminalType::Showdown;

        assert_eq!(showdown1, showdown2);
    }

    #[test]
    fn postflop_state_stores_holdings() {
        let (p1, p2) = sample_holdings();
        let state = PostflopState::new_preflop(p1, p2, 100);

        assert_eq!(state.p1_holding, p1);
        assert_eq!(state.p2_holding, p2);
    }

    #[test]
    fn different_stack_depths() {
        let (p1, p2) = sample_holdings();

        let state_50bb = PostflopState::new_preflop(p1, p2, 50);
        assert_eq!(state_50bb.stacks[0], 4950); // 50 * 100 - 50
        assert_eq!(state_50bb.stacks[1], 4900); // 50 * 100 - 100

        let state_200bb = PostflopState::new_preflop(p1, p2, 200);
        assert_eq!(state_200bb.stacks[0], 19950); // 200 * 100 - 50
        assert_eq!(state_200bb.stacks[1], 19900); // 200 * 100 - 100
    }

    // ==================== HunlPostflop Game trait tests ====================

    fn create_game() -> HunlPostflop {
        HunlPostflop::new(PostflopConfig::default(), None)
    }

    #[test]
    fn preflop_actions_include_fold_call_raise() {
        let (p1, p2) = sample_holdings();
        let state = PostflopState::new_preflop(p1, p2, 100);
        let game = create_game();

        let actions = game.actions(&state);

        // SB facing BB: can fold, call, or raise
        assert!(
            actions.contains(&Action::Fold),
            "Actions should contain Fold"
        );
        assert!(
            actions.contains(&Action::Call),
            "Actions should contain Call"
        );
        // Should have at least one raise size
        assert!(
            actions.iter().any(|a| matches!(a, Action::Raise(_))),
            "Actions should contain at least one Raise"
        );
    }

    #[test]
    fn fold_ends_game() {
        let (p1, p2) = sample_holdings();
        let state = PostflopState::new_preflop(p1, p2, 100);
        let game = create_game();

        let new_state = game.next_state(&state, Action::Fold);

        assert!(game.is_terminal(&new_state), "Fold should end the game");
        assert_eq!(
            new_state.terminal,
            Some(TerminalType::Fold(Player::Player1)),
            "Terminal should be Fold(Player1)"
        );
    }

    #[test]
    fn check_check_advances_street() {
        let (p1, p2) = sample_holdings();
        let state = PostflopState::new_preflop(p1, p2, 100);
        let game = create_game();

        // SB calls (limps) - BB gets option
        let after_limp = game.next_state(&state, Action::Call);
        assert_eq!(
            after_limp.to_act,
            Some(Player::Player2),
            "After SB limp, BB should get option"
        );
        assert_eq!(
            after_limp.street,
            Street::Preflop,
            "Still preflop after limp"
        );

        // BB checks (option) - goes to flop
        let after_bb_check = game.next_state(&after_limp, Action::Check);
        assert_eq!(after_bb_check.street, Street::Flop, "BB check goes to flop");
        assert_eq!(
            after_bb_check.to_act,
            Some(Player::Player1),
            "SB acts first on flop"
        );

        // SB checks on flop
        let after_sb_check = game.next_state(&after_bb_check, Action::Check);
        assert_eq!(
            after_sb_check.to_act,
            Some(Player::Player2),
            "After SB check on flop, BB to act"
        );

        // BB checks on flop - advance to turn
        let after_bb_check_flop = game.next_state(&after_sb_check, Action::Check);
        assert_eq!(
            after_bb_check_flop.street,
            Street::Turn,
            "Both checks should advance street to Turn"
        );
    }

    #[test]
    fn call_after_raise_advances_street() {
        let (p1, p2) = sample_holdings();
        let state = PostflopState::new_preflop(p1, p2, 100);
        let game = create_game();

        // Find a raise action
        let actions = game.actions(&state);
        let raise = actions
            .iter()
            .find(|a| matches!(a, Action::Raise(_)))
            .expect("Should have a raise action");

        // SB raises
        let after_raise = game.next_state(&state, *raise);
        assert_eq!(
            after_raise.to_act,
            Some(Player::Player2),
            "After SB raise, BB should be to act"
        );

        // BB calls
        let after_call = game.next_state(&after_raise, Action::Call);
        // Call after raise should advance to next street (flop)
        assert_eq!(
            after_call.street,
            Street::Flop,
            "Call after raise should advance to Flop"
        );
    }

    #[test]
    fn initial_states_samples_correctly() {
        let mut config = PostflopConfig::default();
        config.samples_per_iteration = 10; // Small for testing
        let game = HunlPostflop::new(config, None);

        let states = game.initial_states();

        assert_eq!(states.len(), 10, "Should have 10 samples");
        for state in &states {
            assert_eq!(state.street, Street::Preflop);
            assert!(state.terminal.is_none());
            assert_eq!(state.to_act, Some(Player::Player1));
        }
    }

    #[test]
    fn fold_utility_correct() {
        let (p1, p2) = sample_holdings();
        let state = PostflopState::new_preflop(p1, p2, 100);
        let game = create_game();

        // SB folds (loses 0.5 BB)
        let folded = game.next_state(&state, Action::Fold);

        let p1_utility = game.utility(&folded, Player::Player1);
        let p2_utility = game.utility(&folded, Player::Player2);

        // SB posted 0.5 BB (50 cents), loses it
        assert!(
            (p1_utility - (-0.5)).abs() < 0.01,
            "P1 should lose 0.5 BB, got {}",
            p1_utility
        );
        assert!(
            (p2_utility - 0.5).abs() < 0.01,
            "P2 should win 0.5 BB, got {}",
            p2_utility
        );
    }

    #[test]
    fn utilities_are_zero_sum() {
        let (p1, p2) = sample_holdings();
        let state = PostflopState::new_preflop(p1, p2, 100);
        let game = create_game();

        // Test fold
        let folded = game.next_state(&state, Action::Fold);
        let u1 = game.utility(&folded, Player::Player1);
        let u2 = game.utility(&folded, Player::Player2);
        assert!(
            (u1 + u2).abs() < 0.001,
            "Fold utilities should be zero-sum: {} + {} = {}",
            u1,
            u2,
            u1 + u2
        );
    }

    #[test]
    fn info_set_key_includes_hand_and_history() {
        let (p1, p2) = sample_holdings();
        let state = PostflopState::new_preflop(p1, p2, 100);
        let game = create_game();

        let info_set = game.info_set_key(&state);
        // Should contain hand chars (AK) and street (P)
        assert!(
            info_set.contains("|P|"),
            "Info set should contain preflop marker: {}",
            info_set
        );

        // After a raise, history should be included
        let actions = game.actions(&state);
        let raise = actions
            .iter()
            .find(|a| matches!(a, Action::Raise(_)))
            .expect("Should have a raise");
        let after_raise = game.next_state(&state, *raise);

        let info_set2 = game.info_set_key(&after_raise);
        assert!(
            info_set2.contains(":r") || info_set2.contains("|r"),
            "Info set should contain raise history: {}",
            info_set2
        );
    }

    #[test]
    fn get_bet_sizes_includes_all_in() {
        let game = create_game();
        let pot = 150; // 1.5 BB
        let stack = 500; // 5 BB remaining

        let sizes = game.get_bet_sizes(pot, stack);

        // Should include all-in (500)
        assert!(
            sizes.contains(&stack),
            "Sizes should include all-in: {:?}",
            sizes
        );
    }

    #[test]
    fn raise_switches_action() {
        let (p1, p2) = sample_holdings();
        let state = PostflopState::new_preflop(p1, p2, 100);
        let game = create_game();

        let actions = game.actions(&state);
        let raise = actions
            .iter()
            .find(|a| matches!(a, Action::Raise(_)))
            .expect("Should have a raise");

        let after_raise = game.next_state(&state, *raise);

        assert!(!game.is_terminal(&after_raise));
        assert_eq!(game.player(&after_raise), Player::Player2);
    }

    #[test]
    fn bet_after_check_allowed() {
        let (p1, p2) = sample_holdings();
        let state = PostflopState::new_preflop(p1, p2, 100);
        let game = create_game();

        // SB calls to go to flop
        let after_call = game.next_state(&state, Action::Call);

        // P2 (BB) should be able to check or bet
        let actions = game.actions(&after_call);
        assert!(actions.contains(&Action::Check), "Should be able to check");
        assert!(
            actions.iter().any(|a| matches!(a, Action::Bet(_))),
            "Should be able to bet"
        );
    }

    #[test]
    fn river_check_check_is_showdown() {
        let (p1, p2) = sample_holdings();
        let mut state = PostflopState::new_preflop(p1, p2, 100);
        state.street = Street::River;
        state.to_call = 0;
        state.to_act = Some(Player::Player1);

        let game = create_game();

        // P1 checks
        let after_p1_check = game.next_state(&state, Action::Check);
        assert!(!game.is_terminal(&after_p1_check));

        // P2 checks
        let after_p2_check = game.next_state(&after_p1_check, Action::Check);
        assert!(
            game.is_terminal(&after_p2_check),
            "River check-check should be showdown"
        );
        assert_eq!(
            after_p2_check.terminal,
            Some(TerminalType::Showdown),
            "Terminal should be Showdown"
        );
    }
}
