//! Heads-Up No-Limit Texas Hold'em Postflop game implementation.
//!
//! Models the complete postflop game tree for heads-up no-limit Texas Hold'em.
//! This is a more complex implementation than `HunlPreflop` that handles
//! flop, turn, and river streets with configurable bet sizing.

use crate::abstraction::Street;
use crate::poker::Card;

use super::Player;

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
}
