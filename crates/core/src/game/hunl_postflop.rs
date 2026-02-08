//! Heads-Up No-Limit Texas Hold'em Postflop game implementation.
//!
//! Models the complete postflop game tree for heads-up no-limit Texas Hold'em.
//! This is a more complex implementation than `HunlPreflop` that handles
//! flop, turn, and river streets with configurable bet sizing.

use std::sync::Arc;

use arrayvec::ArrayVec;
use serde::{Deserialize, Serialize};

use crate::abstraction::{CardAbstraction, Street};
use crate::poker::{Card, Hand, Rankable, Suit, Value};

use super::{Action, Actions, Game, Player};

/// Configuration for the postflop game.
///
/// Controls stack depth, bet sizing options, and sampling parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PostflopConfig {
    /// Stack depth in big blinds
    pub stack_depth: u32,
    /// Available bet sizes as fractions of pot (e.g., 0.5 = half pot)
    pub bet_sizes: Vec<f32>,
    /// Number of samples per iteration for Monte Carlo methods
    pub samples_per_iteration: usize,
    /// Maximum bets/raises allowed per street (default: 3).
    /// After this many bets on a street, only fold/call/check are available.
    /// This keeps the game tree tractable for CFR traversal.
    #[serde(default = "default_max_raises")]
    pub max_raises_per_street: u8,
}

impl Default for PostflopConfig {
    fn default() -> Self {
        Self {
            stack_depth: 100,
            bet_sizes: vec![0.33, 0.5, 0.75, 1.0],
            samples_per_iteration: 1000,
            max_raises_per_street: 3,
        }
    }
}

fn default_max_raises() -> u8 {
    3
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
///
/// Uses stack-allocated `ArrayVec` for `board` and `history` to avoid
/// heap allocations on clone (the hot path during tree building).
#[derive(Debug, Clone)]
pub struct PostflopState {
    /// Community cards (0-5 cards)
    pub board: ArrayVec<Card, 5>,
    /// Player 1's (SB) hole cards
    pub p1_holding: [Card; 2],
    /// Player 2's (BB) hole cards
    pub p2_holding: [Card; 2],
    /// Pre-dealt 5-card board. When Some, `advance_street` reveals cards from here.
    /// When None, board stays empty (used by `all_canonical_hands` for preflop-only analysis).
    pub full_board: Option<[Card; 5]>,
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
    pub history: ArrayVec<(Street, super::Action), 40>,
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
            board: ArrayVec::new(),
            p1_holding,
            p2_holding,
            full_board: None,
            street: Street::Preflop,
            pot: 150,                                      // SB (50) + BB (100) = 150 cents
            stacks: [stack_cents - 50, stack_cents - 100], // SB posted 50, BB posted 100
            to_call: 50,                                   // SB needs to call 50 more to match BB
            to_act: Some(Player::Player1),                 // SB acts first preflop
            history: ArrayVec::new(),
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

    /// Create a new preflop state with a pre-dealt 5-card board.
    ///
    /// The board cards are revealed progressively as streets advance:
    /// flop reveals cards 0-2, turn reveals card 3, river reveals card 4.
    #[must_use]
    pub fn new_preflop_with_board(
        p1_holding: [Card; 2],
        p2_holding: [Card; 2],
        full_board: [Card; 5],
        stack_depth_bb: u32,
    ) -> Self {
        let mut state = Self::new_preflop(p1_holding, p2_holding, stack_depth_bb);
        state.full_board = Some(full_board);
        state
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

    /// Generate all 169 canonical preflop hands.
    ///
    /// Returns states for all pairs (13), suited hands (78), and offsuit hands (78).
    #[must_use]
    pub fn all_canonical_hands(stack_depth: u32) -> Vec<PostflopState> {
        let mut states = Vec::with_capacity(169);

        let values = [
            Value::Ace,
            Value::King,
            Value::Queen,
            Value::Jack,
            Value::Ten,
            Value::Nine,
            Value::Eight,
            Value::Seven,
            Value::Six,
            Value::Five,
            Value::Four,
            Value::Three,
            Value::Two,
        ];

        // Default opponent hand (avoids most collisions)
        let default_p2 = [
            Card::new(Value::Two, Suit::Club),
            Card::new(Value::Three, Suit::Diamond),
        ];
        let alt_p2 = [
            Card::new(Value::Four, Suit::Club),
            Card::new(Value::Five, Suit::Diamond),
        ];

        let get_opponent = |v1: Value, v2: Value| {
            if v1 == Value::Two || v1 == Value::Three || v2 == Value::Two || v2 == Value::Three {
                alt_p2
            } else {
                default_p2
            }
        };

        for (i, &v1) in values.iter().enumerate() {
            for &v2 in &values[i..] {
                if v1 == v2 {
                    // Pair
                    let p1 = [Card::new(v1, Suit::Spade), Card::new(v2, Suit::Heart)];
                    states.push(PostflopState::new_preflop(
                        p1,
                        get_opponent(v1, v2),
                        stack_depth,
                    ));
                } else {
                    // Suited
                    let p1_suited = [Card::new(v1, Suit::Spade), Card::new(v2, Suit::Spade)];
                    states.push(PostflopState::new_preflop(
                        p1_suited,
                        get_opponent(v1, v2),
                        stack_depth,
                    ));

                    // Offsuit
                    let p1_offsuit = [Card::new(v1, Suit::Spade), Card::new(v2, Suit::Heart)];
                    states.push(PostflopState::new_preflop(
                        p1_offsuit,
                        get_opponent(v1, v2),
                        stack_depth,
                    ));
                }
            }
        }

        states
    }

    /// Generate all 52 cards of a standard deck.
    fn full_deck() -> Vec<Card> {
        let values = [
            Value::Two,
            Value::Three,
            Value::Four,
            Value::Five,
            Value::Six,
            Value::Seven,
            Value::Eight,
            Value::Nine,
            Value::Ten,
            Value::Jack,
            Value::Queen,
            Value::King,
            Value::Ace,
        ];
        let suits = [Suit::Spade, Suit::Heart, Suit::Diamond, Suit::Club];

        let mut deck = Vec::with_capacity(52);
        for &value in &values {
            for &suit in &suits {
                deck.push(Card::new(value, suit));
            }
        }
        deck
    }

    /// Generate random deals with pre-dealt boards for MCCFR sampling.
    ///
    /// Each deal consists of 2 hole cards per player and 5 board cards,
    /// all drawn without replacement from a shuffled deck.
    fn generate_random_deals(&self, seed: u64, count: usize) -> Vec<PostflopState> {
        use rand::SeedableRng;
        use rand::prelude::SliceRandom;
        use rand::rngs::StdRng;

        let mut rng = StdRng::seed_from_u64(seed);
        let deck = Self::full_deck();
        let mut deals = Vec::with_capacity(count);

        print!("Starting random deal generation ...");
        for _ in 0..count {
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
            deals.push(PostflopState::new_preflop_with_board(
                p1,
                p2,
                board,
                self.config.stack_depth,
            ));
        }
        println!(" DONE");
        deals
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
    ///
    /// When the state has a `full_board`, board cards are revealed progressively:
    /// preflop→flop reveals 3 cards, flop→turn reveals 1, turn→river reveals 1.
    #[allow(clippy::unused_self)]
    fn advance_street(&self, state: &mut PostflopState) {
        match state.street {
            Street::Preflop => {
                state.street = Street::Flop;
                if let Some(fb) = state.full_board {
                    state.board.push(fb[0]);
                    state.board.push(fb[1]);
                    state.board.push(fb[2]);
                }
                state.street_bets = 0;
                state.to_call = 0;
                // Postflop: P1 (SB) is out of position, acts first
                state.to_act = Some(Player::Player1);
            }
            Street::Flop => {
                state.street = Street::Turn;
                if let Some(fb) = state.full_board {
                    state.board.push(fb[3]);
                }
                state.street_bets = 0;
                state.to_call = 0;
                state.to_act = Some(Player::Player1);
            }
            Street::Turn => {
                state.street = Street::River;
                if let Some(fb) = state.full_board {
                    state.board.push(fb[4]);
                }
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
        self.generate_random_deals(self.rng_seed, self.config.samples_per_iteration)
    }

    fn is_terminal(&self, state: &Self::State) -> bool {
        state.terminal.is_some()
    }

    fn player(&self, state: &Self::State) -> Player {
        state.to_act.unwrap_or(Player::Player1)
    }

    fn actions(&self, state: &Self::State) -> Actions {
        let mut actions = Actions::new();
        if state.terminal.is_some() {
            return actions;
        }

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

        // Raise/bet sizes (only if under the per-street raise cap)
        let effective_stack = stack.saturating_sub(to_call);
        if effective_stack > 0 && state.street_bets < self.config.max_raises_per_street {
            let bet_sizes = self.get_bet_sizes(state.pot, effective_stack);
            for size in bet_sizes {
                if actions.is_full() {
                    break;
                }
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
                use std::cmp::Ordering;

                // Build 7-card hands for evaluation
                let mut p1_hand = Hand::default();
                for &card in &state.board {
                    p1_hand.insert(card);
                }
                for &card in &state.p1_holding {
                    p1_hand.insert(card);
                }
                let p1_rank = p1_hand.rank();

                let mut p2_hand = Hand::default();
                for &card in &state.board {
                    p2_hand.insert(card);
                }
                for &card in &state.p2_holding {
                    p2_hand.insert(card);
                }
                let p2_rank = p2_hand.rank();

                let pot = f64::from(p1_invested + p2_invested) / 100.0;

                // Higher rank is better in rs_poker
                let p1_ev = match p1_rank.cmp(&p2_rank) {
                    Ordering::Greater => {
                        // P1 wins - gets opponent's investment
                        pot - f64::from(p1_invested) / 100.0
                    }
                    Ordering::Less => {
                        // P2 wins - P1 loses investment
                        -f64::from(p1_invested) / 100.0
                    }
                    Ordering::Equal => {
                        // Tie - split pot
                        pot / 2.0 - f64::from(p1_invested) / 100.0
                    }
                };

                if player == Player::Player1 {
                    p1_ev
                } else {
                    -p1_ev
                }
            }
        }
    }

    fn info_set_key(&self, state: &Self::State) -> String {
        let mut buf = String::with_capacity(32);
        self.info_set_key_into(state, &mut buf);
        buf
    }

    fn info_set_key_into(&self, state: &Self::State, buf: &mut String) {
        use std::fmt::Write;
        buf.clear();

        let holding = state.current_holding();

        // Get bucket if abstraction available, otherwise use card chars
        if let Some(ref abstraction) = self.abstraction {
            if state.board.is_empty() {
                write_canonical_hand(holding, buf);
            } else {
                match abstraction.get_bucket(&state.board, (holding[0], holding[1])) {
                    Ok(b) => {
                        let _ = write!(buf, "{b}");
                    }
                    Err(_) => buf.push('?'),
                }
            }
        } else {
            write_canonical_hand(holding, buf);
        }

        let street_char = match state.street {
            Street::Preflop => 'P',
            Street::Flop => 'F',
            Street::Turn => 'T',
            Street::River => 'R',
        };

        buf.push('|');
        buf.push(street_char);
        buf.push('|');

        for (_, a) in &state.history {
            write_action_to_buf(*a, buf);
        }
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

/// Write an action to the info set key buffer (no allocation).
fn write_action_to_buf(action: Action, buf: &mut String) {
    use std::fmt::Write;
    match action {
        Action::Fold => buf.push('f'),
        Action::Check => buf.push('x'),
        Action::Call => buf.push('c'),
        Action::Bet(amt) => {
            let _ = write!(buf, "b{amt}");
        }
        Action::Raise(amt) => {
            let _ = write!(buf, "r{amt}");
        }
    }
}

/// Write a preflop holding in canonical hand notation into the buffer.
///
/// Writes strings like "AA", "AKs" (suited), "`AKo`" (offsuit).
/// Ranks are ordered high to low (AK not KA).
fn write_canonical_hand(holding: [Card; 2], buf: &mut String) {
    let (c1, c2) = (holding[0], holding[1]);
    let (r1, r2) = (card_to_char(c1), card_to_char(c2));

    // Order ranks high to low using poker rank order
    let rank_order = |c: char| match c {
        'A' => 14,
        'K' => 13,
        'Q' => 12,
        'J' => 11,
        'T' => 10,
        '9' => 9,
        '8' => 8,
        '7' => 7,
        '6' => 6,
        '5' => 5,
        '4' => 4,
        '3' => 3,
        '2' => 2,
        _ => 0,
    };

    let (high, low) = if rank_order(r1) >= rank_order(r2) {
        (r1, r2)
    } else {
        (r2, r1)
    };

    buf.push(high);
    buf.push(low);

    if high != low {
        if c1.suit == c2.suit {
            buf.push('s');
        } else {
            buf.push('o');
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::float_cmp, clippy::items_after_statements)]
    use super::*;
    use crate::poker::{Suit, Value};
    use test_macros::timed_test;

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

    #[timed_test]
    fn postflop_config_default_values() {
        let config = PostflopConfig::default();
        assert_eq!(config.stack_depth, 100);
        assert_eq!(config.bet_sizes, vec![0.33, 0.5, 0.75, 1.0]);
        assert_eq!(config.samples_per_iteration, 1000);
    }

    #[timed_test]
    fn postflop_config_custom_values() {
        let config = PostflopConfig {
            stack_depth: 50,
            bet_sizes: vec![0.5, 1.0],
            samples_per_iteration: 500,
            ..PostflopConfig::default()
        };
        assert_eq!(config.stack_depth, 50);
        assert_eq!(config.bet_sizes.len(), 2);
        assert_eq!(config.samples_per_iteration, 500);
    }

    #[timed_test]
    fn new_preflop_sets_correct_pot() {
        let (p1, p2) = sample_holdings();
        let state = PostflopState::new_preflop(p1, p2, 100);

        // Pot should be SB (50) + BB (100) = 150 cents
        assert_eq!(state.pot, 150);
    }

    #[timed_test]
    fn new_preflop_sets_correct_stacks() {
        let (p1, p2) = sample_holdings();
        let state = PostflopState::new_preflop(p1, p2, 100);

        // SB posted 50 cents: 10000 - 50 = 9950
        // BB posted 100 cents: 10000 - 100 = 9900
        assert_eq!(state.stacks[0], 9950);
        assert_eq!(state.stacks[1], 9900);
    }

    #[timed_test]
    fn new_preflop_sb_acts_first() {
        let (p1, p2) = sample_holdings();
        let state = PostflopState::new_preflop(p1, p2, 100);

        assert_eq!(state.to_act, Some(Player::Player1));
    }

    #[timed_test]
    fn new_preflop_to_call_is_half_bb() {
        let (p1, p2) = sample_holdings();
        let state = PostflopState::new_preflop(p1, p2, 100);

        // SB needs to call 50 more cents to match BB's 100
        assert_eq!(state.to_call, 50);
    }

    #[timed_test]
    fn new_preflop_street_is_preflop() {
        let (p1, p2) = sample_holdings();
        let state = PostflopState::new_preflop(p1, p2, 100);

        assert_eq!(state.street, Street::Preflop);
    }

    #[timed_test]
    fn new_preflop_board_is_empty() {
        let (p1, p2) = sample_holdings();
        let state = PostflopState::new_preflop(p1, p2, 100);

        assert!(state.board.is_empty());
    }

    #[timed_test]
    fn new_preflop_is_not_terminal() {
        let (p1, p2) = sample_holdings();
        let state = PostflopState::new_preflop(p1, p2, 100);

        assert!(state.terminal.is_none());
    }

    #[timed_test]
    fn new_preflop_history_is_empty() {
        let (p1, p2) = sample_holdings();
        let state = PostflopState::new_preflop(p1, p2, 100);

        assert!(state.history.is_empty());
    }

    #[timed_test]
    fn new_preflop_street_bets_is_one() {
        let (p1, p2) = sample_holdings();
        let state = PostflopState::new_preflop(p1, p2, 100);

        // BB's post counts as first bet
        assert_eq!(state.street_bets, 1);
    }

    #[timed_test]
    fn current_stack_returns_sb_stack_when_sb_to_act() {
        let (p1, p2) = sample_holdings();
        let state = PostflopState::new_preflop(p1, p2, 100);

        // SB acts first, should return SB's stack
        assert_eq!(state.current_stack(), 9950);
    }

    #[timed_test]
    fn current_stack_returns_bb_stack_when_bb_to_act() {
        let (p1, p2) = sample_holdings();
        let mut state = PostflopState::new_preflop(p1, p2, 100);
        state.to_act = Some(Player::Player2);

        assert_eq!(state.current_stack(), 9900);
    }

    #[timed_test]
    fn opponent_stack_returns_bb_stack_when_sb_to_act() {
        let (p1, p2) = sample_holdings();
        let state = PostflopState::new_preflop(p1, p2, 100);

        // SB acts first, opponent is BB
        assert_eq!(state.opponent_stack(), 9900);
    }

    #[timed_test]
    fn opponent_stack_returns_sb_stack_when_bb_to_act() {
        let (p1, p2) = sample_holdings();
        let mut state = PostflopState::new_preflop(p1, p2, 100);
        state.to_act = Some(Player::Player2);

        assert_eq!(state.opponent_stack(), 9950);
    }

    #[timed_test]
    fn current_holding_returns_p1_holding_when_sb_to_act() {
        let (p1, p2) = sample_holdings();
        let state = PostflopState::new_preflop(p1, p2, 100);

        assert_eq!(state.current_holding(), p1);
    }

    #[timed_test]
    fn current_holding_returns_p2_holding_when_bb_to_act() {
        let (p1, p2) = sample_holdings();
        let mut state = PostflopState::new_preflop(p1, p2, 100);
        state.to_act = Some(Player::Player2);

        assert_eq!(state.current_holding(), p2);
    }

    #[timed_test]
    fn terminal_type_fold_equality() {
        let fold_p1 = TerminalType::Fold(Player::Player1);
        let fold_p1_again = TerminalType::Fold(Player::Player1);
        let fold_p2 = TerminalType::Fold(Player::Player2);

        assert_eq!(fold_p1, fold_p1_again);
        assert_ne!(fold_p1, fold_p2);
    }

    #[timed_test]
    fn terminal_type_showdown_equality() {
        let showdown1 = TerminalType::Showdown;
        let showdown2 = TerminalType::Showdown;

        assert_eq!(showdown1, showdown2);
    }

    #[timed_test]
    fn postflop_state_stores_holdings() {
        let (p1, p2) = sample_holdings();
        let state = PostflopState::new_preflop(p1, p2, 100);

        assert_eq!(state.p1_holding, p1);
        assert_eq!(state.p2_holding, p2);
    }

    #[timed_test]
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

    #[timed_test]
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

    #[timed_test]
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

    #[timed_test]
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

    #[timed_test]
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

    #[timed_test]
    fn initial_states_returns_random_deals() {
        let config = PostflopConfig {
            samples_per_iteration: 50,
            ..PostflopConfig::default()
        };
        let game = HunlPostflop::new(config, None);
        let states = game.initial_states();

        assert_eq!(states.len(), 50, "Should have 50 random deals");

        for state in &states {
            assert_eq!(state.street, Street::Preflop);
            assert!(state.terminal.is_none());
            assert_eq!(state.to_act, Some(Player::Player1));
            assert!(
                state.full_board.is_some(),
                "Random deal should have full_board"
            );
        }
    }

    #[timed_test]
    fn all_canonical_hands_generates_169() {
        let states = HunlPostflop::all_canonical_hands(100);
        assert_eq!(states.len(), 169, "Should have 169 canonical hands");
    }

    #[timed_test]
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
            "P1 should lose 0.5 BB, got {p1_utility}"
        );
        assert!(
            (p2_utility - 0.5).abs() < 0.01,
            "P2 should win 0.5 BB, got {p2_utility}"
        );
    }

    #[timed_test]
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
            "Fold utilities should be zero-sum: {u1} + {u2} = {}",
            u1 + u2
        );
    }

    #[timed_test]
    fn info_set_key_includes_hand_and_history() {
        let (p1, p2) = sample_holdings();
        let state = PostflopState::new_preflop(p1, p2, 100);
        let game = create_game();

        let info_set = game.info_set_key(&state);
        // Should contain hand chars (AK) and street (P)
        assert!(
            info_set.contains("|P|"),
            "Info set should contain preflop marker: {info_set}"
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
            "Info set should contain raise history: {info_set2}"
        );
    }

    #[timed_test]
    fn get_bet_sizes_includes_all_in() {
        let game = create_game();
        let pot = 150; // 1.5 BB
        let stack = 500; // 5 BB remaining

        let sizes = game.get_bet_sizes(pot, stack);

        // Should include all-in (500)
        assert!(
            sizes.contains(&stack),
            "Sizes should include all-in: {sizes:?}"
        );
    }

    #[timed_test]
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

    #[timed_test]
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

    #[timed_test]
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

    #[timed_test]
    fn showdown_utility_uses_hand_evaluation() {
        // P1 has AA (Aces)
        let p1 = [
            make_card(Value::Ace, Suit::Spade),
            make_card(Value::Ace, Suit::Heart),
        ];
        // P2 has 72o (Seven-Two offsuit - worst hand)
        let p2 = [
            make_card(Value::Seven, Suit::Club),
            make_card(Value::Two, Suit::Diamond),
        ];

        // Create a 5-card board that doesn't help P2
        // Board: Kh Qd Jc 8s 3h (no straight, flush, or pair for 72o)
        let board = ArrayVec::from([
            make_card(Value::King, Suit::Heart),
            make_card(Value::Queen, Suit::Diamond),
            make_card(Value::Jack, Suit::Club),
            make_card(Value::Eight, Suit::Spade),
            make_card(Value::Three, Suit::Heart),
        ]);

        let mut state = PostflopState::new_preflop(p1, p2, 100);
        state.board = board;
        state.street = Street::River;
        state.terminal = Some(TerminalType::Showdown);
        state.to_act = None;

        let game = create_game();

        // P1 should win with pair of Aces vs P2's high card Seven
        let p1_utility = game.utility(&state, Player::Player1);
        let p2_utility = game.utility(&state, Player::Player2);

        // P1 wins, so P1 utility should be positive
        assert!(
            p1_utility > 0.0,
            "P1 with AA should have positive utility, got {p1_utility}"
        );

        // P2 loses, so P2 utility should be negative
        assert!(
            p2_utility < 0.0,
            "P2 with 72o should have negative utility, got {p2_utility}"
        );

        // Utilities should be zero-sum
        assert!(
            (p1_utility + p2_utility).abs() < 0.001,
            "Utilities should be zero-sum: {p1_utility} + {p2_utility} = {}",
            p1_utility + p2_utility
        );
    }

    #[timed_test]
    fn showdown_utility_tie_splits_pot() {
        // Both players have same pocket pair
        let p1 = [
            make_card(Value::Nine, Suit::Spade),
            make_card(Value::Nine, Suit::Heart),
        ];
        let p2 = [
            make_card(Value::Nine, Suit::Club),
            make_card(Value::Nine, Suit::Diamond),
        ];

        // Board with higher cards - both make same hand
        // Board: AKQJT (broadway) - both have same straight
        let board = ArrayVec::from([
            make_card(Value::Ace, Suit::Spade),
            make_card(Value::King, Suit::Diamond),
            make_card(Value::Queen, Suit::Club),
            make_card(Value::Jack, Suit::Heart),
            make_card(Value::Ten, Suit::Spade),
        ]);

        let mut state = PostflopState::new_preflop(p1, p2, 100);
        state.board = board;
        state.street = Street::River;
        state.terminal = Some(TerminalType::Showdown);
        state.to_act = None;

        let game = create_game();

        let p1_utility = game.utility(&state, Player::Player1);
        let p2_utility = game.utility(&state, Player::Player2);

        // Both should get zero (split pot equals their investment)
        // P1 invested 0.5BB (50 cents), P2 invested 1BB (100 cents)
        // With a tie, pot is split - this test verifies zero-sum property
        assert!(
            (p1_utility + p2_utility).abs() < 0.001,
            "Tie utilities should be zero-sum: {p1_utility} + {p2_utility} = {}",
            p1_utility + p2_utility
        );
    }

    // ==================== Board dealing tests ====================

    fn sample_board() -> [Card; 5] {
        [
            make_card(Value::King, Suit::Heart),
            make_card(Value::Queen, Suit::Diamond),
            make_card(Value::Jack, Suit::Club),
            make_card(Value::Eight, Suit::Spade),
            make_card(Value::Three, Suit::Heart),
        ]
    }

    #[timed_test]
    fn new_preflop_with_board_stores_full_board() {
        let (p1, p2) = sample_holdings();
        let board = sample_board();
        let state = PostflopState::new_preflop_with_board(p1, p2, board, 100);

        assert_eq!(state.full_board, Some(board));
        assert!(state.board.is_empty(), "Board should be empty at preflop");
        assert_eq!(state.street, Street::Preflop);
    }

    #[timed_test]
    fn advance_street_reveals_flop_cards() {
        let (p1, p2) = sample_holdings();
        let board = sample_board();
        let state = PostflopState::new_preflop_with_board(p1, p2, board, 100);
        let game = create_game();

        // SB calls, BB checks → advance to flop
        let after_limp = game.next_state(&state, Action::Call);
        let after_bb_check = game.next_state(&after_limp, Action::Check);

        assert_eq!(after_bb_check.street, Street::Flop);
        assert_eq!(after_bb_check.board.len(), 3, "Flop should have 3 cards");
        assert_eq!(after_bb_check.board[0], board[0]);
        assert_eq!(after_bb_check.board[1], board[1]);
        assert_eq!(after_bb_check.board[2], board[2]);
    }

    #[timed_test]
    fn advance_street_reveals_turn_card() {
        let (p1, p2) = sample_holdings();
        let board = sample_board();
        let state = PostflopState::new_preflop_with_board(p1, p2, board, 100);
        let game = create_game();

        // Limp → BB check → flop check-check → turn
        let s = game.next_state(&state, Action::Call);
        let s = game.next_state(&s, Action::Check); // → flop
        let s = game.next_state(&s, Action::Check);
        let s = game.next_state(&s, Action::Check); // → turn

        assert_eq!(s.street, Street::Turn);
        assert_eq!(s.board.len(), 4, "Turn should have 4 cards");
        assert_eq!(s.board[3], board[3]);
    }

    #[timed_test]
    fn advance_street_reveals_river_card() {
        let (p1, p2) = sample_holdings();
        let board = sample_board();
        let state = PostflopState::new_preflop_with_board(p1, p2, board, 100);
        let game = create_game();

        // Limp → BB check → check-check (flop) → check-check (turn) → river
        let s = game.next_state(&state, Action::Call);
        let s = game.next_state(&s, Action::Check); // → flop
        let s = game.next_state(&s, Action::Check);
        let s = game.next_state(&s, Action::Check); // → turn
        let s = game.next_state(&s, Action::Check);
        let s = game.next_state(&s, Action::Check); // → river

        assert_eq!(s.street, Street::River);
        assert_eq!(s.board.len(), 5, "River should have 5 cards");
        assert_eq!(s.board[4], board[4]);
    }

    #[timed_test]
    fn full_game_check_through_reaches_showdown() {
        let (p1, p2) = sample_holdings();
        let board = sample_board();
        let state = PostflopState::new_preflop_with_board(p1, p2, board, 100);
        let game = create_game();

        // Play through all 4 streets with limp + check-through
        let s = game.next_state(&state, Action::Call); // SB limps
        let s = game.next_state(&s, Action::Check); // BB checks → flop
        let s = game.next_state(&s, Action::Check); // SB checks
        let s = game.next_state(&s, Action::Check); // BB checks → turn
        let s = game.next_state(&s, Action::Check); // SB checks
        let s = game.next_state(&s, Action::Check); // BB checks → river
        let s = game.next_state(&s, Action::Check); // SB checks
        let s = game.next_state(&s, Action::Check); // BB checks → showdown

        assert!(
            game.is_terminal(&s),
            "Should reach terminal after all streets"
        );
        assert_eq!(s.terminal, Some(TerminalType::Showdown));
        assert_eq!(
            s.board.len(),
            5,
            "Board should have all 5 cards at showdown"
        );
    }

    #[timed_test]
    fn showdown_with_dealt_board_correct_winner() {
        // P1 has AA
        let p1 = [
            make_card(Value::Ace, Suit::Spade),
            make_card(Value::Ace, Suit::Heart),
        ];
        // P2 has 72o
        let p2 = [
            make_card(Value::Seven, Suit::Club),
            make_card(Value::Two, Suit::Diamond),
        ];
        // Board that doesn't help 72o
        let board = [
            make_card(Value::King, Suit::Heart),
            make_card(Value::Queen, Suit::Diamond),
            make_card(Value::Jack, Suit::Club),
            make_card(Value::Eight, Suit::Spade),
            make_card(Value::Three, Suit::Heart),
        ];

        let state = PostflopState::new_preflop_with_board(p1, p2, board, 100);
        let game = create_game();

        // Play through to showdown
        let s = game.next_state(&state, Action::Call);
        let s = game.next_state(&s, Action::Check);
        let s = game.next_state(&s, Action::Check);
        let s = game.next_state(&s, Action::Check);
        let s = game.next_state(&s, Action::Check);
        let s = game.next_state(&s, Action::Check);
        let s = game.next_state(&s, Action::Check);
        let s = game.next_state(&s, Action::Check);

        assert!(game.is_terminal(&s));
        let p1_util = game.utility(&s, Player::Player1);
        assert!(p1_util > 0.0, "AA should beat 72o, got {p1_util}");
    }

    #[timed_test]
    fn random_deals_no_card_conflicts() {
        let config = PostflopConfig {
            samples_per_iteration: 100,
            ..PostflopConfig::default()
        };
        let game = HunlPostflop::new(config, None);
        let deals = game.initial_states();

        for (i, deal) in deals.iter().enumerate() {
            let board = deal.full_board.expect("should have board");
            let mut all_cards = vec![
                deal.p1_holding[0],
                deal.p1_holding[1],
                deal.p2_holding[0],
                deal.p2_holding[1],
                board[0],
                board[1],
                board[2],
                board[3],
                board[4],
            ];
            let orig_len = all_cards.len();
            all_cards.sort_by_key(|c| (c.value as u8, c.suit as u8));
            all_cards.dedup();
            assert_eq!(all_cards.len(), orig_len, "Deal {i} has duplicate cards");
        }
    }

    #[timed_test]
    fn random_deals_deterministic() {
        let config = PostflopConfig {
            samples_per_iteration: 10,
            ..PostflopConfig::default()
        };
        let game = HunlPostflop::new(config, None);

        let deals1 = game.initial_states();
        let deals2 = game.initial_states();

        for (d1, d2) in deals1.iter().zip(deals2.iter()) {
            assert_eq!(d1.p1_holding, d2.p1_holding);
            assert_eq!(d1.p2_holding, d2.p2_holding);
            assert_eq!(d1.full_board, d2.full_board);
        }
    }

    #[timed_test]
    fn new_preflop_has_none_full_board() {
        let (p1, p2) = sample_holdings();
        let state = PostflopState::new_preflop(p1, p2, 100);
        assert!(
            state.full_board.is_none(),
            "Plain new_preflop should have no full_board"
        );
    }

    #[timed_test]
    fn advance_street_without_board_leaves_board_empty() {
        let (p1, p2) = sample_holdings();
        let state = PostflopState::new_preflop(p1, p2, 100);
        let game = create_game();

        // SB calls → BB checks → flop (no board cards dealt without full_board)
        let s = game.next_state(&state, Action::Call);
        let s = game.next_state(&s, Action::Check);

        assert_eq!(s.street, Street::Flop);
        assert!(
            s.board.is_empty(),
            "Board should stay empty without full_board"
        );
    }
}
