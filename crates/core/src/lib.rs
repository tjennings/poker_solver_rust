#![warn(clippy::all)]
#![warn(clippy::pedantic)]

//! Poker Solver Core Library
//!
//! A CFR (Counterfactual Regret Minimization) solver for poker games.
//!
//! # Modules
//!
//! - `game` - Game trait and implementations (Kuhn Poker, HUNL Postflop)
//! - `cfr` - CFR solver implementations (Vanilla, MCCFR)
//! - `error` - Error types
//! - `poker` - Re-exported poker domain types from `rs_poker`

pub mod abstraction;
pub mod agent;
pub mod blueprint_mp;
pub mod blueprint_v2;
pub mod card_utils;
pub mod cfr;
pub mod equity;
pub mod error;
pub mod flops;
pub mod game;
pub mod hand_class;
pub mod hands;
pub mod info_key;
pub mod range;
pub mod showdown_equity;
pub mod simulation;

pub use agent::AgentConfig;
pub use error::SolverError;
pub use flops::{CanonicalFlop, HighCardClass, RankTexture, SuitTexture, all_flops};
pub use game::{Action, Player};
pub use hand_class::{
    ClassifyError, HandClass, HandClassification, classify, intra_class_strength,
};
pub use hands::{CanonicalHand, HandType, all_hands};

/// Re-exported poker domain types from the `rs_poker` crate.
///
/// These types provide efficient representations for cards, hands, and hand evaluation.
pub mod poker {
    pub use rs_poker::core::{Card, CardIter, Deck, FlatDeck, Hand, Rank, Rankable, Suit, Value};

    /// All 13 card values in ascending order (Two through Ace).
    pub const ALL_VALUES: [Value; 13] = [
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

    /// All 4 suits.
    pub const ALL_SUITS: [Suit; 4] = [Suit::Spade, Suit::Heart, Suit::Diamond, Suit::Club];

    /// Generate a standard 52-card deck (value-major order).
    #[must_use]
    pub fn full_deck() -> Vec<Card> {
        let mut deck = Vec::with_capacity(52);
        for &value in &ALL_VALUES {
            for &suit in &ALL_SUITS {
                deck.push(Card::new(value, suit));
            }
        }
        deck
    }

    /// Parse a card from a two-character string like "Ah", "2c", "Td".
    #[must_use]
    pub fn parse_card(s: &str) -> Option<Card> {
        let mut chars = s.chars();
        let value = match chars.next()? {
            '2' => Value::Two,
            '3' => Value::Three,
            '4' => Value::Four,
            '5' => Value::Five,
            '6' => Value::Six,
            '7' => Value::Seven,
            '8' => Value::Eight,
            '9' => Value::Nine,
            'T' | 't' => Value::Ten,
            'J' | 'j' => Value::Jack,
            'Q' | 'q' => Value::Queen,
            'K' | 'k' => Value::King,
            'A' | 'a' => Value::Ace,
            _ => return None,
        };
        let suit = match chars.next()? {
            'h' | 'H' => Suit::Heart,
            'd' | 'D' => Suit::Diamond,
            'c' | 'C' => Suit::Club,
            's' | 'S' => Suit::Spade,
            _ => return None,
        };
        Some(Card::new(value, suit))
    }
}
