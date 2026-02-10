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
pub mod blueprint;
pub mod cfr;
pub mod equity;
pub mod error;
pub mod flops;
pub mod game;
pub mod hand_class;
pub mod hands;
pub mod info_key;
pub mod showdown_equity;
pub mod simulation;

pub use agent::AgentConfig;
pub use error::SolverError;
pub use flops::{CanonicalFlop, HighCardClass, RankTexture, SuitTexture, all_flops};
pub use game::{Action, Game, Player};
pub use hand_class::{ClassifyError, HandClass, HandClassification, classify, intra_class_strength};
pub use hands::{CanonicalHand, HandType, all_hands};

/// Re-exported poker domain types from the `rs_poker` crate.
///
/// These types provide efficient representations for cards, hands, and hand evaluation.
pub mod poker {
    pub use rs_poker::core::{Card, CardIter, Deck, FlatDeck, Hand, Rank, Rankable, Suit, Value};

    /// All 13 card values in ascending order (Two through Ace).
    pub const ALL_VALUES: [Value; 13] = [
        Value::Two, Value::Three, Value::Four, Value::Five,
        Value::Six, Value::Seven, Value::Eight, Value::Nine,
        Value::Ten, Value::Jack, Value::Queen, Value::King, Value::Ace,
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
}
