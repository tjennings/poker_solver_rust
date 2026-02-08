#![deny(clippy::all)]
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
pub mod config;
pub mod equity;
pub mod error;
pub mod flops;
pub mod game;
pub mod hand_class;
pub mod hands;

pub use agent::AgentConfig;
pub use config::Config;
pub use error::SolverError;
pub use flops::{CanonicalFlop, HighCardClass, RankTexture, SuitTexture, all_flops};
pub use game::{Action, Game, Player};
pub use hand_class::{ClassifyError, HandClass, HandClassification, classify};
pub use hands::{CanonicalHand, HandType, all_hands};

/// Re-exported poker domain types from the `rs_poker` crate.
///
/// These types provide efficient representations for cards, hands, and hand evaluation.
pub mod poker {
    pub use rs_poker::core::{Card, CardIter, Deck, FlatDeck, Hand, Rank, Rankable, Suit, Value};
}
