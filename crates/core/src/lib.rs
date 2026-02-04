#![deny(clippy::all)]
#![warn(clippy::pedantic)]

//! Poker Solver Core Library
//!
//! A CFR (Counterfactual Regret Minimization) solver for poker games.
//!
//! # Modules
//!
//! - `game` - Game trait and implementations (Kuhn Poker, HUNL Preflop)
//! - `cfr` - CFR solver implementations (Vanilla, Batched)
//! - `error` - Error types
//! - `poker` - Re-exported poker domain types from `rs_poker`

pub mod cfr;
#[cfg(feature = "gpu")]
pub mod device;
pub mod error;
pub mod game;

pub use error::SolverError;
pub use game::{Action, Game, Player};

/// Re-exported poker domain types from the `rs_poker` crate.
///
/// These types provide efficient representations for cards, hands, and hand evaluation.
pub mod poker {
    pub use rs_poker::core::{Card, CardIter, Deck, FlatDeck, Hand, Rank, Rankable, Suit, Value};
}
