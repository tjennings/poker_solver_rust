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

pub mod cfr;
pub mod error;
pub mod game;

pub use error::SolverError;
pub use game::{Action, Game, Player};
