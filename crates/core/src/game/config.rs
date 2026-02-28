//! Postflop game configuration types.
//!
//! Extracted from `hunl_postflop.rs` so they remain available after the
//! standalone MCCFR game implementations are removed.

use std::sync::Arc;

use serde::{Deserialize, Serialize};

use crate::abstraction::CardAbstraction;

/// Selects which card abstraction to use for postflop info-set keys.
#[derive(Debug, Clone)]
pub enum AbstractionMode {
    /// EHS2-based bucketing (expensive Monte Carlo, fine-grained).
    Ehs2(Arc<CardAbstraction>),
    /// Hand-class V2: class ID + intra-class strength + equity bin + draw flags.
    ///
    /// `strength_bits` and `equity_bits` control quantization (0-4 each).
    /// 0 means that dimension is omitted. With both set to 0, this is equivalent
    /// to the old `HandClass` mode (class ID only).
    HandClassV2 {
        /// Number of bits for intra-class strength (0-4).
        strength_bits: u8,
        /// Number of bits for equity bin (0-4).
        equity_bits: u8,
    },
}

/// Configuration for the postflop game.
///
/// Controls stack depth and bet sizing options.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PostflopConfig {
    /// Stack depth in big blinds
    pub stack_depth: u32,
    /// Available bet sizes as fractions of pot (e.g., 0.5 = half pot)
    pub bet_sizes: Vec<f32>,
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
            max_raises_per_street: 3,
        }
    }
}

fn default_max_raises() -> u8 {
    3
}
