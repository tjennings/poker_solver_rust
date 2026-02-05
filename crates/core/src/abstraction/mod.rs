//! Card Abstraction Module
//!
//! Provides EHS2-based card abstraction for poker hand bucketing.
//! This module enables grouping similar hands into buckets to reduce
//! the game tree size for CFR solving.

mod buckets;
mod error;
mod hand_strength;
mod isomorphism;

use crate::poker::Card;
use std::collections::HashSet;
use std::path::Path;

pub use buckets::{BucketAssigner, BucketBoundaries};
pub use error::AbstractionError;
pub use hand_strength::{HandStrength, HandStrengthCalculator};
pub use isomorphism::{CanonicalBoard, CanonicalSuit, SuitMapping};

/// Street in poker (determines bucket count and calculation method)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Street {
    /// Flop: 3 community cards
    Flop,
    /// Turn: 4 community cards
    Turn,
    /// River: 5 community cards
    River,
}

impl Street {
    /// Determine street from board card count.
    ///
    /// # Arguments
    /// * `len` - Number of community cards on the board
    ///
    /// # Errors
    /// Returns `AbstractionError::InvalidBoardSize` if `len` is not 3, 4, or 5.
    ///
    /// # Examples
    /// ```
    /// use poker_solver_core::abstraction::Street;
    ///
    /// assert_eq!(Street::from_board_len(3).unwrap(), Street::Flop);
    /// assert_eq!(Street::from_board_len(4).unwrap(), Street::Turn);
    /// assert_eq!(Street::from_board_len(5).unwrap(), Street::River);
    /// assert!(Street::from_board_len(2).is_err());
    /// ```
    pub fn from_board_len(len: usize) -> Result<Self, AbstractionError> {
        match len {
            3 => Ok(Street::Flop),
            4 => Ok(Street::Turn),
            5 => Ok(Street::River),
            n => Err(AbstractionError::InvalidBoardSize {
                expected: 3, // Use 3 as the "expected" since that's the minimum valid
                got: n,
            }),
        }
    }

    /// Returns the number of community cards for this street.
    #[must_use]
    pub const fn board_cards(self) -> usize {
        match self {
            Street::Flop => 3,
            Street::Turn => 4,
            Street::River => 5,
        }
    }
}

/// Configuration for abstraction generation
#[derive(Debug, Clone)]
pub struct AbstractionConfig {
    /// Number of buckets to use on the flop
    pub flop_buckets: u16,
    /// Number of buckets to use on the turn
    pub turn_buckets: u16,
    /// Number of buckets to use on the river
    pub river_buckets: u16,
    /// Number of samples to use when computing bucket boundaries
    pub samples_per_street: u32,
}

impl Default for AbstractionConfig {
    fn default() -> Self {
        Self {
            flop_buckets: 5_000,
            turn_buckets: 5_000,
            river_buckets: 20_000,
            samples_per_street: 100_000,
        }
    }
}

/// Main entry point for card abstraction.
///
/// This struct ties together bucket boundaries and hand strength calculation
/// to provide a unified API for mapping hands to buckets.
pub struct CardAbstraction {
    assigner: BucketAssigner,
    calculator: HandStrengthCalculator,
}

impl CardAbstraction {
    /// Create from precomputed boundaries.
    ///
    /// # Arguments
    /// * `boundaries` - Precomputed bucket boundaries for each street
    #[must_use]
    pub fn from_boundaries(boundaries: BucketBoundaries) -> Self {
        Self {
            assigner: BucketAssigner::new(boundaries),
            calculator: HandStrengthCalculator::new(),
        }
    }

    /// Load boundaries from file.
    ///
    /// # Arguments
    /// * `path` - Path to the MessagePack-encoded boundaries file
    ///
    /// # Errors
    /// Returns `AbstractionError::LoadError` if the file cannot be read,
    /// or `AbstractionError::SerializationError` if deserialization fails.
    pub fn load(path: &Path) -> Result<Self, AbstractionError> {
        let data = std::fs::read(path)?;
        let boundaries: BucketBoundaries = rmp_serde::from_slice(&data)
            .map_err(|e| AbstractionError::SerializationError(e.to_string()))?;
        Ok(Self::from_boundaries(boundaries))
    }

    /// Save boundaries to file.
    ///
    /// # Arguments
    /// * `path` - Path where the MessagePack-encoded boundaries will be written
    ///
    /// # Errors
    /// Returns `AbstractionError::LoadError` if the file cannot be written,
    /// or `AbstractionError::SerializationError` if serialization fails.
    pub fn save(&self, path: &Path) -> Result<(), AbstractionError> {
        let data = rmp_serde::to_vec(self.assigner.boundaries())
            .map_err(|e| AbstractionError::SerializationError(e.to_string()))?;
        std::fs::write(path, data)?;
        Ok(())
    }

    /// Get bucket for a hand on a board.
    ///
    /// This is the main API for card abstraction. Given a board and hole cards,
    /// it computes the EHS2 value and maps it to a bucket index.
    ///
    /// # Arguments
    /// * `board` - The community cards (3 for flop, 4 for turn, 5 for river)
    /// * `holding` - The player's two hole cards
    ///
    /// # Errors
    /// Returns an error if:
    /// - The board has an invalid number of cards
    /// - Any card appears more than once (board + holding)
    pub fn get_bucket(
        &self,
        board: &[Card],
        holding: (Card, Card),
    ) -> Result<u16, AbstractionError> {
        let street = Street::from_board_len(board.len())?;

        // Check for duplicate cards
        let mut seen = HashSet::new();
        for &card in board {
            if !seen.insert(card) {
                return Err(AbstractionError::DuplicateCard(card));
            }
        }
        if !seen.insert(holding.0) {
            return Err(AbstractionError::DuplicateCard(holding.0));
        }
        if !seen.insert(holding.1) {
            return Err(AbstractionError::DuplicateCard(holding.1));
        }

        // Canonicalize board and holding
        let canonical_board = CanonicalBoard::from_cards(board)?;
        let canonical_holding = canonical_board.canonicalize_holding(holding.0, holding.1);

        // Calculate EHS2
        let hs = match street {
            Street::Flop => self
                .calculator
                .calculate_flop(&canonical_board.cards, canonical_holding),
            Street::Turn => self
                .calculator
                .calculate_turn(&canonical_board.cards, canonical_holding),
            Street::River => self
                .calculator
                .calculate_river(&canonical_board.cards, canonical_holding),
        };

        Ok(self.assigner.get_bucket(street, hs.ehs2))
    }

    /// Get number of buckets for a street.
    #[must_use]
    pub fn num_buckets(&self, street: Street) -> usize {
        self.assigner.num_buckets(street)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::poker::{Suit, Value};

    #[test]
    fn street_from_board_len_flop() {
        let result = Street::from_board_len(3);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), Street::Flop);
    }

    #[test]
    fn street_from_board_len_turn() {
        let result = Street::from_board_len(4);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), Street::Turn);
    }

    #[test]
    fn street_from_board_len_river() {
        let result = Street::from_board_len(5);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), Street::River);
    }

    #[test]
    fn street_from_board_len_invalid_zero() {
        let result = Street::from_board_len(0);
        assert!(result.is_err());
        match result {
            Err(AbstractionError::InvalidBoardSize { expected: _, got }) => {
                assert_eq!(got, 0);
            }
            _ => panic!("Expected InvalidBoardSize error"),
        }
    }

    #[test]
    fn street_from_board_len_invalid_two() {
        let result = Street::from_board_len(2);
        assert!(result.is_err());
    }

    #[test]
    fn street_from_board_len_invalid_six() {
        let result = Street::from_board_len(6);
        assert!(result.is_err());
    }

    #[test]
    fn street_board_cards_returns_correct_count() {
        assert_eq!(Street::Flop.board_cards(), 3);
        assert_eq!(Street::Turn.board_cards(), 4);
        assert_eq!(Street::River.board_cards(), 5);
    }

    #[test]
    fn street_is_copy_and_clone() {
        let flop = Street::Flop;
        let flop_copy = flop;
        let flop_clone = flop.clone();
        assert_eq!(flop, flop_copy);
        assert_eq!(flop, flop_clone);
    }

    #[test]
    fn street_implements_debug() {
        let debug_str = format!("{:?}", Street::Flop);
        assert_eq!(debug_str, "Flop");
    }

    #[test]
    fn card_abstraction_river_bucket_lookup() {
        // Create simple boundaries for testing
        let boundaries = BucketBoundaries {
            flop: (1..100).map(|i| i as f32 / 100.0).collect(),
            turn: (1..100).map(|i| i as f32 / 100.0).collect(),
            river: (1..100).map(|i| i as f32 / 100.0).collect(),
        };
        let abstraction = CardAbstraction::from_boundaries(boundaries);

        // River board
        let board = vec![
            Card::new(Value::Ace, Suit::Spade),
            Card::new(Value::King, Suit::Spade),
            Card::new(Value::Queen, Suit::Heart),
            Card::new(Value::Jack, Suit::Diamond),
            Card::new(Value::Two, Suit::Club),
        ];
        let holding = (
            Card::new(Value::Ten, Suit::Spade),
            Card::new(Value::Nine, Suit::Spade),
        );

        let bucket = abstraction.get_bucket(&board, holding).unwrap();
        // Should return a valid bucket (0-99 for 100 buckets)
        assert!(bucket < 100, "Bucket {} should be < 100", bucket);
    }

    #[test]
    fn card_abstraction_rejects_duplicate_card() {
        let boundaries = BucketBoundaries {
            flop: vec![0.5],
            turn: vec![0.5],
            river: vec![0.5],
        };
        let abstraction = CardAbstraction::from_boundaries(boundaries);

        let board = vec![
            Card::new(Value::Ace, Suit::Spade),
            Card::new(Value::King, Suit::Spade),
            Card::new(Value::Queen, Suit::Heart),
            Card::new(Value::Jack, Suit::Diamond),
            Card::new(Value::Two, Suit::Club),
        ];
        // Holding has Ace of Spades which is on board
        let holding = (
            Card::new(Value::Ace, Suit::Spade),
            Card::new(Value::Nine, Suit::Heart),
        );

        let result = abstraction.get_bucket(&board, holding);
        assert!(result.is_err(), "Should reject duplicate card");
    }

    #[test]
    fn abstraction_config_default() {
        let config = AbstractionConfig::default();
        assert_eq!(config.flop_buckets, 5_000);
        assert_eq!(config.turn_buckets, 5_000);
        assert_eq!(config.river_buckets, 20_000);
        assert_eq!(config.samples_per_street, 100_000);
    }

    #[test]
    fn card_abstraction_num_buckets() {
        let boundaries = BucketBoundaries {
            flop: vec![0.25, 0.5, 0.75], // 4 buckets
            turn: vec![0.5],             // 2 buckets
            river: vec![],               // 1 bucket
        };
        let abstraction = CardAbstraction::from_boundaries(boundaries);

        assert_eq!(abstraction.num_buckets(Street::Flop), 4);
        assert_eq!(abstraction.num_buckets(Street::Turn), 2);
        assert_eq!(abstraction.num_buckets(Street::River), 1);
    }

    #[test]
    fn card_abstraction_rejects_invalid_board_size() {
        let boundaries = BucketBoundaries {
            flop: vec![0.5],
            turn: vec![0.5],
            river: vec![0.5],
        };
        let abstraction = CardAbstraction::from_boundaries(boundaries);

        // 2 cards is invalid
        let board = vec![
            Card::new(Value::Ace, Suit::Spade),
            Card::new(Value::King, Suit::Spade),
        ];
        let holding = (
            Card::new(Value::Queen, Suit::Heart),
            Card::new(Value::Jack, Suit::Diamond),
        );

        let result = abstraction.get_bucket(&board, holding);
        assert!(result.is_err(), "Should reject invalid board size");
    }

    #[test]
    fn card_abstraction_rejects_duplicate_in_holding() {
        let boundaries = BucketBoundaries {
            flop: vec![0.5],
            turn: vec![0.5],
            river: vec![0.5],
        };
        let abstraction = CardAbstraction::from_boundaries(boundaries);

        let board = vec![
            Card::new(Value::Ace, Suit::Spade),
            Card::new(Value::King, Suit::Spade),
            Card::new(Value::Queen, Suit::Heart),
            Card::new(Value::Jack, Suit::Diamond),
            Card::new(Value::Two, Suit::Club),
        ];
        // Holding has same card twice
        let holding = (
            Card::new(Value::Nine, Suit::Heart),
            Card::new(Value::Nine, Suit::Heart),
        );

        let result = abstraction.get_bucket(&board, holding);
        assert!(result.is_err(), "Should reject duplicate card in holding");
    }
}
