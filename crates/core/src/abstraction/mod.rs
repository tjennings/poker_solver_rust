//! Card Abstraction Module
//!
//! Provides EHS2-based card abstraction for poker hand bucketing.
//! This module enables grouping similar hands into buckets to reduce
//! the game tree size for CFR solving.

mod buckets;
mod error;
mod hand_strength;
mod isomorphism;

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

#[cfg(test)]
mod tests {
    use super::*;

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
}
