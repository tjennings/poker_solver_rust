use crate::poker::Card;
use thiserror::Error;

/// Errors that can occur in card abstraction operations
#[derive(Debug, Error)]
pub enum AbstractionError {
    /// The board has an invalid number of cards for the expected street
    #[error("Invalid board size: expected {expected} cards, got {got}")]
    InvalidBoardSize { expected: usize, got: usize },

    /// A card appears more than once (in hand + board combination)
    #[error("Duplicate card: {0}")]
    DuplicateCard(Card),

    /// Failed to load abstraction data from disk
    #[error("Failed to load abstraction: {0}")]
    LoadError(#[from] std::io::Error),

    /// The bucket boundaries are invalid (not sorted, wrong count, etc.)
    #[error("Invalid boundary data: {reason}")]
    InvalidBoundaries { reason: String },

    /// Serialization or deserialization failed
    #[error("Serialization error: {0}")]
    SerializationError(String),
}

#[cfg(test)]
mod tests {
    use super::*;
    use test_macros::timed_test;

    #[timed_test]
    fn invalid_board_size_error_displays_expected_and_got() {
        let err = AbstractionError::InvalidBoardSize {
            expected: 3,
            got: 2,
        };
        let msg = err.to_string();
        assert!(msg.contains("expected 3"), "should show expected count");
        assert!(msg.contains("got 2"), "should show actual count");
    }

    #[timed_test]
    fn invalid_boundaries_error_includes_reason() {
        let err = AbstractionError::InvalidBoundaries {
            reason: "boundaries not sorted".to_string(),
        };
        let msg = err.to_string();
        assert!(
            msg.contains("boundaries not sorted"),
            "should include the reason"
        );
    }

    #[timed_test]
    fn serialization_error_includes_message() {
        let err = AbstractionError::SerializationError("invalid JSON".to_string());
        let msg = err.to_string();
        assert!(msg.contains("invalid JSON"), "should include the message");
    }

    #[timed_test]
    fn load_error_converts_from_io_error() {
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "file not found");
        let err: AbstractionError = io_err.into();
        let msg = err.to_string();
        assert!(msg.contains("file not found"), "should include io error");
    }
}
