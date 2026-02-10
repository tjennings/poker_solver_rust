use thiserror::Error;

/// Errors that can occur in blueprint operations
#[derive(Debug, Error)]
pub enum BlueprintError {
    /// I/O operation failed
    #[error("I/O error: {0}")]
    IoError(String),

    /// Serialization or deserialization failed
    #[error("Serialization error: {0}")]
    SerializationError(String),

    /// Strategy data is invalid or corrupted
    #[error("Invalid strategy: {0}")]
    InvalidStrategy(String),

    /// Cache operation failed
    #[error("Cache error: {0}")]
    CacheError(String),

    /// Information set not found in strategy
    #[error("Information set not found: {0:#018x}")]
    InfoSetNotFound(u64),
}

impl From<std::io::Error> for BlueprintError {
    fn from(err: std::io::Error) -> Self {
        BlueprintError::IoError(err.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use test_macros::timed_test;

    #[timed_test]
    fn io_error_displays_message() {
        let err = BlueprintError::IoError("file not found".to_string());
        let msg = err.to_string();
        assert!(msg.contains("I/O error"), "should contain error type");
        assert!(msg.contains("file not found"), "should contain message");
    }

    #[timed_test]
    fn serialization_error_displays_message() {
        let err = BlueprintError::SerializationError("invalid JSON".to_string());
        let msg = err.to_string();
        assert!(
            msg.contains("Serialization error"),
            "should contain error type"
        );
        assert!(msg.contains("invalid JSON"), "should contain message");
    }

    #[timed_test]
    fn invalid_strategy_error_displays_message() {
        let err = BlueprintError::InvalidStrategy("probabilities don't sum to 1".to_string());
        let msg = err.to_string();
        assert!(
            msg.contains("Invalid strategy"),
            "should contain error type"
        );
        assert!(
            msg.contains("probabilities don't sum to 1"),
            "should contain message"
        );
    }

    #[timed_test]
    fn cache_error_displays_message() {
        let err = BlueprintError::CacheError("cache full".to_string());
        let msg = err.to_string();
        assert!(msg.contains("Cache error"), "should contain error type");
        assert!(msg.contains("cache full"), "should contain message");
    }

    #[timed_test]
    fn info_set_not_found_error_displays_message() {
        let err = BlueprintError::InfoSetNotFound(0x0000_DEAD_BEEF_CAFE);
        let msg = err.to_string();
        assert!(
            msg.contains("Information set not found"),
            "should contain error type"
        );
        assert!(msg.contains("deadbeefcafe"), "should contain hex key: {msg}");
    }

    #[timed_test]
    fn from_io_error_converts_correctly() {
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "file missing");
        let blueprint_err: BlueprintError = io_err.into();

        match blueprint_err {
            BlueprintError::IoError(msg) => {
                assert!(
                    msg.contains("file missing"),
                    "should preserve io error message"
                );
            }
            _ => panic!("Expected IoError variant"),
        }
    }

    #[timed_test]
    fn error_is_debug() {
        let err = BlueprintError::IoError("test".to_string());
        let debug_str = format!("{err:?}");
        assert!(debug_str.contains("IoError"), "should be debuggable");
    }
}
