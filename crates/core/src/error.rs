use thiserror::Error;

/// Errors that can occur in the solver
#[derive(Debug, Error)]
pub enum SolverError {
    #[error("Invalid game state: {0}")]
    InvalidState(String),

    #[error("Configuration error: {0}")]
    Config(String),

    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("batch serialization error: {0}")]
    BatchSerialize(String),

    #[error("batch version mismatch: expected {expected}, got {actual}")]
    BatchVersionMismatch { expected: u32, actual: u32 },

    #[error("no batch files found in {0}")]
    NoBatchFiles(String),
}
