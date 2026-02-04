use thiserror::Error;

/// Errors that can occur in the solver
#[derive(Debug, Error)]
pub enum SolverError {
    #[error("Invalid game state: {0}")]
    InvalidState(String),

    #[error("Configuration error: {0}")]
    Config(String),
}
