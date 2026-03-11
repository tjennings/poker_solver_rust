//! Subgame solver configuration.
//!
//! Provides the `SubgameConfig` used by the `SubgameCfrSolver` and other
//! subgame-related components.

/// Configuration for subgame solving
#[derive(Debug, Clone)]
pub struct SubgameConfig {
    /// Maximum depth to search in the game tree
    pub depth_limit: usize,
    /// Time budget for solving in milliseconds
    pub time_budget_ms: u64,
    /// Maximum number of CFR iterations
    pub max_iterations: u32,
}

impl Default for SubgameConfig {
    fn default() -> Self {
        Self {
            depth_limit: 4,
            time_budget_ms: 200,
            max_iterations: 1000,
        }
    }
}
