//! Hybrid solver dispatch logic.
//!
//! Given a street and live combo count, decides whether to use full-depth
//! or depth-limited solving. River always uses full-depth (no future streets).
//! Earlier streets switch to depth-limited when the combo count exceeds a
//! per-street threshold.

use serde::{Deserialize, Serialize};

/// Configuration controlling solver dispatch thresholds and iteration counts.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SolverConfig {
    /// Maximum live combos on the flop before switching to depth-limited.
    pub flop_combo_threshold: usize,
    /// Maximum live combos on the turn before switching to depth-limited.
    pub turn_combo_threshold: usize,
    /// Iteration count for depth-limited solves.
    pub depth_limited_iterations: u32,
    /// Iteration count for full-depth solves.
    pub full_solve_iterations: u32,
    /// Target exploitability (epsilon) for early termination.
    pub target_exploitability: f64,
}

impl Default for SolverConfig {
    fn default() -> Self {
        Self {
            flop_combo_threshold: 200,
            turn_combo_threshold: 300,
            depth_limited_iterations: 200,
            full_solve_iterations: 1000,
            target_exploitability: 0.005,
        }
    }
}

/// Which solver strategy to use for a given subgame.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SolverChoice {
    /// Solve the full game tree to the terminal nodes.
    FullDepth,
    /// Solve to a limited depth, using blueprint values at leaf nodes.
    DepthLimited,
}

/// Postflop street identifier for dispatch decisions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Street {
    Flop,
    Turn,
    River,
}

/// Decide which solver to use based on street and live combo count.
///
/// - **River**: always `FullDepth` (no future streets to truncate).
/// - **Turn**: `FullDepth` if `live_combos <= turn_combo_threshold`, else `DepthLimited`.
/// - **Flop**: `FullDepth` if `live_combos <= flop_combo_threshold`, else `DepthLimited`.
#[must_use]
pub fn dispatch_decision(
    config: &SolverConfig,
    street: Street,
    live_combos: usize,
) -> SolverChoice {
    match street {
        Street::River => SolverChoice::FullDepth,
        Street::Turn => {
            if live_combos <= config.turn_combo_threshold {
                SolverChoice::FullDepth
            } else {
                SolverChoice::DepthLimited
            }
        }
        Street::Flop => {
            if live_combos <= config.flop_combo_threshold {
                SolverChoice::FullDepth
            } else {
                SolverChoice::DepthLimited
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_river_always_full_depth() {
        let config = SolverConfig::default();
        // Even with an absurdly high combo count, river is always full-depth.
        assert_eq!(
            dispatch_decision(&config, Street::River, 1000),
            SolverChoice::FullDepth,
        );
        assert_eq!(
            dispatch_decision(&config, Street::River, 0),
            SolverChoice::FullDepth,
        );
    }

    #[test]
    fn test_flop_wide_depth_limited() {
        let config = SolverConfig {
            flop_combo_threshold: 200,
            ..SolverConfig::default()
        };
        assert_eq!(
            dispatch_decision(&config, Street::Flop, 500),
            SolverChoice::DepthLimited,
        );
    }

    #[test]
    fn test_flop_narrow_full_depth() {
        let config = SolverConfig {
            flop_combo_threshold: 200,
            ..SolverConfig::default()
        };
        assert_eq!(
            dispatch_decision(&config, Street::Flop, 100),
            SolverChoice::FullDepth,
        );
    }

    #[test]
    fn test_turn_at_threshold_boundary() {
        let config = SolverConfig {
            turn_combo_threshold: 300,
            ..SolverConfig::default()
        };
        // Exactly at threshold: full-depth.
        assert_eq!(
            dispatch_decision(&config, Street::Turn, 300),
            SolverChoice::FullDepth,
        );
        // One above threshold: depth-limited.
        assert_eq!(
            dispatch_decision(&config, Street::Turn, 301),
            SolverChoice::DepthLimited,
        );
    }

    #[test]
    fn test_default_config() {
        let config = SolverConfig::default();
        assert_eq!(config.flop_combo_threshold, 200);
        assert_eq!(config.turn_combo_threshold, 300);
        assert_eq!(config.depth_limited_iterations, 200);
        assert_eq!(config.full_solve_iterations, 1000);
        assert!((config.target_exploitability - 0.005).abs() < f64::EPSILON);
    }
}
