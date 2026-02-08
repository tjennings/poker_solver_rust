//! Configuration loading for poker solver games.
//!
//! Provides types and functions for loading game configuration from YAML files.
//! Configuration includes bet sizing, stack depths, and game tree parameters.

use std::path::Path;

use serde::Deserialize;
use thiserror::Error;

/// Game configuration loaded from YAML.
///
/// Defines the parameters for a HUNL preflop game tree including
/// bet sizes and constraints.
#[derive(Debug, Clone, Deserialize)]
pub struct Config {
    /// Human-readable name for this configuration
    pub name: String,
    /// Stack depths to solve for (in big blinds)
    pub stack_depths: Vec<u32>,
    /// Available raise sizes (in big blinds)
    pub raise_sizes: Vec<f64>,
    /// Maximum number of bets allowed per round
    #[serde(default = "default_max_bets")]
    pub max_bets_per_round: u8,
}

fn default_max_bets() -> u8 {
    4
}

impl Config {
    /// Load configuration from a YAML file.
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be read or parsed.
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self, ConfigError> {
        let path = path.as_ref();
        let content =
            std::fs::read_to_string(path).map_err(|e| ConfigError::Io(path.to_path_buf(), e))?;
        Self::from_yaml(&content)
    }

    /// Parse configuration from a YAML string.
    ///
    /// # Errors
    ///
    /// Returns an error if the YAML is invalid or missing required fields.
    pub fn from_yaml(yaml: &str) -> Result<Self, ConfigError> {
        let config: Self = serde_yaml::from_str(yaml).map_err(ConfigError::Parse)?;
        config.validate()?;
        Ok(config)
    }

    /// Validate the configuration.
    fn validate(&self) -> Result<(), ConfigError> {
        if self.stack_depths.is_empty() {
            return Err(ConfigError::EmptyStackDepths);
        }

        for &depth in &self.stack_depths {
            if depth == 0 {
                return Err(ConfigError::InvalidStackDepth(depth));
            }
        }

        for &size in &self.raise_sizes {
            if size <= 0.0 {
                return Err(ConfigError::InvalidRaiseSize(size));
            }
        }

        if self.max_bets_per_round == 0 {
            return Err(ConfigError::InvalidMaxBets(0));
        }

        Ok(())
    }

    /// Get legal raise sizes given the current betting state.
    ///
    /// # Arguments
    ///
    /// * `current_bet` - The current bet to call (in BB)
    /// * `stack` - The player's remaining stack (in BB)
    /// * `bets_this_round` - Number of bets already made this round
    ///
    /// # Returns
    ///
    /// Vector of valid raise sizes. If at max bets, only all-in is available.
    #[must_use]
    pub fn get_legal_raise_sizes(
        &self,
        current_bet: f64,
        stack: f64,
        bets_this_round: u8,
    ) -> Vec<f64> {
        if bets_this_round >= self.max_bets_per_round {
            // At max bets - no more raises allowed (can only call or fold)
            return vec![];
        }

        // Minimum raise is typically 2x the current bet (or 1BB from 0)
        let min_raise = if current_bet < 0.001 {
            1.0
        } else {
            current_bet * 2.0
        };

        let mut sizes: Vec<f64> = self
            .raise_sizes
            .iter()
            .filter(|&&size| size >= min_raise && size < stack)
            .copied()
            .collect();

        // Always include all-in if we have chips
        if stack > current_bet && !sizes.iter().any(|&s| (s - stack).abs() < 0.001) {
            sizes.push(stack);
        }

        sizes.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        sizes
    }

    /// Check if a raise size is valid given the betting state.
    #[must_use]
    pub fn is_valid_raise(
        &self,
        raise_size: f64,
        current_bet: f64,
        stack: f64,
        bets_this_round: u8,
    ) -> bool {
        if bets_this_round >= self.max_bets_per_round {
            return false;
        }

        let legal_sizes = self.get_legal_raise_sizes(current_bet, stack, bets_this_round);
        legal_sizes.iter().any(|&s| (s - raise_size).abs() < 0.001)
    }
}

impl Default for Config {
    fn default() -> Self {
        Self {
            name: "Default".to_string(),
            stack_depths: vec![100],
            raise_sizes: vec![2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 15.0, 20.0, 25.0],
            max_bets_per_round: 4,
        }
    }
}

/// Errors that can occur when loading or validating configuration.
#[derive(Debug, Error)]
pub enum ConfigError {
    /// IO error reading config file
    #[error("failed to read config file {0}: {1}")]
    Io(std::path::PathBuf, #[source] std::io::Error),

    /// YAML parsing error
    #[error("failed to parse YAML: {0}")]
    Parse(#[from] serde_yaml::Error),

    /// No stack depths specified
    #[error("stack_depths cannot be empty")]
    EmptyStackDepths,

    /// Invalid stack depth value
    #[error("invalid stack depth: {0} (must be > 0)")]
    InvalidStackDepth(u32),

    /// Invalid raise size value
    #[error("invalid raise size: {0} (must be > 0)")]
    InvalidRaiseSize(f64),

    /// Invalid max bets value
    #[error("invalid max_bets_per_round: {0} (must be > 0)")]
    InvalidMaxBets(u8),
}

#[cfg(test)]
mod tests {
    use super::*;
    use test_macros::timed_test;

    const VALID_YAML: &str = r#"
name: "Test Config"
stack_depths: [20, 50, 100]
raise_sizes: [2.5, 3.0, 4.0, 5.0, 10.0]
max_bets_per_round: 3
"#;

    #[timed_test]
    fn parse_valid_config() {
        let config = Config::from_yaml(VALID_YAML).unwrap();
        assert_eq!(config.name, "Test Config");
        assert_eq!(config.stack_depths, vec![20, 50, 100]);
        assert_eq!(config.raise_sizes.len(), 5);
        assert_eq!(config.max_bets_per_round, 3);
    }

    #[timed_test]
    fn default_max_bets_is_four() {
        let yaml = r#"
name: "No Max Bets"
stack_depths: [100]
raise_sizes: [3.0]
"#;
        let config = Config::from_yaml(yaml).unwrap();
        assert_eq!(config.max_bets_per_round, 4);
    }

    #[timed_test]
    fn empty_stack_depths_fails() {
        let yaml = r#"
name: "Bad Config"
stack_depths: []
raise_sizes: [3.0]
"#;
        let result = Config::from_yaml(yaml);
        assert!(matches!(result, Err(ConfigError::EmptyStackDepths)));
    }

    #[timed_test]
    fn zero_stack_depth_fails() {
        let yaml = r#"
name: "Bad Config"
stack_depths: [0, 100]
raise_sizes: [3.0]
"#;
        let result = Config::from_yaml(yaml);
        assert!(matches!(result, Err(ConfigError::InvalidStackDepth(0))));
    }

    #[timed_test]
    fn negative_raise_size_fails() {
        let yaml = r#"
name: "Bad Config"
stack_depths: [100]
raise_sizes: [-1.0, 3.0]
"#;
        let result = Config::from_yaml(yaml);
        assert!(matches!(result, Err(ConfigError::InvalidRaiseSize(_))));
    }

    #[timed_test]
    fn get_legal_raise_sizes_filters_correctly() {
        let config = Config {
            name: "Test".to_string(),
            stack_depths: vec![100],
            raise_sizes: vec![2.0, 2.5, 3.0, 5.0, 10.0, 20.0],
            max_bets_per_round: 4,
        };

        // From 0 bet, min raise is 1BB
        let sizes = config.get_legal_raise_sizes(0.0, 100.0, 0);
        assert!(sizes.contains(&2.0));
        assert!(sizes.contains(&100.0)); // All-in

        // From 3BB bet, min raise is 6BB
        let sizes = config.get_legal_raise_sizes(3.0, 100.0, 1);
        assert!(!sizes.contains(&2.0));
        assert!(!sizes.contains(&3.0));
        assert!(sizes.contains(&10.0));
        assert!(sizes.contains(&20.0));
    }

    #[timed_test]
    fn max_bets_stops_raises() {
        let config = Config {
            name: "Test".to_string(),
            stack_depths: vec![100],
            raise_sizes: vec![3.0, 5.0, 10.0],
            max_bets_per_round: 2,
        };

        // At max bets, no raises allowed
        let sizes = config.get_legal_raise_sizes(10.0, 100.0, 2);
        assert!(sizes.is_empty());
    }

    #[timed_test]
    fn is_valid_raise_checks_correctly() {
        let config = Config {
            name: "Test".to_string(),
            stack_depths: vec![100],
            raise_sizes: vec![3.0, 5.0, 10.0],
            max_bets_per_round: 3,
        };

        assert!(config.is_valid_raise(5.0, 2.0, 100.0, 1));
        assert!(!config.is_valid_raise(4.0, 2.0, 100.0, 1)); // Not in raise_sizes
        assert!(!config.is_valid_raise(5.0, 2.0, 100.0, 3)); // At max bets
    }

    #[timed_test]
    fn all_in_always_available() {
        let config = Config {
            name: "Test".to_string(),
            stack_depths: vec![25],
            raise_sizes: vec![3.0, 5.0],
            max_bets_per_round: 4,
        };

        let sizes = config.get_legal_raise_sizes(0.0, 25.0, 0);
        assert!(sizes.contains(&25.0)); // All-in
    }

    #[timed_test]
    fn default_config_is_valid() {
        let config = Config::default();
        assert!(!config.stack_depths.is_empty());
        assert!(!config.raise_sizes.is_empty());
        assert!(config.max_bets_per_round > 0);
    }
}
