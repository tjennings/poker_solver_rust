//! Preflop strategy bundle persistence.
//!
//! Saves and loads a `PreflopConfig` + `PreflopStrategy` pair:
//! ```text
//! preflop_bundle/
//! ├── config.yaml      # Human-readable PreflopConfig
//! └── strategy.bin     # Bincode-serialized PreflopStrategy
//! ```

use std::fs;
use std::path::Path;

use super::config::PreflopConfig;
use super::solver::PreflopStrategy;

/// A paired config and trained strategy, ready for persistence.
#[derive(Debug)]
pub struct PreflopBundle {
    pub config: PreflopConfig,
    pub strategy: PreflopStrategy,
}

impl PreflopBundle {
    /// Create a new bundle from a config and a trained strategy.
    #[must_use]
    pub fn new(config: PreflopConfig, strategy: PreflopStrategy) -> Self {
        Self { config, strategy }
    }

    /// Save the bundle to a directory (created if needed).
    ///
    /// # Errors
    ///
    /// Returns an error if directory creation, serialization, or file I/O fails.
    pub fn save(&self, dir: &Path) -> Result<(), std::io::Error> {
        fs::create_dir_all(dir)?;

        let config_yaml = serde_yaml::to_string(&self.config).map_err(std::io::Error::other)?;
        fs::write(dir.join("config.yaml"), config_yaml)?;

        let strategy_bytes = bincode::serialize(&self.strategy).map_err(std::io::Error::other)?;
        fs::write(dir.join("strategy.bin"), strategy_bytes)?;

        Ok(())
    }

    /// Load a bundle from a directory.
    ///
    /// # Errors
    ///
    /// Returns an error if files are missing, unreadable, or deserialization fails.
    pub fn load(dir: &Path) -> Result<Self, std::io::Error> {
        let config_yaml = fs::read_to_string(dir.join("config.yaml"))?;
        let config: PreflopConfig =
            serde_yaml::from_str(&config_yaml).map_err(std::io::Error::other)?;

        let strategy_bytes = fs::read(dir.join("strategy.bin"))?;
        let strategy: PreflopStrategy =
            bincode::deserialize(&strategy_bytes).map_err(std::io::Error::other)?;

        Ok(Self { config, strategy })
    }

    /// Check whether a directory contains a complete bundle.
    #[must_use]
    pub fn exists(dir: &Path) -> bool {
        dir.join("config.yaml").exists() && dir.join("strategy.bin").exists()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::preflop::{PreflopConfig, PreflopSolver, RaiseSize};
    use tempfile::TempDir;
    use test_macros::timed_test;

    /// A minimal config with a tiny tree for fast unit tests.
    fn tiny_config() -> PreflopConfig {
        let mut config = PreflopConfig::heads_up(3);
        config.raise_sizes = vec![vec![RaiseSize::Bb(3.0)]];
        config.raise_cap = 1;
        config
    }

    #[timed_test]
    fn preflop_bundle_roundtrip() {
        let config = tiny_config();
        let mut solver = PreflopSolver::new(&config);
        solver.train(1);
        let strategy = solver.strategy();
        let bundle = PreflopBundle::new(config, strategy);

        let dir = TempDir::new().unwrap();
        let path = dir.path().join("preflop_test");
        bundle.save(&path).unwrap();
        assert!(PreflopBundle::exists(&path));

        let loaded = PreflopBundle::load(&path).unwrap();
        assert_eq!(loaded.config.num_players(), 2);
        assert!(!loaded.strategy.is_empty());
    }

    #[timed_test]
    fn preflop_bundle_files_on_disk() {
        let config = tiny_config();
        let mut solver = PreflopSolver::new(&config);
        solver.train(1);
        let bundle = PreflopBundle::new(config, solver.strategy());

        let dir = TempDir::new().unwrap();
        let path = dir.path().join("test_bundle");
        bundle.save(&path).unwrap();
        assert!(path.join("config.yaml").exists());
        assert!(path.join("strategy.bin").exists());
    }

    #[timed_test]
    fn preflop_bundle_exists_false_for_missing() {
        let dir = TempDir::new().unwrap();
        assert!(!PreflopBundle::exists(&dir.path().join("nonexistent")));
    }
}
