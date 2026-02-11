//! Strategy Bundle
//!
//! Manages the directory structure for paired blueprint and config files:
//! ```text
//! my_strategy/
//! ├── config.yaml       # PostflopConfig + AbstractionConfig
//! ├── blueprint.bin     # BlueprintStrategy
//! └── boundaries.bin    # BucketBoundaries
//! ```

use std::fs;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::abstraction::{AbstractionConfig, BucketBoundaries};
use crate::game::PostflopConfig;

use super::{BlueprintError, BlueprintStrategy};

/// Which abstraction mode a bundle was trained with.
///
/// Determines how postflop hands are bucketed for info-set key construction.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AbstractionModeConfig {
    /// EHS2-based bucketing (expensive Monte Carlo, fine-grained).
    #[default]
    Ehs2,
    /// Hand-class V2: class ID + intra-class strength + equity bin + draw flags.
    ///
    /// With `strength_bits=0, equity_bits=0`, equivalent to the old `hand_class` mode.
    /// Legacy configs with `hand_class` are deserialized as `HandClassV2`.
    #[serde(alias = "hand_class")]
    HandClassV2,
}

impl AbstractionModeConfig {
    /// Returns true if this mode uses hand-class based bucketing.
    #[must_use]
    pub fn is_hand_class(self) -> bool {
        matches!(self, Self::HandClassV2)
    }
}

/// Combined configuration stored in config.yaml
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct BundleConfig {
    /// Game configuration (stack depth, bet sizes)
    pub game: PostflopConfig,
    /// Abstraction configuration (bucket counts). None for `hand_class` mode.
    pub abstraction: Option<AbstractionConfig>,
    /// Which abstraction mode was used.
    #[serde(default)]
    pub abstraction_mode: AbstractionModeConfig,
    /// Number of bits for intra-class strength (0-4). Only used with `hand_class_v2`.
    #[serde(default)]
    pub strength_bits: u8,
    /// Number of bits for equity bin (0-4). Only used with `hand_class_v2`.
    #[serde(default)]
    pub equity_bits: u8,
}

/// A complete strategy bundle containing all data needed for exploration.
#[derive(Debug)]
pub struct StrategyBundle {
    /// Combined configuration
    pub config: BundleConfig,
    /// Trained strategies
    pub blueprint: BlueprintStrategy,
    /// Bucket boundaries for card abstraction (None for `hand_class` mode)
    pub boundaries: Option<BucketBoundaries>,
}

impl StrategyBundle {
    /// Create a new bundle from components.
    #[must_use]
    pub fn new(
        config: BundleConfig,
        blueprint: BlueprintStrategy,
        boundaries: Option<BucketBoundaries>,
    ) -> Self {
        Self {
            config,
            blueprint,
            boundaries,
        }
    }

    /// Save the bundle to a directory.
    ///
    /// Creates the directory if it doesn't exist. Writes:
    /// - `config.yaml` - Human-readable configuration
    /// - `blueprint.bin` - Binary strategy data
    /// - `boundaries.bin` - Binary bucket boundaries (only for EHS2 mode)
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Directory creation fails
    /// - Any file write fails
    /// - Serialization fails
    pub fn save(&self, dir: &Path) -> Result<(), BlueprintError> {
        // Create directory if needed
        fs::create_dir_all(dir)?;

        // Save config as YAML
        let config_path = dir.join("config.yaml");
        let config_yaml = serde_yaml::to_string(&self.config)
            .map_err(|e| BlueprintError::SerializationError(e.to_string()))?;
        fs::write(&config_path, config_yaml)?;

        // Save blueprint as bincode
        let blueprint_path = dir.join("blueprint.bin");
        self.blueprint.save(&blueprint_path)?;

        // Save boundaries as bincode (only if present)
        if let Some(ref boundaries) = self.boundaries {
            let boundaries_path = dir.join("boundaries.bin");
            let boundaries_data = bincode::serialize(boundaries)
                .map_err(|e| BlueprintError::SerializationError(e.to_string()))?;
            fs::write(&boundaries_path, boundaries_data)?;
        }

        Ok(())
    }

    /// Load a bundle from a directory.
    ///
    /// Expects the directory to contain:
    /// - `config.yaml`
    /// - `blueprint.bin`
    /// - `boundaries.bin` (optional — absent for `hand_class` bundles)
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Directory doesn't exist
    /// - `config.yaml` or `blueprint.bin` is missing
    /// - Deserialization fails
    pub fn load(dir: &Path) -> Result<Self, BlueprintError> {
        // Load config from YAML
        let config_path = dir.join("config.yaml");
        let config_yaml = fs::read_to_string(&config_path)?;
        let config: BundleConfig = serde_yaml::from_str(&config_yaml)
            .map_err(|e| BlueprintError::SerializationError(e.to_string()))?;

        // Load blueprint
        let blueprint_path = dir.join("blueprint.bin");
        let blueprint = BlueprintStrategy::load(&blueprint_path)?;

        // Load boundaries (optional)
        let boundaries_path = dir.join("boundaries.bin");
        let boundaries = if boundaries_path.exists() {
            let boundaries_data = fs::read(&boundaries_path)?;
            let b: BucketBoundaries = bincode::deserialize(&boundaries_data)
                .map_err(|e| BlueprintError::SerializationError(e.to_string()))?;
            Some(b)
        } else {
            None
        };

        Ok(Self {
            config,
            blueprint,
            boundaries,
        })
    }

    /// Check if a directory contains a valid bundle.
    ///
    /// A bundle requires `config.yaml` and `blueprint.bin`.
    /// `boundaries.bin` is optional (absent for `hand_class` bundles).
    #[must_use]
    pub fn exists(dir: &Path) -> bool {
        dir.join("config.yaml").exists() && dir.join("blueprint.bin").exists()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;
    use test_macros::timed_test;

    fn create_test_config() -> BundleConfig {
        BundleConfig {
            game: PostflopConfig {
                stack_depth: 20,
                bet_sizes: vec![0.5, 1.0],
                ..PostflopConfig::default()
            },
            abstraction: Some(AbstractionConfig {
                flop_buckets: 100,
                turn_buckets: 100,
                river_buckets: 100,
                samples_per_street: 1000,
            }),
            abstraction_mode: AbstractionModeConfig::Ehs2,
            strength_bits: 0,
            equity_bits: 0,
        }
    }

    fn create_test_boundaries() -> BucketBoundaries {
        BucketBoundaries {
            flop: vec![0.25, 0.5, 0.75],
            turn: vec![0.25, 0.5, 0.75],
            river: vec![0.25, 0.5, 0.75],
        }
    }

    #[timed_test]
    fn bundle_roundtrip_save_load() {
        let config = create_test_config();
        let mut blueprint = BlueprintStrategy::new();
        blueprint.insert(100, vec![0.3, 0.7]);
        blueprint.insert(200, vec![0.1, 0.4, 0.5]);
        blueprint.set_iterations(1000);
        let boundaries = create_test_boundaries();

        let bundle = StrategyBundle::new(config, blueprint, Some(boundaries));

        let temp_dir = TempDir::new().expect("failed to create temp dir");
        let bundle_path = temp_dir.path().join("test_strategy");

        bundle.save(&bundle_path).expect("save should succeed");
        assert!(StrategyBundle::exists(&bundle_path));

        let loaded = StrategyBundle::load(&bundle_path).expect("load should succeed");

        assert_eq!(loaded.config.game.stack_depth, 20);
        assert_eq!(loaded.config.game.bet_sizes, vec![0.5, 1.0]);
        assert_eq!(
            loaded.config.abstraction.as_ref().unwrap().flop_buckets,
            100
        );
        assert_eq!(loaded.blueprint.len(), 2);
        assert_eq!(loaded.blueprint.iterations_trained(), 1000);
        assert_eq!(loaded.boundaries.as_ref().unwrap().flop.len(), 3);
    }

    #[timed_test]
    fn bundle_creates_directory() {
        let config = create_test_config();
        let blueprint = BlueprintStrategy::new();
        let boundaries = create_test_boundaries();
        let bundle = StrategyBundle::new(config, blueprint, Some(boundaries));

        let temp_dir = TempDir::new().expect("failed to create temp dir");
        let bundle_path = temp_dir.path().join("nested").join("strategy");

        bundle
            .save(&bundle_path)
            .expect("save should create nested dirs");
        assert!(bundle_path.exists());
        assert!(bundle_path.join("config.yaml").exists());
        assert!(bundle_path.join("blueprint.bin").exists());
        assert!(bundle_path.join("boundaries.bin").exists());
    }

    #[timed_test]
    fn bundle_hand_class_no_boundaries_file() {
        let config = BundleConfig {
            game: PostflopConfig {
                stack_depth: 20,
                bet_sizes: vec![0.5, 1.0],
                ..PostflopConfig::default()
            },
            abstraction: None,
            abstraction_mode: AbstractionModeConfig::HandClassV2,
            strength_bits: 0,
            equity_bits: 0,
        };
        let blueprint = BlueprintStrategy::new();
        let bundle = StrategyBundle::new(config, blueprint, None);

        let temp_dir = TempDir::new().expect("failed to create temp dir");
        let bundle_path = temp_dir.path().join("hand_class_bundle");

        bundle.save(&bundle_path).expect("save should succeed");
        assert!(StrategyBundle::exists(&bundle_path));
        assert!(bundle_path.join("config.yaml").exists());
        assert!(bundle_path.join("blueprint.bin").exists());
        assert!(
            !bundle_path.join("boundaries.bin").exists(),
            "hand_class bundle should not have boundaries.bin"
        );

        let loaded = StrategyBundle::load(&bundle_path).expect("load should succeed");
        assert!(loaded.boundaries.is_none());
        assert_eq!(loaded.config.abstraction_mode, AbstractionModeConfig::HandClassV2);
    }

    #[timed_test]
    fn bundle_exists_returns_false_for_missing() {
        let temp_dir = TempDir::new().expect("failed to create temp dir");
        let missing_path = temp_dir.path().join("nonexistent");

        assert!(!StrategyBundle::exists(&missing_path));
    }

    #[timed_test]
    fn bundle_exists_returns_false_for_partial() {
        let temp_dir = TempDir::new().expect("failed to create temp dir");
        let partial_path = temp_dir.path().join("partial");
        fs::create_dir_all(&partial_path).expect("create dir");
        fs::write(partial_path.join("config.yaml"), "test").expect("write config");
        // Missing blueprint.bin

        assert!(!StrategyBundle::exists(&partial_path));
    }

    #[timed_test]
    fn config_yaml_is_human_readable() {
        let config = create_test_config();
        let blueprint = BlueprintStrategy::new();
        let boundaries = create_test_boundaries();
        let bundle = StrategyBundle::new(config, blueprint, Some(boundaries));

        let temp_dir = TempDir::new().expect("failed to create temp dir");
        let bundle_path = temp_dir.path().join("readable_test");

        bundle.save(&bundle_path).expect("save should succeed");

        let yaml_content = fs::read_to_string(bundle_path.join("config.yaml")).expect("read yaml");

        // Verify it contains human-readable keys
        assert!(
            yaml_content.contains("stack_depth"),
            "Should contain stack_depth"
        );
        assert!(
            yaml_content.contains("bet_sizes"),
            "Should contain bet_sizes"
        );
        assert!(
            yaml_content.contains("flop_buckets"),
            "Should contain flop_buckets"
        );
    }

    #[timed_test]
    fn load_nonexistent_returns_error() {
        let result = StrategyBundle::load(Path::new("/nonexistent/path"));
        assert!(result.is_err());
    }
}
