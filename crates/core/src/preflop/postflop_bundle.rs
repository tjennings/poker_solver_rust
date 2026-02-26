//! Postflop solve bundle persistence.
//!
//! Saves and loads a `PostflopModelConfig` + solved data (values, flops, spr):
//! ```text
//! postflop_bundle/
//! ├── config.yaml      # Human-readable PostflopModelConfig
//! └── solve.bin        # Bincode-serialized PostflopBundleData
//! ```
//!
//! The `PostflopTree` is NOT serialized — it is cheap to rebuild from config.

use std::fs;
use std::path::Path;

use serde::{Deserialize, Serialize};

use super::postflop_abstraction::{
    PostflopAbstraction, PostflopAbstractionError, PostflopValues, compute_hand_avg_values,
};
use super::postflop_model::PostflopModelConfig;
use crate::poker::Card;

/// Serializable payload for a postflop solve bundle.
#[derive(Serialize, Deserialize)]
struct PostflopBundleData {
    values: PostflopValues,
    /// Hand-averaged EV: `[pos0: 169×169, pos1: 169×169]`.
    hand_avg_values: Vec<f64>,
    flops: Vec<[Card; 3]>,
    spr: f64,
}

/// A paired config and solved postflop data, ready for persistence.
pub struct PostflopBundle {
    pub config: PostflopModelConfig,
    data: PostflopBundleData,
}

impl PostflopBundle {
    /// Create a bundle from a config and a built `PostflopAbstraction`.
    #[must_use]
    pub fn from_abstraction(config: &PostflopModelConfig, abs: &PostflopAbstraction) -> Self {
        Self {
            config: config.clone(),
            data: PostflopBundleData {
                values: PostflopValues::from_raw(
                    abs.values.values.clone(),
                    abs.values.num_buckets,
                    abs.values.num_flops,
                ),
                hand_avg_values: abs.hand_avg_values.clone(),
                flops: abs.flops.clone(),
                spr: abs.spr,
            },
        }
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

        let data_bytes = bincode::serialize(&self.data).map_err(std::io::Error::other)?;
        fs::write(dir.join("solve.bin"), data_bytes)?;

        Ok(())
    }

    /// Load a bundle from a directory.
    ///
    /// # Errors
    ///
    /// Returns an error if files are missing, unreadable, or deserialization fails.
    pub fn load(dir: &Path) -> Result<Self, std::io::Error> {
        let config_yaml = fs::read_to_string(dir.join("config.yaml"))?;
        let config: PostflopModelConfig =
            serde_yaml::from_str(&config_yaml).map_err(std::io::Error::other)?;

        let data_bytes = fs::read(dir.join("solve.bin"))?;
        let data: PostflopBundleData =
            bincode::deserialize(&data_bytes).map_err(std::io::Error::other)?;

        Ok(Self { config, data })
    }

    /// Reconstruct a `PostflopAbstraction` from the bundle data.
    ///
    /// Rebuilds the `PostflopTree` from the stored config (cheap).
    /// Uses the precomputed `hand_avg_values` from the bundle if present,
    /// otherwise recomputes from values.
    ///
    /// # Errors
    ///
    /// Returns an error if tree building fails.
    pub fn into_abstraction(self) -> Result<PostflopAbstraction, PostflopAbstractionError> {
        let hand_avg = if self.data.hand_avg_values.is_empty() {
            // Backward compat: recompute if bundle was saved without avg values.
            compute_hand_avg_values(&self.data.values)
        } else {
            self.data.hand_avg_values
        };
        PostflopAbstraction::build_from_cached(
            &self.config,
            self.data.values,
            hand_avg,
            self.data.flops,
        )
    }

    /// Return a reference to the hand-averaged EV table.
    ///
    /// Layout: `[pos0: N×N, pos1: N×N]` where N = number of canonical hands (169).
    /// Index: `pos * N * N + hero_hand * N + opp_hand`.
    #[must_use]
    pub fn hand_avg_values(&self) -> &[f64] {
        &self.data.hand_avg_values
    }

    /// Check whether a directory contains a complete bundle.
    #[must_use]
    pub fn exists(dir: &Path) -> bool {
        dir.join("config.yaml").exists() && dir.join("solve.bin").exists()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;
    use test_macros::timed_test;

    fn minimal_bundle() -> PostflopBundle {
        let config = PostflopModelConfig::fast();
        let values = PostflopValues::from_raw(vec![0.5; 8], 1, 1);
        let hand_avg_values = compute_hand_avg_values(&values);
        let flops = vec![];
        PostflopBundle {
            config,
            data: PostflopBundleData {
                values,
                hand_avg_values,
                flops,
                spr: 3.5,
            },
        }
    }

    #[timed_test]
    fn postflop_bundle_roundtrip() {
        let bundle = minimal_bundle();
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("pf_test");
        bundle.save(&path).unwrap();
        assert!(PostflopBundle::exists(&path));

        let loaded = PostflopBundle::load(&path).unwrap();
        assert_eq!(loaded.config, PostflopModelConfig::fast());
        assert_eq!(loaded.data.values.num_flops(), 1);
        assert!((loaded.data.spr - 3.5).abs() < 1e-9);
        assert!(!loaded.data.hand_avg_values.is_empty());
    }

    #[timed_test]
    fn postflop_bundle_files_on_disk() {
        let bundle = minimal_bundle();
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("test_bundle");
        bundle.save(&path).unwrap();
        assert!(path.join("config.yaml").exists());
        assert!(path.join("solve.bin").exists());
    }

    #[timed_test]
    fn postflop_bundle_exists_false_for_missing() {
        let dir = TempDir::new().unwrap();
        assert!(!PostflopBundle::exists(&dir.path().join("nonexistent")));
    }

    #[timed_test]
    fn postflop_bundle_into_abstraction() {
        let bundle = minimal_bundle();
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("pf_abs");
        bundle.save(&path).unwrap();

        let loaded = PostflopBundle::load(&path).unwrap();
        let abs = loaded.into_abstraction().unwrap();
        assert!((abs.spr - 3.5).abs() < 1e-9);
        assert!(!abs.hand_avg_values.is_empty());
    }

    #[timed_test]
    fn hand_avg_values_getter() {
        let bundle = minimal_bundle();
        let vals = bundle.hand_avg_values();
        assert!(!vals.is_empty());
        assert_eq!(vals.len(), bundle.data.hand_avg_values.len());
    }

    #[timed_test]
    fn hand_avg_values_roundtrip_matches() {
        let bundle = minimal_bundle();
        let original_avg = bundle.data.hand_avg_values.clone();

        let dir = TempDir::new().unwrap();
        let path = dir.path().join("pf_avg");
        bundle.save(&path).unwrap();

        let loaded = PostflopBundle::load(&path).unwrap();
        assert_eq!(loaded.data.hand_avg_values.len(), original_avg.len());
        for (a, b) in loaded.data.hand_avg_values.iter().zip(&original_avg) {
            assert!((a - b).abs() < 1e-12);
        }
    }
}
