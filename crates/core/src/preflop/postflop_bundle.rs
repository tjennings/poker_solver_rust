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
        // Always recompute with correct flop weights — cached hand_avg_values
        // from older bundles used equal weighting which overweights monotone
        // boards ~3× and underweights rainbow ~0.5×.
        let flop_weights = crate::flops::lookup_flop_weights(&self.data.flops);
        let hand_avg = compute_hand_avg_values(&self.data.values, &flop_weights);
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

    /// Return a reference to the per-flop postflop values table.
    #[must_use]
    pub fn values(&self) -> &PostflopValues {
        &self.data.values
    }

    /// Return the solved flop boards.
    #[must_use]
    pub fn flops(&self) -> &[[Card; 3]] {
        &self.data.flops
    }

    /// Load just the hand-averaged EV table from a solve.bin file.
    ///
    /// Useful when the companion config.yaml has been overwritten
    /// (e.g. by a co-located `PreflopBundle`).
    ///
    /// # Errors
    ///
    /// Returns an error if the file is missing, unreadable, or deserialization fails.
    pub fn load_hand_avg_values(solve_bin: &Path) -> Result<Vec<f64>, std::io::Error> {
        let data_bytes = fs::read(solve_bin)?;
        let data: PostflopBundleData =
            bincode::deserialize(&data_bytes).map_err(std::io::Error::other)?;
        Ok(data.hand_avg_values)
    }

    /// Check whether a directory contains a complete bundle.
    #[must_use]
    pub fn exists(dir: &Path) -> bool {
        dir.join("config.yaml").exists() && dir.join("solve.bin").exists()
    }

    /// Save multiple SPR abstractions to a directory.
    ///
    /// Layout:
    /// ```text
    /// dir/
    /// ├── config.yaml
    /// ├── spr_2.0/solve.bin
    /// ├── spr_6.0/solve.bin
    /// └── spr_20.0/solve.bin
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error if directory creation, serialization, or file I/O fails.
    pub fn save_multi(
        config: &PostflopModelConfig,
        abstractions: &[&PostflopAbstraction],
        dir: &Path,
    ) -> Result<(), std::io::Error> {
        fs::create_dir_all(dir)?;
        let config_yaml = serde_yaml::to_string(config).map_err(std::io::Error::other)?;
        fs::write(dir.join("config.yaml"), config_yaml)?;

        for abs in abstractions {
            let spr_dir = dir.join(format!("spr_{}", abs.spr));
            fs::create_dir_all(&spr_dir)?;
            let data = PostflopBundleData {
                values: PostflopValues::from_raw(
                    abs.values.values.clone(),
                    abs.values.num_buckets,
                    abs.values.num_flops,
                ),
                hand_avg_values: abs.hand_avg_values.clone(),
                flops: abs.flops.clone(),
                spr: abs.spr,
            };
            let bytes = bincode::serialize(&data).map_err(std::io::Error::other)?;
            fs::write(spr_dir.join("solve.bin"), bytes)?;
        }
        Ok(())
    }

    /// Load multi-SPR abstractions from a directory.
    ///
    /// Handles both new multi-SPR layout (`spr_*/solve.bin`) and legacy
    /// single-file layout (`solve.bin` at root).
    ///
    /// # Errors
    ///
    /// Returns an error if no bundle is found, or if deserialization/tree building fails.
    pub fn load_multi(
        config: &PostflopModelConfig,
        dir: &Path,
    ) -> Result<Vec<PostflopAbstraction>, std::io::Error> {
        // Try new layout: look for spr_* subdirectories
        let mut spr_dirs: Vec<_> = fs::read_dir(dir)?
            .filter_map(|e| e.ok())
            .filter(|e| {
                e.file_name()
                    .to_str()
                    .map_or(false, |n| n.starts_with("spr_"))
                    && e.path().join("solve.bin").exists()
            })
            .collect();

        if !spr_dirs.is_empty() {
            spr_dirs.sort_by_key(|e| e.file_name());
            let mut result = Vec::with_capacity(spr_dirs.len());
            for entry in &spr_dirs {
                let data_bytes = fs::read(entry.path().join("solve.bin"))?;
                let data: PostflopBundleData =
                    bincode::deserialize(&data_bytes).map_err(std::io::Error::other)?;
                let flop_weights = crate::flops::lookup_flop_weights(&data.flops);
                let hand_avg = compute_hand_avg_values(&data.values, &flop_weights);
                let abs = PostflopAbstraction::build_from_cached_spr(
                    config,
                    data.spr,
                    data.values,
                    hand_avg,
                    data.flops,
                )
                .map_err(|e| std::io::Error::other(format!("{e}")))?;
                result.push(abs);
            }
            return Ok(result);
        }

        // Legacy: single solve.bin at root
        if dir.join("solve.bin").exists() {
            let bundle = Self::load(dir)?;
            let abs = bundle
                .into_abstraction()
                .map_err(|e| std::io::Error::other(format!("{e}")))?;
            return Ok(vec![abs]);
        }

        Err(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            format!("no postflop bundle found in {}", dir.display()),
        ))
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
        let flops: Vec<[Card; 3]> = vec![];
        let flop_weights = crate::flops::lookup_flop_weights(&flops);
        let hand_avg_values = compute_hand_avg_values(&values, &flop_weights);
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
    fn load_hand_avg_values_from_solve_bin() {
        let bundle = minimal_bundle();
        let original_len = bundle.hand_avg_values().len();
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("pf_havg");
        bundle.save(&path).unwrap();

        let loaded = PostflopBundle::load_hand_avg_values(&path.join("solve.bin")).unwrap();
        assert_eq!(loaded.len(), original_len);
        assert!(!loaded.is_empty());
    }

    #[timed_test]
    fn multi_spr_bundle_roundtrip() {
        let config = PostflopModelConfig::fast();
        let values1 = PostflopValues::from_raw(vec![0.2; 8], 1, 1);
        let values2 = PostflopValues::from_raw(vec![0.6; 8], 1, 1);
        let flops: Vec<[Card; 3]> = vec![];
        let flop_weights = crate::flops::lookup_flop_weights(&flops);
        let hand_avg1 = compute_hand_avg_values(&values1, &flop_weights);
        let hand_avg2 = compute_hand_avg_values(&values2, &flop_weights);

        let abs1 = PostflopAbstraction::build_from_cached_spr(
            &config, 2.0, values1, hand_avg1, vec![],
        ).unwrap();
        let abs2 = PostflopAbstraction::build_from_cached_spr(
            &config, 6.0, values2, hand_avg2, vec![],
        ).unwrap();

        let dir = TempDir::new().unwrap();
        let path = dir.path().join("multi_spr");

        PostflopBundle::save_multi(&config, &[&abs1, &abs2], &path).unwrap();
        let loaded = PostflopBundle::load_multi(&config, &path).unwrap();

        assert_eq!(loaded.len(), 2);
        assert!((loaded[0].spr - 2.0).abs() < 1e-9);
        assert!((loaded[1].spr - 6.0).abs() < 1e-9);
    }

    #[timed_test]
    fn legacy_single_bundle_loads_via_load_multi() {
        let bundle = minimal_bundle();
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("legacy");
        bundle.save(&path).unwrap();

        let config = PostflopModelConfig::fast();
        let loaded = PostflopBundle::load_multi(&config, &path).unwrap();
        assert_eq!(loaded.len(), 1);
        assert!((loaded[0].spr - 3.5).abs() < 1e-9);
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
