//! Disk cache for postflop abstraction data (board, hand buckets, equity).
//!
//! Caches the expensive abstraction phases (board clustering, hand bucketing,
//! equity computation) keyed by the config fields that affect them.
//! Solve-dependent data (trees, CFR values) is always rebuilt.
//!
//! # Layout
//!
//! ```text
//! cache/postflop/<hex_key>/
//!   key.yaml             # Human-readable cache key
//!   abstraction.bin      # bincode: AbstractionCacheData
//! ```

use std::fs;
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use super::hand_buckets::{StreetBuckets, StreetEquity, TransitionMatrices};
use super::postflop_model::PostflopModelConfig;

// ──────────────────────────────────────────────────────────────────────────────
// Cache key
// ──────────────────────────────────────────────────────────────────────────────

/// The subset of config fields that determine the abstraction output.
#[derive(Debug, Clone, Serialize, Deserialize, Hash, PartialEq, Eq)]
pub struct AbstractionCacheKey {
    pub num_hand_buckets_flop: u16,
    pub num_hand_buckets_turn: u16,
    pub num_hand_buckets_river: u16,
    pub has_equity_table: bool,
}

/// Build a cache key from config and whether an equity table is present.
#[must_use]
pub fn cache_key(config: &PostflopModelConfig, has_equity_table: bool) -> AbstractionCacheKey {
    AbstractionCacheKey {
        num_hand_buckets_flop: config.num_hand_buckets_flop,
        num_hand_buckets_turn: config.num_hand_buckets_turn,
        num_hand_buckets_river: config.num_hand_buckets_river,
        has_equity_table,
    }
}

/// Compute the cache directory for a given key: `<base>/<hex_hash>/`.
#[must_use]
pub fn cache_dir(base: &Path, key: &AbstractionCacheKey) -> PathBuf {
    base.join(hex_hash(key))
}

// ──────────────────────────────────────────────────────────────────────────────
// Serializable payload
// ──────────────────────────────────────────────────────────────────────────────

#[derive(Serialize, Deserialize)]
struct AbstractionCacheData {
    buckets: StreetBuckets,
    equity: StreetEquity,
    transitions: TransitionMatrices,
}

// ──────────────────────────────────────────────────────────────────────────────
// Public API
// ──────────────────────────────────────────────────────────────────────────────

/// Error type for cache operations.
#[derive(Debug, thiserror::Error)]
pub enum CacheError {
    #[error("cache I/O: {0}")]
    Io(#[from] std::io::Error),
    #[error("cache serialization: {0}")]
    Serialize(String),
}

/// Save abstraction data to the cache directory.
///
/// Creates the directory if needed, writes `key.yaml` and `abstraction.bin`.
///
/// # Errors
///
/// Returns `CacheError` on I/O or serialization failure.
pub fn save(
    base: &Path,
    key: &AbstractionCacheKey,
    buckets: &StreetBuckets,
    equity: &StreetEquity,
    transitions: &TransitionMatrices,
) -> Result<(), CacheError> {
    let dir = cache_dir(base, key);
    fs::create_dir_all(&dir)?;

    let key_yaml = serde_yaml::to_string(key).map_err(|e| CacheError::Serialize(e.to_string()))?;
    fs::write(dir.join("key.yaml"), key_yaml)?;

    let data = AbstractionCacheData {
        buckets: clone_street_buckets(buckets),
        equity: clone_street_equity(equity),
        transitions: TransitionMatrices {
            flop_to_turn: transitions.flop_to_turn.clone(),
            turn_to_river: transitions.turn_to_river.clone(),
        },
    };
    let bytes = bincode::serialize(&data).map_err(|e| CacheError::Serialize(e.to_string()))?;
    fs::write(dir.join("abstraction.bin"), bytes)?;

    Ok(())
}

/// Load cached abstraction data, returning `None` if missing or corrupt.
#[must_use]
pub fn load(
    base: &Path,
    key: &AbstractionCacheKey,
) -> Option<(StreetBuckets, StreetEquity, TransitionMatrices)> {
    let dir = cache_dir(base, key);
    let bytes = fs::read(dir.join("abstraction.bin")).ok()?;
    let data: AbstractionCacheData = bincode::deserialize(&bytes).ok()?;
    Some((data.buckets, data.equity, data.transitions))
}

/// Check whether a cache entry exists for the given key.
#[must_use]
pub fn exists(base: &Path, key: &AbstractionCacheKey) -> bool {
    let dir = cache_dir(base, key);
    dir.join("abstraction.bin").exists()
}

// ──────────────────────────────────────────────────────────────────────────────
// Helpers
// ──────────────────────────────────────────────────────────────────────────────

/// FNV-1a hash of the key, formatted as a 16-char hex string.
fn hex_hash(key: &AbstractionCacheKey) -> String {
    let mut hasher = super::fnv::FnvHasher::new();
    key.hash(&mut hasher);
    format!("{:016x}", hasher.finish())
}

/// Clone `StreetBuckets` by copying fields.
fn clone_street_buckets(b: &StreetBuckets) -> StreetBuckets {
    StreetBuckets {
        flop: b.flop.clone(),
        num_flop_buckets: b.num_flop_buckets,
        turn: b.turn.clone(),
        num_turn_buckets: b.num_turn_buckets,
        river: b.river.clone(),
        num_river_buckets: b.num_river_buckets,
    }
}

/// Clone `StreetEquity` by copying fields.
fn clone_street_equity(e: &StreetEquity) -> StreetEquity {
    use super::hand_buckets::BucketEquity;
    let clone_eq = |eq: &BucketEquity| BucketEquity {
        equity: eq.equity.clone(),
        num_buckets: eq.num_buckets,
    };
    StreetEquity {
        flop: e.flop.iter().map(clone_eq).collect(),
        turn: e.turn.iter().map(clone_eq).collect(),
        river: e.river.iter().map(clone_eq).collect(),
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;
    use test_macros::timed_test;

    fn minimal_key() -> AbstractionCacheKey {
        AbstractionCacheKey {
            num_hand_buckets_flop: 50,
            num_hand_buckets_turn: 50,
            num_hand_buckets_river: 50,
            has_equity_table: false,
        }
    }

    fn minimal_buckets() -> StreetBuckets {
        StreetBuckets {
            flop: vec![vec![0u16]],
            num_flop_buckets: 1,
            turn: vec![vec![0u16]],
            num_turn_buckets: 1,
            river: vec![vec![0u16]],
            num_river_buckets: 1,
        }
    }

    fn minimal_equity() -> StreetEquity {
        use super::super::hand_buckets::BucketEquity;
        let eq = BucketEquity {
            equity: vec![vec![0.5f32]],
            num_buckets: 1,
        };
        StreetEquity {
            flop: vec![BucketEquity { equity: eq.equity.clone(), num_buckets: eq.num_buckets }],
            turn: vec![BucketEquity { equity: eq.equity.clone(), num_buckets: eq.num_buckets }],
            river: vec![eq],
        }
    }

    fn minimal_transitions() -> TransitionMatrices {
        TransitionMatrices {
            flop_to_turn: vec![vec![vec![1.0]]],
            turn_to_river: vec![vec![vec![1.0]]],
        }
    }

    #[timed_test]
    fn cache_roundtrip() {
        let dir = TempDir::new().unwrap();
        let key = minimal_key();
        let buckets = minimal_buckets();
        let equity = minimal_equity();
        let transitions = minimal_transitions();

        save(dir.path(), &key, &buckets, &equity, &transitions).unwrap();
        let (loaded_buckets, loaded_equity, loaded_transitions) =
            load(dir.path(), &key).expect("cache load should succeed");

        assert_eq!(loaded_buckets.num_flop_buckets, buckets.num_flop_buckets);
        assert_eq!(loaded_buckets.flop, buckets.flop);
        assert_eq!(loaded_buckets.flop.len(), buckets.flop.len());
        assert_eq!(loaded_equity.river[0].num_buckets, equity.river[0].num_buckets);
        assert!((loaded_equity.river[0].get(0, 0) - 0.5).abs() < 1e-6);
        assert_eq!(loaded_transitions.flop_to_turn.len(), 1);
        assert!((loaded_transitions.flop_to_turn[0][0][0] - 1.0).abs() < 1e-9);
    }

    #[timed_test]
    fn cache_exists_false_for_missing() {
        let dir = TempDir::new().unwrap();
        let key = minimal_key();
        assert!(!exists(dir.path(), &key));
        assert!(load(dir.path(), &key).is_none());
    }

    #[timed_test]
    fn cache_key_deterministic() {
        let config = PostflopModelConfig::fast();
        let k1 = cache_key(&config, false);
        let k2 = cache_key(&config, false);
        assert_eq!(hex_hash(&k1), hex_hash(&k2));
    }

    #[timed_test]
    fn cache_key_differs_with_equity_flag() {
        let config = PostflopModelConfig::fast();
        let k_without = cache_key(&config, false);
        let k_with = cache_key(&config, true);
        assert_ne!(hex_hash(&k_without), hex_hash(&k_with));
    }

    #[timed_test]
    fn cache_key_differs_with_bucket_count() {
        let config1 = PostflopModelConfig::fast();
        let mut config2 = PostflopModelConfig::fast();
        config2.num_hand_buckets_flop = 100;
        let k1 = cache_key(&config1, false);
        let k2 = cache_key(&config2, false);
        assert_ne!(hex_hash(&k1), hex_hash(&k2));
    }

    #[timed_test]
    fn exists_true_after_save() {
        let dir = TempDir::new().unwrap();
        let key = minimal_key();
        save(dir.path(), &key, &minimal_buckets(), &minimal_equity(), &minimal_transitions()).unwrap();
        assert!(exists(dir.path(), &key));
    }

    #[timed_test]
    fn key_yaml_written_on_save() {
        let dir = TempDir::new().unwrap();
        let key = minimal_key();
        save(dir.path(), &key, &minimal_buckets(), &minimal_equity(), &minimal_transitions()).unwrap();
        let yaml_path = cache_dir(dir.path(), &key).join("key.yaml");
        assert!(yaml_path.exists());
        let contents = fs::read_to_string(yaml_path).unwrap();
        assert!(contents.contains("num_hand_buckets_flop"));
    }
}
