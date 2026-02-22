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

use super::hand_buckets::{StreetBuckets, StreetEquity};
use super::postflop_model::PostflopModelConfig;

// ──────────────────────────────────────────────────────────────────────────────
// Cache key
// ──────────────────────────────────────────────────────────────────────────────

/// The subset of config fields that determine the abstraction output.
#[derive(Debug, Clone, Serialize, Deserialize, Hash, PartialEq, Eq)]
pub struct AbstractionCacheKey {
    pub num_flop_textures: u16,
    pub num_turn_transitions: u16,
    pub num_river_transitions: u16,
    pub num_hand_buckets_flop: u16,
    pub num_hand_buckets_turn: u16,
    pub num_hand_buckets_river: u16,
    pub ehs_samples: u32,
    pub has_equity_table: bool,
}

/// Build a cache key from config and whether an equity table is present.
#[must_use]
pub fn cache_key(config: &PostflopModelConfig, has_equity_table: bool) -> AbstractionCacheKey {
    AbstractionCacheKey {
        num_flop_textures: config.num_flop_textures,
        num_turn_transitions: config.num_turn_transitions,
        num_river_transitions: config.num_river_transitions,
        num_hand_buckets_flop: config.num_hand_buckets_flop,
        num_hand_buckets_turn: config.num_hand_buckets_turn,
        num_hand_buckets_river: config.num_hand_buckets_river,
        ehs_samples: config.ehs_samples,
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
) -> Result<(), CacheError> {
    let dir = cache_dir(base, key);
    fs::create_dir_all(&dir)?;

    let key_yaml = serde_yaml::to_string(key).map_err(|e| CacheError::Serialize(e.to_string()))?;
    fs::write(dir.join("key.yaml"), key_yaml)?;

    let data = AbstractionCacheData {
        buckets: clone_street_buckets(buckets),
        equity: clone_street_equity(equity),
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
) -> Option<(StreetBuckets, StreetEquity)> {
    let dir = cache_dir(base, key);
    let bytes = fs::read(dir.join("abstraction.bin")).ok()?;
    let data: AbstractionCacheData = bincode::deserialize(&bytes).ok()?;
    Some((data.buckets, data.equity))
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
        num_flop_boards: b.num_flop_boards,
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
        flop: clone_eq(&e.flop),
        turn: clone_eq(&e.turn),
        river: clone_eq(&e.river),
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
            num_flop_textures: 5,
            num_turn_transitions: 3,
            num_river_transitions: 3,
            num_hand_buckets_flop: 50,
            num_hand_buckets_turn: 50,
            num_hand_buckets_river: 50,
            ehs_samples: 200,
            has_equity_table: false,
        }
    }

    fn minimal_buckets() -> StreetBuckets {
        StreetBuckets {
            flop: vec![0u16],
            num_flop_buckets: 1,
            num_flop_boards: 1,
            turn: vec![0u16],
            num_turn_buckets: 1,
            river: vec![0u16],
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
            flop: BucketEquity { equity: eq.equity.clone(), num_buckets: eq.num_buckets },
            turn: BucketEquity { equity: eq.equity.clone(), num_buckets: eq.num_buckets },
            river: eq,
        }
    }

    #[timed_test]
    fn cache_roundtrip() {
        let dir = TempDir::new().unwrap();
        let key = minimal_key();
        let buckets = minimal_buckets();
        let equity = minimal_equity();

        save(dir.path(), &key, &buckets, &equity).unwrap();
        let (loaded_buckets, loaded_equity) =
            load(dir.path(), &key).expect("cache load should succeed");

        assert_eq!(loaded_buckets.num_flop_buckets, buckets.num_flop_buckets);
        assert_eq!(loaded_buckets.flop, buckets.flop);
        assert_eq!(loaded_buckets.num_flop_boards, buckets.num_flop_boards);
        assert_eq!(loaded_equity.river.num_buckets, equity.river.num_buckets);
        assert!((loaded_equity.river.get(0, 0) - 0.5).abs() < 1e-6);
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
        save(dir.path(), &key, &minimal_buckets(), &minimal_equity()).unwrap();
        assert!(exists(dir.path(), &key));
    }

    #[timed_test]
    fn key_yaml_written_on_save() {
        let dir = TempDir::new().unwrap();
        let key = minimal_key();
        save(dir.path(), &key, &minimal_buckets(), &minimal_equity()).unwrap();
        let yaml_path = cache_dir(dir.path(), &key).join("key.yaml");
        assert!(yaml_path.exists());
        let contents = fs::read_to_string(yaml_path).unwrap();
        assert!(contents.contains("num_flop_textures"));
    }
}
