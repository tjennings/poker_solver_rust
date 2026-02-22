//! Disk cache for postflop solve results (`PostflopValues`).
//!
//! Caches the expensive postflop CFR solve keyed by the full `PostflopModelConfig`
//! plus whether a real equity table was used. On cache hit, the preflop solver
//! skips phases 5-7 entirely.
//!
//! # Layout
//!
//! ```text
//! cache/postflop/solve_<hex_key>/
//!   key.yaml        # human-readable config for debugging
//!   values.bin      # bincode: PostflopValues
//! ```

use std::fs;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use super::postflop_abstraction::PostflopValues;
use super::postflop_model::PostflopModelConfig;

/// Cache key: the full config that determines the postflop solve output.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SolveCacheKey {
    pub config: PostflopModelConfig,
    pub has_equity_table: bool,
}

/// Error type for solve cache operations.
#[derive(Debug, thiserror::Error)]
pub enum SolveCacheError {
    #[error("solve cache I/O: {0}")]
    Io(#[from] std::io::Error),
    #[error("solve cache serialization: {0}")]
    Serialize(String),
}

/// Build a cache key from config and equity table flag.
#[must_use]
pub fn cache_key(config: &PostflopModelConfig, has_equity_table: bool) -> SolveCacheKey {
    SolveCacheKey {
        config: config.clone(),
        has_equity_table,
    }
}

/// Compute the cache directory for a given key.
#[must_use]
pub fn cache_dir(base: &Path, key: &SolveCacheKey) -> PathBuf {
    base.join(format!("solve_{}", hex_hash(key)))
}

/// Save postflop values to the cache.
///
/// # Errors
///
/// Returns `SolveCacheError` on I/O or serialization failure.
pub fn save(
    base: &Path,
    key: &SolveCacheKey,
    values: &PostflopValues,
) -> Result<(), SolveCacheError> {
    let dir = cache_dir(base, key);
    fs::create_dir_all(&dir)?;

    let key_yaml =
        serde_yaml::to_string(key).map_err(|e| SolveCacheError::Serialize(e.to_string()))?;
    fs::write(dir.join("key.yaml"), key_yaml)?;

    let bytes =
        bincode::serialize(values).map_err(|e| SolveCacheError::Serialize(e.to_string()))?;
    fs::write(dir.join("values.bin"), bytes)?;

    Ok(())
}

/// Load cached postflop values, returning `None` if missing or corrupt.
#[must_use]
pub fn load(base: &Path, key: &SolveCacheKey) -> Option<PostflopValues> {
    let dir = cache_dir(base, key);
    let bytes = fs::read(dir.join("values.bin")).ok()?;
    bincode::deserialize(&bytes).ok()
}

/// FNV-1a hash of the key's YAML serialization, as a 16-char hex string.
///
/// Uses YAML rather than `Hash` trait because `PostflopModelConfig` contains
/// `Vec<f32>` which doesn't implement `Hash`.
fn hex_hash(key: &SolveCacheKey) -> String {
    use std::hash::Hasher;
    let yaml = serde_yaml::to_string(key).expect("SolveCacheKey should always serialize");
    let mut hasher = super::fnv::FnvHasher::new();
    hasher.write(yaml.as_bytes());
    format!("{:016x}", hasher.finish())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;
    use test_macros::timed_test;

    fn fast_key(has_eq: bool) -> SolveCacheKey {
        cache_key(&PostflopModelConfig::fast(), has_eq)
    }

    #[timed_test]
    fn solve_cache_roundtrip() {
        let dir = TempDir::new().unwrap();
        let key = fast_key(false);
        let values = PostflopValues::empty();
        save(dir.path(), &key, &values).unwrap();
        let loaded = load(dir.path(), &key).expect("should load");
        assert!(loaded.is_empty());
    }

    #[timed_test]
    fn solve_cache_miss_returns_none() {
        let dir = TempDir::new().unwrap();
        let key = fast_key(false);
        assert!(load(dir.path(), &key).is_none());
    }

    #[timed_test]
    fn solve_cache_key_deterministic() {
        let k1 = fast_key(false);
        let k2 = fast_key(false);
        assert_eq!(hex_hash(&k1), hex_hash(&k2));
    }

    #[timed_test]
    fn solve_cache_key_differs_with_equity_flag() {
        let k1 = fast_key(false);
        let k2 = fast_key(true);
        assert_ne!(hex_hash(&k1), hex_hash(&k2));
    }

    #[timed_test]
    fn solve_cache_key_differs_with_config() {
        let k1 = fast_key(false);
        let mut config2 = PostflopModelConfig::fast();
        config2.postflop_solve_iterations = 999;
        let k2 = cache_key(&config2, false);
        assert_ne!(hex_hash(&k1), hex_hash(&k2));
    }

    #[timed_test]
    fn solve_cache_key_yaml_written() {
        let dir = TempDir::new().unwrap();
        let key = fast_key(false);
        let values = PostflopValues::empty();
        save(dir.path(), &key, &values).unwrap();
        let yaml_path = cache_dir(dir.path(), &key).join("key.yaml");
        assert!(yaml_path.exists());
        let contents = std::fs::read_to_string(yaml_path).unwrap();
        assert!(contents.contains("has_equity_table"));
    }
}
