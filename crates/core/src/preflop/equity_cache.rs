//! Disk cache for the 169x169 preflop equity table.
//!
//! Caches the expensive Monte Carlo equity computation keyed by sample count.
//!
//! # Layout
//!
//! ```text
//! cache/postflop/equity_<samples>/
//!   equity.bin      # bincode: EquityTable
//! ```

use std::fs;
use std::path::{Path, PathBuf};

use super::equity::EquityTable;

/// Error type for equity cache operations.
#[derive(Debug, thiserror::Error)]
pub enum EquityCacheError {
    #[error("equity cache I/O: {0}")]
    Io(#[from] std::io::Error),
    #[error("equity cache serialization: {0}")]
    Serialize(String),
}

/// Compute the cache directory for a given sample count.
#[must_use]
pub fn cache_dir(base: &Path, equity_samples: u32) -> PathBuf {
    base.join(format!("equity_{equity_samples}"))
}

/// Save an equity table to the cache.
///
/// # Errors
///
/// Returns `EquityCacheError` on I/O or serialization failure.
pub fn save(
    base: &Path,
    equity_samples: u32,
    table: &EquityTable,
) -> Result<(), EquityCacheError> {
    let dir = cache_dir(base, equity_samples);
    fs::create_dir_all(&dir)?;
    let bytes =
        bincode::serialize(table).map_err(|e| EquityCacheError::Serialize(e.to_string()))?;
    fs::write(dir.join("equity.bin"), bytes)?;
    Ok(())
}

/// Load a cached equity table, returning `None` if missing or corrupt.
#[must_use]
pub fn load(base: &Path, equity_samples: u32) -> Option<EquityTable> {
    let dir = cache_dir(base, equity_samples);
    let bytes = fs::read(dir.join("equity.bin")).ok()?;
    bincode::deserialize(&bytes).ok()
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;
    use test_macros::timed_test;

    #[timed_test]
    fn equity_cache_roundtrip() {
        let dir = TempDir::new().unwrap();
        let table = EquityTable::new_uniform();
        save(dir.path(), 20000, &table).unwrap();
        let loaded = load(dir.path(), 20000).expect("should load");
        assert_eq!(loaded.num_hands(), 169);
        assert!((loaded.equity(0, 1) - 0.5).abs() < 1e-9);
    }

    #[timed_test]
    fn equity_cache_miss_returns_none() {
        let dir = TempDir::new().unwrap();
        assert!(load(dir.path(), 99999).is_none());
    }

    #[timed_test]
    fn equity_cache_different_samples_different_dirs() {
        let dir = TempDir::new().unwrap();
        let table = EquityTable::new_uniform();
        save(dir.path(), 1000, &table).unwrap();
        assert!(load(dir.path(), 1000).is_some());
        assert!(load(dir.path(), 2000).is_none());
    }
}
