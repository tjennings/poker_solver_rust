//! Subgame cache with two-tier storage (memory + disk)
//!
//! Provides caching for subgame solver results with an in-memory LRU cache
//! backed by optional disk persistence via sled.

use std::num::NonZeroUsize;
use std::path::PathBuf;
use std::sync::Mutex;

use lru::LruCache;
use serde::{Deserialize, Serialize};

use super::BlueprintError;

/// Key for identifying cached subgame results
#[derive(Debug, Clone, Copy, Hash, Eq, PartialEq, Serialize, Deserialize)]
pub struct SubgameKey {
    /// Canonical board hash
    board_hash: u64,
    /// Bucket from `CardAbstraction`
    holding_bucket: u16,
    /// Hash of action history
    history_hash: u64,
}

impl SubgameKey {
    /// Create a new subgame key
    #[must_use]
    pub fn new(board_hash: u64, holding_bucket: u16, history_hash: u64) -> Self {
        Self {
            board_hash,
            holding_bucket,
            history_hash,
        }
    }

    /// Convert to bytes for disk storage using bincode
    ///
    /// # Panics
    ///
    /// This function will not panic in practice as `SubgameKey` contains only
    /// fixed-size primitive types that are always serializable by bincode.
    #[must_use]
    pub fn to_bytes(&self) -> Vec<u8> {
        // bincode::serialize is infallible for this type since all fields are fixed-size primitives
        bincode::serialize(self)
            .expect("SubgameKey serialization is infallible for primitive types")
    }
}

/// Configuration for the subgame cache
#[derive(Debug, Clone)]
pub struct CacheConfig {
    /// Maximum number of entries in the in-memory LRU cache
    pub max_memory_entries: usize,
    /// Optional path for disk-backed storage
    pub disk_path: Option<PathBuf>,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            max_memory_entries: 100_000,
            disk_path: None,
        }
    }
}

/// Two-tier cache for subgame solver results
///
/// Provides fast in-memory access via LRU cache with optional
/// disk persistence via sled for cross-session caching.
pub struct SubgameCache {
    /// In-memory LRU cache
    memory: Mutex<LruCache<SubgameKey, Vec<f32>>>,
    /// Optional disk-backed storage
    disk: Option<sled::Db>,
}

impl SubgameCache {
    /// Create a new subgame cache with the given configuration
    ///
    /// # Errors
    ///
    /// Returns `BlueprintError::CacheError` if:
    /// - `max_memory_entries` is zero
    /// - The disk database cannot be opened at the specified path
    pub fn new(config: CacheConfig) -> Result<Self, BlueprintError> {
        let capacity = NonZeroUsize::new(config.max_memory_entries).ok_or_else(|| {
            BlueprintError::CacheError("max_memory_entries must be > 0".to_string())
        })?;

        let memory = Mutex::new(LruCache::new(capacity));

        let disk = match config.disk_path {
            Some(path) => {
                let db = sled::open(&path).map_err(|e| {
                    BlueprintError::CacheError(format!(
                        "failed to open sled db at {}: {e}",
                        path.display()
                    ))
                })?;
                Some(db)
            }
            None => None,
        };

        Ok(Self { memory, disk })
    }

    /// Get cached probabilities for a subgame key
    ///
    /// Checks memory first, then disk. On disk hit, promotes to memory.
    ///
    /// # Panics
    ///
    /// Panics if the memory lock is poisoned (indicating a panic in another thread
    /// while holding the lock).
    pub fn get(&self, key: &SubgameKey) -> Option<Vec<f32>> {
        // Check memory first
        {
            let mut memory = self.memory.lock().expect("memory lock poisoned");
            if let Some(probs) = memory.get(key) {
                return Some(probs.clone());
            }
        }

        // Check disk if available
        let disk = self.disk.as_ref()?;
        let key_bytes = key.to_bytes();
        let value_bytes = disk.get(&key_bytes).ok()??;
        let probs = bincode::deserialize::<Vec<f32>>(&value_bytes).ok()?;
        // Promote to memory cache
        let mut memory = self.memory.lock().expect("memory lock poisoned");
        memory.put(*key, probs.clone());
        Some(probs)
    }

    /// Insert probabilities into the cache
    ///
    /// Writes to both memory and disk (if configured).
    ///
    /// # Errors
    ///
    /// Returns `BlueprintError::SerializationError` if probabilities cannot be serialized,
    /// or `BlueprintError::CacheError` if disk write fails.
    ///
    /// # Panics
    ///
    /// Panics if the memory lock is poisoned (indicating a panic in another thread
    /// while holding the lock).
    #[allow(clippy::needless_pass_by_value)]
    pub fn insert(&self, key: SubgameKey, probs: Vec<f32>) -> Result<(), BlueprintError> {
        // Write to memory (clone probs since we may need it for disk)
        let probs_for_disk = if self.disk.is_some() {
            Some(probs.clone())
        } else {
            None
        };
        {
            let mut memory = self.memory.lock().expect("memory lock poisoned");
            memory.put(key, probs);
        }

        // Write to disk if available
        if let Some(ref disk) = self.disk {
            let key_bytes = key.to_bytes();
            // Safe to unwrap: we checked self.disk.is_some() above
            let probs = probs_for_disk.expect("probs_for_disk should exist when disk is Some");
            let value_bytes = bincode::serialize(&probs).map_err(|e| {
                BlueprintError::SerializationError(format!("failed to serialize probs: {e}"))
            })?;
            disk.insert(key_bytes, value_bytes).map_err(|e| {
                BlueprintError::CacheError(format!("failed to insert into sled: {e}"))
            })?;
        }

        Ok(())
    }

    /// Get the number of entries in the memory cache
    ///
    /// # Panics
    ///
    /// Panics if the memory lock is poisoned (indicating a panic in another thread
    /// while holding the lock).
    #[must_use]
    pub fn memory_len(&self) -> usize {
        let memory = self.memory.lock().expect("memory lock poisoned");
        memory.len()
    }

    /// Clear all entries from both memory and disk
    ///
    /// # Errors
    ///
    /// Returns `BlueprintError::CacheError` if clearing the disk database fails.
    ///
    /// # Panics
    ///
    /// Panics if the memory lock is poisoned (indicating a panic in another thread
    /// while holding the lock).
    pub fn clear(&self) -> Result<(), BlueprintError> {
        // Clear memory
        {
            let mut memory = self.memory.lock().expect("memory lock poisoned");
            memory.clear();
        }

        // Clear disk if available
        if let Some(ref disk) = self.disk {
            disk.clear()
                .map_err(|e| BlueprintError::CacheError(format!("failed to clear sled db: {e}")))?;
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;
    use test_macros::timed_test;

    #[timed_test]
    fn cache_miss_returns_none() {
        let config = CacheConfig::default();
        let cache = SubgameCache::new(config).expect("cache creation should succeed");

        let key = SubgameKey::new(123, 45, 678);
        let result = cache.get(&key);

        assert!(result.is_none(), "cache miss should return None");
    }

    #[timed_test]
    fn cache_insert_then_get() {
        let config = CacheConfig::default();
        let cache = SubgameCache::new(config).expect("cache creation should succeed");

        let key = SubgameKey::new(123, 45, 678);
        let probs = vec![0.25, 0.35, 0.40];

        cache
            .insert(key, probs.clone())
            .expect("insert should succeed");

        let result = cache.get(&key);
        assert_eq!(result, Some(probs), "should retrieve inserted probs");
    }

    #[timed_test]
    fn cache_with_disk_persists() {
        let dir = tempdir().expect("tempdir creation should succeed");
        let db_path = dir.path().join("cache_db");

        let key = SubgameKey::new(999, 88, 777);
        let probs = vec![0.1, 0.2, 0.3, 0.4];

        // Insert into cache with disk backing
        {
            let config = CacheConfig {
                max_memory_entries: 100,
                disk_path: Some(db_path.clone()),
            };
            let cache = SubgameCache::new(config).expect("cache creation should succeed");

            cache
                .insert(key, probs.clone())
                .expect("insert should succeed");
        }

        // Reopen cache and verify data persists
        {
            let config = CacheConfig {
                max_memory_entries: 100,
                disk_path: Some(db_path),
            };
            let cache = SubgameCache::new(config).expect("cache reopen should succeed");

            let result = cache.get(&key);
            assert_eq!(
                result,
                Some(probs),
                "data should persist across cache instances"
            );
        }
    }

    #[timed_test]
    fn memory_len_tracks_entries() {
        let config = CacheConfig {
            max_memory_entries: 1000,
            disk_path: None,
        };
        let cache = SubgameCache::new(config).expect("cache creation should succeed");

        assert_eq!(cache.memory_len(), 0, "empty cache should have length 0");

        cache
            .insert(SubgameKey::new(1, 1, 1), vec![0.5, 0.5])
            .expect("insert should succeed");
        assert_eq!(
            cache.memory_len(),
            1,
            "should have 1 entry after first insert"
        );

        cache
            .insert(SubgameKey::new(2, 2, 2), vec![0.3, 0.7])
            .expect("insert should succeed");
        assert_eq!(
            cache.memory_len(),
            2,
            "should have 2 entries after second insert"
        );

        cache
            .insert(SubgameKey::new(3, 3, 3), vec![0.1, 0.9])
            .expect("insert should succeed");
        assert_eq!(
            cache.memory_len(),
            3,
            "should have 3 entries after third insert"
        );
    }

    #[timed_test]
    fn clear_removes_all_entries() {
        let dir = tempdir().expect("tempdir creation should succeed");
        let db_path = dir.path().join("cache_db");

        let config = CacheConfig {
            max_memory_entries: 100,
            disk_path: Some(db_path),
        };
        let cache = SubgameCache::new(config).expect("cache creation should succeed");

        // Insert some entries
        cache
            .insert(SubgameKey::new(1, 1, 1), vec![0.5, 0.5])
            .expect("insert should succeed");
        cache
            .insert(SubgameKey::new(2, 2, 2), vec![0.3, 0.7])
            .expect("insert should succeed");

        assert_eq!(cache.memory_len(), 2, "should have 2 entries before clear");

        cache.clear().expect("clear should succeed");

        assert_eq!(cache.memory_len(), 0, "memory should be empty after clear");

        // Verify keys are no longer accessible
        assert!(
            cache.get(&SubgameKey::new(1, 1, 1)).is_none(),
            "key 1 should be gone after clear"
        );
        assert!(
            cache.get(&SubgameKey::new(2, 2, 2)).is_none(),
            "key 2 should be gone after clear"
        );
    }

    #[timed_test]
    fn subgame_key_to_bytes_is_deterministic() {
        let key1 = SubgameKey::new(100, 50, 200);
        let key2 = SubgameKey::new(100, 50, 200);

        assert_eq!(
            key1.to_bytes(),
            key2.to_bytes(),
            "same key should produce same bytes"
        );
    }

    #[timed_test]
    fn zero_capacity_returns_error() {
        let config = CacheConfig {
            max_memory_entries: 0,
            disk_path: None,
        };
        let result = SubgameCache::new(config);

        assert!(result.is_err(), "zero capacity should return error");
        if let Err(BlueprintError::CacheError(msg)) = result {
            assert!(
                msg.contains("must be > 0"),
                "error message should mention capacity requirement"
            );
        } else {
            panic!("expected CacheError variant");
        }
    }

    #[timed_test]
    fn lru_eviction_works() {
        let config = CacheConfig {
            max_memory_entries: 2,
            disk_path: None,
        };
        let cache = SubgameCache::new(config).expect("cache creation should succeed");

        let key1 = SubgameKey::new(1, 1, 1);
        let key2 = SubgameKey::new(2, 2, 2);
        let key3 = SubgameKey::new(3, 3, 3);

        cache
            .insert(key1, vec![0.1])
            .expect("insert should succeed");
        cache
            .insert(key2, vec![0.2])
            .expect("insert should succeed");

        // Access key1 to make it recently used
        cache.get(&key1);

        // Insert key3, should evict key2 (least recently used)
        cache
            .insert(key3, vec![0.3])
            .expect("insert should succeed");

        assert_eq!(cache.memory_len(), 2, "cache should be at capacity");
        assert!(cache.get(&key1).is_some(), "key1 should still be present");
        assert!(cache.get(&key2).is_none(), "key2 should be evicted");
        assert!(cache.get(&key3).is_some(), "key3 should be present");
    }

    #[timed_test]
    fn disk_promotion_to_memory() {
        let dir = tempdir().expect("tempdir creation should succeed");
        let db_path = dir.path().join("cache_db");

        let key = SubgameKey::new(42, 42, 42);
        let probs = vec![0.25, 0.25, 0.25, 0.25];

        // Insert directly to disk
        {
            let db = sled::open(&db_path).expect("sled open should succeed");
            let key_bytes = key.to_bytes();
            let value_bytes = bincode::serialize(&probs).expect("serialize should succeed");
            db.insert(key_bytes, value_bytes)
                .expect("insert should succeed");
            db.flush().expect("flush should succeed");
        }

        // Create cache and verify promotion
        let config = CacheConfig {
            max_memory_entries: 100,
            disk_path: Some(db_path),
        };
        let cache = SubgameCache::new(config).expect("cache creation should succeed");

        assert_eq!(cache.memory_len(), 0, "memory should be empty initially");

        let result = cache.get(&key);
        assert_eq!(result, Some(probs), "should retrieve from disk");

        assert_eq!(cache.memory_len(), 1, "should be promoted to memory");
    }
}
