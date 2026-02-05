# Blueprint Solver Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement a blueprint poker solver with full HUNL game, strategy storage, and subgame solving with caching.

**Architecture:** New `HunlPostflop` game type implementing the `Game` trait, `BlueprintStrategy` for storing/loading trained strategies, `SubgameSolver` for real-time depth-limited solving, and `SubgameCache` for persistent caching of solved subgames.

**Tech Stack:** Rust, Burn (GPU), existing CardAbstraction, existing GpuCfrSolver, sled (disk cache), lru (memory cache)

---

## Task 1: Add Dependencies

**Files:**
- Modify: `crates/core/Cargo.toml`

**Step 1: Add lru and sled dependencies**

Edit `crates/core/Cargo.toml` to add:

```toml
[dependencies]
# ... existing deps ...
lru = "0.12"
sled = "0.34"
bincode = "1.3"
```

**Step 2: Verify compilation**

Run: `cargo check -p poker-solver-core --features gpu`
Expected: Compiles successfully

**Step 3: Commit**

```bash
git add crates/core/Cargo.toml
git commit -m "deps: add lru, sled, bincode for blueprint caching"
```

---

## Task 2: Create Blueprint Module Structure

**Files:**
- Create: `crates/core/src/blueprint/mod.rs`
- Create: `crates/core/src/blueprint/error.rs`
- Modify: `crates/core/src/lib.rs`

**Step 1: Write the failing test**

Add to `crates/core/src/blueprint/mod.rs`:

```rust
//! Blueprint Strategy Module
//!
//! Provides strategy storage, subgame solving, and caching for HUNL poker.

mod error;

pub use error::BlueprintError;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn blueprint_error_display() {
        let err = BlueprintError::IoError("test".to_string());
        let msg = format!("{}", err);
        assert!(msg.contains("test"));
    }
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test -p poker-solver-core blueprint_error_display --features gpu`
Expected: FAIL with "cannot find type `BlueprintError`"

**Step 3: Write minimal implementation**

Create `crates/core/src/blueprint/error.rs`:

```rust
//! Blueprint error types.

use std::fmt;

/// Errors that can occur in blueprint operations.
#[derive(Debug)]
pub enum BlueprintError {
    /// I/O error reading or writing files
    IoError(String),
    /// Serialization/deserialization error
    SerializationError(String),
    /// Invalid strategy data
    InvalidStrategy(String),
    /// Cache error
    CacheError(String),
    /// Info set not found
    InfoSetNotFound(String),
}

impl fmt::Display for BlueprintError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::IoError(msg) => write!(f, "IO error: {}", msg),
            Self::SerializationError(msg) => write!(f, "Serialization error: {}", msg),
            Self::InvalidStrategy(msg) => write!(f, "Invalid strategy: {}", msg),
            Self::CacheError(msg) => write!(f, "Cache error: {}", msg),
            Self::InfoSetNotFound(key) => write!(f, "Info set not found: {}", key),
        }
    }
}

impl std::error::Error for BlueprintError {}

impl From<std::io::Error> for BlueprintError {
    fn from(err: std::io::Error) -> Self {
        Self::IoError(err.to_string())
    }
}
```

**Step 4: Update lib.rs to include blueprint module**

Add to `crates/core/src/lib.rs` after the `pub mod game;` line:

```rust
pub mod blueprint;
```

**Step 5: Run test to verify it passes**

Run: `cargo test -p poker-solver-core blueprint_error_display --features gpu`
Expected: PASS

**Step 6: Commit**

```bash
git add crates/core/src/blueprint crates/core/src/lib.rs
git commit -m "feat(blueprint): add module structure and error types"
```

---

## Task 3: Implement BlueprintStrategy Storage

**Files:**
- Create: `crates/core/src/blueprint/strategy.rs`
- Modify: `crates/core/src/blueprint/mod.rs`

**Step 1: Write the failing test**

Add to `crates/core/src/blueprint/strategy.rs`:

```rust
//! Blueprint strategy storage and lookup.

use std::collections::HashMap;
use std::path::Path;

use serde::{Deserialize, Serialize};

use super::BlueprintError;

/// Stored blueprint strategy for HUNL poker.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlueprintStrategy {
    /// Map from info set key to action probabilities
    strategies: HashMap<String, Vec<f32>>,
    /// Number of training iterations
    iterations_trained: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_strategy_returns_none() {
        let strategy = BlueprintStrategy::new();
        assert!(strategy.lookup("nonexistent").is_none());
    }

    #[test]
    fn strategy_lookup_returns_inserted_values() {
        let mut strategy = BlueprintStrategy::new();
        strategy.insert("test|F|xbc".to_string(), vec![0.3, 0.7]);

        let probs = strategy.lookup("test|F|xbc").unwrap();
        assert_eq!(probs.len(), 2);
        assert!((probs[0] - 0.3).abs() < 1e-6);
        assert!((probs[1] - 0.7).abs() < 1e-6);
    }

    #[test]
    fn strategy_roundtrip_save_load() {
        let mut strategy = BlueprintStrategy::new();
        strategy.insert("k1".to_string(), vec![0.5, 0.5]);
        strategy.insert("k2".to_string(), vec![0.2, 0.3, 0.5]);
        strategy.set_iterations(1000);

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test_strategy.bin");

        strategy.save(&path).unwrap();
        let loaded = BlueprintStrategy::load(&path).unwrap();

        assert_eq!(loaded.iterations_trained(), 1000);
        assert_eq!(loaded.lookup("k1").unwrap(), &[0.5, 0.5]);
        assert_eq!(loaded.lookup("k2").unwrap(), &[0.2, 0.3, 0.5]);
    }
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test -p poker-solver-core strategy:: --features gpu`
Expected: FAIL with "cannot find function `new`"

**Step 3: Write minimal implementation**

Complete `crates/core/src/blueprint/strategy.rs`:

```rust
//! Blueprint strategy storage and lookup.

use std::collections::HashMap;
use std::path::Path;

use serde::{Deserialize, Serialize};

use super::BlueprintError;

/// Stored blueprint strategy for HUNL poker.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlueprintStrategy {
    /// Map from info set key to action probabilities
    strategies: HashMap<String, Vec<f32>>,
    /// Number of training iterations
    iterations_trained: u64,
}

impl BlueprintStrategy {
    /// Create an empty strategy.
    #[must_use]
    pub fn new() -> Self {
        Self {
            strategies: HashMap::new(),
            iterations_trained: 0,
        }
    }

    /// Create from a HashMap of strategies (from solver output).
    #[must_use]
    pub fn from_strategies(strategies: HashMap<String, Vec<f64>>, iterations: u64) -> Self {
        let strategies = strategies
            .into_iter()
            .map(|(k, v)| (k, v.into_iter().map(|x| x as f32).collect()))
            .collect();
        Self {
            strategies,
            iterations_trained: iterations,
        }
    }

    /// Insert a strategy for an info set.
    pub fn insert(&mut self, info_set: String, probs: Vec<f32>) {
        self.strategies.insert(info_set, probs);
    }

    /// Set the number of training iterations.
    pub fn set_iterations(&mut self, iterations: u64) {
        self.iterations_trained = iterations;
    }

    /// Lookup action probabilities for an info set.
    #[must_use]
    pub fn lookup(&self, info_set: &str) -> Option<&[f32]> {
        self.strategies.get(info_set).map(|v| v.as_slice())
    }

    /// Get number of training iterations.
    #[must_use]
    pub fn iterations_trained(&self) -> u64 {
        self.iterations_trained
    }

    /// Get number of info sets stored.
    #[must_use]
    pub fn len(&self) -> usize {
        self.strategies.len()
    }

    /// Check if strategy is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.strategies.is_empty()
    }

    /// Save strategy to file using bincode.
    pub fn save(&self, path: &Path) -> Result<(), BlueprintError> {
        let data = bincode::serialize(self)
            .map_err(|e| BlueprintError::SerializationError(e.to_string()))?;
        std::fs::write(path, data)?;
        Ok(())
    }

    /// Load strategy from file.
    pub fn load(path: &Path) -> Result<Self, BlueprintError> {
        let data = std::fs::read(path)?;
        bincode::deserialize(&data)
            .map_err(|e| BlueprintError::SerializationError(e.to_string()))
    }
}

impl Default for BlueprintStrategy {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_strategy_returns_none() {
        let strategy = BlueprintStrategy::new();
        assert!(strategy.lookup("nonexistent").is_none());
    }

    #[test]
    fn strategy_lookup_returns_inserted_values() {
        let mut strategy = BlueprintStrategy::new();
        strategy.insert("test|F|xbc".to_string(), vec![0.3, 0.7]);

        let probs = strategy.lookup("test|F|xbc").unwrap();
        assert_eq!(probs.len(), 2);
        assert!((probs[0] - 0.3).abs() < 1e-6);
        assert!((probs[1] - 0.7).abs() < 1e-6);
    }

    #[test]
    fn strategy_roundtrip_save_load() {
        let mut strategy = BlueprintStrategy::new();
        strategy.insert("k1".to_string(), vec![0.5, 0.5]);
        strategy.insert("k2".to_string(), vec![0.2, 0.3, 0.5]);
        strategy.set_iterations(1000);

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test_strategy.bin");

        strategy.save(&path).unwrap();
        let loaded = BlueprintStrategy::load(&path).unwrap();

        assert_eq!(loaded.iterations_trained(), 1000);
        assert_eq!(loaded.lookup("k1").unwrap(), &[0.5, 0.5]);
        assert_eq!(loaded.lookup("k2").unwrap(), &[0.2, 0.3, 0.5]);
    }

    #[test]
    fn from_strategies_converts_f64_to_f32() {
        let mut map = HashMap::new();
        map.insert("k".to_string(), vec![0.25f64, 0.75f64]);

        let strategy = BlueprintStrategy::from_strategies(map, 500);

        assert_eq!(strategy.iterations_trained(), 500);
        let probs = strategy.lookup("k").unwrap();
        assert!((probs[0] - 0.25).abs() < 1e-6);
    }

    #[test]
    fn len_and_is_empty() {
        let mut strategy = BlueprintStrategy::new();
        assert!(strategy.is_empty());
        assert_eq!(strategy.len(), 0);

        strategy.insert("k".to_string(), vec![1.0]);
        assert!(!strategy.is_empty());
        assert_eq!(strategy.len(), 1);
    }
}
```

**Step 4: Update mod.rs to export strategy**

Add to `crates/core/src/blueprint/mod.rs`:

```rust
//! Blueprint Strategy Module
//!
//! Provides strategy storage, subgame solving, and caching for HUNL poker.

mod error;
mod strategy;

pub use error::BlueprintError;
pub use strategy::BlueprintStrategy;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn blueprint_error_display() {
        let err = BlueprintError::IoError("test".to_string());
        let msg = format!("{}", err);
        assert!(msg.contains("test"));
    }
}
```

**Step 5: Run tests to verify they pass**

Run: `cargo test -p poker-solver-core blueprint:: --features gpu`
Expected: All tests PASS

**Step 6: Commit**

```bash
git add crates/core/src/blueprint/
git commit -m "feat(blueprint): implement BlueprintStrategy with save/load"
```

---

## Task 4: Implement SubgameCache

**Files:**
- Create: `crates/core/src/blueprint/cache.rs`
- Modify: `crates/core/src/blueprint/mod.rs`

**Step 1: Write the failing test**

Create `crates/core/src/blueprint/cache.rs`:

```rust
//! Subgame cache with LRU memory and sled disk backing.

use std::num::NonZeroUsize;
use std::path::Path;

use lru::LruCache;
use serde::{Deserialize, Serialize};

use super::BlueprintError;

/// Key for subgame cache lookups.
#[derive(Debug, Clone, Hash, Eq, PartialEq, Serialize, Deserialize)]
pub struct SubgameKey {
    /// Canonical board hash (suit-isomorphic)
    pub board_hash: u64,
    /// Holding bucket (not exact cards for better cache hits)
    pub holding_bucket: u16,
    /// Hash of action history
    pub history_hash: u64,
}

/// Configuration for subgame cache.
#[derive(Debug, Clone)]
pub struct CacheConfig {
    /// Maximum entries in memory LRU cache
    pub max_memory_entries: usize,
    /// Path to sled database for disk cache
    pub disk_path: Option<std::path::PathBuf>,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            max_memory_entries: 100_000,
            disk_path: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cache_miss_returns_none() {
        let config = CacheConfig::default();
        let cache = SubgameCache::new(config).unwrap();

        let key = SubgameKey {
            board_hash: 12345,
            holding_bucket: 100,
            history_hash: 999,
        };

        assert!(cache.get(&key).is_none());
    }

    #[test]
    fn cache_insert_then_get() {
        let config = CacheConfig::default();
        let mut cache = SubgameCache::new(config).unwrap();

        let key = SubgameKey {
            board_hash: 12345,
            holding_bucket: 100,
            history_hash: 999,
        };
        let probs = vec![0.3f32, 0.7f32];

        cache.insert(key.clone(), probs.clone()).unwrap();

        let retrieved = cache.get(&key).unwrap();
        assert_eq!(retrieved, probs);
    }

    #[test]
    fn cache_with_disk_persists() {
        let dir = tempfile::tempdir().unwrap();
        let disk_path = dir.path().join("cache_db");

        let key = SubgameKey {
            board_hash: 111,
            holding_bucket: 222,
            history_hash: 333,
        };
        let probs = vec![0.5f32, 0.5f32];

        // Insert and drop cache
        {
            let config = CacheConfig {
                max_memory_entries: 100,
                disk_path: Some(disk_path.clone()),
            };
            let mut cache = SubgameCache::new(config).unwrap();
            cache.insert(key.clone(), probs.clone()).unwrap();
        }

        // Reopen and verify persisted
        {
            let config = CacheConfig {
                max_memory_entries: 100,
                disk_path: Some(disk_path),
            };
            let cache = SubgameCache::new(config).unwrap();
            let retrieved = cache.get(&key).unwrap();
            assert_eq!(retrieved, probs);
        }
    }
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test -p poker-solver-core cache:: --features gpu`
Expected: FAIL with "cannot find type `SubgameCache`"

**Step 3: Write minimal implementation**

Complete the implementation in `crates/core/src/blueprint/cache.rs`:

```rust
//! Subgame cache with LRU memory and sled disk backing.

use std::num::NonZeroUsize;
use std::path::Path;
use std::sync::Mutex;

use lru::LruCache;
use serde::{Deserialize, Serialize};

use super::BlueprintError;

/// Key for subgame cache lookups.
#[derive(Debug, Clone, Hash, Eq, PartialEq, Serialize, Deserialize)]
pub struct SubgameKey {
    /// Canonical board hash (suit-isomorphic)
    pub board_hash: u64,
    /// Holding bucket (not exact cards for better cache hits)
    pub holding_bucket: u16,
    /// Hash of action history
    pub history_hash: u64,
}

impl SubgameKey {
    /// Create a new subgame key.
    #[must_use]
    pub fn new(board_hash: u64, holding_bucket: u16, history_hash: u64) -> Self {
        Self {
            board_hash,
            holding_bucket,
            history_hash,
        }
    }

    /// Convert to bytes for disk storage.
    fn to_bytes(&self) -> Vec<u8> {
        bincode::serialize(self).unwrap_or_default()
    }
}

/// Configuration for subgame cache.
#[derive(Debug, Clone)]
pub struct CacheConfig {
    /// Maximum entries in memory LRU cache
    pub max_memory_entries: usize,
    /// Path to sled database for disk cache
    pub disk_path: Option<std::path::PathBuf>,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            max_memory_entries: 100_000,
            disk_path: None,
        }
    }
}

/// Two-tier subgame cache: LRU memory + optional sled disk.
pub struct SubgameCache {
    /// In-memory LRU cache (mutex for interior mutability in get)
    memory: Mutex<LruCache<SubgameKey, Vec<f32>>>,
    /// Optional disk-backed cache
    disk: Option<sled::Db>,
}

impl SubgameCache {
    /// Create a new cache with the given configuration.
    pub fn new(config: CacheConfig) -> Result<Self, BlueprintError> {
        let capacity = NonZeroUsize::new(config.max_memory_entries)
            .unwrap_or(NonZeroUsize::new(1).unwrap());
        let memory = Mutex::new(LruCache::new(capacity));

        let disk = if let Some(path) = config.disk_path {
            Some(
                sled::open(&path)
                    .map_err(|e| BlueprintError::CacheError(e.to_string()))?,
            )
        } else {
            None
        };

        Ok(Self { memory, disk })
    }

    /// Get cached action probabilities for a subgame.
    /// Checks memory first, then disk if available.
    #[must_use]
    pub fn get(&self, key: &SubgameKey) -> Option<Vec<f32>> {
        // Check memory cache first
        {
            let mut memory = self.memory.lock().ok()?;
            if let Some(probs) = memory.get(key) {
                return Some(probs.clone());
            }
        }

        // Check disk cache
        if let Some(ref disk) = self.disk {
            let key_bytes = key.to_bytes();
            if let Ok(Some(data)) = disk.get(&key_bytes) {
                if let Ok(probs) = bincode::deserialize::<Vec<f32>>(&data) {
                    // Promote to memory cache
                    if let Ok(mut memory) = self.memory.lock() {
                        memory.put(key.clone(), probs.clone());
                    }
                    return Some(probs);
                }
            }
        }

        None
    }

    /// Insert action probabilities for a subgame.
    /// Writes to both memory and disk (if available).
    pub fn insert(&mut self, key: SubgameKey, probs: Vec<f32>) -> Result<(), BlueprintError> {
        // Insert into memory
        {
            let mut memory = self
                .memory
                .lock()
                .map_err(|e| BlueprintError::CacheError(e.to_string()))?;
            memory.put(key.clone(), probs.clone());
        }

        // Insert into disk if available
        if let Some(ref disk) = self.disk {
            let key_bytes = key.to_bytes();
            let value_bytes = bincode::serialize(&probs)
                .map_err(|e| BlueprintError::SerializationError(e.to_string()))?;
            disk.insert(key_bytes, value_bytes)
                .map_err(|e| BlueprintError::CacheError(e.to_string()))?;
        }

        Ok(())
    }

    /// Get number of entries in memory cache.
    #[must_use]
    pub fn memory_len(&self) -> usize {
        self.memory.lock().map(|m| m.len()).unwrap_or(0)
    }

    /// Clear both memory and disk caches.
    pub fn clear(&mut self) -> Result<(), BlueprintError> {
        if let Ok(mut memory) = self.memory.lock() {
            memory.clear();
        }
        if let Some(ref disk) = self.disk {
            disk.clear()
                .map_err(|e| BlueprintError::CacheError(e.to_string()))?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cache_miss_returns_none() {
        let config = CacheConfig::default();
        let cache = SubgameCache::new(config).unwrap();

        let key = SubgameKey {
            board_hash: 12345,
            holding_bucket: 100,
            history_hash: 999,
        };

        assert!(cache.get(&key).is_none());
    }

    #[test]
    fn cache_insert_then_get() {
        let config = CacheConfig::default();
        let mut cache = SubgameCache::new(config).unwrap();

        let key = SubgameKey {
            board_hash: 12345,
            holding_bucket: 100,
            history_hash: 999,
        };
        let probs = vec![0.3f32, 0.7f32];

        cache.insert(key.clone(), probs.clone()).unwrap();

        let retrieved = cache.get(&key).unwrap();
        assert_eq!(retrieved, probs);
    }

    #[test]
    fn cache_with_disk_persists() {
        let dir = tempfile::tempdir().unwrap();
        let disk_path = dir.path().join("cache_db");

        let key = SubgameKey {
            board_hash: 111,
            holding_bucket: 222,
            history_hash: 333,
        };
        let probs = vec![0.5f32, 0.5f32];

        // Insert and drop cache
        {
            let config = CacheConfig {
                max_memory_entries: 100,
                disk_path: Some(disk_path.clone()),
            };
            let mut cache = SubgameCache::new(config).unwrap();
            cache.insert(key.clone(), probs.clone()).unwrap();
        }

        // Reopen and verify persisted
        {
            let config = CacheConfig {
                max_memory_entries: 100,
                disk_path: Some(disk_path),
            };
            let cache = SubgameCache::new(config).unwrap();
            let retrieved = cache.get(&key).unwrap();
            assert_eq!(retrieved, probs);
        }
    }

    #[test]
    fn memory_len_tracks_entries() {
        let config = CacheConfig {
            max_memory_entries: 10,
            disk_path: None,
        };
        let mut cache = SubgameCache::new(config).unwrap();

        assert_eq!(cache.memory_len(), 0);

        for i in 0..5 {
            let key = SubgameKey::new(i, 0, 0);
            cache.insert(key, vec![0.5, 0.5]).unwrap();
        }

        assert_eq!(cache.memory_len(), 5);
    }

    #[test]
    fn clear_removes_all_entries() {
        let config = CacheConfig::default();
        let mut cache = SubgameCache::new(config).unwrap();

        let key = SubgameKey::new(1, 2, 3);
        cache.insert(key.clone(), vec![1.0]).unwrap();
        assert!(cache.get(&key).is_some());

        cache.clear().unwrap();
        assert!(cache.get(&key).is_none());
    }
}
```

**Step 4: Update mod.rs to export cache**

Update `crates/core/src/blueprint/mod.rs`:

```rust
//! Blueprint Strategy Module
//!
//! Provides strategy storage, subgame solving, and caching for HUNL poker.

mod cache;
mod error;
mod strategy;

pub use cache::{CacheConfig, SubgameCache, SubgameKey};
pub use error::BlueprintError;
pub use strategy::BlueprintStrategy;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn blueprint_error_display() {
        let err = BlueprintError::IoError("test".to_string());
        let msg = format!("{}", err);
        assert!(msg.contains("test"));
    }
}
```

**Step 5: Run tests to verify they pass**

Run: `cargo test -p poker-solver-core blueprint:: --features gpu`
Expected: All tests PASS

**Step 6: Commit**

```bash
git add crates/core/src/blueprint/
git commit -m "feat(blueprint): implement SubgameCache with LRU + sled"
```

---

## Task 5: Create HunlPostflop Game - State and Basic Structure

**Files:**
- Create: `crates/core/src/game/hunl_postflop.rs`
- Modify: `crates/core/src/game/mod.rs`

**Step 1: Write the failing test**

Create `crates/core/src/game/hunl_postflop.rs`:

```rust
//! Full Heads-Up No-Limit Texas Hold'em game with postflop streets.

use crate::abstraction::{CardAbstraction, Street};
use crate::poker::Card;

use super::{Action, Game, Player};

/// Configuration for HUNL postflop game.
#[derive(Debug, Clone)]
pub struct PostflopConfig {
    /// Stack depth in big blinds
    pub stack_depth: u32,
    /// Bet sizes as fractions of pot (e.g., [0.33, 0.5, 0.75, 1.0])
    pub bet_sizes: Vec<f32>,
    /// Number of random deals to sample per iteration
    pub samples_per_iteration: usize,
}

impl Default for PostflopConfig {
    fn default() -> Self {
        Self {
            stack_depth: 100,
            bet_sizes: vec![0.33, 0.5, 0.75, 1.0],
            samples_per_iteration: 1000,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn postflop_config_default() {
        let config = PostflopConfig::default();
        assert_eq!(config.stack_depth, 100);
        assert_eq!(config.bet_sizes.len(), 4);
        assert_eq!(config.samples_per_iteration, 1000);
    }

    #[test]
    fn postflop_state_initial_preflop() {
        let state = PostflopState::new_preflop(
            [Card::default(), Card::default()],
            [Card::default(), Card::default()],
            100,
        );
        assert_eq!(state.street, Street::Preflop);
        assert_eq!(state.pot, 15); // 0.5 + 1.0 BB in cents
    }
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test -p poker-solver-core hunl_postflop --features gpu`
Expected: FAIL - missing `PostflopState` and `Street::Preflop`

**Step 3: Add Street::Preflop to abstraction module**

First, modify `crates/core/src/abstraction/mod.rs` to add Preflop variant:

```rust
/// Street in poker (determines bucket count and calculation method)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Street {
    /// Preflop: 0 community cards
    Preflop,
    /// Flop: 3 community cards
    Flop,
    /// Turn: 4 community cards
    Turn,
    /// River: 5 community cards
    River,
}

impl Street {
    /// Determine street from board card count.
    pub fn from_board_len(len: usize) -> Result<Self, AbstractionError> {
        match len {
            0 => Ok(Street::Preflop),
            3 => Ok(Street::Flop),
            4 => Ok(Street::Turn),
            5 => Ok(Street::River),
            n => Err(AbstractionError::InvalidBoardSize {
                expected: 3,
                got: n,
            }),
        }
    }

    /// Returns the number of community cards for this street.
    #[must_use]
    pub const fn board_cards(self) -> usize {
        match self {
            Street::Preflop => 0,
            Street::Flop => 3,
            Street::Turn => 4,
            Street::River => 5,
        }
    }
}
```

**Step 4: Write PostflopState**

Add to `crates/core/src/game/hunl_postflop.rs`:

```rust
//! Full Heads-Up No-Limit Texas Hold'em game with postflop streets.

use std::sync::Arc;

use crate::abstraction::Street;
use crate::poker::Card;

use super::{Action, Game, Player};

/// Configuration for HUNL postflop game.
#[derive(Debug, Clone)]
pub struct PostflopConfig {
    /// Stack depth in big blinds (stored as cents for precision)
    pub stack_depth: u32,
    /// Bet sizes as fractions of pot (e.g., [0.33, 0.5, 0.75, 1.0])
    pub bet_sizes: Vec<f32>,
    /// Number of random deals to sample per iteration
    pub samples_per_iteration: usize,
}

impl Default for PostflopConfig {
    fn default() -> Self {
        Self {
            stack_depth: 100,
            bet_sizes: vec![0.33, 0.5, 0.75, 1.0],
            samples_per_iteration: 1000,
        }
    }
}

/// Terminal state type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TerminalType {
    /// Player folded
    Fold(Player),
    /// Showdown reached
    Showdown,
}

/// State of a HUNL postflop hand.
#[derive(Debug, Clone)]
pub struct PostflopState {
    /// Community cards (0-5)
    pub board: Vec<Card>,
    /// Player 1's hole cards
    pub p1_holding: [Card; 2],
    /// Player 2's hole cards
    pub p2_holding: [Card; 2],
    /// Current street
    pub street: Street,
    /// Total pot in cents (1 BB = 100 cents)
    pub pot: u32,
    /// Player stacks in cents [P1, P2]
    pub stacks: [u32; 2],
    /// Amount to call in cents
    pub to_call: u32,
    /// Who acts next
    pub to_act: Option<Player>,
    /// Action history for info set key
    pub history: Vec<(Street, Action)>,
    /// Terminal state if game over
    pub terminal: Option<TerminalType>,
    /// Number of bets on current street
    pub street_bets: u8,
}

impl PostflopState {
    /// Create initial preflop state.
    /// Stack depth is in BB, internally stored as cents.
    #[must_use]
    pub fn new_preflop(
        p1_holding: [Card; 2],
        p2_holding: [Card; 2],
        stack_depth_bb: u32,
    ) -> Self {
        let stack_cents = stack_depth_bb * 100;
        Self {
            board: Vec::new(),
            p1_holding,
            p2_holding,
            street: Street::Preflop,
            pot: 150, // SB (50) + BB (100)
            stacks: [stack_cents - 50, stack_cents - 100], // P1=SB, P2=BB
            to_call: 50, // SB needs to call 50 more
            to_act: Some(Player::Player1), // SB acts first preflop
            history: Vec::new(),
            terminal: None,
            street_bets: 1, // BB counts as first bet
        }
    }

    /// Get current player's stack.
    #[must_use]
    pub fn current_stack(&self) -> u32 {
        match self.to_act {
            Some(Player::Player1) => self.stacks[0],
            Some(Player::Player2) => self.stacks[1],
            None => 0,
        }
    }

    /// Get opponent's stack.
    #[must_use]
    pub fn opponent_stack(&self) -> u32 {
        match self.to_act {
            Some(Player::Player1) => self.stacks[1],
            Some(Player::Player2) => self.stacks[0],
            None => 0,
        }
    }

    /// Get the current player's holding.
    #[must_use]
    pub fn current_holding(&self) -> [Card; 2] {
        match self.to_act {
            Some(Player::Player1) => self.p1_holding,
            _ => self.p2_holding,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::poker::{Suit, Value};

    fn make_card(value: Value, suit: Suit) -> Card {
        Card::new(value, suit)
    }

    #[test]
    fn postflop_config_default() {
        let config = PostflopConfig::default();
        assert_eq!(config.stack_depth, 100);
        assert_eq!(config.bet_sizes.len(), 4);
        assert_eq!(config.samples_per_iteration, 1000);
    }

    #[test]
    fn postflop_state_initial_preflop() {
        let p1 = [
            make_card(Value::Ace, Suit::Spade),
            make_card(Value::King, Suit::Spade),
        ];
        let p2 = [
            make_card(Value::Queen, Suit::Heart),
            make_card(Value::Jack, Suit::Heart),
        ];
        let state = PostflopState::new_preflop(p1, p2, 100);

        assert_eq!(state.street, Street::Preflop);
        assert_eq!(state.pot, 150); // SB + BB
        assert_eq!(state.stacks[0], 9950); // SB posted 50
        assert_eq!(state.stacks[1], 9900); // BB posted 100
        assert_eq!(state.to_call, 50);
        assert_eq!(state.to_act, Some(Player::Player1));
    }

    #[test]
    fn current_and_opponent_stack() {
        let p1 = [
            make_card(Value::Ace, Suit::Spade),
            make_card(Value::King, Suit::Spade),
        ];
        let p2 = [
            make_card(Value::Queen, Suit::Heart),
            make_card(Value::Jack, Suit::Heart),
        ];
        let state = PostflopState::new_preflop(p1, p2, 100);

        // P1 to act
        assert_eq!(state.current_stack(), 9950);
        assert_eq!(state.opponent_stack(), 9900);
    }
}
```

**Step 5: Update game/mod.rs to export**

Add to `crates/core/src/game/mod.rs`:

```rust
mod hunl_postflop;
mod hunl_preflop;
mod kuhn;

pub use hunl_postflop::{PostflopConfig, PostflopState, TerminalType as PostflopTerminal};
pub use hunl_preflop::HunlPreflop;
pub use kuhn::KuhnPoker;
```

**Step 6: Run tests to verify they pass**

Run: `cargo test -p poker-solver-core hunl_postflop --features gpu`
Expected: All tests PASS

**Step 7: Commit**

```bash
git add crates/core/src/game/ crates/core/src/abstraction/mod.rs
git commit -m "feat(game): add HunlPostflop state and config structures"
```

---

## Task 6: Implement HunlPostflop Game Trait - Actions and Transitions

**Files:**
- Modify: `crates/core/src/game/hunl_postflop.rs`

**Step 1: Write the failing test**

Add tests to `crates/core/src/game/hunl_postflop.rs`:

```rust
#[test]
fn preflop_actions_include_fold_call_raise() {
    let p1 = [
        make_card(Value::Ace, Suit::Spade),
        make_card(Value::King, Suit::Spade),
    ];
    let p2 = [
        make_card(Value::Queen, Suit::Heart),
        make_card(Value::Jack, Suit::Heart),
    ];
    let state = PostflopState::new_preflop(p1, p2, 100);
    let config = PostflopConfig::default();
    let game = HunlPostflop::new(config, None);

    let actions = game.actions(&state);

    // SB facing BB: can fold, call, or raise
    assert!(actions.contains(&Action::Fold));
    assert!(actions.contains(&Action::Call));
    // Should have at least one raise size
    assert!(actions.iter().any(|a| matches!(a, Action::Raise(_))));
}

#[test]
fn fold_ends_game() {
    let p1 = [
        make_card(Value::Ace, Suit::Spade),
        make_card(Value::King, Suit::Spade),
    ];
    let p2 = [
        make_card(Value::Queen, Suit::Heart),
        make_card(Value::Jack, Suit::Heart),
    ];
    let state = PostflopState::new_preflop(p1, p2, 100);
    let config = PostflopConfig::default();
    let game = HunlPostflop::new(config, None);

    let new_state = game.next_state(&state, Action::Fold);

    assert!(game.is_terminal(&new_state));
    assert_eq!(new_state.terminal, Some(PostflopTerminal::Fold(Player::Player1)));
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test -p poker-solver-core preflop_actions --features gpu`
Expected: FAIL - `HunlPostflop` not found

**Step 3: Implement HunlPostflop with Game trait**

Add to `crates/core/src/game/hunl_postflop.rs`:

```rust
use std::sync::Arc;
use rand::prelude::*;
use crate::abstraction::{CardAbstraction, Street};
use crate::poker::{Card, FlatDeck};

/// Full HUNL game including postflop streets.
pub struct HunlPostflop {
    config: PostflopConfig,
    abstraction: Option<Arc<CardAbstraction>>,
    rng_seed: u64,
}

impl HunlPostflop {
    /// Create a new HUNL postflop game.
    #[must_use]
    pub fn new(config: PostflopConfig, abstraction: Option<Arc<CardAbstraction>>) -> Self {
        Self {
            config,
            abstraction,
            rng_seed: 42,
        }
    }

    /// Set RNG seed for reproducible sampling.
    pub fn set_seed(&mut self, seed: u64) {
        self.rng_seed = seed;
    }

    /// Get bet sizes based on pot.
    fn get_bet_sizes(&self, pot: u32, stack: u32) -> Vec<u32> {
        let mut sizes = Vec::new();
        for &fraction in &self.config.bet_sizes {
            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
            let size = (pot as f32 * fraction).round() as u32;
            if size > 0 && size <= stack {
                sizes.push(size);
            }
        }
        // Always include all-in if not already
        if !sizes.contains(&stack) && stack > 0 {
            sizes.push(stack);
        }
        sizes
    }
}

impl Game for HunlPostflop {
    type State = PostflopState;

    fn initial_states(&self) -> Vec<Self::State> {
        let mut rng = StdRng::seed_from_u64(self.rng_seed);
        let mut states = Vec::with_capacity(self.config.samples_per_iteration);
        let stack = self.config.stack_depth;

        for _ in 0..self.config.samples_per_iteration {
            let mut deck = FlatDeck::default();
            deck.shuffle(&mut rng);

            let p1 = [deck.deal().unwrap(), deck.deal().unwrap()];
            let p2 = [deck.deal().unwrap(), deck.deal().unwrap()];

            states.push(PostflopState::new_preflop(p1, p2, stack));
        }

        states
    }

    fn is_terminal(&self, state: &Self::State) -> bool {
        state.terminal.is_some()
    }

    fn player(&self, state: &Self::State) -> Player {
        state.to_act.unwrap_or(Player::Player1)
    }

    fn actions(&self, state: &Self::State) -> Vec<Action> {
        if state.terminal.is_some() {
            return vec![];
        }

        let mut actions = Vec::new();
        let stack = state.current_stack();
        let to_call = state.to_call;

        // Can fold if facing a bet
        if to_call > 0 {
            actions.push(Action::Fold);
        }

        // Check or call
        if to_call == 0 {
            actions.push(Action::Check);
        } else if stack >= to_call {
            actions.push(Action::Call);
        }

        // Raise/bet sizes
        let effective_stack = stack.saturating_sub(to_call);
        if effective_stack > 0 {
            let bet_sizes = self.get_bet_sizes(state.pot, effective_stack);
            for size in bet_sizes {
                if to_call == 0 {
                    actions.push(Action::Bet(size));
                } else {
                    actions.push(Action::Raise(to_call + size));
                }
            }
        }

        actions
    }

    fn next_state(&self, state: &Self::State, action: Action) -> Self::State {
        let mut new_state = state.clone();
        let is_p1 = state.to_act == Some(Player::Player1);
        let player_idx = if is_p1 { 0 } else { 1 };

        new_state.history.push((state.street, action));

        match action {
            Action::Fold => {
                let folder = state.to_act.unwrap_or(Player::Player1);
                new_state.terminal = Some(TerminalType::Fold(folder));
                new_state.to_act = None;
            }

            Action::Check => {
                // If both players checked, advance street or showdown
                let last_action = state.history.last().map(|(_, a)| a);
                if matches!(last_action, Some(Action::Check)) {
                    // Both checked - advance street
                    self.advance_street(&mut new_state);
                } else {
                    // First check, switch to opponent
                    new_state.to_act = Some(state.to_act.unwrap().opponent());
                }
            }

            Action::Call => {
                let call_amount = state.to_call.min(state.current_stack());
                new_state.stacks[player_idx] -= call_amount;
                new_state.pot += call_amount;
                new_state.to_call = 0;

                // Advance street or showdown
                self.advance_street(&mut new_state);
            }

            Action::Bet(amount) | Action::Raise(amount) => {
                let actual_amount = amount.min(state.current_stack());
                new_state.stacks[player_idx] -= actual_amount;
                new_state.pot += actual_amount;
                new_state.to_call = actual_amount - state.to_call;
                new_state.street_bets += 1;
                new_state.to_act = Some(state.to_act.unwrap().opponent());

                // Check if opponent is all-in
                if new_state.opponent_stack() == 0 {
                    new_state.terminal = Some(TerminalType::Showdown);
                    new_state.to_act = None;
                }
            }
        }

        new_state
    }

    fn utility(&self, state: &Self::State, player: Player) -> f64 {
        let Some(terminal) = state.terminal else {
            return 0.0;
        };

        let stack_cents = self.config.stack_depth * 100;
        let p1_invested = stack_cents - state.stacks[0];
        let p2_invested = stack_cents - state.stacks[1];

        match terminal {
            TerminalType::Fold(folder) => {
                if folder == Player::Player1 {
                    if player == Player::Player1 {
                        -(p1_invested as f64) / 100.0
                    } else {
                        p1_invested as f64 / 100.0
                    }
                } else if player == Player::Player2 {
                    -(p2_invested as f64) / 100.0
                } else {
                    p2_invested as f64 / 100.0
                }
            }
            TerminalType::Showdown => {
                // For now, use 50% equity (will be replaced with actual hand eval)
                let pot = (p1_invested + p2_invested) as f64 / 100.0;
                let p1_equity = 0.5; // TODO: actual hand evaluation
                let p1_ev = p1_equity * pot - (p1_invested as f64 / 100.0);

                if player == Player::Player1 {
                    p1_ev
                } else {
                    -p1_ev
                }
            }
        }
    }

    fn info_set_key(&self, state: &Self::State) -> String {
        let holding = state.current_holding();

        // Get bucket if abstraction available, otherwise use placeholder
        let bucket = if let Some(ref abstraction) = self.abstraction {
            if state.board.is_empty() {
                // Preflop: no bucket, use hand string
                format!("{}{}", card_to_char(holding[0]), card_to_char(holding[1]))
            } else {
                abstraction
                    .get_bucket(&state.board, (holding[0], holding[1]))
                    .map(|b| b.to_string())
                    .unwrap_or_else(|_| "?".to_string())
            }
        } else {
            format!("{}{}", card_to_char(holding[0]), card_to_char(holding[1]))
        };

        let street_char = match state.street {
            Street::Preflop => 'P',
            Street::Flop => 'F',
            Street::Turn => 'T',
            Street::River => 'R',
        };

        let history_str: String = state
            .history
            .iter()
            .map(|(_, a)| action_to_char(a))
            .collect();

        format!("{}|{}|{}", bucket, street_char, history_str)
    }
}

impl HunlPostflop {
    fn advance_street(&self, state: &mut PostflopState) {
        match state.street {
            Street::Preflop => {
                state.street = Street::Flop;
                // Deal flop (3 cards) - for now placeholder
                // In MCCFR, board cards are sampled when needed
                state.street_bets = 0;
                state.to_call = 0;
                // Postflop: P1 (SB) is OOP, acts first
                state.to_act = Some(Player::Player1);
            }
            Street::Flop => {
                state.street = Street::Turn;
                state.street_bets = 0;
                state.to_call = 0;
                state.to_act = Some(Player::Player1);
            }
            Street::Turn => {
                state.street = Street::River;
                state.street_bets = 0;
                state.to_call = 0;
                state.to_act = Some(Player::Player1);
            }
            Street::River => {
                // River checked through - showdown
                state.terminal = Some(TerminalType::Showdown);
                state.to_act = None;
            }
        }
    }
}

fn card_to_char(card: Card) -> char {
    // Simple representation for info set key
    let rank = match card.value {
        crate::poker::Value::Two => '2',
        crate::poker::Value::Three => '3',
        crate::poker::Value::Four => '4',
        crate::poker::Value::Five => '5',
        crate::poker::Value::Six => '6',
        crate::poker::Value::Seven => '7',
        crate::poker::Value::Eight => '8',
        crate::poker::Value::Nine => '9',
        crate::poker::Value::Ten => 'T',
        crate::poker::Value::Jack => 'J',
        crate::poker::Value::Queen => 'Q',
        crate::poker::Value::King => 'K',
        crate::poker::Value::Ace => 'A',
    };
    rank
}

fn action_to_char(action: &Action) -> String {
    match action {
        Action::Fold => "f".to_string(),
        Action::Check => "x".to_string(),
        Action::Call => "c".to_string(),
        Action::Bet(amt) => format!("b{}", amt),
        Action::Raise(amt) => format!("r{}", amt),
    }
}
```

**Step 4: Run tests to verify they pass**

Run: `cargo test -p poker-solver-core hunl_postflop --features gpu`
Expected: All tests PASS

**Step 5: Commit**

```bash
git add crates/core/src/game/hunl_postflop.rs
git commit -m "feat(game): implement HunlPostflop Game trait with actions and transitions"
```

---

## Task 7: Implement Showdown Equity Calculation

**Files:**
- Modify: `crates/core/src/game/hunl_postflop.rs`

**Step 1: Write the failing test**

Add test to `crates/core/src/game/hunl_postflop.rs`:

```rust
#[test]
fn showdown_utility_uses_hand_evaluation() {
    // AA vs 72o on Kh Qh Jh 2d 3c - AA should have high equity
    let p1 = [
        make_card(Value::Ace, Suit::Spade),
        make_card(Value::Ace, Suit::Club),
    ];
    let p2 = [
        make_card(Value::Seven, Suit::Diamond),
        make_card(Value::Two, Suit::Spade),
    ];

    let board = vec![
        make_card(Value::King, Suit::Heart),
        make_card(Value::Queen, Suit::Heart),
        make_card(Value::Jack, Suit::Heart),
        make_card(Value::Two, Suit::Diamond),
        make_card(Value::Three, Suit::Club),
    ];

    let mut state = PostflopState::new_preflop(p1, p2, 100);
    state.board = board;
    state.street = Street::River;
    state.terminal = Some(PostflopTerminal::Showdown);
    state.pot = 2000; // 20 BB pot

    let config = PostflopConfig::default();
    let game = HunlPostflop::new(config, None);

    // AA has pair of aces with K-Q-J kickers vs 72 has two pair (2s and small pair)
    // Actually 72 made two pair with 2d on board - let me recalc
    // P1: AA -> pair of aces with K Q J kickers
    // P2: 72 -> two pair 2s and... wait 7 doesn't pair. Just pair of 2s.
    // AA wins, P1 should have positive utility
    let p1_utility = game.utility(&state, Player::Player1);
    assert!(p1_utility > 0.0, "AA should beat 72o, got {}", p1_utility);
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test -p poker-solver-core showdown_utility --features gpu`
Expected: May fail if using 50% equity placeholder

**Step 3: Implement proper hand evaluation**

Update the `utility` function in `HunlPostflop` to use rs_poker's hand evaluation:

```rust
fn utility(&self, state: &Self::State, player: Player) -> f64 {
    let Some(terminal) = state.terminal else {
        return 0.0;
    };

    let stack_cents = self.config.stack_depth * 100;
    let p1_invested = stack_cents - state.stacks[0];
    let p2_invested = stack_cents - state.stacks[1];

    match terminal {
        TerminalType::Fold(folder) => {
            if folder == Player::Player1 {
                if player == Player::Player1 {
                    -(p1_invested as f64) / 100.0
                } else {
                    p1_invested as f64 / 100.0
                }
            } else if player == Player::Player2 {
                -(p2_invested as f64) / 100.0
            } else {
                p2_invested as f64 / 100.0
            }
        }
        TerminalType::Showdown => {
            use crate::poker::{Hand, Rankable};

            // Build hands for evaluation
            let mut p1_cards = state.board.clone();
            p1_cards.extend_from_slice(&state.p1_holding);
            let p1_hand = Hand::new_with_cards(p1_cards);
            let p1_rank = p1_hand.rank();

            let mut p2_cards = state.board.clone();
            p2_cards.extend_from_slice(&state.p2_holding);
            let p2_hand = Hand::new_with_cards(p2_cards);
            let p2_rank = p2_hand.rank();

            let pot = (p1_invested + p2_invested) as f64 / 100.0;

            // Determine winner (lower rank is better in rs_poker)
            let p1_ev = if p1_rank < p2_rank {
                // P1 wins
                pot - (p1_invested as f64 / 100.0)
            } else if p1_rank > p2_rank {
                // P2 wins
                -(p1_invested as f64 / 100.0)
            } else {
                // Tie - split pot
                pot / 2.0 - (p1_invested as f64 / 100.0)
            };

            if player == Player::Player1 {
                p1_ev
            } else {
                -p1_ev
            }
        }
    }
}
```

**Step 4: Run tests to verify they pass**

Run: `cargo test -p poker-solver-core showdown_utility --features gpu`
Expected: PASS

**Step 5: Commit**

```bash
git add crates/core/src/game/hunl_postflop.rs
git commit -m "feat(game): implement proper showdown hand evaluation"
```

---

## Task 8: Integration Test - HunlPostflop with GPU Solver

**Files:**
- Create: `crates/core/tests/hunl_postflop_integration.rs`

**Step 1: Write the failing test**

Create `crates/core/tests/hunl_postflop_integration.rs`:

```rust
//! Integration tests for HunlPostflop with GPU solver.

#![cfg(feature = "gpu")]

use poker_solver_core::game::{HunlPostflop, PostflopConfig};
use poker_solver_core::cfr::GpuCfrSolver;
use burn::backend::ndarray::NdArray;

type TestBackend = NdArray;

#[test]
fn hunl_postflop_compiles_and_trains() {
    let config = PostflopConfig {
        stack_depth: 20, // Short stack for faster test
        bet_sizes: vec![0.5, 1.0],
        samples_per_iteration: 10, // Very few samples for test
    };
    let game = HunlPostflop::new(config, None);
    let device = Default::default();

    let mut solver = GpuCfrSolver::<TestBackend>::new(&game, device);
    solver.train(5);

    assert_eq!(solver.iterations(), 5);
}

#[test]
fn hunl_postflop_produces_valid_strategies() {
    let config = PostflopConfig {
        stack_depth: 20,
        bet_sizes: vec![0.5, 1.0],
        samples_per_iteration: 10,
    };
    let game = HunlPostflop::new(config, None);
    let device = Default::default();

    let mut solver = GpuCfrSolver::<TestBackend>::new(&game, device);
    solver.train(10);

    let strategies = solver.all_strategies();

    // Should have some strategies
    assert!(!strategies.is_empty(), "Should have some strategies");

    // Each strategy should sum to ~1.0
    for (key, probs) in &strategies {
        let sum: f64 = probs.iter().sum();
        assert!(
            (sum - 1.0).abs() < 0.01,
            "Strategy {} should sum to 1.0, got {}",
            key,
            sum
        );
    }
}
```

**Step 2: Run test to verify it compiles and runs**

Run: `cargo test -p poker-solver-core hunl_postflop_integration --features gpu -- --nocapture`
Expected: Tests should pass (may take a few seconds)

**Step 3: Commit**

```bash
git add crates/core/tests/hunl_postflop_integration.rs
git commit -m "test: add HunlPostflop integration tests with GPU solver"
```

---

## Task 9: Implement SubgameSolver Structure

**Files:**
- Create: `crates/core/src/blueprint/subgame.rs`
- Modify: `crates/core/src/blueprint/mod.rs`

**Step 1: Write the failing test**

Create `crates/core/src/blueprint/subgame.rs`:

```rust
//! Subgame solver for real-time depth-limited solving.

use std::sync::Arc;

use crate::abstraction::CardAbstraction;
use crate::poker::Card;
use crate::game::Action;

use super::{BlueprintStrategy, SubgameCache, CacheConfig, SubgameKey, BlueprintError};

/// Configuration for subgame solving.
#[derive(Debug, Clone)]
pub struct SubgameConfig {
    /// Maximum depth to solve (number of actions)
    pub depth_limit: usize,
    /// Time budget in milliseconds
    pub time_budget_ms: u64,
    /// Number of CFR iterations if time permits
    pub max_iterations: u32,
}

impl Default for SubgameConfig {
    fn default() -> Self {
        Self {
            depth_limit: 4,
            time_budget_ms: 200,
            max_iterations: 1000,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn subgame_config_default() {
        let config = SubgameConfig::default();
        assert_eq!(config.depth_limit, 4);
        assert_eq!(config.time_budget_ms, 200);
        assert_eq!(config.max_iterations, 1000);
    }

    #[test]
    fn subgame_solver_creates() {
        let blueprint = BlueprintStrategy::new();
        let config = SubgameConfig::default();
        let cache_config = CacheConfig::default();

        let solver = SubgameSolver::new(
            Arc::new(blueprint),
            None,
            config,
            cache_config,
        ).unwrap();

        assert_eq!(solver.config().depth_limit, 4);
    }
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test -p poker-solver-core subgame:: --features gpu`
Expected: FAIL - `SubgameSolver` not found

**Step 3: Write minimal implementation**

Complete `crates/core/src/blueprint/subgame.rs`:

```rust
//! Subgame solver for real-time depth-limited solving.

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use crate::abstraction::CardAbstraction;
use crate::game::Action;
use crate::poker::Card;

use super::{BlueprintError, BlueprintStrategy, CacheConfig, SubgameCache, SubgameKey};

/// Configuration for subgame solving.
#[derive(Debug, Clone)]
pub struct SubgameConfig {
    /// Maximum depth to solve (number of actions)
    pub depth_limit: usize,
    /// Time budget in milliseconds
    pub time_budget_ms: u64,
    /// Number of CFR iterations if time permits
    pub max_iterations: u32,
}

impl Default for SubgameConfig {
    fn default() -> Self {
        Self {
            depth_limit: 4,
            time_budget_ms: 200,
            max_iterations: 1000,
        }
    }
}

/// Real-time subgame solver using blueprint and caching.
pub struct SubgameSolver {
    blueprint: Arc<BlueprintStrategy>,
    abstraction: Option<Arc<CardAbstraction>>,
    cache: SubgameCache,
    config: SubgameConfig,
}

impl SubgameSolver {
    /// Create a new subgame solver.
    pub fn new(
        blueprint: Arc<BlueprintStrategy>,
        abstraction: Option<Arc<CardAbstraction>>,
        config: SubgameConfig,
        cache_config: CacheConfig,
    ) -> Result<Self, BlueprintError> {
        let cache = SubgameCache::new(cache_config)?;
        Ok(Self {
            blueprint,
            abstraction,
            cache,
            config,
        })
    }

    /// Get the configuration.
    #[must_use]
    pub fn config(&self) -> &SubgameConfig {
        &self.config
    }

    /// Solve for action probabilities at a given game state.
    ///
    /// First checks the cache, then falls back to blueprint if no cache hit.
    /// Full subgame solving is a TODO for future implementation.
    pub fn solve(
        &mut self,
        board: &[Card],
        holding: [Card; 2],
        history: &[(crate::abstraction::Street, Action)],
    ) -> Result<Vec<f32>, BlueprintError> {
        // Create cache key
        let key = self.make_cache_key(board, holding, history)?;

        // Check cache first
        if let Some(probs) = self.cache.get(&key) {
            return Ok(probs);
        }

        // Fall back to blueprint lookup
        let info_set = self.make_info_set_key(board, holding, history)?;
        if let Some(probs) = self.blueprint.lookup(&info_set) {
            let probs = probs.to_vec();
            self.cache.insert(key, probs.clone())?;
            return Ok(probs);
        }

        // If no blueprint entry, return uniform
        // TODO: Implement actual subgame solving
        Err(BlueprintError::InfoSetNotFound(info_set))
    }

    fn make_cache_key(
        &self,
        board: &[Card],
        holding: [Card; 2],
        history: &[(crate::abstraction::Street, Action)],
    ) -> Result<SubgameKey, BlueprintError> {
        // Hash board canonically
        let mut board_hasher = DefaultHasher::new();
        for card in board {
            card.hash(&mut board_hasher);
        }
        let board_hash = board_hasher.finish();

        // Get bucket for holding
        let holding_bucket = if let Some(ref abstraction) = self.abstraction {
            if board.is_empty() {
                0 // Preflop placeholder
            } else {
                abstraction.get_bucket(board, (holding[0], holding[1]))?
            }
        } else {
            0
        };

        // Hash history
        let mut history_hasher = DefaultHasher::new();
        history.hash(&mut history_hasher);
        let history_hash = history_hasher.finish();

        Ok(SubgameKey::new(board_hash, holding_bucket, history_hash))
    }

    fn make_info_set_key(
        &self,
        board: &[Card],
        holding: [Card; 2],
        history: &[(crate::abstraction::Street, Action)],
    ) -> Result<String, BlueprintError> {
        let bucket = if let Some(ref abstraction) = self.abstraction {
            if board.is_empty() {
                "P".to_string() // Preflop
            } else {
                abstraction
                    .get_bucket(board, (holding[0], holding[1]))?
                    .to_string()
            }
        } else {
            "?".to_string()
        };

        let street_char = if board.is_empty() {
            'P'
        } else {
            match board.len() {
                3 => 'F',
                4 => 'T',
                5 => 'R',
                _ => '?',
            }
        };

        let history_str: String = history
            .iter()
            .map(|(_, a)| match a {
                Action::Fold => "f".to_string(),
                Action::Check => "x".to_string(),
                Action::Call => "c".to_string(),
                Action::Bet(amt) => format!("b{}", amt),
                Action::Raise(amt) => format!("r{}", amt),
            })
            .collect();

        Ok(format!("{}|{}|{}", bucket, street_char, history_str))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn subgame_config_default() {
        let config = SubgameConfig::default();
        assert_eq!(config.depth_limit, 4);
        assert_eq!(config.time_budget_ms, 200);
        assert_eq!(config.max_iterations, 1000);
    }

    #[test]
    fn subgame_solver_creates() {
        let blueprint = BlueprintStrategy::new();
        let config = SubgameConfig::default();
        let cache_config = CacheConfig::default();

        let solver = SubgameSolver::new(Arc::new(blueprint), None, config, cache_config).unwrap();

        assert_eq!(solver.config().depth_limit, 4);
    }

    #[test]
    fn subgame_solver_returns_cached() {
        let mut blueprint = BlueprintStrategy::new();
        blueprint.insert("100|F|xbc".to_string(), vec![0.3, 0.7]);

        let config = SubgameConfig::default();
        let cache_config = CacheConfig::default();

        let mut solver =
            SubgameSolver::new(Arc::new(blueprint), None, config, cache_config).unwrap();

        // Manually insert into cache
        let key = SubgameKey::new(123, 0, 456);
        solver.cache.insert(key.clone(), vec![0.5, 0.5]).unwrap();

        // Should get cached value (not blueprint)
        let probs = solver.cache.get(&key).unwrap();
        assert_eq!(probs, vec![0.5, 0.5]);
    }
}
```

**Step 4: Update mod.rs to export subgame**

Update `crates/core/src/blueprint/mod.rs`:

```rust
//! Blueprint Strategy Module
//!
//! Provides strategy storage, subgame solving, and caching for HUNL poker.

mod cache;
mod error;
mod strategy;
mod subgame;

pub use cache::{CacheConfig, SubgameCache, SubgameKey};
pub use error::BlueprintError;
pub use strategy::BlueprintStrategy;
pub use subgame::{SubgameConfig, SubgameSolver};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn blueprint_error_display() {
        let err = BlueprintError::IoError("test".to_string());
        let msg = format!("{}", err);
        assert!(msg.contains("test"));
    }
}
```

**Step 5: Run tests to verify they pass**

Run: `cargo test -p poker-solver-core blueprint:: --features gpu`
Expected: All tests PASS

**Step 6: Commit**

```bash
git add crates/core/src/blueprint/
git commit -m "feat(blueprint): implement SubgameSolver with cache integration"
```

---

## Task 10: Full Integration Test

**Files:**
- Create: `crates/core/tests/blueprint_integration.rs`

**Step 1: Write the integration test**

Create `crates/core/tests/blueprint_integration.rs`:

```rust
//! Integration tests for the full blueprint system.

#![cfg(feature = "gpu")]

use std::sync::Arc;

use poker_solver_core::blueprint::{
    BlueprintStrategy, CacheConfig, SubgameConfig, SubgameSolver,
};
use poker_solver_core::cfr::GpuCfrSolver;
use poker_solver_core::game::{HunlPostflop, PostflopConfig};
use burn::backend::ndarray::NdArray;

type TestBackend = NdArray;

#[test]
fn full_blueprint_pipeline() {
    // 1. Create game
    let game_config = PostflopConfig {
        stack_depth: 20,
        bet_sizes: vec![0.5, 1.0],
        samples_per_iteration: 10,
    };
    let game = HunlPostflop::new(game_config, None);

    // 2. Train with GPU solver
    let device = Default::default();
    let mut solver = GpuCfrSolver::<TestBackend>::new(&game, device);
    solver.train(20);

    // 3. Extract blueprint strategy
    let strategies = solver.all_strategies();
    let blueprint = BlueprintStrategy::from_strategies(strategies, solver.iterations());

    assert!(blueprint.len() > 0, "Blueprint should have strategies");
    assert_eq!(blueprint.iterations_trained(), 20);

    // 4. Create subgame solver
    let subgame_config = SubgameConfig::default();
    let cache_config = CacheConfig::default();
    let _subgame_solver = SubgameSolver::new(
        Arc::new(blueprint),
        None,
        subgame_config,
        cache_config,
    )
    .unwrap();
}

#[test]
fn blueprint_save_and_load() {
    let game_config = PostflopConfig {
        stack_depth: 20,
        bet_sizes: vec![0.5, 1.0],
        samples_per_iteration: 10,
    };
    let game = HunlPostflop::new(game_config, None);

    let device = Default::default();
    let mut solver = GpuCfrSolver::<TestBackend>::new(&game, device);
    solver.train(10);

    let strategies = solver.all_strategies();
    let blueprint = BlueprintStrategy::from_strategies(strategies, solver.iterations());

    // Save
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("blueprint.bin");
    blueprint.save(&path).unwrap();

    // Load
    let loaded = BlueprintStrategy::load(&path).unwrap();
    assert_eq!(loaded.iterations_trained(), blueprint.iterations_trained());
    assert_eq!(loaded.len(), blueprint.len());
}
```

**Step 2: Run tests**

Run: `cargo test -p poker-solver-core blueprint_integration --features gpu -- --nocapture`
Expected: All tests PASS

**Step 3: Commit**

```bash
git add crates/core/tests/blueprint_integration.rs
git commit -m "test: add full blueprint system integration tests"
```

---

## Task 11: Run All Tests and Clippy

**Step 1: Run all tests**

Run: `cargo test -p poker-solver-core --features gpu`
Expected: All tests PASS

**Step 2: Run clippy**

Run: `cargo clippy -p poker-solver-core --features gpu -- -D warnings`
Expected: No warnings

**Step 3: Fix any clippy issues**

If clippy reports issues, fix them.

**Step 4: Final commit**

```bash
git add -A
git commit -m "chore: fix clippy warnings and finalize blueprint module"
```

---

## Summary

This plan implements:

1. **BlueprintStrategy** - Storage for trained strategies with save/load
2. **SubgameCache** - Two-tier LRU + sled caching
3. **HunlPostflop** - Full HUNL game with Game trait implementation
4. **SubgameSolver** - Structure for real-time solving (basic implementation)

**Not implemented (future work):**
- Actual depth-limited subgame CFR solving
- Board card sampling during MCCFR
- Multi-valued opponent states
- Cache warming utilities

**Total: 11 tasks**
