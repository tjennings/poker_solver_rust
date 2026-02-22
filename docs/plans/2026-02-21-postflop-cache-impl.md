# Postflop Cache Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Cache the equity table and postflop solve results so `solve-preflop` re-runs skip phases 1-7 when config hasn't changed.

**Architecture:** Two new cache modules (`equity_cache`, `solve_cache`) following the existing `abstraction_cache` pattern. Each uses content-addressed disk caching with bincode serialization under `cache/postflop/`.

**Tech Stack:** Rust, serde/bincode for serialization, FNV-1a hashing for cache keys, existing `abstraction_cache` as reference pattern.

---

### Task 1: Add Serialize/Deserialize to EquityTable

**Files:**
- Modify: `crates/core/src/preflop/equity.rs:18` (derive line)

**Step 1: Add serde derives to EquityTable**

Change line 18 from:
```rust
#[derive(Debug, Clone)]
pub struct EquityTable {
```
to:
```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EquityTable {
```

Add the serde import at the top of the file:
```rust
use serde::{Deserialize, Serialize};
```

**Step 2: Run tests to verify nothing breaks**

Run: `cargo test -p poker-solver-core --lib preflop::equity`
Expected: All existing equity tests pass.

**Step 3: Commit**

```bash
git add crates/core/src/preflop/equity.rs
git commit -m "feat: add Serialize/Deserialize to EquityTable"
```

---

### Task 2: Create equity_cache module

**Files:**
- Create: `crates/core/src/preflop/equity_cache.rs`
- Modify: `crates/core/src/preflop/mod.rs:1` (add module declaration)

**Step 1: Write the failing test**

Create `crates/core/src/preflop/equity_cache.rs` with the full module including tests at the bottom:

```rust
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
    let bytes = bincode::serialize(table)
        .map_err(|e| EquityCacheError::Serialize(e.to_string()))?;
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
```

**Step 2: Add module to mod.rs**

In `crates/core/src/preflop/mod.rs`, add after the `abstraction_cache` line:

```rust
pub mod equity_cache;
```

**Step 3: Run tests to verify they pass**

Run: `cargo test -p poker-solver-core --lib preflop::equity_cache`
Expected: 3 tests pass.

**Step 4: Commit**

```bash
git add crates/core/src/preflop/equity_cache.rs crates/core/src/preflop/mod.rs
git commit -m "feat: add equity_cache module for 169x169 equity table caching"
```

---

### Task 3: Add Serialize/Deserialize to PostflopValues

**Files:**
- Modify: `crates/core/src/preflop/postflop_abstraction.rs:45-48` (PostflopValues struct)

**Step 1: Add serde derives to PostflopValues**

The file already imports serde via other structs. Add derives to `PostflopValues`:

```rust
#[derive(Serialize, Deserialize)]
pub struct PostflopValues {
    values: Vec<f64>,
    num_buckets: usize,
}
```

Add serde import if not already present:
```rust
use serde::{Deserialize, Serialize};
```

**Step 2: Run tests to verify nothing breaks**

Run: `cargo test -p poker-solver-core --lib preflop::postflop_abstraction`
Expected: All existing tests pass.

**Step 3: Commit**

```bash
git add crates/core/src/preflop/postflop_abstraction.rs
git commit -m "feat: add Serialize/Deserialize to PostflopValues"
```

---

### Task 4: Create solve_cache module

**Files:**
- Create: `crates/core/src/preflop/solve_cache.rs`
- Modify: `crates/core/src/preflop/mod.rs` (add module declaration)

**Step 1: Write the module with tests**

Create `crates/core/src/preflop/solve_cache.rs`:

```rust
//! Disk cache for postflop solve results (PostflopValues).
//!
//! Caches the expensive postflop CFR solve keyed by the full PostflopModelConfig
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
use std::hash::Hasher;
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

    let key_yaml = serde_yaml::to_string(key)
        .map_err(|e| SolveCacheError::Serialize(e.to_string()))?;
    fs::write(dir.join("key.yaml"), key_yaml)?;

    let bytes = bincode::serialize(values)
        .map_err(|e| SolveCacheError::Serialize(e.to_string()))?;
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
    let yaml = serde_yaml::to_string(key).unwrap_or_default();
    let mut hasher = FnvHasher::new();
    hasher.write(yaml.as_bytes());
    format!("{:016x}", hasher.finish())
}

/// FNV-1a 64-bit hasher.
struct FnvHasher(u64);

impl FnvHasher {
    const OFFSET_BASIS: u64 = 0xcbf2_9ce4_8422_2325;
    const PRIME: u64 = 0x0000_0100_0000_01B3;

    fn new() -> Self {
        Self(Self::OFFSET_BASIS)
    }
}

impl Hasher for FnvHasher {
    fn finish(&self) -> u64 {
        self.0
    }

    fn write(&mut self, bytes: &[u8]) {
        for &byte in bytes {
            self.0 ^= u64::from(byte);
            self.0 = self.0.wrapping_mul(Self::PRIME);
        }
    }
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
```

**Step 2: Add PostflopValues::empty() constructor for testing**

In `crates/core/src/preflop/postflop_abstraction.rs`, add to `impl PostflopValues`:

```rust
    /// Create an empty value table (for testing).
    #[must_use]
    pub fn empty() -> Self {
        Self {
            values: Vec::new(),
            num_buckets: 0,
        }
    }
```

**Step 3: Add module to mod.rs**

In `crates/core/src/preflop/mod.rs`, add:

```rust
pub mod solve_cache;
```

**Step 4: Run tests**

Run: `cargo test -p poker-solver-core --lib preflop::solve_cache`
Expected: 6 tests pass.

**Step 5: Commit**

```bash
git add crates/core/src/preflop/solve_cache.rs crates/core/src/preflop/mod.rs crates/core/src/preflop/postflop_abstraction.rs
git commit -m "feat: add solve_cache module for postflop CFR values caching"
```

---

### Task 5: Refactor PostflopAbstraction::build to support cached values

**Files:**
- Modify: `crates/core/src/preflop/postflop_abstraction.rs:279-365`

**Step 1: Add a `build_with_cached_values` constructor**

Add a new method to `impl PostflopAbstraction` that accepts pre-loaded values instead of solving:

```rust
    /// Build PostflopAbstraction using pre-cached values, skipping the solve phase.
    ///
    /// Loads abstraction from cache, rebuilds trees (instant), and uses the
    /// provided `PostflopValues` directly.
    ///
    /// # Errors
    ///
    /// Returns an error if tree building fails.
    pub fn build_from_cached(
        config: &PostflopModelConfig,
        board: BoardAbstraction,
        buckets: HandBucketMapping,
        bucket_equity: BucketEquity,
        values: PostflopValues,
    ) -> Result<Self, PostflopAbstractionError> {
        let trees = build_all_trees(config)?;
        Ok(Self {
            board,
            buckets,
            bucket_equity,
            trees,
            values,
        })
    }
```

**Step 2: Extract solve phase from build() into a public method**

Extract the solve + value computation into a standalone method so the trainer can call it independently and cache the result. Add to `impl PostflopAbstraction`:

```rust
    /// Run the postflop CFR solve and compute the value table.
    ///
    /// This is the expensive phase (minutes). The result can be cached via `solve_cache`.
    pub fn solve_values(
        config: &PostflopModelConfig,
        trees: &FxHashMap<PotType, PostflopTree>,
        bucket_equity: &BucketEquity,
        num_flop_buckets: usize,
        num_turn_buckets: usize,
        num_river_buckets: usize,
        on_progress: impl Fn(BuildPhase) + Sync,
    ) -> PostflopValues {
        let node_streets: FxHashMap<PotType, Vec<Street>> = trees
            .iter()
            .map(|(&pt, tree)| (pt, annotate_streets(tree)))
            .collect();

        let pt_layouts = build_per_pot_type_layouts(
            trees,
            &node_streets,
            num_flop_buckets,
            num_turn_buckets,
            num_river_buckets,
        );

        let total_iters = config.postflop_solve_iterations as usize;
        let samples = if config.postflop_solve_samples > 0 {
            config.postflop_solve_samples as usize
        } else {
            num_flop_buckets
        };
        let total_steps = total_iters * NUM_POT_TYPES;
        on_progress(BuildPhase::SolvingPostflop(0, total_steps));

        let pt_strategy_sums = solve_postflop_per_pot_type(
            trees,
            &pt_layouts,
            bucket_equity,
            num_flop_buckets,
            total_iters,
            samples,
            |step, total| on_progress(BuildPhase::SolvingPostflop(step, total)),
        );

        on_progress(BuildPhase::ComputingValues);
        compute_postflop_values(
            trees,
            &pt_layouts,
            bucket_equity,
            &pt_strategy_sums,
            num_flop_buckets,
        )
    }
```

**Step 3: Refactor existing `build()` to use `solve_values`**

Replace the solve section in `build()` (lines ~312-355) with:

```rust
        on_progress(BuildPhase::Trees);
        let trees = build_all_trees(config)?;

        let values = Self::solve_values(
            config,
            &trees,
            &bucket_equity,
            num_b,
            buckets.num_turn_buckets as usize,
            buckets.num_river_buckets as usize,
            &on_progress,
        );

        Ok(Self {
            board,
            buckets,
            bucket_equity,
            trees,
            values,
        })
```

This removes the duplicated layout/solve/value code from `build()`.

**Step 4: Run tests**

Run: `cargo test -p poker-solver-core --lib preflop::postflop_abstraction`
Expected: All existing tests still pass.

**Step 5: Commit**

```bash
git add crates/core/src/preflop/postflop_abstraction.rs
git commit -m "refactor: extract solve_values and build_from_cached on PostflopAbstraction"
```

---

### Task 6: Wire caching into run_solve_preflop

**Files:**
- Modify: `crates/trainer/src/main.rs:755-821` (the equity + postflop build section)

**Step 1: Wire equity cache**

Replace the equity computation block (lines ~755-775) with cache-aware version:

```rust
    let cache_base = std::path::Path::new("cache/postflop");

    let equity = if equity_samples > 0 {
        // Try equity cache first.
        if let Some(cached) = poker_solver_core::preflop::equity_cache::load(cache_base, equity_samples) {
            eprintln!("Equity cache hit: {}", poker_solver_core::preflop::equity_cache::cache_dir(cache_base, equity_samples).display());
            cached
        } else {
            let total_pairs = 169 * 168 / 2;
            let eq_pb = ProgressBar::new(total_pairs as u64);
            eq_pb.set_style(
                ProgressStyle::default_bar()
                    .template("{spinner:.green} Computing equities [{bar:40.cyan/blue}] {pos}/{len} ({eta})")
                    .expect("valid template")
                    .progress_chars("#>-"),
            );
            let eq_start = Instant::now();
            let table = EquityTable::new_computed(equity_samples, |done| {
                eq_pb.set_position(done as u64);
            });
            eq_pb.finish_with_message("done");
            println!("Equity table built in {:.1?}", eq_start.elapsed());

            if let Err(e) = poker_solver_core::preflop::equity_cache::save(cache_base, equity_samples, &table) {
                eprintln!("Warning: failed to save equity cache: {e}");
            } else {
                eprintln!("Equity cache saved: {}", poker_solver_core::preflop::equity_cache::cache_dir(cache_base, equity_samples).display());
            }

            lhe_viz::print_equity_matrix(&table);
            table
        }
    } else {
        println!("Using uniform equities (--equity-samples 0)");
        EquityTable::new_uniform()
    };
```

**Step 2: Wire solve cache**

Replace the postflop build block (lines ~784-821) with cache-aware version:

```rust
    let postflop = if let Some(pf_config) = &config.postflop_model {
        use poker_solver_core::preflop::solve_cache;
        use poker_solver_core::preflop::abstraction_cache;
        use poker_solver_core::preflop::postflop_abstraction::{PostflopAbstraction, BuildPhase};

        let has_eq = equity_samples > 0;
        let sk = solve_cache::cache_key(pf_config, has_eq);

        // Try full solve cache (phases 2-7 all cached).
        let abstraction_key = abstraction_cache::cache_key(pf_config, has_eq);
        if let Some(values) = solve_cache::load(cache_base, &sk)
            && let Some((board, buckets, bucket_equity)) = abstraction_cache::load(cache_base, &abstraction_key)
        {
            let dir = solve_cache::cache_dir(cache_base, &sk);
            eprintln!("Solve cache hit: {}", dir.display());
            Some(PostflopAbstraction::build_from_cached(pf_config, board, buckets, bucket_equity, values)
                .map_err(|e| format!("postflop from cache: {e}"))?)
        } else {
            // Build with progress, then cache the values.
            let pf_pb = ProgressBar::new(0);
            pf_pb.set_style(
                ProgressStyle::default_bar()
                    .template("{spinner:.green} {msg} [{bar:40.cyan/blue}] {pos}/{len}")
                    .expect("valid template")
                    .progress_chars("#>-"),
            );
            pf_pb.set_message("Building postflop abstraction");
            let pf_start = Instant::now();
            let abstraction = PostflopAbstraction::build(pf_config, Some(&equity), Some(cache_base), |phase| {
                match &phase {
                    BuildPhase::HandBuckets(done, total) => {
                        pf_pb.set_length(*total as u64);
                        pf_pb.set_position(*done as u64);
                        pf_pb.set_message("Hand buckets");
                    }
                    BuildPhase::SolvingPostflop(iter, total) => {
                        pf_pb.set_length(*total as u64);
                        pf_pb.set_position(*iter as u64);
                        pf_pb.set_message("Solving postflop");
                    }
                    _ => {
                        pf_pb.set_message(format!("{phase}..."));
                    }
                }
            })
            .map_err(|e| format!("postflop abstraction: {e}"))?;
            pf_pb.finish_with_message(format!(
                "done in {:.1?} (values: {} entries)",
                pf_start.elapsed(),
                abstraction.values.len(),
            ));

            // Cache the solve values for next time.
            if let Err(e) = solve_cache::save(cache_base, &sk, &abstraction.values) {
                eprintln!("Warning: failed to save solve cache: {e}");
            } else {
                eprintln!("Solve cache saved: {}", solve_cache::cache_dir(cache_base, &sk).display());
            }

            Some(abstraction)
        }
    } else {
        None
    };
```

**Step 3: Remove the now-redundant `cache_base` definition that was inside the old postflop block**

The `let cache_base = ...` line that was at line 794 is now defined earlier (in the equity section). Remove the duplicate.

**Step 4: Run full test suite**

Run: `cargo test -p poker-solver-core && cargo test -p poker-solver-trainer`
Expected: All tests pass.

**Step 5: Manual smoke test**

Run: `cargo run -p poker-solver-trainer --release -- solve-preflop --stack-depth 20 --iterations 100 --postflop-model fast -o /tmp/test_preflop`
Expected: First run computes everything. Second run shows cache hits for equity, abstraction, and solve.

**Step 6: Commit**

```bash
git add crates/trainer/src/main.rs
git commit -m "feat: wire equity_cache and solve_cache into solve-preflop pipeline"
```

---

### Task 7: Add PostflopAbstractionError variant for solve cache

**Files:**
- Modify: `crates/core/src/preflop/postflop_abstraction.rs:229-240`

**Step 1: Add SolveCache variant**

Add to the `PostflopAbstractionError` enum:

```rust
    #[error("solve cache: {0}")]
    SolveCache(#[from] super::solve_cache::SolveCacheError),
```

This allows `?` propagation from solve cache operations in the core library if needed in future.

**Step 2: Run tests**

Run: `cargo test -p poker-solver-core --lib preflop`
Expected: All pass.

**Step 3: Commit**

```bash
git add crates/core/src/preflop/postflop_abstraction.rs
git commit -m "feat: add SolveCache error variant to PostflopAbstractionError"
```
