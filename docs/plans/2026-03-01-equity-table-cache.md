# Equity Table Cache Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Precompute all 1755 canonical flop equity tables to disk, then auto-load them during postflop solves to eliminate per-flop startup latency.

**Architecture:** New `equity_table_cache` module in `core` handles serialization/deserialization of a binary cache file. A new `precompute-equity` CLI subcommand computes all tables with a progress bar. The existing `solve-postflop` handler auto-detects the cache at `./cache/equity_tables.bin` and passes tables via the existing `pre_equity_tables` parameter.

**Tech Stack:** bincode for serialization, indicatif for progress bar, rayon for parallel computation. No new dependencies.

---

### Task 1: Create `EquityTableCache` module in core

**Files:**
- Create: `crates/core/src/preflop/equity_table_cache.rs`
- Modify: `crates/core/src/preflop/mod.rs`

**Step 1: Create the cache module with struct and serialization**

Create `crates/core/src/preflop/equity_table_cache.rs`:

```rust
//! Disk cache for precomputed flop equity tables.
//!
//! Each canonical flop requires an expensive O(169² × combos² × 990 runouts) equity
//! computation. Since equity depends only on flop cards (not SPR, bet sizing, or solver
//! parameters), these tables can be computed once and reused across all training runs.
//!
//! File format: bincode-serialized `CacheData` with a 4-byte magic header and version.
//! All 1755 canonical flops are stored in deterministic order (`canonical_flops()`).
//! Flops are encoded as `[u8; 3]` using `value * 4 + suit` per card.

use std::io;
use std::path::Path;

use super::postflop_hands::{canonical_flops, build_combo_map, NUM_CANONICAL_HANDS};
use super::postflop_exhaustive::compute_equity_table;
use rs_poker::core::Card;
use serde::{Serialize, Deserialize};

const MAGIC: [u8; 4] = *b"EQTC";
const VERSION: u32 = 1;

/// On-disk representation (bincode-serialized).
#[derive(Serialize, Deserialize)]
struct CacheData {
    magic: [u8; 4],
    version: u32,
    num_flops: u32,
    /// Flops encoded as [value*4+suit; 3] per flop, flattened.
    flop_bytes: Vec<u8>,
    /// Equity tables flattened: num_flops × 169 × 169 f64s.
    tables_flat: Vec<f64>,
}

/// In-memory equity table cache.
pub struct EquityTableCache {
    /// Canonical flops in deterministic order.
    flops: Vec<[Card; 3]>,
    /// Flat array: flops × 169 × 169.
    tables_flat: Vec<f64>,
}

fn encode_card(c: Card) -> u8 {
    c.value as u8 * 4 + c.suit as u8
}

fn decode_card(b: u8) -> Card {
    use rs_poker::core::{Suit, Value};
    let value = unsafe { std::mem::transmute::<u8, Value>(b / 4) };
    let suit = unsafe { std::mem::transmute::<u8, Suit>(b % 4) };
    Card { value, suit }
}

impl EquityTableCache {
    /// Number of entries per equity table (169 × 169).
    const TABLE_SIZE: usize = NUM_CANONICAL_HANDS * NUM_CANONICAL_HANDS;

    /// Build the cache by computing equity tables for all canonical flops.
    ///
    /// Calls `on_progress(completed, total)` after each flop finishes.
    pub fn build(on_progress: impl Fn(usize, usize) + Sync) -> Self {
        use rayon::prelude::*;
        use std::sync::atomic::{AtomicUsize, Ordering};

        let flops = canonical_flops();
        let total = flops.len();
        let completed = AtomicUsize::new(0);

        let tables: Vec<Vec<f64>> = flops
            .par_iter()
            .map(|flop| {
                let combo_map = build_combo_map(flop);
                let table = compute_equity_table(&combo_map, *flop);
                let done = completed.fetch_add(1, Ordering::Relaxed) + 1;
                on_progress(done, total);
                table
            })
            .collect();

        let mut tables_flat = Vec::with_capacity(total * Self::TABLE_SIZE);
        for table in tables {
            tables_flat.extend_from_slice(&table);
        }

        Self { flops, tables_flat }
    }

    /// Save the cache to a binary file.
    pub fn save(&self, path: &Path) -> io::Result<()> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let mut flop_bytes = Vec::with_capacity(self.flops.len() * 3);
        for flop in &self.flops {
            flop_bytes.push(encode_card(flop[0]));
            flop_bytes.push(encode_card(flop[1]));
            flop_bytes.push(encode_card(flop[2]));
        }
        let data = CacheData {
            magic: MAGIC,
            version: VERSION,
            num_flops: self.flops.len() as u32,
            flop_bytes,
            tables_flat: self.tables_flat.clone(),
        };
        let bytes = bincode::serialize(&data)
            .map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;
        std::fs::write(path, bytes)
    }

    /// Load a cache from a binary file.
    ///
    /// Returns `None` if the file doesn't exist, has an invalid magic/version,
    /// or fails to deserialize.
    pub fn load(path: &Path) -> Option<Self> {
        let bytes = std::fs::read(path).ok()?;
        let data: CacheData = bincode::deserialize(&bytes).ok()?;
        if data.magic != MAGIC || data.version != VERSION {
            return None;
        }
        let num_flops = data.num_flops as usize;
        if data.flop_bytes.len() != num_flops * 3 {
            return None;
        }
        if data.tables_flat.len() != num_flops * Self::TABLE_SIZE {
            return None;
        }
        let mut flops = Vec::with_capacity(num_flops);
        for i in 0..num_flops {
            let base = i * 3;
            flops.push([
                decode_card(data.flop_bytes[base]),
                decode_card(data.flop_bytes[base + 1]),
                decode_card(data.flop_bytes[base + 2]),
            ]);
        }
        Some(Self { flops, tables_flat: data.tables_flat })
    }

    /// Number of flops in the cache.
    pub fn num_flops(&self) -> usize {
        self.flops.len()
    }

    /// Extract equity tables for a specific set of flops (in the order given).
    ///
    /// For each requested flop, looks up its index in the cache and copies
    /// the corresponding 169×169 table. Returns `None` for any flop not
    /// found in the cache.
    pub fn extract_tables(&self, requested_flops: &[[Card; 3]]) -> Option<Vec<Vec<f64>>> {
        use rustc_hash::FxHashMap;
        let mut index_map: FxHashMap<[u8; 3], usize> = FxHashMap::default();
        for (i, flop) in self.flops.iter().enumerate() {
            let key = [encode_card(flop[0]), encode_card(flop[1]), encode_card(flop[2])];
            index_map.insert(key, i);
        }

        let mut tables = Vec::with_capacity(requested_flops.len());
        for flop in requested_flops {
            let key = [encode_card(flop[0]), encode_card(flop[1]), encode_card(flop[2])];
            let idx = index_map.get(&key)?;
            let start = idx * Self::TABLE_SIZE;
            let end = start + Self::TABLE_SIZE;
            tables.push(self.tables_flat[start..end].to_vec());
        }
        Some(tables)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn card_round_trip() {
        use rs_poker::core::{Suit, Value};
        let card = Card { value: Value::Ace, suit: Suit::Spade };
        let encoded = encode_card(card);
        let decoded = decode_card(encoded);
        assert_eq!(card, decoded);
    }

    #[test]
    fn save_load_round_trip() {
        // Build a tiny cache with just 1 flop to test serialization
        let flops = canonical_flops();
        let flop = flops[0];
        let combo_map = build_combo_map(&flop);
        let table = compute_equity_table(&combo_map, flop);

        let cache = EquityTableCache {
            flops: vec![flop],
            tables_flat: table.clone(),
        };

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test_equity.bin");
        cache.save(&path).unwrap();

        let loaded = EquityTableCache::load(&path).unwrap();
        assert_eq!(loaded.num_flops(), 1);
        assert_eq!(loaded.flops[0], flop);
        assert_eq!(loaded.tables_flat.len(), table.len());

        // Verify values match (NaN-aware)
        for (a, b) in loaded.tables_flat.iter().zip(table.iter()) {
            if a.is_nan() {
                assert!(b.is_nan());
            } else {
                assert!((a - b).abs() < 1e-15);
            }
        }
    }

    #[test]
    fn extract_tables_finds_cached_flop() {
        let flops = canonical_flops();
        let flop = flops[0];
        let combo_map = build_combo_map(&flop);
        let table = compute_equity_table(&combo_map, flop);

        let cache = EquityTableCache {
            flops: vec![flop],
            tables_flat: table,
        };

        let extracted = cache.extract_tables(&[flop]).unwrap();
        assert_eq!(extracted.len(), 1);
        assert_eq!(extracted[0].len(), NUM_CANONICAL_HANDS * NUM_CANONICAL_HANDS);
    }

    #[test]
    fn extract_tables_returns_none_for_missing_flop() {
        let flops = canonical_flops();
        let cache = EquityTableCache {
            flops: vec![flops[0]],
            tables_flat: vec![0.0; NUM_CANONICAL_HANDS * NUM_CANONICAL_HANDS],
        };
        // Ask for a different flop
        assert!(cache.extract_tables(&[flops[1]]).is_none());
    }

    #[test]
    fn load_returns_none_for_missing_file() {
        assert!(EquityTableCache::load(Path::new("/tmp/nonexistent_equity_cache_xyz.bin")).is_none());
    }

    #[test]
    fn load_returns_none_for_bad_magic() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("bad_magic.bin");
        let data = CacheData {
            magic: *b"XXXX",
            version: VERSION,
            num_flops: 0,
            flop_bytes: vec![],
            tables_flat: vec![],
        };
        let bytes = bincode::serialize(&data).unwrap();
        std::fs::write(&path, bytes).unwrap();
        assert!(EquityTableCache::load(&path).is_none());
    }
}
```

**Step 2: Register the module in mod.rs**

In `crates/core/src/preflop/mod.rs`, add:
```rust
pub mod equity_table_cache;
```
And add to the pub use block in the same file — no pub use needed; callers use the full path.

**Step 3: Run tests**

Run: `cargo test -p poker-solver-core equity_table_cache`
Expected: All 5 tests pass.

**Step 4: Commit**

```bash
git add crates/core/src/preflop/equity_table_cache.rs crates/core/src/preflop/mod.rs
git commit -m "feat(core): add EquityTableCache for precomputed flop equity tables"
```

---

### Task 2: Add `precompute-equity` CLI subcommand

**Files:**
- Modify: `crates/trainer/src/main.rs`

**Step 1: Add the subcommand variant to `Commands` enum**

In `crates/trainer/src/main.rs`, add after the `TraceHand` variant (before the closing `}`):

```rust
    /// Precompute equity tables for all 1,755 canonical flops and save to disk.
    /// These tables are auto-loaded by solve-postflop to skip the expensive
    /// per-flop equity computation at startup.
    PrecomputeEquity {
        /// Output file path
        #[arg(short, long, default_value = "cache/equity_tables.bin")]
        output: PathBuf,
    },
```

**Step 2: Add the import**

Add to the imports at the top of `main.rs`:

```rust
use poker_solver_core::preflop::equity_table_cache::EquityTableCache;
```

**Step 3: Add the handler in the `match cli.command` block**

Find the match arm pattern in main. Add a new arm:

```rust
        Commands::PrecomputeEquity { output } => {
            let total = 1755; // all canonical flops
            let pb = ProgressBar::new(total as u64);
            pb.set_style(
                ProgressStyle::default_bar()
                    .template("{spinner:.green} [{bar:40.cyan/blue}] {pos}/{len} flops ({eta})")
                    .unwrap()
                    .progress_chars("=>-"),
            );
            pb.set_message("Computing equity tables...");

            let start = Instant::now();
            let cache = EquityTableCache::build(|completed, _total| {
                pb.set_position(completed as u64);
            });
            pb.finish_with_message("done");

            let elapsed = start.elapsed();
            eprintln!(
                "Computed {} equity tables in {:.1}s",
                cache.num_flops(),
                elapsed.as_secs_f64()
            );

            cache.save(&output)?;
            eprintln!("Saved to {}", output.display());
        }
```

**Step 4: Verify it compiles**

Run: `cargo build -p poker-solver-trainer`
Expected: Compiles without errors.

**Step 5: Commit**

```bash
git add crates/trainer/src/main.rs
git commit -m "feat(trainer): add precompute-equity CLI subcommand"
```

---

### Task 3: Auto-detect equity cache in `solve-postflop` handler

**Files:**
- Modify: `crates/trainer/src/main.rs`

**Step 1: Add cache auto-detection before the SPR solve loop**

In the `Commands::SolvePostflop` handler, find the block that starts with:
```rust
    // Pre-compute equity tables once when solving multiple SPRs with the
```

Replace that entire pre-computation block (lines ~368-391) with cache-aware logic:

```rust
    // Try loading pre-computed equity tables from disk cache.
    // Falls back to inline computation if cache is missing/invalid.
    let cache_path = Path::new("cache/equity_tables.bin");
    let equity_cache = if pf_config.solve_type == PostflopSolveType::Exhaustive {
        match EquityTableCache::load(cache_path) {
            Some(cache) => {
                eprintln!(
                    "Loaded equity table cache ({} flops) from {}",
                    cache.num_flops(),
                    cache_path.display()
                );
                Some(cache)
            }
            None => {
                if cache_path.exists() {
                    eprintln!(
                        "Warning: equity cache at {} is invalid, falling back to inline computation",
                        cache_path.display()
                    );
                }
                None
            }
        }
    } else {
        None
    };

    // Extract tables for the specific flops we need, or fall back to
    // the old inline pre-computation for multi-SPR runs.
    let equity_tables: Vec<Vec<f64>> = if let Some(ref cache) = equity_cache {
        match cache.extract_tables(&flops) {
            Some(tables) => {
                eprintln!(
                    "Using cached equity tables for all {} flops",
                    flops.len()
                );
                tables
            }
            None => {
                eprintln!(
                    "Warning: cache missing some requested flops, falling back to inline computation"
                );
                vec![]
            }
        }
    } else if pf_config.solve_type == PostflopSolveType::Exhaustive && sprs.len() > 1 {
        eprintln!(
            "Pre-computing equity tables for {} flops...",
            flops.len()
        );
        flops
            .par_iter()
            .map(|flop| {
                let combo_map = build_combo_map(flop);
                compute_equity_table(&combo_map, *flop)
            })
            .collect()
    } else {
        vec![]
    };
    let pre_tables = if equity_tables.is_empty() {
        None
    } else {
        Some(equity_tables.as_slice())
    };
```

**Step 2: Verify it compiles**

Run: `cargo build -p poker-solver-trainer`
Expected: Compiles without errors.

**Step 3: Run existing tests**

Run: `cargo test -p poker-solver-trainer`
Expected: All existing tests pass.

**Step 4: Commit**

```bash
git add crates/trainer/src/main.rs
git commit -m "feat(trainer): auto-detect equity table cache in solve-postflop"
```

---

### Task 4: Always pass pre_equity_tables for single-SPR exhaustive solves

**Files:**
- Modify: `crates/core/src/preflop/postflop_exhaustive.rs` (no change needed — already handles `pre_equity_tables`)
- Modify: `crates/trainer/src/main.rs` (already done in Task 3 — cache provides tables for single-SPR too)

This task is already handled by Task 3's logic: when the cache is loaded, `equity_tables` is populated regardless of `sprs.len()`. The existing `pre_equity_tables` parameter in `build_exhaustive` handles it. No additional changes needed.

**Step 1: Verify single-SPR cache path works**

Run: `cargo build -p poker-solver-trainer --release`
Expected: Compiles. The cache auto-detection now covers single-SPR runs too.

**Step 2: Commit (skip if no changes)**

No commit needed — Task 3 already covers this.

---

### Task 5: Add `cache/` to .gitignore

**Files:**
- Modify: `.gitignore`

**Step 1: Add cache directory to .gitignore**

Append to `.gitignore`:
```
# Precomputed equity table cache
cache/
```

**Step 2: Commit**

```bash
git add .gitignore
git commit -m "chore: add cache/ to .gitignore"
```

---

### Task 6: Integration test — build, save, load round-trip

**Files:**
- Modify: `crates/core/src/preflop/equity_table_cache.rs` (add test)

**Step 1: Add an integration-style test that exercises build → save → load → extract**

Add to the `#[cfg(test)] mod tests` block in `equity_table_cache.rs`:

```rust
    #[test]
    fn build_small_save_load_extract() {
        // Build cache for all canonical flops (but we only verify a few)
        // This test is slow (~minutes) so we just build for the first flop
        // by constructing manually, then verify extract works.
        let flops = canonical_flops();
        let first_three: Vec<[Card; 3]> = flops[..3].to_vec();

        // Compute tables for 3 flops
        let mut tables_flat = Vec::new();
        for flop in &first_three {
            let combo_map = build_combo_map(flop);
            let table = compute_equity_table(&combo_map, *flop);
            tables_flat.extend_from_slice(&table);
        }

        let cache = EquityTableCache {
            flops: first_three.clone(),
            tables_flat,
        };

        // Save and reload
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("equity.bin");
        cache.save(&path).unwrap();

        let loaded = EquityTableCache::load(&path).unwrap();
        assert_eq!(loaded.num_flops(), 3);

        // Extract in different order
        let reversed: Vec<[Card; 3]> = vec![first_three[2], first_three[0]];
        let extracted = loaded.extract_tables(&reversed).unwrap();
        assert_eq!(extracted.len(), 2);

        // Verify second extracted table matches first flop's original table
        let combo_map = build_combo_map(&first_three[0]);
        let original = compute_equity_table(&combo_map, first_three[0]);
        for (a, b) in extracted[1].iter().zip(original.iter()) {
            if a.is_nan() {
                assert!(b.is_nan());
            } else {
                assert!((a - b).abs() < 1e-15);
            }
        }
    }
```

**Step 2: Run the test**

Run: `cargo test -p poker-solver-core build_small_save_load_extract`
Expected: PASS

**Step 3: Commit**

```bash
git add crates/core/src/preflop/equity_table_cache.rs
git commit -m "test(core): add integration test for equity table cache round-trip"
```

---

### Task 7: Update documentation

**Files:**
- Modify: `docs/training.md`

**Step 1: Add precompute-equity docs**

Add a new section to `docs/training.md` under CLI commands:

```markdown
### Precompute Equity Tables

Precompute all 1,755 canonical flop equity tables and save to disk. This
eliminates the expensive per-flop equity computation during postflop solves.

```bash
cargo run -p poker-solver-trainer --release -- precompute-equity [-o cache/equity_tables.bin]
```

The `solve-postflop` command auto-detects the cache at `cache/equity_tables.bin`
and loads it if present. No additional flags needed.

**Recommended workflow:**
1. Run `precompute-equity` once (takes several minutes)
2. Run `solve-postflop` as many times as needed — equity tables load instantly from cache
```

**Step 2: Commit**

```bash
git add docs/training.md
git commit -m "docs: add precompute-equity command documentation"
```
