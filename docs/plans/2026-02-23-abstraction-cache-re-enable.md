# Re-enable Abstraction Cache for solve-preflop and trace-hand

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Re-enable the abstraction cache so both `solve-preflop` and `trace-hand` load cached hand buckets + equity instead of rebuilding from scratch every run.

**Architecture:** Fix the `AbstractionCacheKey` to include `max_flop_boards` (currently missing — two runs with different `max_flop_boards` would collide). Re-enable the load-or-build pattern in `load_or_build_abstraction()`. Change the default cache directory from `cache/postflop` to `local_data/cache`. Add a `hands` parameter to `load_or_build_abstraction` so tests can pass a small hand set (e.g., 1 hand) for fast end-to-end validation.

**Tech Stack:** Rust, bincode, serde, tempfile (tests)

---

### Task 1: Fix `AbstractionCacheKey` to include `max_flop_boards`

**Files:**
- Modify: `crates/core/src/preflop/abstraction_cache.rs:29-46`
- Test: `crates/core/src/preflop/abstraction_cache.rs` (existing tests module)

**Step 1: Write the failing test**

Add to the existing `tests` module in `abstraction_cache.rs`:

```rust
#[timed_test]
fn cache_key_differs_with_max_flop_boards() {
    let config1 = PostflopModelConfig::fast();
    let mut config2 = PostflopModelConfig::fast();
    config2.max_flop_boards = config1.max_flop_boards + 10;
    let k1 = cache_key(&config1, false);
    let k2 = cache_key(&config2, false);
    assert_ne!(hex_hash(&k1), hex_hash(&k2));
}
```

**Step 2: Run test to verify failure**

Run: `cargo test -p poker-solver-core cache_key_differs_with_max_flop_boards -- --nocapture`
Expected: FAIL — `max_flop_boards` isn't in the key, so hashes are equal.

**Step 3: Add `max_flop_boards` to `AbstractionCacheKey`**

In `abstraction_cache.rs`, add the field to the struct and the constructor:

```rust
#[derive(Debug, Clone, Serialize, Deserialize, Hash, PartialEq, Eq)]
pub struct AbstractionCacheKey {
    pub num_hand_buckets_flop: u16,
    pub num_hand_buckets_turn: u16,
    pub num_hand_buckets_river: u16,
    pub max_flop_boards: usize,
    pub has_equity_table: bool,
}

pub fn cache_key(config: &PostflopModelConfig, has_equity_table: bool) -> AbstractionCacheKey {
    AbstractionCacheKey {
        num_hand_buckets_flop: config.num_hand_buckets_flop,
        num_hand_buckets_turn: config.num_hand_buckets_turn,
        num_hand_buckets_river: config.num_hand_buckets_river,
        max_flop_boards: config.max_flop_boards,
        has_equity_table,
    }
}
```

**Step 4: Run tests**

Run: `cargo test -p poker-solver-core abstraction_cache -- --nocapture`
Expected: All pass (including the new test).

**Step 5: Commit**

```bash
git add crates/core/src/preflop/abstraction_cache.rs
git commit -m "fix: add max_flop_boards to AbstractionCacheKey to prevent cache collisions"
```

---

### Task 2: Add `num_hands` to `AbstractionCacheKey` and accept `hands` slice in `load_or_build_abstraction`

The cache key must account for the number of hands used (169 for production, fewer for tests). The `load_or_build_abstraction` function needs to accept `&[CanonicalHand]` instead of hardcoding `all_hands()`.

**Files:**
- Modify: `crates/core/src/preflop/abstraction_cache.rs:29-46`
- Modify: `crates/core/src/preflop/postflop_abstraction.rs:373-411` (`load_or_build_abstraction`)
- Modify: `crates/core/src/preflop/postflop_abstraction.rs:276-288` (`build` method)

**Step 1: Add `num_hands` to `AbstractionCacheKey`**

```rust
#[derive(Debug, Clone, Serialize, Deserialize, Hash, PartialEq, Eq)]
pub struct AbstractionCacheKey {
    pub num_hand_buckets_flop: u16,
    pub num_hand_buckets_turn: u16,
    pub num_hand_buckets_river: u16,
    pub max_flop_boards: usize,
    pub num_hands: usize,
    pub has_equity_table: bool,
}

pub fn cache_key(
    config: &PostflopModelConfig,
    num_hands: usize,
    has_equity_table: bool,
) -> AbstractionCacheKey {
    AbstractionCacheKey {
        num_hand_buckets_flop: config.num_hand_buckets_flop,
        num_hand_buckets_turn: config.num_hand_buckets_turn,
        num_hand_buckets_river: config.num_hand_buckets_river,
        max_flop_boards: config.max_flop_boards,
        num_hands,
        has_equity_table,
    }
}
```

**Step 2: Update `load_or_build_abstraction` signature**

Add `hands` and `cache_base` parameters:

```rust
fn load_or_build_abstraction(
    config: &PostflopModelConfig,
    hands: &[CanonicalHand],
    cache_base: Option<&std::path::Path>,
    on_progress: &(impl Fn(BuildPhase) + Sync),
) -> Result<(StreetBuckets, StreetEquity), PostflopAbstractionError> {
    use super::abstraction_cache;

    // Try cache first
    if let Some(base) = cache_base {
        let key = abstraction_cache::cache_key(config, hands.len(), false);
        if let Some(cached) = abstraction_cache::load(base, &key) {
            return Ok(cached);
        }
    }

    on_progress(BuildPhase::HandBuckets(0, hands.len()));
    let flops = crate::preflop::ehs::sample_canonical_flops(config.max_flop_boards);

    let buckets = hand_buckets::build_street_buckets_independent(
        hands,
        &flops,
        config.num_hand_buckets_flop,
        config.num_hand_buckets_turn,
        config.num_hand_buckets_river,
        &|progress| {
            use hand_buckets::BuildProgress;
            match progress {
                BuildProgress::FlopFeatures(done, total) => {
                    on_progress(BuildPhase::HandBuckets(done, total));
                }
                _ => {}
            }
        },
    );

    let street_equity = build_street_equity_from_buckets(&buckets);
    on_progress(BuildPhase::EquityTable);

    // Save to cache
    if let Some(base) = cache_base {
        let key = abstraction_cache::cache_key(config, hands.len(), false);
        if let Err(e) = abstraction_cache::save(base, &key, &buckets, &street_equity) {
            tracing::warn!("failed to save abstraction cache: {e}");
        }
    }

    Ok((buckets, street_equity))
}
```

**Step 3: Update `PostflopAbstraction::build` to pass `hands` and `cache_base` through**

Change the signature to accept `hands: &[CanonicalHand]` and un-underscore `_cache_base`:

```rust
pub fn build(
    config: &PostflopModelConfig,
    hands: &[CanonicalHand],
    _equity_table: Option<&EquityTable>,
    cache_base: Option<&std::path::Path>,
    on_progress: impl Fn(BuildPhase) + Sync,
) -> Result<Self, PostflopAbstractionError> {
    if config.canonical_sprs.is_empty() {
        return Err(PostflopAbstractionError::EmptyCanonicalSprs);
    }
    let (buckets, street_equity) = load_or_build_abstraction(
        config,
        hands,
        cache_base,
        &on_progress,
    )?;
    // ... rest unchanged
}
```

**Step 4: Fix all callers of `PostflopAbstraction::build`**

Every caller currently passes `None` for cache_base and doesn't pass hands. Update them to pass `&all_hands().collect::<Vec<_>>()` and the actual cache path. The callers are:
- `crates/trainer/src/main.rs` line ~838 (solve-preflop)
- `crates/trainer/src/main.rs` line ~685 (trace-hand)

**Step 5: Run tests**

Run: `cargo test -p poker-solver-core -- --nocapture`
Expected: All pass.

**Step 6: Commit**

```bash
git add crates/core/src/preflop/abstraction_cache.rs crates/core/src/preflop/postflop_abstraction.rs crates/trainer/src/main.rs
git commit -m "feat: re-enable abstraction cache with hands and max_flop_boards in cache key"
```

---

### Task 3: Update `bucket_diagnostics` and default cache path to `local_data/cache`

**Files:**
- Modify: `crates/trainer/src/bucket_diagnostics.rs:90-129` (`load_or_build`)
- Modify: `crates/trainer/src/main.rs` (cache path defaults)

**Step 1: Update `bucket_diagnostics::load_or_build` to pass `num_hands` to `cache_key`**

The `bucket_diagnostics` module has its own `load_or_build` that calls `abstraction_cache::cache_key(config, false)`. Update it to also pass `hands.len()`:

```rust
fn load_or_build(
    config: &PostflopModelConfig,
    cache_base: &Path,
) -> (StreetBuckets, StreetEquity) {
    let hands: Vec<CanonicalHand> = all_hands().collect();
    let key = abstraction_cache::cache_key(config, hands.len(), false);
    if let Some(cached) = abstraction_cache::load(cache_base, &key) {
        eprintln!("Cache hit: {}", abstraction_cache::cache_dir(cache_base, &key).display());
        return cached;
    }
    // ... rest builds from scratch, then saves with updated key
}
```

**Step 2: Change default cache paths from `cache/postflop` to `local_data/cache`**

In `main.rs`:
- Line 243 (`DiagBuckets`): change default to `local_data/cache`
- Line 255 (`TraceHand`): change default to `local_data/cache`
- Line 765 (`run_solve_preflop`): change hardcoded `cache/postflop` to `local_data/cache`

**Step 3: Run tests**

Run: `cargo test -p poker-solver-trainer -- --nocapture`
Expected: All pass.

**Step 4: Commit**

```bash
git add crates/trainer/src/main.rs crates/trainer/src/bucket_diagnostics.rs
git commit -m "refactor: change default cache path to local_data/cache and fix bucket_diagnostics cache key"
```

---

### Task 4: End-to-end integration test with minimal config (1 hand, 1 flop, 1 bucket)

This test validates the full cache round-trip: build → save → load (cache hit) → produce same result.

**Files:**
- Modify: `crates/core/src/preflop/postflop_abstraction.rs` (add test to existing `tests` module)

**Step 1: Write the failing test**

```rust
#[timed_test(30)]
fn build_caches_and_reuses_abstraction() {
    use crate::hands::CanonicalHand;

    let config = PostflopModelConfig {
        num_hand_buckets_flop: 1,
        num_hand_buckets_turn: 1,
        num_hand_buckets_river: 1,
        max_flop_boards: 1,
        canonical_sprs: vec![5.0],
        postflop_solve_iterations: 1,
        postflop_solve_samples: 1,
        ..PostflopModelConfig::fast()
    };

    // Use just 2 hands (minimum for k-means with 1 bucket)
    let hands: Vec<CanonicalHand> = crate::hands::all_hands().take(2).collect();
    let cache_dir = tempfile::TempDir::new().unwrap();

    // First build: should miss cache and compute
    let a1 = PostflopAbstraction::build(
        &config, &hands, None, Some(cache_dir.path()), |_| {},
    ).unwrap();

    // Second build: should hit cache
    let a2 = PostflopAbstraction::build(
        &config, &hands, None, Some(cache_dir.path()), |_| {},
    ).unwrap();

    // Bucket assignments should be identical
    assert_eq!(a1.buckets.flop, a2.buckets.flop);
    assert_eq!(a1.buckets.turn, a2.buckets.turn);
    assert_eq!(a1.buckets.river, a2.buckets.river);
    assert_eq!(a1.buckets.num_flop_buckets, a2.buckets.num_flop_buckets);

    // Values should also be populated (solve ran)
    assert!(!a1.values.is_empty());
}
```

**Step 2: Run test to verify failure**

Run: `cargo test -p poker-solver-core build_caches_and_reuses_abstraction -- --nocapture`
Expected: FAIL (compilation error if the signature hasn't been updated yet, or pass if Tasks 1-2 are done).

**Step 3: Run test to verify pass (after Tasks 1-3)**

Run: `cargo test -p poker-solver-core build_caches_and_reuses_abstraction -- --nocapture`
Expected: PASS — first call builds and caches, second call loads from cache.

**Step 4: Commit**

```bash
git add crates/core/src/preflop/postflop_abstraction.rs
git commit -m "test: end-to-end abstraction cache round-trip with minimal config"
```

---

### Task 5: Verify full pipeline and run clippy

**Step 1: Run all core tests**

Run: `cargo test -p poker-solver-core -- --nocapture 2>&1 | tail -5`
Expected: All pass.

**Step 2: Run trainer tests**

Run: `cargo test -p poker-solver-trainer -- --nocapture 2>&1 | tail -5`
Expected: All pass.

**Step 3: Run clippy**

Run: `cargo clippy --all-targets 2>&1 | tail -20`
Expected: No new warnings.

**Step 4: Final commit (if any fixups)**

```bash
git add -A && git commit -m "chore: clippy fixes"
```
