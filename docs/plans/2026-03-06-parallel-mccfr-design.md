# Parallel Blueprint V2 MCCFR Training — Design Document

**Date**: 2026-03-06
**Status**: Approved

## Overview

Parallelize the Blueprint V2 external-sampling MCCFR training loop to use all available CPU cores. Currently single-threaded (~150 iter/sec), the solver processes one deal at a time. This change enables concurrent traversal of multiple deals using Rayon's thread pool, with lock-free atomic shared storage following the Pluribus paper's approach.

## Architecture

**Approach: Atomic Shared Buffers (Pluribus-style)**

Replace `Vec<i32>` regrets and `Vec<i64>` strategy sums with `Vec<AtomicI32>` and `Vec<AtomicI64>`. Multiple threads traverse different deals concurrently, updating the same storage with `Relaxed` atomic operations. This matches what the Pluribus paper describes: workers read slightly stale regrets (acceptable for MCCFR convergence) and accumulate updates without locks.

**Why not snapshot+delta-merge:** The existing postflop solver uses snapshot+merge, but that requires copying the entire storage per thread. Blueprint V2 storage can be hundreds of MB — copying is wasteful when atomic adds suffice.

## Synchronization Model

**Batched parallel execution:**

```
loop {
    // 1. Generate batch of 200 deals (sequential, fast)
    let deals: Vec<Deal> = (0..batch_size).map(|_| sample_deal(&mut rng)).collect();

    // 2. Parallel traversal (Rayon par_iter)
    deals.par_iter().for_each(|deal| {
        let mut rng = SmallRng::from_entropy();
        let prune = should_prune_for_iteration(...);
        traverse_external(&tree, &storage, &buckets, deal, 0, ...);
        traverse_external(&tree, &storage, &buckets, deal, 1, ...);
    });

    // 3. Sequential timed actions (LCFR discount, snapshots, logging)
    iterations += batch_size;
    check_timed_actions()?;
}
```

- **Batch size**: Fixed at 200 iterations (configurable via `batch_size` in `TrainingConfig`)
- **Thread count**: Rayon auto-detects cores (no config needed)
- **LCFR discount**: Runs between batches when no traversals are active — safe sequential pass over atomics
- **Timed actions**: Checked once per batch, not per iteration

## Storage Changes

### `BlueprintStorage` (`storage.rs`)

```rust
pub struct BlueprintStorage {
    pub regrets: Vec<AtomicI32>,        // was Vec<i32>
    pub strategy_sums: Vec<AtomicI64>,  // was Vec<i64>
    pub bucket_counts: [u16; 4],
    layout: Vec<NodeLayout>,
}
```

New atomic accessor methods:

```rust
// Read regret for a single action (Relaxed load)
fn get_regret_atomic(&self, node_idx: u32, bucket: u16, action: usize) -> i32

// Add delta to regret (Relaxed fetch_add)
fn add_regret_atomic(&self, node_idx: u32, bucket: u16, action: usize, delta: i32)

// Add delta to strategy sum (Relaxed fetch_add)
fn add_strategy_sum_atomic(&self, node_idx: u32, bucket: u16, action: usize, delta: i64)

// Read regrets into caller buffer (for regret matching)
fn current_strategy_into(&self, node_idx: u32, bucket: u16, out: &mut [f64])
// ^ This already reads individual elements; just changes from slice indexing to atomic loads
```

The existing `get_regrets()` → `&[i32]` slice methods cannot return references to atomics. They are replaced by:
- `current_strategy_into()` which reads atomics into a stack buffer (already exists, just needs atomic loads)
- Per-element `add_regret_atomic()` / `add_strategy_sum_atomic()` for writes

### `traverse_external` (`mccfr.rs`)

Signature changes from `&mut BlueprintStorage` to `&BlueprintStorage`:

```rust
pub fn traverse_external(
    tree: &GameTree,
    storage: &BlueprintStorage,   // was &mut
    buckets: &AllBuckets,
    deal: &Deal,
    traverser: u8,
    node_idx: u32,
    prune: bool,
    prune_threshold: i32,
    rng: &mut impl Rng,
) -> f64
```

Inside `traverse_traverser`, regret and strategy sum updates change from slice mutation to atomic adds:

```rust
// Before:
let regrets = storage.get_regrets_mut(node_idx, bucket);
regrets[a] += (delta * 1000.0) as i32;

// After:
storage.add_regret_atomic(node_idx, bucket, a, (delta * 1000.0) as i32);
```

### LCFR Discount

Runs between batches (no concurrent access). Uses `Relaxed` load + store:

```rust
fn apply_lcfr_discount(&self) {
    for atom in &self.storage.regrets {
        let v = atom.load(Relaxed);
        atom.store((v as f64 * d) as i32, Relaxed);
    }
    // Same for strategy_sums
}
```

## Deal Sampling

The main RNG (`StdRng`) stays on the coordinator thread for deal generation. Each worker thread gets a `SmallRng::from_entropy()` for opponent sampling within `traverse_external`.

```rust
// Coordinator generates deals
let deals: Vec<Deal> = (0..batch_size)
    .map(|_| self.sample_deal())
    .collect();

// Workers use thread-local RNG
deals.par_iter().for_each(|deal| {
    let mut rng = SmallRng::from_entropy();
    // ... traverse with &mut rng
});
```

## Serialization

`save_regrets` / `load_regrets` need adaptation:
- **Save**: Collect atomics into plain `Vec<i32>` / `Vec<i64>`, then serialize (existing bincode format unchanged)
- **Load**: Deserialize plain vecs, convert to atomic vecs

This preserves backward compatibility with existing snapshots.

## Config Changes

New field in `TrainingConfig`:

```yaml
training:
  batch_size: 200  # iterations per parallel batch (default: 200)
```

## Pruning

`should_prune()` is currently per-iteration with RNG. In batched mode, we pre-decide for each deal whether it uses pruning (based on elapsed time + random roll from the coordinator RNG).

## Testing

- Existing tests continue to work (single-threaded traversal is a special case of batch_size=1)
- New test: parallel batch produces non-zero regret updates
- New test: LCFR discount works on atomic storage
- New test: snapshot round-trip with atomic storage
- Convergence: run 1000 iterations single-threaded vs parallel, assert mean positive regret is similar (not identical due to ordering differences)

## Performance Expectations

- Current: ~150 iter/sec (1 core)
- Expected: ~150 × num_cores × efficiency_factor
- On 8 cores with ~70% parallel efficiency: ~840 iter/sec
- Bottleneck will be `compute_equity()` in bucket lookup (pure CPU, scales well)
- Atomic contention is minimal: different deals hit different (node, bucket) slots
