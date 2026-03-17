# Per-Flop Regret Tables Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use hex:executing-plans to implement this plan task-by-task.

**Goal:** Split MCCFR regret storage into global preflop + per-flop postflop tables with disk-backed FIFO cache, enabling per-flop bucketing to produce coherent strategies.

**Architecture:** The existing `BlueprintStorage` is reused as-is for both preflop (global) and per-flop (local) contexts — same API, different instances. A new `PerFlopRegretStore` manages the FIFO cache of per-flop `BlueprintStorage` instances. The training loop changes from random deal sampling to epoch-based weighted round-robin with atomic work-stealing.

**Tech Stack:** Rust, rayon, existing BlueprintStorage, bincode for serialization

**Design doc:** `docs/plans/2026-03-17-per-flop-regrets-design.md`

---

### Task 1: PerFlopRegretStore — FIFO Cache

Create the disk-backed FIFO cache that manages per-flop `BlueprintStorage` instances.

**Files:**
- Create: `crates/core/src/blueprint_v2/per_flop_regret_store.rs`
- Modify: `crates/core/src/blueprint_v2/mod.rs` (add module)

**The struct:**

```rust
pub struct PerFlopRegretStore {
    /// Directory where per-flop regret files are stored
    dir: PathBuf,
    /// The game tree (shared across all flops — same action structure)
    tree: Arc<GameTree>,
    /// Bucket counts for postflop streets [flop_k, turn_k, river_k]
    postflop_bucket_counts: [u16; 3],
    /// FIFO cache: VecDeque of (flop_index, storage, dirty)
    cache: Mutex<VecDeque<CacheEntry>>,
    /// Max entries in cache
    capacity: usize,
}

struct CacheEntry {
    flop_index: u16,
    storage: BlueprintStorage,
    dirty: bool,
}
```

**Key methods:**

```rust
impl PerFlopRegretStore {
    /// Create a new store. Creates the directory if needed.
    pub fn new(dir: PathBuf, tree: Arc<GameTree>, postflop_bucket_counts: [u16; 3], capacity: usize) -> Self;

    /// Get or load a flop's storage. If cache is full, flush oldest.
    /// Returns a reference that borrows the cache lock.
    ///
    /// Since threads work on different flops (atomic work-stealing),
    /// we can take the entry OUT of the cache, give it to the thread,
    /// and have the thread put it back when done.
    pub fn checkout(&self, flop_index: u16) -> BlueprintStorage;

    /// Return a storage to the cache after use. Marks as dirty.
    pub fn checkin(&self, flop_index: u16, storage: BlueprintStorage);

    /// Flush all dirty entries to disk.
    pub fn flush_all(&self) -> std::io::Result<()>;

    /// Save a single storage to disk.
    fn save_flop(&self, flop_index: u16, storage: &BlueprintStorage) -> std::io::Result<()>;

    /// Load a storage from disk (or create empty if file doesn't exist).
    fn load_flop(&self, flop_index: u16) -> BlueprintStorage;
}
```

The `checkout`/`checkin` pattern avoids holding the Mutex during traversal. The thread takes ownership of the `BlueprintStorage`, runs traversal with it, then returns it.

**Serialization:** Use the existing `BlueprintStorage::save_regrets` / `load_regrets` methods but adapted for i16. Or simpler: just write the raw `Vec<AtomicI32>` values to disk (they'll be i32 on disk even if we clamp to i16 range — i16 quantization can come later as an optimization).

**Storage sizing for one flop:**
The `BlueprintStorage::new` takes a `GameTree` and `bucket_counts`. For per-flop, the tree is the same but we only use postflop nodes. The `bucket_counts` would be `[0, flop_k, turn_k, river_k]` where preflop=0 (no preflop nodes in per-flop storage). Actually, simpler: use the full tree but set preflop buckets to 0 — preflop decision nodes will have 0 slots allocated.

Actually, even simpler: use `bucket_counts = [1, flop_k, turn_k, river_k]` with preflop=1. The preflop slots (169 × 1 = tiny) will exist but never be used. This avoids any tree modifications.

Wait — the layout computes `buckets * num_actions` per decision node. If preflop buckets = 1 instead of 169, the preflop slots shrink from 169 × actions to 1 × actions. That's fine — a few wasted bytes.

**Tests:**
- Create store, checkout a flop, checkin, verify it's cached
- Checkout beyond capacity, verify oldest is flushed to disk
- Flush all, reload, verify regrets are preserved
- Concurrent checkout of different flops (no conflicts)

**Step 1:** Write tests
**Step 2:** Implement `PerFlopRegretStore`
**Step 3:** Verify: `cargo test -p poker-solver-core per_flop_regret`
**Step 4:** Commit

---

### Task 2: Epoch Schedule Builder

Create the weighted round-robin schedule for training epochs.

**Files:**
- Create: `crates/core/src/blueprint_v2/epoch_schedule.rs`
- Modify: `crates/core/src/blueprint_v2/mod.rs`

**The struct:**

```rust
pub struct EpochSchedule {
    /// Entries: (flop_index, canonical_flop_cards)
    /// Each flop appears `weight` times.
    entries: Vec<ScheduleEntry>,
}

pub struct ScheduleEntry {
    pub flop_index: u16,
    pub flop_cards: [Card; 3],
    pub weight: u32,  // combinatorial weight of this canonical flop
}

impl EpochSchedule {
    /// Build a schedule from all canonical flops.
    /// Total entries = sum of weights = C(52,3) = 22,100.
    pub fn new() -> Self;

    /// Shuffle the schedule order (for a new epoch).
    pub fn shuffle(&mut self, rng: &mut impl Rng);

    /// Number of entries (total deals in epoch).
    pub fn len(&self) -> usize;

    /// Get entry by index.
    pub fn get(&self, idx: usize) -> &ScheduleEntry;
}
```

Uses `enumerate_canonical_flops()` from `cluster_pipeline.rs` to build the schedule. Each canonical flop with weight W appears W times in the schedule (or appears once with `weight` field indicating how many deals to run).

Actually, simpler: each entry appears ONCE but carries its `weight`. The training thread runs `weight` deals on that flop before moving to the next entry. This way the schedule has 1,755 entries (not 22,100), and each flop is visited exactly once per epoch.

```rust
pub struct EpochSchedule {
    entries: Vec<ScheduleEntry>,
}

impl EpochSchedule {
    pub fn new() -> Self {
        let flops = enumerate_canonical_flops();
        let entries = flops.iter().enumerate().map(|(i, wb)| {
            ScheduleEntry {
                flop_index: i as u16,
                flop_cards: wb.cards,
                weight: wb.weight,
            }
        }).collect();
        Self { entries }
    }
}
```

**Tests:**
- Schedule has 1,755 entries
- Sum of weights = 22,100
- Shuffle produces different order

**Step 1:** Write tests
**Step 2:** Implement
**Step 3:** Verify: `cargo test -p poker-solver-core epoch_schedule`
**Step 4:** Commit

---

### Task 3: Split MCCFR Traversal for Two-Level Storage

Modify the MCCFR traversal functions to accept both a preflop storage and a postflop storage, selecting the right one based on the current street.

**Files:**
- Modify: `crates/core/src/blueprint_v2/mccfr.rs`

**Current traversal signature:**
```rust
fn traverse_mccfr(
    tree: &GameTree,
    storage: &BlueprintStorage,
    node_idx: u32,
    deal: &DealWithBuckets,
    traverser: usize,
    ...
) -> f64
```

**New traversal signature:**
```rust
fn traverse_mccfr(
    tree: &GameTree,
    preflop_storage: &BlueprintStorage,
    postflop_storage: &BlueprintStorage,
    node_idx: u32,
    deal: &DealWithBuckets,
    traverser: usize,
    ...
) -> f64
```

At each Decision node, select the storage based on the node's street:
```rust
let storage = if street == Street::Preflop {
    preflop_storage
} else {
    postflop_storage
};
```

The rest of the traversal logic stays identical — `storage.get_regret(...)`, `storage.add_regret(...)`, etc.

**Also update:**
- `traverse_mccfr_prune` (the pruning variant)
- `update_strategy` (strategy sum accumulation)
- Any other functions that take `&BlueprintStorage`

**Important:** The `DealWithBuckets` precomputes bucket IDs for all 4 streets. The preflop bucket comes from the global bucket lookup. The postflop buckets come from the per-flop bucket lookup (already implemented in `AllBuckets`). No changes needed to bucket lookup — it already uses per-flop files.

**Tests:**
- Existing traversal tests should pass with `preflop_storage == postflop_storage` (same object for both)
- New test: preflop and postflop use different storage objects, regrets accumulate to the correct one

**Step 1:** Update traverse signatures
**Step 2:** Add street-based storage selection
**Step 3:** Update all callers
**Step 4:** Verify existing tests pass
**Step 5:** Add new test for split storage
**Step 6:** Commit

---

### Task 4: Epoch-Based Training Loop

Replace the current batch-based training loop in `trainer.rs` with the epoch-based weighted round-robin loop.

**Files:**
- Modify: `crates/core/src/blueprint_v2/trainer.rs`

**Current loop (simplified):**
```rust
loop {
    // Sample random deals in parallel
    let batch_results: Vec<_> = thread_seeds.into_par_iter().map(|seed| {
        let deal = sample_deal();
        let buckets = all_buckets.precompute_buckets(&deal);
        traverse_mccfr(&tree, &storage, ...);
    }).collect();

    // Apply DCFR discounting periodically
    iterations += batch_size;
}
```

**New loop:**
```rust
let preflop_storage = BlueprintStorage::new(&tree, [169, 1, 1, 1]);  // preflop only
let per_flop_store = PerFlopRegretStore::new(regret_dir, tree.clone(), [500, 500, 500], cache_capacity);
let mut schedule = EpochSchedule::new();
let epoch_counter = AtomicUsize::new(0);

loop {
    schedule.shuffle(&mut rng);
    let work_idx = AtomicUsize::new(0);

    // Process epoch in parallel
    rayon::scope(|s| {
        for _ in 0..num_threads {
            s.spawn(|_| {
                let mut rng = thread_rng();
                loop {
                    let idx = work_idx.fetch_add(1, Ordering::Relaxed);
                    if idx >= schedule.len() { break; }

                    let entry = schedule.get(idx);
                    let mut postflop_storage = per_flop_store.checkout(entry.flop_index);

                    // Run `weight` deals on this flop
                    for _ in 0..entry.weight {
                        let deal = sample_deal_for_flop(entry.flop_cards, &mut rng);
                        let buckets = all_buckets.precompute_buckets(&deal);
                        let deal = DealWithBuckets { deal, buckets };

                        traverse_mccfr(&tree, &preflop_storage, &postflop_storage, ...);
                        traverse_mccfr(&tree, &preflop_storage, &postflop_storage, ...); // both traversers
                    }

                    per_flop_store.checkin(entry.flop_index, postflop_storage);
                }
            });
        }
    });

    per_flop_store.flush_all()?;
    epochs += 1;

    // DCFR discounting, snapshots, etc.
}
```

**Key new function: `sample_deal_for_flop`**

Given a specific canonical flop, sample random hole cards for both players:
```rust
fn sample_deal_for_flop(flop: [Card; 3], rng: &mut impl Rng) -> Deal {
    // Remove flop cards from deck
    // Sample 4 cards: 2 for each player
    // Sample 2 more: turn + river
    // Return Deal { hole_cards, board: [flop + turn + river] }
}
```

**Tests:**
- Epoch processes all 1,755 flops
- Each flop gets `weight` deals
- Preflop regrets accumulate globally
- Per-flop regrets are independent per flop
- Snapshots save/load correctly

**Step 1:** Add `sample_deal_for_flop`
**Step 2:** Add epoch loop (can coexist with old loop behind `per_flop_regrets` config flag)
**Step 3:** Wire up preflop + per-flop storage
**Step 4:** Verify with a short training run
**Step 5:** Commit

---

### Task 5: Config Changes

Add the new training config fields.

**Files:**
- Modify: `crates/core/src/blueprint_v2/config.rs`

Add to `TrainingConfig`:
```rust
/// Enable per-flop regret tables (requires per-flop bucket files).
#[serde(default)]
pub per_flop_regrets: bool,

/// Number of per-flop regret tables to keep in memory.
#[serde(default = "default_flop_cache_capacity")]
pub flop_cache_capacity: usize,
```

Add to `PerFlopConfig`:
```rust
/// Per-flop flop buckets (in addition to turn/river).
#[serde(default = "default_per_flop_buckets")]
pub flop_buckets: u16,
```

**Tests:**
- Parse YAML with `per_flop_regrets: true`
- Default `flop_cache_capacity` = 1000
- Backwards compatible (old configs without these fields still parse)

**Step 1:** Add fields with defaults
**Step 2:** Update tests
**Step 3:** Commit

---

### Task 6: Snapshot Save/Load for Per-Flop Regrets

Update snapshot saving to write preflop.bin + per-flop regret files, and loading to restore them.

**Files:**
- Modify: `crates/core/src/blueprint_v2/trainer.rs`
- Modify: `crates/core/src/blueprint_v2/bundle.rs` (if strategy export needs changes)

**Save:**
```rust
// Save preflop regrets
preflop_storage.save_regrets(&snapshot_dir.join("preflop_regrets.bin"))?;

// Per-flop regrets are already on disk in regret_dir/flop_NNNN.bin
// Just flush the cache to ensure all are written
per_flop_store.flush_all()?;

// Copy or symlink the regret dir into the snapshot
```

**Load (resume):**
```rust
// Load preflop regrets
preflop_storage = BlueprintStorage::load_regrets(&snapshot_dir.join("preflop_regrets.bin"), ...)?;

// Per-flop regrets load lazily from the regret dir (already on disk)
```

**Tests:**
- Save snapshot, load snapshot, verify regrets match
- Resume training from snapshot produces continued convergence

**Step 1:** Update save logic
**Step 2:** Update resume logic
**Step 3:** Verify round-trip
**Step 4:** Commit

---

### Task 7: Integration Test — End to End

Run a short training session with per-flop regrets and verify SB develops a raising strategy.

**Files:** None (manual testing)

**Step 1:** Create config with `per_flop_regrets: true`
**Step 2:** Run training for ~5 minutes
**Step 3:** Check TUI: SB Open scenario should show raising for strong hands
**Step 4:** If passive, debug bucket lookups with the panic check
**Step 5:** Commit any fixes

---

### Task 8: Remove Debug Code

Remove the temporary debug prints and panics added during the per-flop debugging session.

**Files:**
- Modify: `crates/core/src/blueprint_v2/mccfr.rs`

**Step 1:** Remove the `[DEBUG]` print in `precompute_buckets`
**Step 2:** Remove the `PRINTED` atomic
**Step 3:** Remove the `Per-flop index: ...` print in `with_per_flop_dir`
**Step 4:** Keep the panic in `lookup_bucket` (it's a valid safety check)
**Step 5:** Verify: `cargo test`
**Step 6:** Commit
