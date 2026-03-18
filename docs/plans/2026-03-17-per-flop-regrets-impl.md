# Per-Flop Regret Tables Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use hex:executing-plans to implement this plan task-by-task.

**Goal:** Split MCCFR regret storage into global preflop + per-flop postflop tables with i16-quantized compact storage and a preload buffer for concurrent disk I/O.

**Architecture:** A new `CompactStorage` (i16 regrets, i32 strategy sums) replaces `BlueprintStorage` for per-flop use, cutting per-flop memory from 1GB to 500MB. A `PerFlopRegretStore` manages a preload buffer using crossbeam channels for true multi-consumer support. Workers grab ready flops from the buffer, process them, and hand back dirty storage for async write.

**Tech Stack:** Rust, rayon, crossbeam-channel, existing BlueprintStorage API

**Design doc:** `docs/plans/2026-03-17-per-flop-regrets-design.md`

---

### Task 1: CompactStorage — i16 Regrets, i32 Strategy Sums

Create a compact storage struct that uses `AtomicI16` for regrets and `AtomicI32` for strategy sums. Same API surface as `BlueprintStorage` so the MCCFR traversal can use either.

**Files:**
- Create: `crates/core/src/blueprint_v2/compact_storage.rs`
- Modify: `crates/core/src/blueprint_v2/mod.rs`

**The struct:**
```rust
pub struct CompactStorage {
    pub regrets: Vec<AtomicI16>,
    pub strategy_sums: Vec<AtomicI32>,
    pub bucket_counts: [u16; 4],
    layout: Vec<NodeLayout>,  // same NodeLayout as BlueprintStorage
}
```

**Same API as BlueprintStorage:**
- `new(tree, bucket_counts) -> Self`
- `get_regret(node_idx, bucket, action) -> i32` (widens i16 to i32 on read)
- `add_regret(node_idx, bucket, action, delta: i32)` (clamps to i16 range)
- `current_strategy(node_idx, bucket) -> Vec<f64>`
- `current_strategy_into(node_idx, bucket, out: &mut [f64])`
- `get_strategy_sum(node_idx, bucket, action) -> i64` (widens i32 to i64)
- `add_strategy_sum(node_idx, bucket, action, delta: i64)` (clamps to i32)
- `save_regrets(path)` / `load_regrets(path, tree, bucket_counts)`
- `num_slots() -> usize`

**Key: clamping.** `add_regret` adds delta but clamps result to `[-32768, 32767]`. Use `fetch_update` or load-compute-store with CAS loop for atomic clamped add. Or simpler: just let it wrap (the MCCFR algorithm is tolerant of regret magnitude, only signs matter for strategy).

Actually simplest: use `AtomicI16::fetch_add` and let it saturate. Rust's `AtomicI16` wraps on overflow, but we can use `.fetch_add(delta as i16)` which truncates. For MCCFR this is fine — large regrets mean "strongly prefer this action" and clamping to ±32K preserves that signal.

**Serialization:** Binary format: magic `CMP1`, bucket_counts, then raw i16 regrets, then raw i32 strategy sums. Much smaller files than BlueprintStorage (~500MB vs ~1GB per flop).

**Tests:**
- `new` creates zeroed storage
- `add_regret` + `get_regret` round-trip
- `current_strategy` produces valid distribution
- `save/load` round-trip preserves regrets and strategy sums
- Size is ~half of equivalent BlueprintStorage

---

### Task 2: Storage Trait for MCCFR

Extract a trait that both `BlueprintStorage` and `CompactStorage` implement, so the traversal code works with either.

**Files:**
- Create or modify: `crates/core/src/blueprint_v2/storage.rs`
- Modify: `crates/core/src/blueprint_v2/mccfr.rs`

**The trait:**
```rust
pub trait RegretStorage: Sync {
    fn get_regret(&self, node_idx: u32, bucket: u16, action: usize) -> i32;
    fn add_regret(&self, node_idx: u32, bucket: u16, action: usize, delta: i32);
    fn current_strategy_into(&self, node_idx: u32, bucket: u16, out: &mut [f64]);
    fn get_strategy_sum(&self, node_idx: u32, bucket: u16, action: usize) -> i64;
    fn add_strategy_sum(&self, node_idx: u32, bucket: u16, action: usize, delta: i64);
    fn num_actions(&self, node_idx: u32) -> usize;
}
```

Implement for both `BlueprintStorage` and `CompactStorage`.

Change `traverse_external` and related functions to be generic over `S: RegretStorage` instead of taking `&BlueprintStorage` directly.

**Backwards compatible:** Existing code using `&BlueprintStorage` continues to work since it implements the trait.

**Tests:**
- Existing traversal tests pass unchanged
- New test: traversal works with `CompactStorage`

---

### Task 3: Split MCCFR Traversal for Two-Level Storage

Modify traversal to accept separate preflop and postflop storage, selecting by street.

**Files:**
- Modify: `crates/core/src/blueprint_v2/mccfr.rs`

**New signature (using the trait from Task 2):**
```rust
fn traverse_external<P: RegretStorage, F: RegretStorage>(
    tree: &GameTree,
    preflop_storage: &P,
    postflop_storage: &F,
    deal: &DealWithBuckets,
    traverser: usize,
    ...
) -> (f64, PruneStats)
```

At Decision nodes:
```rust
let storage: &dyn RegretStorage = if street == Street::Preflop {
    preflop_storage
} else {
    postflop_storage
};
```

Or use an enum dispatch to avoid dynamic dispatch overhead on the hot path.

**Backwards compat:** Callers can pass the same storage for both args.

**Tests:**
- All existing tests pass (same storage for both)
- New test: preflop regrets go to preflop storage, postflop to postflop

---

### Task 4: PerFlopRegretStore — Preload Buffer

The store that manages loading/saving per-flop `CompactStorage` with a background preloader.

**Files:**
- Create: `crates/core/src/blueprint_v2/per_flop_regret_store.rs`
- Modify: `crates/core/src/blueprint_v2/mod.rs`
- Add dependency: `crossbeam-channel` to `crates/core/Cargo.toml`

**Architecture:**
```
Schedule:    [flop_823, flop_12, flop_1504, ...]
                          ^cursor

Reader thread:  walks schedule from cursor+1, loads CompactStorage from disk
                sends (flop_index, flop_cards, weight, storage) through bounded channel

Workers:        recv from channel (multi-consumer, no mutex needed)
                process deals, send dirty storage through write channel

Writer thread:  recv dirty (flop_index, storage), save to disk
```

Uses `crossbeam::channel::bounded` which supports **multiple consumers** (unlike `std::mpsc`). No Mutex needed for the ready channel.

```rust
pub struct PerFlopRegretStore {
    dir: PathBuf,
    tree: Arc<GameTree>,
    bucket_counts: [u16; 4],
}

impl PerFlopRegretStore {
    pub fn new(dir, tree, bucket_counts) -> Self;

    /// Start an epoch. Spawns reader + writer threads.
    /// Returns (ready_rx, dirty_tx, join_handles).
    ///
    /// Workers call ready_rx.recv() to get next flop (blocks until ready).
    /// Workers call dirty_tx.send() to return dirty storage.
    /// Multiple workers can recv concurrently (crossbeam is MPMC).
    pub fn start_epoch(&self, schedule: &EpochSchedule, buffer_size: usize)
        -> (Receiver<FlopWork>, Sender<(u16, CompactStorage)>, Vec<JoinHandle<()>>);
}

pub struct FlopWork {
    pub flop_index: u16,
    pub flop_cards: [Card; 3],
    pub weight: u32,
    pub storage: CompactStorage,
}
```

The key difference from the failed attempt: `start_epoch` returns the channels directly to the caller. Workers call `ready_rx.recv()` themselves — no Mutex wrapper. crossbeam's `Receiver` is `Clone` and supports concurrent recv.

**Tests:**
- Load/save round-trip preserves regrets
- Multiple consumers can recv concurrently
- Writer persists dirty storage to disk

---

### Task 5: sample_deal_for_flop

Add a function to sample random deals conditioned on a specific flop.

**Files:**
- Modify: `crates/core/src/blueprint_v2/mccfr.rs`

```rust
pub fn sample_deal_for_flop(flop: [Card; 3], rng: &mut impl Rng) -> Deal {
    // Build remaining deck (49 cards), shuffle first 6,
    // assign to hole cards (4) + turn + river
}
```

**Tests:**
- Board starts with the given flop cards
- All 7 dealt cards are unique
- Hole cards don't overlap with flop

---

### Task 6: Epoch Schedule

Create the weighted round-robin schedule.

**Files:**
- Create: `crates/core/src/blueprint_v2/epoch_schedule.rs`
- Modify: `crates/core/src/blueprint_v2/mod.rs`

```rust
pub struct ScheduleEntry {
    pub flop_index: u16,
    pub flop_cards: [Card; 3],
    pub weight: u32,
}

pub struct EpochSchedule { pub entries: Vec<ScheduleEntry> }

impl EpochSchedule {
    pub fn new() -> Self;  // 1755 entries from enumerate_canonical_flops()
    pub fn shuffle(&mut self, rng: &mut impl Rng);
}
```

**Tests:**
- 1755 entries
- Sum of weights = 22100
- Shuffle changes order

---

### Task 7: Config Changes

**Files:**
- Modify: `crates/core/src/blueprint_v2/config.rs`

Add to `TrainingConfig`:
```rust
pub per_flop_regrets: bool,        // default false
pub preload_buffer_size: usize,    // default 32
```

Add `flop_buckets: u16` to `PerFlopConfig`.

---

### Task 8: Epoch-Based Training Loop

Wire everything together in `trainer.rs`.

**Files:**
- Modify: `crates/core/src/blueprint_v2/trainer.rs`

```rust
fn train_per_flop(&mut self) -> Result<(), Box<dyn Error>> {
    let preflop_storage = BlueprintStorage::new(&self.tree, [169, 1, 1, 1]);
    let store = PerFlopRegretStore::new(regret_dir, tree, [1, 500, 500, 500]);
    let mut schedule = EpochSchedule::new();

    loop {
        schedule.shuffle(&mut self.rng);
        let (ready_rx, dirty_tx, handles) = store.start_epoch(
            &schedule, self.config.training.preload_buffer_size);

        rayon::scope(|s| {
            for _ in 0..rayon::current_num_threads() {
                let rx = ready_rx.clone();  // crossbeam Receiver is Clone
                let tx = dirty_tx.clone();
                s.spawn(move |_| {
                    let mut rng = SmallRng::from_entropy();
                    while let Ok(work) = rx.recv() {
                        for _ in 0..work.weight {
                            let deal = sample_deal_for_flop(work.flop_cards, &mut rng);
                            let buckets = bucket_lookup.precompute_buckets(&deal);
                            let deal = DealWithBuckets { deal, buckets };
                            traverse_external(
                                &tree, &preflop_storage, &work.storage,
                                &deal, 0, ...);
                            traverse_external(
                                &tree, &preflop_storage, &work.storage,
                                &deal, 1, ...);
                        }
                        tx.send((work.flop_index, work.storage)).ok();
                    }
                });
            }
        });
        drop(dirty_tx);  // signal writer to finish
        for h in handles { h.join().ok(); }

        epoch += 1;
        // time/iteration checks, progress reporting
    }
}
```

Wire into `train()`: `if self.config.training.per_flop_regrets { return self.train_per_flop(); }`

---

### Task 9: Snapshot Save/Load

- Preflop: save/load `preflop.bin` using existing `BlueprintStorage::save_regrets`
- Per-flop: already on disk (writer thread saves after each flop)
- Resume: just load preflop.bin, per-flop files load lazily through the preload buffer

---

### Task 10: Integration Test + Cleanup

- Run training, verify SB develops raising strategy
- Remove any debug prints/panics from prior attempts
- Update docs
