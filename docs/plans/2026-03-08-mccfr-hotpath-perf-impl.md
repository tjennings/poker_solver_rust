# MCCFR Hot-Path Performance Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use hex:executing-plans to implement this plan task-by-task.

**Goal:** Eliminate the four biggest performance bottlenecks in the MCCFR training loop: redundant equity computation on every node visit, deck rebuilding per deal, OS entropy syscalls per thread, and global atomic contention on prune counters.

**Architecture:** All changes are in `crates/core/src/blueprint_v2/{mccfr.rs, trainer.rs}`. The traversal signature changes to accept pre-computed buckets instead of computing them inline. Deal generation and parallel dispatch are optimised in the trainer. No public API changes beyond this module.

**Tech Stack:** Rust, rayon, rand (SmallRng), std::sync::atomic

---

### Task 1: Pre-compute buckets per deal

The biggest win. Currently `get_bucket()` calls `compute_equity()` (enumerating ~990 opponent combos) on every decision node visit. We pre-compute all buckets when the deal is sampled and pass them through traversal as array lookups.

**Files:**
- Modify: `crates/core/src/blueprint_v2/mccfr.rs`
- Modify: `crates/core/src/blueprint_v2/trainer.rs`

**Step 1: Add `DealWithBuckets` struct to mccfr.rs**

Add this after the existing `Deal` struct (line 45):

```rust
/// A deal with pre-computed bucket assignments for all streets and players.
///
/// Avoids calling `compute_equity()` on every decision node visit during
/// MCCFR traversal — buckets are computed once when the deal is sampled.
#[derive(Debug, Clone)]
pub struct DealWithBuckets {
    pub deal: Deal,
    /// Pre-computed bucket indices: `buckets[player][street]`.
    pub buckets: [[u16; 4]; 2],
}
```

**Step 2: Add `AllBuckets::precompute_buckets` method**

Add this method to the `impl AllBuckets` block, after `board_for_street`:

```rust
/// Pre-compute bucket assignments for all 4 streets × 2 players.
///
/// Called once per deal at sampling time. Returns the bucket array
/// to embed in [`DealWithBuckets`].
#[must_use]
pub fn precompute_buckets(&self, deal: &Deal) -> [[u16; 4]; 2] {
    let mut result = [[0u16; 4]; 2];
    for player in 0..2 {
        for street_idx in 0..4u8 {
            let street = match street_idx {
                0 => Street::Preflop,
                1 => Street::Flop,
                2 => Street::Turn,
                _ => Street::River,
            };
            let board = Self::board_for_street(&deal.board, street);
            result[player][street_idx as usize] =
                self.get_bucket(street, deal.hole_cards[player], board);
        }
    }
    result
}
```

**Step 3: Change `traverse_external` signature**

Replace the `deal: &Deal` + `buckets: &AllBuckets` parameters with `deal: &DealWithBuckets`. Remove `buckets` parameter entirely. Update the function signature at line 121:

```rust
pub fn traverse_external(
    tree: &GameTree,
    storage: &BlueprintStorage,
    deal: &DealWithBuckets,
    traverser: u8,
    node_idx: u32,
    prune: bool,
    prune_threshold: i32,
    rng: &mut impl Rng,
) -> f64 {
```

In the `Decision` match arm (around line 162-163), replace the bucket lookup:

```rust
// Old:
let visible_board = AllBuckets::board_for_street(&deal.board, street);
let bucket = buckets.get_bucket(street, deal.hole_cards[player as usize], visible_board);

// New:
let bucket = deal.buckets[player as usize][street as usize];
```

Remove `buckets` from all recursive `traverse_external` calls in this function (the `Chance` arm and both `traverse_traverser`/`traverse_opponent` calls).

**Step 4: Update `traverse_traverser` and `traverse_opponent` signatures**

Remove `buckets: &AllBuckets` parameter from both functions. Change `deal: &Deal` to `deal: &DealWithBuckets`. Update their recursive `traverse_external` calls to drop the `buckets` argument. The `deal` parameter they pass through is already the `&DealWithBuckets`.

`traverse_traverser` (line 232):
```rust
fn traverse_traverser(
    tree: &GameTree,
    storage: &BlueprintStorage,
    deal: &DealWithBuckets,
    traverser: u8,
    node_idx: u32,
    bucket: u16,
    children: &[u32],
    num_actions: usize,
    prune: bool,
    prune_threshold: i32,
    rng: &mut impl Rng,
) -> f64 {
```

`traverse_opponent` (line 296):
```rust
fn traverse_opponent(
    tree: &GameTree,
    storage: &BlueprintStorage,
    deal: &DealWithBuckets,
    traverser: u8,
    node_idx: u32,
    bucket: u16,
    children: &[u32],
    num_actions: usize,
    prune: bool,
    prune_threshold: i32,
    rng: &mut impl Rng,
) -> f64 {
```

**Step 5: Update `terminal_value` to use `deal.deal`**

`terminal_value` takes `deal: &Deal`. Callers now pass `&deal_with_buckets.deal` instead. Update the call site in `traverse_external`:

```rust
GameNode::Terminal { kind, invested, .. } => {
    terminal_value(*kind, invested, traverser, &deal.deal)
}
```

**Step 6: Update trainer.rs to build `DealWithBuckets`**

In `trainer.rs`, the `train()` method at line 383:

```rust
// Old:
let deals: Vec<Deal> = (0..this_batch).map(|_| self.sample_deal()).collect();

// New:
let deals: Vec<DealWithBuckets> = (0..this_batch)
    .map(|_| {
        let deal = self.sample_deal();
        let buckets = self.buckets.precompute_buckets(&deal);
        DealWithBuckets { deal, buckets }
    })
    .collect();
```

Update the `par_iter` traversal calls (lines 398-403) to drop `buckets`:

```rust
let ev0 = traverse_external(
    tree, storage, deal, 0, tree.root, prune, threshold, &mut rng,
);
let ev1 = traverse_external(
    tree, storage, deal, 1, tree.root, prune, threshold, &mut rng,
);
```

Update the EV accumulation lines (406-413) to use `deal.deal.hole_cards`:

```rust
let idx0 = CanonicalHand::from_cards(
    deal.deal.hole_cards[0][0],
    deal.deal.hole_cards[0][1],
).index();
let idx1 = CanonicalHand::from_cards(
    deal.deal.hole_cards[1][0],
    deal.deal.hole_cards[1][1],
).index();
```

**Step 7: Update all tests in mccfr.rs**

Every test that calls `traverse_external` currently passes `&buckets` and `&deal` separately. Update them to build a `DealWithBuckets` and drop the `&buckets` argument.

For each test, replace:
```rust
// Old:
let deal = make_deal();
// ... later:
traverse_external(&tree, &storage, &buckets, &deal, 0, tree.root, false, -310_000_000, &mut rng);

// New:
let deal = make_deal();
let precomputed = DealWithBuckets {
    buckets: buckets.precompute_buckets(&deal),
    deal,
};
// ... later:
traverse_external(&tree, &storage, &precomputed, 0, tree.root, false, -310_000_000, &mut rng);
```

Tests affected: `traverse_returns_finite`, `traverse_both_players`, `traverse_updates_regrets`, `traverse_updates_strategy_sums`, `multiple_iterations_change_strategy`, `traverse_with_pruning`.

The `multiple_iterations_change_strategy` test also calls `buckets.get_bucket()` directly at line 558 — keep that call as-is since it's checking bucket values, not traversing.

**Step 8: Run tests and commit**

Run: `cargo test -p poker-solver-core -- blueprint_v2::mccfr`
Expected: all 7 tests pass

Run: `cargo clippy -p poker-solver-core`
Expected: clean

Commit: `perf: pre-compute bucket assignments per deal in MCCFR traversal`

---

### Task 2: Static canonical deck

**Files:**
- Modify: `crates/core/src/blueprint_v2/trainer.rs`

**Step 1: Add a `const` canonical deck**

Add near the top of `trainer.rs`, after the imports (around line 32). Since `Card::new` may not be `const`, use `std::sync::LazyLock`:

```rust
use std::sync::LazyLock;

/// Pre-initialized canonical deck, copied into the trainer's deck buffer
/// via memcpy instead of rebuilding from VALUE×SUIT loops each deal.
static CANONICAL_DECK: LazyLock<[Card; 52]> = LazyLock::new(|| {
    let mut deck = [Card::new(ALL_VALUES[0], ALL_SUITS[0]); 52];
    let mut idx = 0;
    for &v in &ALL_VALUES {
        for &s in &ALL_SUITS {
            deck[idx] = Card::new(v, s);
            idx += 1;
        }
    }
    deck
});
```

**Step 2: Simplify `sample_deal`**

Replace the deck-building loop in `sample_deal` (lines 435-441):

```rust
// Old:
let mut idx = 0;
for &v in &ALL_VALUES {
    for &s in &ALL_SUITS {
        self.deck[idx] = Card::new(v, s);
        idx += 1;
    }
}

// New:
self.deck = *CANONICAL_DECK;
```

**Step 3: Simplify `new()` deck initialisation**

In the `new()` constructor (lines 183-190), replace the loop with:

```rust
let deck = *CANONICAL_DECK;
```

**Step 4: Run tests and commit**

Run: `cargo test -p poker-solver-core -- blueprint_v2`
Expected: all pass

Commit: `perf: use static canonical deck instead of rebuilding per deal`

---

### Task 3: Thread-local RNG in parallel traversal

**Files:**
- Modify: `crates/core/src/blueprint_v2/trainer.rs`

**Step 1: Replace `from_os_rng()` with thread-local seeded RNG**

In the `train()` method, before the `par_iter` block (around line 395), seed per-thread RNGs from the batch. Replace the `par_iter` closure:

```rust
// Old (inside par_iter):
let mut rng = SmallRng::from_os_rng();

// New: seed RNGs sequentially, then use in parallel
let thread_seeds: Vec<u64> = (0..this_batch).map(|_| self.rng.random()).collect();

// Then in par_iter, index into seeds:
deals.par_iter().enumerate().for_each(|(i, deal)| {
    let mut rng = SmallRng::seed_from_u64(thread_seeds[i]);
    // ... rest unchanged
});
```

This eliminates the `from_os_rng()` syscall while keeping each deal's RNG independently seeded from the master RNG. The sequential seed generation is negligible (one `random()` call per deal on the fast `StdRng`).

**Step 2: Run tests and commit**

Run: `cargo test -p poker-solver-core -- blueprint_v2`
Expected: all pass

Commit: `perf: pre-seed RNGs sequentially to avoid OS entropy syscalls in par_iter`

---

### Task 4: Local prune counters

**Files:**
- Modify: `crates/core/src/blueprint_v2/mccfr.rs`
- Modify: `crates/core/src/blueprint_v2/trainer.rs`

**Step 1: Add `PruneStats` struct and accumulator**

In `mccfr.rs`, add after the `DealWithBuckets` struct:

```rust
/// Prune statistics accumulated during a single traversal.
///
/// Collected locally to avoid global atomic contention, then merged
/// into the global counters after the batch completes.
#[derive(Debug, Default, Clone, Copy)]
pub struct PruneStats {
    pub hits: u64,
    pub total: u64,
}

impl PruneStats {
    pub fn merge(&mut self, other: PruneStats) {
        self.hits += other.hits;
        self.total += other.total;
    }
}
```

**Step 2: Change `traverse_external` return type**

Change return type from `f64` to `(f64, PruneStats)`. Add a `stats` accumulator:

```rust
pub fn traverse_external(
    tree: &GameTree,
    storage: &BlueprintStorage,
    deal: &DealWithBuckets,
    traverser: u8,
    node_idx: u32,
    prune: bool,
    prune_threshold: i32,
    rng: &mut impl Rng,
) -> (f64, PruneStats) {
    match &tree.nodes[node_idx as usize] {
        GameNode::Terminal { kind, invested, .. } => {
            (terminal_value(*kind, invested, traverser, &deal.deal), PruneStats::default())
        }

        GameNode::Chance { child, .. } => {
            traverse_external(
                tree, storage, deal, traverser, *child, prune, prune_threshold, rng,
            )
        }

        GameNode::Decision {
            player,
            street,
            children,
            ..
        } => {
            let player = *player;
            let street = *street;
            let num_actions = children.len();
            let bucket = deal.buckets[player as usize][street as usize];

            if player == traverser {
                traverse_traverser(
                    tree, storage, deal, traverser, node_idx, bucket, children,
                    num_actions, prune, prune_threshold, rng,
                )
            } else {
                traverse_opponent(
                    tree, storage, deal, traverser, node_idx, bucket, children,
                    num_actions, prune, prune_threshold, rng,
                )
            }
        }
    }
}
```

**Step 3: Update `traverse_traverser` to accumulate locally**

Change return type to `(f64, PruneStats)`. Replace the global atomic increments with local counter:

```rust
fn traverse_traverser(
    tree: &GameTree,
    storage: &BlueprintStorage,
    deal: &DealWithBuckets,
    traverser: u8,
    node_idx: u32,
    bucket: u16,
    children: &[u32],
    num_actions: usize,
    prune: bool,
    prune_threshold: i32,
    rng: &mut impl Rng,
) -> (f64, PruneStats) {
    debug_assert!(num_actions <= MAX_ACTIONS);
    let mut strategy_buf = [0.0f64; MAX_ACTIONS];
    storage.current_strategy_into(node_idx, bucket, &mut strategy_buf[..num_actions]);
    let strategy = &strategy_buf[..num_actions];

    let mut action_values_buf = [0.0f64; MAX_ACTIONS];
    let action_values = &mut action_values_buf[..num_actions];
    let mut node_value = 0.0f64;
    let mut stats = PruneStats::default();

    for (a, &child_idx) in children.iter().enumerate() {
        if prune {
            stats.total += 1;
            if storage.get_regret(node_idx, bucket, a) < prune_threshold {
                stats.hits += 1;
                continue;
            }
        }

        let (child_ev, child_stats) = traverse_external(
            tree, storage, deal, traverser, child_idx, prune, prune_threshold, rng,
        );
        action_values[a] = child_ev;
        node_value += strategy[a] * child_ev;
        stats.merge(child_stats);
    }

    for a in 0..num_actions {
        let delta = action_values[a] - node_value;
        storage.add_regret(node_idx, bucket, a, (delta * 1000.0) as i32);
    }

    for a in 0..num_actions {
        storage.add_strategy_sum(node_idx, bucket, a, (strategy[a] * 1000.0) as i64);
    }

    (node_value, stats)
}
```

**Step 4: Update `traverse_opponent` similarly**

Change return type to `(f64, PruneStats)` and propagate child stats:

```rust
fn traverse_opponent(
    tree: &GameTree,
    storage: &BlueprintStorage,
    deal: &DealWithBuckets,
    traverser: u8,
    node_idx: u32,
    bucket: u16,
    children: &[u32],
    num_actions: usize,
    prune: bool,
    prune_threshold: i32,
    rng: &mut impl Rng,
) -> (f64, PruneStats) {
    debug_assert!(num_actions <= MAX_ACTIONS);
    let mut strategy_buf = [0.0f64; MAX_ACTIONS];
    storage.current_strategy_into(node_idx, bucket, &mut strategy_buf[..num_actions]);
    let strategy = &strategy_buf[..num_actions];

    let r: f64 = rng.random();
    let mut cumulative = 0.0;
    let mut chosen = num_actions - 1;
    for (a, &prob) in strategy.iter().enumerate() {
        cumulative += prob;
        if r < cumulative {
            chosen = a;
            break;
        }
    }

    traverse_external(
        tree, storage, deal, traverser, children[chosen], prune, prune_threshold, rng,
    )
}
```

**Step 5: Update trainer.rs to aggregate prune stats**

In the `train()` method's `par_iter`, collect stats locally and do a single atomic update after the batch. Replace the `par_iter` block:

```rust
// Use par_iter with map+reduce to collect prune stats
let batch_prune_stats: PruneStats = deals.par_iter().enumerate().map(|(i, deal)| {
    let mut rng = SmallRng::seed_from_u64(thread_seeds[i]);
    let mut stats = PruneStats::default();

    let (ev0, s0) = traverse_external(
        tree, storage, deal, 0, tree.root, prune, threshold, &mut rng,
    );
    stats.merge(s0);
    let (ev1, s1) = traverse_external(
        tree, storage, deal, 1, tree.root, prune, threshold, &mut rng,
    );
    stats.merge(s1);

    let idx0 = CanonicalHand::from_cards(
        deal.deal.hole_cards[0][0],
        deal.deal.hole_cards[0][1],
    ).index();
    let idx1 = CanonicalHand::from_cards(
        deal.deal.hole_cards[1][0],
        deal.deal.hole_cards[1][1],
    ).index();
    ev_sum[idx0].fetch_add((ev0 * 1000.0) as i64, Ordering::Relaxed);
    ev_count[idx0].fetch_add(1, Ordering::Relaxed);
    ev_sum[idx1].fetch_add((ev1 * 1000.0) as i64, Ordering::Relaxed);
    ev_count[idx1].fetch_add(1, Ordering::Relaxed);

    stats
}).reduce(PruneStats::default, |mut a, b| { a.merge(b); a });

// Single atomic update for the whole batch
PRUNE_HITS.fetch_add(batch_prune_stats.hits, AtomicOrdering::Relaxed);
PRUNE_TOTAL.fetch_add(batch_prune_stats.total, AtomicOrdering::Relaxed);
```

Note: `PRUNE_HITS` and `PRUNE_TOTAL` are still read by `check_timed_actions` for the TUI. The globals remain but are now updated once per batch instead of per-action.

**Step 6: Update tests in mccfr.rs**

All tests that call `traverse_external` now get `(f64, PruneStats)`. Destructure the return:

```rust
// Old:
let ev = traverse_external(...);
assert!(ev.is_finite());

// New:
let (ev, _stats) = traverse_external(...);
assert!(ev.is_finite());
```

Update all 6 tests similarly.

**Step 7: Run tests and commit**

Run: `cargo test -p poker-solver-core -- blueprint_v2`
Expected: all pass

Run: `cargo clippy -p poker-solver-core`
Expected: clean

Commit: `perf: accumulate prune stats locally to eliminate global atomic contention`

---

## Final Verification

After all 4 tasks:

Run: `cargo test -p poker-solver-core`
Expected: all tests pass

Run: `cargo clippy -p poker-solver-core`
Expected: clean

Run: `cargo test` (full suite)
Expected: all pass, under 1 minute
