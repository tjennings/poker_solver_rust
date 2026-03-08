# MCCFR Hot-Path Performance Optimizations

## Problem

Profiling the Blueprint V2 MCCFR training loop reveals several hot-path inefficiencies that compound across 2000+ deals per batch × 50+ decision nodes × 2 traversers.

## Optimizations (Ranked by Impact)

### Batch A — Traversal Hot Path (mccfr.rs)

#### 1. Pre-compute buckets per deal (HIGH)

**Problem**: `get_bucket()` calls `compute_equity()` on every non-preflop decision node visit. Each call enumerates ~900-1000 opponent combos. With ~50 nodes × 2000 deals × 2 traversers = 200K equity computations per batch.

**Fix**: Create `DealWithBuckets` that pre-computes all 8 buckets (2 players × 4 streets) when the deal is sampled. Pass pre-computed buckets into `traverse_external`, replacing `get_bucket()` calls with array lookups.

- `DealWithBuckets { deal: Deal, buckets: [[u16; 4]; 2] }` — `buckets[player][street]`
- Compute after sampling in `sample_deal()` or a wrapper
- `traverse_external` takes `&DealWithBuckets` instead of `&Deal` + `&AllBuckets`
- Removes `AllBuckets` parameter from `traverse_traverser` / `traverse_opponent`
- Net: O(streets × players) equity calls per deal instead of O(nodes)

#### 4. Local prune counters (MEDIUM)

**Problem**: `PRUNE_TOTAL` and `PRUNE_HITS` are global `AtomicU64` counters incremented on every pruned action across all rayon threads. Adjacent cache lines get hammered.

**Fix**: Accumulate prune stats locally per deal traversal, return them from `traverse_external`, single `fetch_add` per deal at the end.

- Add `PruneStats { hits: u64, total: u64 }` return alongside `f64` EV
- Or: return `(f64, u64, u64)` tuple
- Aggregate in `par_iter` with local counters, single atomic update after batch

### Batch B — Deal Generation & Parallel Setup (trainer.rs)

#### 2. Static canonical deck (HIGH)

**Problem**: `sample_deal()` rebuilds the 52-card deck from scratch each call (nested loop over VALUES × SUITS). At 2000 deals/batch = 104K pointless `Card::new` writes.

**Fix**: Copy from a pre-initialized `const` or `static` canonical deck array.

- `const CANONICAL_DECK: [Card; 52] = [...]` or `LazyLock`
- `sample_deal` does `self.deck = CANONICAL_DECK;` (single memcpy of 52 bytes)

#### 3. Thread-local RNG (MEDIUM)

**Problem**: `SmallRng::from_os_rng()` called per deal in `par_iter` — 2000 syscalls to `/dev/urandom` per batch.

**Fix**: Use `thread_local!` with `SmallRng` seeded once per rayon thread.

- `thread_local! { static THREAD_RNG: RefCell<SmallRng> = RefCell::new(SmallRng::from_os_rng()); }`
- Or seed from the batch RNG sequentially before `par_iter`

### Deferred (Measure After A+B)

#### 5. ev_sum/ev_count false sharing (LOW)

Adjacent `AtomicI64` entries for different canonical hands share cache lines. At 1 write per deal this is likely negligible. Measure before adding padding.

#### 6. current_strategy_into SIMD (LOW)

Regret loading in `current_strategy_into` is fine for 3-6 actions. SIMD possible but measure first.

## Implementation Order

1. Item 1 (pre-compute buckets) — biggest impact, changes traversal signature
2. Item 2 (static deck) — simple, no API change
3. Item 3 (thread-local RNG) — simple, no API change
4. Item 4 (local prune counters) — changes traversal return type
5. Items 5-6 — only if benchmarks show remaining contention

## Testing

- All existing `mccfr::tests` must pass
- `cargo test -p poker-solver-core` clean
- Benchmark: compare `bench_mccfr` before/after (baseline ~67ms)
