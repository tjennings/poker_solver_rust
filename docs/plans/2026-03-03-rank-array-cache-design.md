# Rank Array Cache + Equity Table Restructuring

**Date**: 2026-03-03
**Status**: Approved

## Problem

`compute_equity_table` is the dominant bottleneck in postflop solving (~98% of runtime). The current loop structure evaluates `rank_hand()` redundantly — the same hand on the same board is re-evaluated for every opponent matchup. The `rs_poker` algorithmic evaluator (~40M evals/sec) is adequate but the redundant work amplifies the cost.

**Current complexity**: O(169² × combos² × 990 × eval)
where eval = `rank_hand()` via `rs_poker` (~25ns)

## Design

Three changes, applied together:

### 1. Restructure `compute_equity_table` Loop

**Key insight**: For a fixed board (flop + turn + river), each concrete combo's rank is constant regardless of which opponent it faces. Evaluate each combo **once per board**, then do pairwise integer comparisons.

**New structure**:
```
for each (turn, river) board:                     // ~990 boards
  for each concrete combo not conflicting w/ board:
    rank[combo] = rank_hand(combo, board)          // one eval per combo
  for hero_canonical in 0..169:
    for opp_canonical in 0..169:
      for h_combo in hero_combos (non-conflicting):
        for o_combo in opp_combos (non-conflicting):
          if h_combo conflicts with o_combo: skip
          equity[hero][opp] += compare(rank[h_combo], rank[o_combo])
```

**New complexity**: O(~1200 combos × 990 × eval) + O(169² × combos² × 990 × compare)
Since eval >> compare, this eliminates the dominant cost.

**Combo handling**: There are ~1,200 concrete combos in a 52-card deck (C(52,2) = 1,326 minus board-conflicting ones). Each canonical hand maps to 4-16 concrete combos. The rank array is indexed by concrete combo, and the comparison loop iterates concrete combos grouped by canonical hand.

### 2. Cache Rank Arrays to Disk

For each canonical flop, persist the per-board rank data so subsequent training runs skip all hand evaluation.

**Cached data per flop**:
- List of ~990 valid (turn, river) boards
- For each board: array of `u16` ranks indexed by concrete combo
- Mapping from canonical hand index → concrete combo indices (board-independent, but filtered per board for conflicts)

**Format**: Bincode-serialized with magic header + version, same pattern as existing `EquityTableCache`.

**Size**: ~990 boards × ~1,200 combos × 2 bytes ≈ **2.4 MB per flop**, or **~4.2 GB** for all 1,755 canonical flops.

*Note*: This is larger than the equity table cache (~169² × 8 bytes × 1,755 = ~390 MB) because we're storing per-board granularity. The trade-off is that deriving equity tables from cached rank arrays is near-instant (pure integer comparison).

**Alternative**: If 4.2 GB is too large, we could cache only the final equity tables (as the existing `EquityTableCache` does) but compute them via the restructured loop. This still gives the ~50-100x speedup on first run, just without the instant-rebuild capability.

### 3. Keep `rs_poker` as Evaluator

Since rank arrays are cached to disk and only computed once per flop ever, the evaluator speed is not critical. `rs_poker` at ~40M evals/sec is adequate:
- ~1,200 combos × 990 boards = ~1.19M evaluations per flop
- At 40M/s = ~30ms per flop (trivially fast)
- All 1,755 flops (parallel): ~1 minute total on first run

No new dependencies. No new hand evaluation code to maintain.

## Cache Integration

**Workflow**:
1. Check for cached rank arrays on disk
2. If found: load and derive equity tables via integer comparison (~milliseconds per flop)
3. If not found: compute rank arrays using `rs_poker`, save to disk, then derive equity tables
4. The existing `EquityTableCache` becomes redundant (equity tables are trivially derived from rank arrays) but can be kept for backwards compatibility

**Cache location**: Same directory as equity table cache, configurable via config.

**Cache invalidation**: Version field in header. Bump version if card encoding or rank semantics change.

## Files Modified

| File | Change |
|------|--------|
| `crates/core/src/preflop/postflop_exhaustive.rs` | Restructure `compute_equity_table` |
| `crates/core/src/preflop/equity_table_cache.rs` | Add rank array cache (or new file) |
| `crates/core/src/preflop/mod.rs` | Export new cache types |
| `crates/core/benches/equity_table_bench.rs` | Update benchmark for new structure |

## Expected Performance

| Metric | Before | After (first run) | After (cached) |
|--------|--------|-------------------|----------------|
| Evals per flop | ~billions | ~1.19M | 0 |
| Time per flop | ~15-25s | ~30ms compute + derive | ~1ms (load + compare) |
| All 1,755 flops | hours | ~1 min | seconds |

## Risks

- **Cache size**: ~4.2 GB for all flops. Acceptable for desktop/cloud but worth monitoring.
- **Correctness**: Restructured loop must produce identical equity values. Verify with a comparison test against the old implementation before removing it.
- **Combo conflict logic**: Must correctly handle card conflicts between hero combos, opponent combos, and board cards when iterating per-board.
