# Replace EHS2 Buckets with Canonical 169 Hands

**Date:** 2026-02-25
**Status:** Design approved

## Problem

The current postflop abstraction clusters 169 canonical hands into N EHS2 buckets via histogram k-means. This adds complexity (clustering, transition matrices, rebucketing) and loses information. With 169 hands the strategy tables are still tractable (~57K entries per flop), so we can skip clustering entirely and index strategies directly by canonical hand.

## Solution

Remove the bucketed CFR backend entirely. Keep two postflop solve backends, both indexed by canonical hand (0-168):

- **MCCFR** — external-sampling MCCFR with sampled concrete hands and real showdown eval
- **Exhaustive** — vanilla CFR with pre-computed equity tables and full chance-node enumeration

Both produce the same `PostflopValues` output consumed by the preflop solver.

## Section 2: MCCFR Backend Changes

Current flow per flop:
1. EHS cluster 169 hands -> N flop buckets
2. Build FlopBucketMap (bucket -> concrete combos)
3. Sample (hero_bucket, opp_bucket) pair, pick random concrete hands from each bucket
4. Sample random turn + river cards
5. Traverse postflop tree, strategies indexed by flop bucket
6. At showdown: rank_hand() on concrete cards

New flow per flop:
1. Build valid combo list per canonical hand (filter board conflicts) - replaces FlopBucketMap
2. Sample (hero_hand_169, opp_hand_169) pair, pick random concrete combos from each
3. Sample random turn + river cards
4. Traverse postflop tree, strategies indexed by canonical hand index (0-168)
5. At showdown: rank_hand() on concrete cards (unchanged)

The core MCCFR traversal logic stays the same - the only change is replacing bucket lookups with direct hand index. regret_matching_into, weighted_avg_strategy_delta, and the training loop structure are unchanged.

Sampling: mccfr_sample_pct still controls what fraction of (hand_pair x turn x river) space to sample per iteration. With 169 hands instead of ~10-30 buckets, there are more info sets, but the sample space per flop is the same.

Value extraction: Post-convergence, still uses Monte Carlo sampling to estimate values[flop_idx][hero_pos][hero_hand][opp_hand] from the converged strategy. Same approach, just 169x169 output instead of NxN buckets.

## Section 3: Exhaustive Backend

New backend replacing postflop_bucketed.rs.

Flow per flop:
1. Pre-compute equity table: equity[street][hero_169][opp_169] -> f64 for flop, turn, and river
   - For each (hero, opp) canonical pair, enumerate all non-conflicting concrete combos
   - For flop equity: enumerate all turn+river runouts, evaluate with rank_hand(), average
   - For turn equity: given a turn card, enumerate river cards, evaluate, average
   - For river equity: direct rank_hand() evaluation, average across combos
2. Build full game tree for the flop (same PostflopTree as today)
3. Vanilla CFR traversal - both players every iteration:
   - At chance nodes (street transitions): enumerate all next cards, weight by 1/num_cards
   - At decision nodes: strategies indexed by canonical hand (0-168)
   - At showdown terminals: look up pre-computed equity table -> O(1)
   - At fold terminals: standard pot arithmetic
4. Extract converged EV table: values[hero_pos][hero_169][opp_169]

Key difference from MCCFR: No sampling at chance nodes. Every turn card (47 remaining) and every river card (46 remaining) is enumerated. This is exact but more expensive per iteration - compensated by faster convergence (no variance).

Performance consideration: The equity table pre-computation is the expensive part. For a single flop, computing all-streets equity for 169x169 pairs with full runout enumeration is ~169x169 x 47x46 evaluations. This should be parallelized with rayon per canonical hand pair. The CFR iterations themselves are fast since showdown is O(1) table lookup.

Convergence: Use existing cfr_exploitability_threshold for early stopping, measured via exploitability (best-response calculation against the average strategy).

## Section 4: Integration & Shared Types

**PostflopSolveType** - two variants, Bucketed removed:

```rust
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PostflopSolveType {
    Mccfr,
    Exhaustive,
}
```

Default: Mccfr.

**Dead code removal** - delete entirely:
- `postflop_bucketed.rs` (981 lines)
- `hand_buckets.rs` (2270 lines) - EHS histogram clustering
- `ehs.rs` (748 lines) - equity histograms
- `SolveEquity`, `BucketEquity`, `SingleFlopAbstraction` types
- Transition matrix infrastructure
- `StreetBuckets`
- `rebucket_rounds` config and EV rebucketing loop

**Canonical hand index utility** - shared:

```rust
pub fn canonical_hand_index(c1: Card, c2: Card) -> u8  // 0-168
```

Extracted from existing 169-hand enumeration before deleting hand_buckets.rs.

**Combo map** - shared:

```rust
fn build_combo_map(flop: &[Card; 3]) -> Vec<Vec<(Card, Card)>>  // length 169
```

Non-conflicting concrete combos per canonical hand.

**PostflopLayout** - unchanged structurally, always 169:

```rust
PostflopLayout::build(&tree, &streets, 169, 169, 169)
```

**PostflopValues** - num_buckets always 169:

```
values[flop_idx * 2 * 169^2 + hero_pos * 169^2 + hero_hand * 169 + opp_hand]
```

~57K entries per flop. 200 flops ~ 91MB.

**PostflopAbstraction** struct simplifies:
- Remove `buckets: StreetBuckets`
- Remove `street_equity` / `transitions` diagnostic fields
- Keep `values`, `tree`, `hand_avg_values`

**Preflop consumer** - maps hole cards directly to canonical index via `canonical_hand_index()` instead of going through StreetBuckets.

**build() dispatch**:

```rust
match config.solve_type {
    PostflopSolveType::Mccfr => build_mccfr(config, ...),
    PostflopSolveType::Exhaustive => build_exhaustive(config, ...),
}
```

**Config cleanup** - fields removed:
- `num_hand_buckets_flop/turn/river`
- `equity_rollout_fraction`
- `rebucket_rounds`
- `postflop_solve_samples`

Fields kept:
- `solve_type`, `postflop_solve_iterations`, `cfr_exploitability_threshold`
- `postflop_sprs`, `max_flop_boards`, `fixed_flops`
- `bet_sizes`, `max_raises_per_street`
- `mccfr_sample_pct`, `value_extraction_samples` (MCCFR-specific)

**FlopStage** simplifies:

```rust
pub enum FlopStage {
    Solving { iteration, max_iterations, exploitability },
    EstimatingEv { sample, total_samples },  // MCCFR only
    Done,
}
```

**BuildPhase** - remove `Rebucketing` variant.
