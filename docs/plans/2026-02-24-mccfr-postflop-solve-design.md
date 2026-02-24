# MCCFR Postflop Solve with Flop-Only Buckets

**Date:** 2026-02-24
**Status:** Design approved, not yet implemented

## Problem

The current postflop solve uses exhaustive bucket CFR with per-street buckets and transition matrices. At each chance node it iterates over all `turn² × river²` bucket pairs, making the cost per iteration `O(flop² × turn² × river² × tree)`. This is impractically slow at useful bucket counts (20/20/50 is ~400x slower than 10/10/10).

## Solution

Add external-sampling MCCFR as a new postflop solve backend alongside the existing bucket CFR. Selected via a `type` field in the postflop model config:

- `Bucketed` — existing per-street bucket CFR with transition matrices (default, backward compatible)
- `MCCFR` — flop-only imperfect recall with sampled hands and real showdown evaluation

Both backends produce the same `PostflopValues` output consumed by the preflop solver.

## Key Design Decisions

| Decision | Choice | Rationale |
|-|-|-|
| Backend selection | Config `type` field | Allows comparison and gradual migration |
| Strategy key at turn/river (MCCFR) | Flop bucket only | Imperfect recall — simpler, smaller tables, no transition matrices |
| Chance node sampling | External sampling | Sample one turn + one river card per traversal. Both players see same board. |
| Showdown evaluation | Real 7-card hand eval | MCCFR samples actual hands, no bucket equity needed |
| Sample count | Percentage of total space | `mccfr_sample_pct` as fraction of all valid (hand, hand, turn, river) combos per flop |
| Bucket pair iteration | Exhaustive | All (hero_bucket, opp_bucket) matchups every iteration |

## Architecture

Both backends share the same pipeline entry/exit:

```
PostflopAbstraction::build()
    ↓
  match config.solve_type:
    Bucketed → existing per-flop pipeline (process_single_flop + bucket CFR)
    MCCFR    → new per-flop pipeline (cluster_flop_buckets + mccfr solve)
    ↓
  PostflopValues → preflop solver (unchanged consumer)
```

### MCCFR pipeline per flop

```
  cluster_flop_buckets()         →  Vec<u16> (169 flop bucket assignments)
       ↓
  mccfr_solve_one_flop()         →  converged strategy_sum buffer
       ↓
  mccfr_extract_values()         →  Vec<f64> (2 × n × n values)
```

### Bucketed pipeline per flop (unchanged)

```
  process_single_flop()          →  SingleFlopAbstraction (buckets + equity + transitions)
       ↓
  stream_solve_and_extract()     →  Vec<f64> (2 × n × n values)
```

## What Stays Unchanged

- All existing bucket CFR code (retained for `Bucketed` type)
- `PostflopValues` format and interface
- `postflop_showdown_value` in the preflop solver
- `PostflopTree` and tree building
- EV rebucketing loop (extract EV histograms → recluster → re-solve)
- Solve caching (`solve_cache.rs`)
- Progress reporting interface (`FlopStage::Bucketing` → `Solving` → `Done`)

## New Components (MCCFR backend)

### `PostflopSolveType` enum

```rust
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PostflopSolveType {
    Bucketed,
    Mccfr,
}
```

Added to `PostflopModelConfig` as `solve_type` field, defaulting to `Bucketed`.

### `FlopBucketMap`

Maps `flop_bucket → Vec<(Card, Card)>` for sampling hands from a bucket. Built once from the 169 canonical hands and their flop bucket assignments.

### `mccfr_solve_one_flop()`

Training loop:

```
fn mccfr_solve_one_flop(tree, layout, bucket_map, flop, config, on_progress):
    regret_sum = vec![0.0; layout.total_size]   // flop_buckets × actions per node
    strategy_sum = vec![0.0; layout.total_size]

    // Compute sample budget
    total_space = valid_hand_pairs × live_turn_cards × live_river_cards
    num_samples = (total_space * config.mccfr_sample_pct) as usize
    samples_per_iter = num_samples / config.num_iterations

    emit Solving { iteration: 0, ... }

    for iter in 0..num_iterations:
        for _ in 0..samples_per_iter:
            (hero_hand, opp_hand) = sample_non_conflicting_hands(bucket_map, flop)
            turn_card  = random from remaining deck
            river_card = random from remaining deck
            board = [flop[0], flop[1], flop[2], turn_card, river_card]
            hero_bucket = bucket_of(hero_hand)
            opp_bucket  = bucket_of(opp_hand)

            for hero_pos in [0, 1]:
                mccfr_traverse(
                    tree, layout, &mut regret_sum, &mut strategy_sum,
                    node=0, hero_bucket, opp_bucket, hero_pos,
                    hero_hand, opp_hand, board,
                    reach_hero=1.0, reach_opp=1.0, iteration
                )

        emit Solving { iteration: iter+1, delta }

    return FlopSolveResult { strategy_sum, ... }
```

### `mccfr_traverse()`

Single tree traversal:

```
fn mccfr_traverse(node, hero_bucket, opp_bucket, hero_pos,
                  hero_hand, opp_hand, board, ...):
    match node:
        Terminal::Fold { folder }:
            return ±pot_fraction / 2.0

        Terminal::Showdown:
            // Evaluate real 7-card hands
            hero_rank = evaluate_hand(hero_hand, board)
            opp_rank  = evaluate_hand(opp_hand, board)
            equity = if hero_rank > opp_rank { 1.0 }
                     else if hero_rank == opp_rank { 0.5 }
                     else { 0.0 }
            return equity * pot_fraction - pot_fraction / 2.0

        Chance { children, .. }:
            // Board already dealt — just pass through to child
            return mccfr_traverse(children[0], ...)

        Decision { position, children }:
            bucket = if position == hero_pos { hero_bucket } else { opp_bucket }
            strategy = regret_matching(regret_sum, layout.slot(node, bucket))

            if position == hero_pos:
                action_values = [mccfr_traverse(child, ...) for child in children]
                node_value = sum(strategy[i] * action_values[i])
                // Update regrets and strategy sums
                for i in actions:
                    regret_sum[slot + i] += reach_opp * (action_values[i] - node_value)
                    strategy_sum[slot + i] += reach_hero * strategy[i] * iteration
                return node_value
            else:
                // Traverse all opponent actions (external sampling)
                return sum(strategy[i] * mccfr_traverse(child, ...))
```

### `mccfr_extract_values()`

Post-convergence Monte Carlo evaluation:

```
fn mccfr_extract_values(tree, layout, strategy_sum, bucket_map, flop, M):
    values = vec![0.0; 2 * n * n]
    counts = vec![0u32; 2 * n * n]

    for _ in 0..M:
        (hero_hand, opp_hand) = sample_non_conflicting_hands(bucket_map, flop)
        turn, river = random from remaining deck
        board = [flop + turn + river]
        hb = bucket_of(hero_hand)
        ob = bucket_of(opp_hand)

        for hero_pos in [0, 1]:
            ev = eval_with_avg_strategy(
                tree, layout, strategy_sum,
                node=0, hb, ob, hero_pos,
                hero_hand, opp_hand, board
            )
            idx = hero_pos * n*n + hb * n + ob
            values[idx] += ev
            counts[idx] += 1

    // Normalize
    for i in 0..values.len():
        if counts[i] > 0: values[i] /= counts[i]

    return values
```

`M` is controlled by `value_extraction_samples` config (default 10,000).

## Config Changes

### `PostflopModelConfig` additions

```yaml
# New field — selects solve backend
solve_type: mccfr        # or "bucketed" (default)

# MCCFR-specific (ignored when solve_type: bucketed)
mccfr_sample_pct: 0.01            # fraction of total sample space per flop (default: 1%)
value_extraction_samples: 10000   # samples for post-convergence evaluation
```

### Existing fields by backend

| Field | Bucketed | MCCFR |
|-|-|-|
| `num_hand_buckets_flop` | Used | Used (only bucket count) |
| `num_hand_buckets_turn` | Used | Ignored |
| `num_hand_buckets_river` | Used | Ignored |
| `equity_rollout_fraction` | Used | Ignored |
| `postflop_solve_iterations` | Used | Used |
| `postflop_solve_samples` | Used | Ignored (uses `mccfr_sample_pct`) |
| `cfr_delta_threshold` | Used | Used |
| `postflop_sprs` | Used | Used |
| `max_flop_boards` / `fixed_flops` | Used | Used |
| `rebucket_rounds` | Used | Used |
| `mccfr_sample_pct` | Ignored | Used |
| `value_extraction_samples` | Ignored | Used |

No existing config files break — `solve_type` defaults to `Bucketed`.

## Dispatch in `PostflopAbstraction::build()`

The top-level `build()` dispatches based on `config.solve_type`:

```rust
match config.solve_type {
    PostflopSolveType::Bucketed => {
        // Existing pipeline: process_single_flop + bucket CFR
        // (current code, unchanged)
    }
    PostflopSolveType::Mccfr => {
        // New pipeline: cluster flop buckets + MCCFR solve + extract values
        // Layout uses num_flop_buckets for all streets
    }
}
```

Both paths produce the same `PostflopAbstraction` struct with `PostflopValues`.

## Performance

| Metric | Bucketed (20/20/50) | MCCFR (20 buckets) |
|-|-|-|
| Bucketing per flop | 6 steps (3 streets × bucket + equity) | 1 step (flop clustering) |
| Per-iteration cost | O(flop² × turn² × river² × tree) | O(samples × tree) |
| Chance node cost | O(turn² × river²) per traversal | O(1) per traversal |
| Memory (strategy buffers) | flop_b + turn_b + river_b per node | flop_b per node |

With 20 flop buckets: current exhaustive iteration ≈ 400M traversals. MCCFR with 1% sampling ≈ thousands of traversals per iteration. Orders of magnitude faster.

## Example Configs

### Bucketed (existing behavior, unchanged)

```yaml
postflop_model:
  solve_type: bucketed
  num_hand_buckets_flop: 10
  num_hand_buckets_turn: 10
  num_hand_buckets_river: 10
  postflop_solve_iterations: 200
```

### MCCFR

```yaml
postflop_model:
  solve_type: mccfr
  num_hand_buckets_flop: 30
  mccfr_sample_pct: 0.01
  value_extraction_samples: 10000
  postflop_solve_iterations: 500
  cfr_delta_threshold: 0.001
```
