# MCCFR Postflop Solve with Flop-Only Buckets

**Date:** 2026-02-24
**Status:** Design approved, not yet implemented

## Problem

The current postflop solve uses exhaustive bucket CFR with per-street buckets and transition matrices. At each chance node it iterates over all `turn² × river²` bucket pairs, making the cost per iteration `O(flop² × turn² × river² × tree)`. This is impractically slow at useful bucket counts (20/20/50 is ~400x slower than 10/10/10).

## Solution

Replace exhaustive bucket CFR with external-sampling MCCFR using flop-only imperfect recall:

- Strategy tables indexed by **flop bucket only** at all streets
- Sample actual hands and board runouts per traversal
- Evaluate real 7-card hands at showdown
- Cost per sample: `O(tree_nodes)` with no quadratic blowup
- Replaces the current bucket CFR entirely (not kept as an option)

## Key Design Decisions

| Decision | Choice | Rationale |
|-|-|-|
| Strategy key at turn/river | Flop bucket only | Imperfect recall — simpler, smaller tables, no transition matrices |
| Chance node sampling | External sampling | Sample one turn + one river card per traversal. Both players see same board. |
| Showdown evaluation | Real 7-card hand eval | MCCFR samples actual hands, no bucket equity needed |
| Sample count | Percentage of total space | `mccfr_sample_pct` as fraction of all valid (hand, hand, turn, river) combos per flop |
| Bucket pair iteration | Exhaustive | All (hero_bucket, opp_bucket) matchups every iteration |

## Architecture

```
Per-flop pipeline (unchanged interface):

  cluster_flop_buckets()         →  Vec<u16> (169 flop bucket assignments)
       ↓
  mccfr_solve_one_flop()         →  converged strategy_sum buffer
       ↓
  mccfr_extract_values()         →  Vec<f64> (2 × n × n values)
       ↓
  PostflopValues                 →  preflop solver (unchanged consumer)
```

## What Gets Removed

- Turn/river bucketing (clustering, equity computation)
- Flop/turn/river pairwise bucket equity tables (`BucketEquity`)
- Transition matrices (`flop_to_turn`, `turn_to_river`)
- `SingleFlopAbstraction` struct (replaced by `Vec<u16>`)
- `SolveEquity` struct
- `PostflopLayout` per-street bucket counts (all nodes use `num_flop_buckets`)
- `exhaustive_cfr_iteration`, `sampled_cfr_iteration`
- `solve_one_flop`, `solve_cfr_traverse` and related functions

## What Stays Unchanged

- `PostflopValues` format and interface
- `postflop_showdown_value` in the preflop solver
- `PostflopTree` and tree building
- EV rebucketing loop (extract EV histograms → recluster → re-solve)
- Solve caching (`solve_cache.rs`)
- Overall `PostflopAbstraction::build()` flow
- Progress reporting interface (`FlopStage::Bucketing` → `Solving` → `Done`)

## What's Simplified

- `process_single_flop` → returns just `Vec<u16>` (flop bucket assignments)
- `PostflopLayout::build` takes one bucket count (used at all streets)
- Bucketing progress: 1 step instead of 6

## New Components

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

**Removed:**
- `num_hand_buckets_turn`
- `num_hand_buckets_river`
- `equity_rollout_fraction`
- `postflop_solve_samples`

**Renamed:**
- `num_hand_buckets_flop` → `num_hand_buckets` (only one bucket count now)

**Retained:**
- `postflop_solve_iterations`
- `cfr_delta_threshold`
- `postflop_sprs`
- `max_flop_boards` / `fixed_flops`
- `rebucket_rounds`

**New:**
- `mccfr_sample_pct: f64` — fraction of total (hand_pair × turn × river) space to sample per flop (default: 0.01 = 1%)
- `value_extraction_samples: usize` — samples for post-convergence evaluation (default: 10,000)

## Performance

| Metric | Current (20/20/50) | MCCFR (20 buckets) |
|-|-|-|
| Bucketing per flop | 6 steps (3 streets × bucket + equity) | 1 step (flop clustering) |
| Per-iteration cost | O(flop² × turn² × river² × tree) | O(samples × tree) |
| Chance node cost | O(turn² × river²) per traversal | O(1) per traversal |
| Memory (strategy buffers) | flop_b + turn_b + river_b per node | flop_b per node |

With 20 flop buckets: current exhaustive iteration = ~400M traversals. MCCFR with 1% sampling ≈ thousands of traversals per iteration. Orders of magnitude faster.
