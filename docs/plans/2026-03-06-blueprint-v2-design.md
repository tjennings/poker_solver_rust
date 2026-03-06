# Blueprint V2: Pluribus-Style Full-Game Solver

**Date:** 2026-03-06
**Status:** Approved

## Overview

A new, independent solver pipeline that trains all 4 streets (preflop through river) in a single MCCFR run, following the Pluribus architecture. Completely separate from the existing preflop LCFR + postflop abstraction pipeline -- no shared solver code paths, though it reuses low-level utilities (hand eval, card types, EHS2 computation).

**Key properties:**
- External-sampling MCCFR with LCFR weighting
- Potential-aware card abstraction with EMD clustering (bottom-up, per street)
- Configurable action abstraction per street and per raise depth
- Time-based snapshots compatible with the strategy explorer
- Designed for HU now, multi-way later (N-player parameterized)

**Non-goals for MVP:**
- Real-time subgame solving (bundle format supports it as future leaf values)
- Multi-way training (architecture supports it, not tested)
- GPU acceleration

## Architecture

```
[cluster command]          [train-blueprint command]         [explorer]
      |                          |                               |
  bucket assignments -----> Full-Game MCCFR -----------> Blueprint Bundle
  (per street,              (external sampling,           (time-based snapshots,
   bottom-up clustering)     LCFR weighting)              explorer-loadable)
```

Three new CLI subcommands in `crates/trainer/`:
- `cluster` -- compute and save bucket assignments
- `train-blueprint` -- run full-game MCCFR using bucket files
- `diag-clusters` -- evaluate clustering quality

New module in core: `crates/core/src/blueprint_v2/`

Existing `solve-preflop`, `solve-postflop`, and all explorer commands remain untouched.

## 1. Clustering Pipeline

### Algorithm: Potential-Aware Abstraction with EMD

Built bottom-up. Each street's features are distributions over the next street's buckets:

| Street | Feature Vector | Dimensionality | Distance Metric |
|-|-|-|-|
| River | Equity (scalar hand strength) | 1 | L1 |
| Turn | Distribution over K_river buckets | K_river | EMD (= L1 of CDFs) |
| Flop | Distribution over K_turn buckets | K_turn | EMD |
| Preflop | Distribution over K_flop buckets | K_flop | EMD |

EMD on ordered 1D histograms: `EMD(p,q) = sum |CDF_p(i) - CDF_q(i)|`, O(K).

K-means uses EMD for assignment, component-wise mean for centroid update.

### Process

1. **Cluster river** -- for all (hand, full board) situations, compute equity, k-means into K_river buckets
2. **Cluster turn** -- for each (hand, flop+turn), enumerate ~46 river cards, build histogram over river buckets, k-means with EMD into K_turn buckets
3. **Cluster flop** -- for each (hand, flop), enumerate ~1,081 turn+river completions, build histogram over turn buckets, k-means with EMD into K_flop buckets
4. **Cluster preflop** -- for each hand, enumerate flop samples, build histogram over flop buckets, k-means with EMD into K_preflop buckets

Parallelized per board at each stage.

### Config

```yaml
clustering:
  algorithm: potential_aware_emd
  preflop: { buckets: 169 }
  flop: { buckets: 200 }
  turn: { buckets: 200 }
  river: { buckets: 200 }
  seed: 42
  kmeans_iterations: 100
```

### Output Format

```
clusters/
  config.yaml              # clustering config (reproducibility)
  preflop.buckets          # u8[169] -- identity mapping for lossless HU
  flop.buckets             # binary: per canonical flop, u8[1326] mapping each hole pair combo to bucket ID
  turn.buckets             # binary: per (flop, turn), u8[1326]
  river.buckets            # binary: per (flop, turn, river), u8[1326]
  metadata.json            # bucket counts, hand count, timestamp
```

Bucket files map concrete hole card pairs (not 169 canonical hands) to bucket IDs, because suit isomorphism breaks when conditioned on a specific board. For preflop (no board), the 169 canonical mapping suffices.

Binary format: little-endian, header with magic bytes, version, street, bucket count, board count, combo count. u8 for <=256 buckets, u16 for >256. Memory-mapped at training time.

### Diagnostics (`diag-clusters`)

- **Intra-bucket equity variance** -- hands in same bucket should have similar equity
- **Bucket size distribution** -- detect degenerate/empty clusters
- **EMD between adjacent bucket centroids** -- are buckets well-separated?
- **Sample hands per bucket** -- representative hands for a given bucket on a given board
- **Cross-street transition matrix** -- do strong flop buckets map to strong turn buckets?

Output as human-readable tables or `--json`.

## 2. Action Abstraction & Game Tree

### Config

```yaml
game:
  players: 2                    # HU for now, multi-way later
  stack_depth: 100              # BB
  small_blind: 0.5              # BB
  big_blind: 1.0                # BB

action_abstraction:
  preflop:
    - ["2.5bb"]                 # depth 0: open raise
    - ["3.0x"]                  # depth 1: 3-bet (multiplier of previous raise)
    - ["2.5x"]                  # depth 2: 4-bet
    - ["2.0x"]                  # depth 3: 5-bet
  flop:
    - [0.33, 0.67, 1.0]        # depth 0: first bet (pot fractions)
    - [0.5, 1.0]               # depth 1: raise
    - [1.0]                    # depth 2: re-raise
  turn:
    - [0.5, 1.0]
    - [1.0]
  river:
    - [0.5, 1.0]
    - [1.0]
  max_raises: 3                 # global cap per street
```

### Sizing Conventions

- **Preflop**: absolute sizes (`"2.5bb"`) and multipliers of previous raise (`"3.0x"`)
- **Postflop**: pot fractions (0.33 = 33% pot)
- **All-in**: always implicitly available at every decision point

### Tree Structure

Single game tree covering all 4 streets, built once from config:

- **Decision nodes** -- player to act, available actions (fold, check, call, bet/raise sizes, all-in)
- **Chance nodes** -- street transitions (deal flop/turn/river), sampled during training
- **Terminal nodes** -- fold (pot arithmetic) or showdown (real 7-card hand eval)

Tree is shared across all training iterations. Strategy/regret storage indexed by `(node_id, bucket_id, action_idx)`.

## 3. MCCFR Training Engine

### Algorithm

External-sampling MCCFR with LCFR weighting.

### Per Iteration

1. Sample a complete deal (hole cards for both players + 5 board cards)
2. For the traversing player: traverse all their actions at decision nodes
3. For the opponent: sample one action according to current strategy
4. At chance nodes: board cards already sampled, reveal progressively
5. At terminals: fold = pot arithmetic, showdown = real 7-card hand eval
6. Map each (hand, board_at_street) to bucket ID via pre-computed cluster files
7. Update regrets in the bucket's regret buffer

Alternating traversals: P1 as traverser, then P2, both per "iteration."

### LCFR Weighting

During configurable warmup period, at regular discount intervals multiply regrets and strategy sums by `T/interval / (T/interval + 1)`. Stop discounting after warmup.

### Negative-Regret Pruning

After configurable warmup: in 95% of iterations, skip traverser actions with regret below floor (-310M). In remaining 5%, explore everything. Never prune on the river or at actions leading directly to terminals.

### Storage

- **Regrets**: `Vec<i32>` flat buffer, indexed by `(node_id, bucket_id, action_idx)`. Floor at -310M.
- **Strategy sums**: `Vec<i64>` for snapshot averaging
- **Lazy allocation**: memory for an action sequence allocated only when first encountered

### Config

```yaml
training:
  cluster_path: clusters/
  iterations: 100000            # max iterations (or use time limit)
  time_limit_minutes: 480       # alternative stopping criterion
  lcfr_warmup_minutes: 400      # discount during this period
  lcfr_discount_interval: 10    # discount every N minutes
  prune_after_minutes: 200      # enable pruning after this
  prune_threshold: -310000000   # regret floor
  prune_explore_pct: 0.05       # fraction of iterations that skip pruning
  print_every_minutes: 10       # log convergence metrics
```

### Convergence Metrics

Logged at each print interval:
- Strategy L1 delta (mean absolute change across infosets)
- Mean positive regret (should decrease monotonically)
- Iteration count and elapsed time

## 4. Snapshots & Blueprint Bundle

### Snapshot Schedule

```yaml
snapshots:
  warmup_minutes: 60
  snapshot_every_minutes: 30
  output_dir: training_run_01/
```

### Bundle Structure

```
training_run_01/
  config.yaml                   # full training config
  clusters/                     # full copy of cluster directory (self-contained)
    config.yaml
    preflop.buckets
    flop.buckets
    turn.buckets
    river.buckets
    metadata.json
  snapshot_0060/
    strategy.bin                # bucket-indexed normalized strategy
    metadata.json               # iteration count, elapsed time, metrics
    regrets.bin                 # enables resume from any snapshot
  snapshot_0090/
    ...
  final/
    strategy.bin                # final averaged strategy
    metadata.json
    regrets.bin
```

### Final Strategy Construction

- **Preflop**: running average strategy (lossless abstraction converges cleanly)
- **Postflop**: uniform average of all snapshot strategies taken after warmup (snapshot averaging, Pluribus-style)

### Resume Support

If `regrets.bin` is present, training can resume from any snapshot.

## 5. Explorer Integration

New command: `load_blueprint_bundle` -- distinct from existing `load_bundle` and `load_preflop_solve`.

### How the 13x13 Matrix Works

1. **Preflop** -- bucket ID = canonical hand index (0-168 for lossless HU). Strategy lookup is direct. Works identically to today's preflop explorer.

2. **Postflop** -- user enters a board. For each of the 1326 hole card combos:
   - Look up `street.buckets[board_index][combo]` -> bucket ID
   - Get that bucket's strategy at the current tree node
   - Cell shows weighted average across combos that map to that cell
   - Multiple combos in a cell may map to different buckets

### What the Explorer Does NOT Need to Know

The explorer doesn't need to understand buckets. It resolves hand -> bucket -> strategy, and displays the result in the familiar 13x13 matrix. Bucket inspection lives in `diag-clusters`, not the explorer.

## 6. Implementation Notes

### Code Organization

- New module: `crates/core/src/blueprint_v2/`
- New trainer subcommands in `crates/trainer/src/`
- Existing code untouched: `blueprint/`, `preflop/`, exploration commands

### Parallelism

- **Clustering**: parallelized per board
- **Training**: single-threaded MCCFR (external sampling is sequential per traversal), lock-free regret updates

### Multi-Way Readiness

- Game tree builder parameterized by player count (tested with 2 only)
- Bucket files are per-player (for multi-way, positional asymmetry may need separate buckets)
- External sampling extends to N players (traverse one, sample N-1)

### Sample Configs

**Toy preset** (validates pipeline in minutes):
```yaml
game:
  players: 2
  stack_depth: 10
  small_blind: 0.5
  big_blind: 1.0

clustering:
  algorithm: potential_aware_emd
  preflop: { buckets: 50 }
  flop: { buckets: 50 }
  turn: { buckets: 50 }
  river: { buckets: 50 }
  seed: 42
  kmeans_iterations: 50

action_abstraction:
  preflop:
    - ["2.5bb"]
    - ["3.0x"]
  flop:
    - [0.5, 1.0]
    - [1.0]
  turn:
    - [0.5, 1.0]
  river:
    - [0.5, 1.0]
  max_raises: 2

training:
  cluster_path: clusters/toy/
  time_limit_minutes: 5
  lcfr_warmup_minutes: 2
  lcfr_discount_interval: 1
  prune_after_minutes: 2
  print_every_minutes: 1

snapshots:
  warmup_minutes: 1
  snapshot_every_minutes: 1
  output_dir: runs/toy/
```

**Realistic preset** (full HU solve):
```yaml
game:
  players: 2
  stack_depth: 100
  small_blind: 0.5
  big_blind: 1.0

clustering:
  algorithm: potential_aware_emd
  preflop: { buckets: 169 }
  flop: { buckets: 200 }
  turn: { buckets: 200 }
  river: { buckets: 200 }
  seed: 42
  kmeans_iterations: 100

action_abstraction:
  preflop:
    - ["2.5bb"]
    - ["3.0x"]
    - ["2.5x"]
    - ["2.0x"]
  flop:
    - [0.33, 0.67, 1.0]
    - [0.5, 1.0]
    - [1.0]
  turn:
    - [0.5, 1.0]
    - [1.0]
  river:
    - [0.5, 1.0]
    - [1.0]
  max_raises: 3

training:
  cluster_path: clusters/hu_200/
  time_limit_minutes: 480
  lcfr_warmup_minutes: 200
  lcfr_discount_interval: 10
  prune_after_minutes: 100
  print_every_minutes: 10

snapshots:
  warmup_minutes: 60
  snapshot_every_minutes: 30
  output_dir: runs/hu_full/
```
