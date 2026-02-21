# Preflop Solver: Abstracted Postflop Tree Model

**Status:** Approved — Option 2
**Date:** 2026-02-20
**Scope:** HU only
**Branch:** `feature/preflop-postflop-model`

## Problem

The preflop-only solver with raw equity showdowns finds a limp-trap equilibrium instead of the raise-heavy GTO equilibrium. Suited bluff-raises (T2s, 94s, 93s) have no incentive to raise without postflop streets.

## Solution

Replace showdown terminals in the preflop tree with abstracted 3-street postflop subtrees. The solver discovers equity realization endogenously by solving preflop + postflop jointly. Only the preflop strategy is saved.

## Architecture

```
Precompute (one-time, parallelized):
  1. Board abstraction: canonical flops → k-means → flop textures
  2. Hand abstraction: (hand_169 × flop_tex) → EHS → k-means → buckets
  3. Bucket equity: river_bucket × river_bucket → equity

Solve (iterative LCFR):
  Preflop tree (169 canonical hands, full enumeration)
    ↓ at showdown terminals
  Sample flop texture → map hands to flop buckets
    ↓
  Flop betting tree (bucket-level strategies, 2 sizes)
    ↓ sample turn transition → re-bucket
  Turn betting tree (bucket-level strategies, 2 sizes)
    ↓ sample river transition → re-bucket
  River betting tree → bucket equity at showdown
```

## Board Abstraction

**Flop textures** (configurable, default 200):
1. Enumerate ~1,755 canonical flops (suit-isomorphic)
2. Feature vector per flop: `[flush_type(3), connectivity(4), high_card_rank, pairing(3), avg_ehs_spread]`
3. K-means cluster into N buckets, weighted by combo count
4. Store: `flop_texture_id` + probability weight per texture

**Turn transitions** (default 10 per flop texture):
For each flop texture, 47 possible turn cards clustered by board-change features:
- `[completes_flush, pairs_board, completes_straight, rank_relative_to_board]`
- K-means into ~10 transition types with probability weights

**River transitions** (default 10 per turn state): same approach.

## Hand Abstraction (EHS K-Means)

**Per-texture bucketing** — same hand gets different buckets on different textures:

1. For each `(canonical_hand_169, flop_texture)` pair:
   - Sample ~200 opponent hands, compute average equity = EHS
   - Compute hand potential: P(improve on turn+river)
   - Feature vector: `[EHS, positive_potential, negative_potential]`
2. K-means cluster the ~33,800 vectors (169 × 200) into `num_hand_buckets_flop` (default 2000)
3. Store mapping: `hand_to_flop_bucket[169][200] → u16`

Repeat for turn (re-bucket given turn transition) and river.

**Info set key for postflop**: `(pot_type, street, action_node_idx, hand_bucket)` — all hands in the same bucket share one strategy.

## Postflop Tree Templates

Group preflop showdown terminals into **pot types** by `(pot_size_bucket, who_is_IP)`:

| Pot type | Example pot | IP |
|-|-|-|
| Limped | 4 | BB |
| Open-raised | 7-10 | raiser |
| 3-bet | 20-30 | depends |
| 4-bet+ | 50+ | depends |
| All-in | varies | n/a |

Each pot type (except all-in) gets one postflop tree template:
- 2 bet sizes per street (default: 0.5× pot, 1× pot)
- 1 raise per street (default: 2.5× the bet)
- ~15 action sequences per street × 3 streets = ~45 postflop nodes per template

## Solver Integration

Full enumeration at preflop, Monte Carlo sampling at postflop:

```
for each (hand_p1, hand_p2) in 169² pairs:
    traverse preflop tree (existing LCFR):
        at showdown terminals:
            if all-in: use raw equity (no postflop)
            sample flop_texture ~ P(flop_tex)
            bucket_p1 = hand_to_flop_bucket[hand_p1][flop_tex]
            bucket_p2 = hand_to_flop_bucket[hand_p2][flop_tex]
            traverse flop tree with (bucket_p1, bucket_p2):
                at flop end:
                    sample turn_transition ~ P(turn_tx | flop_tex)
                    re-bucket both hands
                    traverse turn tree:
                        at turn end:
                            sample river_transition
                            re-bucket both hands
                            traverse river tree:
                                at showdown: bucket_equity[b1][b2] * pot
```

Postflop regrets/strategies stored in flat buffers indexed by `(pot_type, street, node, bucket, action)`.

## YAML Configuration

```yaml
preflop:
  stack_depth_bb: 100
  raise_sizes: [[2.5], [3.0]]
  raise_cap: 4
  iterations: 10000

  postflop_model:
    # Board abstraction
    num_flop_textures: 200
    num_turn_transitions: 10
    num_river_transitions: 10

    # Hand abstraction (EHS k-means)
    num_hand_buckets_flop: 2000
    num_hand_buckets_turn: 2000
    num_hand_buckets_river: 2000
    ehs_samples: 200

    # Postflop tree
    bet_sizes: [0.5, 1.0]
    raises_per_street: 1

    # Sampling
    flop_samples_per_iter: 1
```

Presets:
```yaml
postflop_model: "fast"     # 50 textures, 500 buckets (~5 min)
postflop_model: "standard" # 200 textures, 2000 buckets (~30 min)
postflop_model: "accurate" # 500 textures, 5000 buckets (~2 hrs)
```

## Memory & Compute

| Component | Size |
|-|-|
| Preflop regrets/strategy | ~8.6 MB |
| Postflop regrets/strategy | ~17 MB |
| Hand-to-bucket mappings | ~150 KB |
| River bucket equity | 16 MB |
| **Total** | **~42 MB** |

Compute (10K iterations, "standard"):
- Precompute EHS + clustering: ~5-10 min
- Per iteration: ~19M node evals (3.3× current)
- Total: ~16 min on 8 cores

## Implementation Order

### Phase 1 — Abstractions (parallel)
1. **Board abstraction**: flop texture clustering, turn/river transitions
2. **EHS computation + hand bucketing**: EHS calculator, k-means, bucket mappings
3. **Postflop tree builder + config**: tree templates per pot type, YAML schema

### Phase 2 — Integration (sequential, depends on Phase 1)
4. **Solver integration**: extend `cfr_traverse` with postflop sampling, flat buffer storage
5. **Bucket equity precomputation**: river bucket-vs-bucket showdown values

### Phase 3 — Validation
6. **Convergence tests**: AA raises 95%+, suited bluffs raise, regression vs GTO reference

## Key Files (new)

```
crates/core/src/preflop/
  board_abstraction.rs    — flop texture clustering, turn/river transitions
  ehs.rs                  — expected hand strength computation
  hand_buckets.rs         — EHS k-means clustering, bucket mappings
  postflop_tree.rs        — postflop tree templates per pot type
  postflop_model.rs       — PostflopModelConfig, presets, orchestration
  solver.rs               — extended with postflop traversal (modify existing)
  config.rs               — extended with postflop_model field (modify existing)
```

## Sources

- [Simple Preflop Holdem](https://simplepoker.com/en/Solutions/Simple_Preflop_Holdem) — 10K/10K/10K buckets, 2 postflop sizes
- [HRC v3](https://www.holdemresources.net/blog/2023-hrc-v3-release/) — imperfect recall, 64-16K buckets
- [DeepStack (Moravcik et al., 2017)](https://arxiv.org/abs/1701.01724) — learned value functions
- [Depth-Limited Solving (Brown & Sandholm, 2018)](https://arxiv.org/pdf/1805.08195) — depth-limited CFR
- Waugh et al. (2009) "A Practical Use of Imperfect Recall"
- Johanson et al. (2013) "Measuring the Size of Large No-Limit Poker Games"
