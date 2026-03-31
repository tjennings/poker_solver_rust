# Per-Flop Clustering and Blueprint Training Pipeline

## Summary

Extend the training pipeline to support per-flop mode: clustering turn/river buckets specific to each canonical flop, then training per-flop blueprints with a locked preflop strategy. This eliminates global bucket abstraction error at postflop streets, enabling suit-aware bucketing that distinguishes flush draws from non-flush draws on specific board textures.

## Problem

Global buckets cluster hands across all 1,755 canonical flops. On a specific board like QhTh7d, combos like 6h2h (flush draw) and 6d2d (nothing) may share a bucket because the global clustering optimizes for average-case board textures, not this specific one. The subgame solver inherits this limitation through bucketed boundary values.

## Solution

### Config

Add `mode` and `flops` to the existing `clustering` config section:

```yaml
clustering:
  algorithm: potential_aware_emd
  mode: per_flop           # "global" (default) or "per_flop"
  flops: all               # "all" (1,755 canonical), or ["QhTh7d", "AsKs2d"]
  output_dir: ./per_flop_buckets
  preflop:
    buckets: 169
  turn:
    buckets: 500
  river:
    buckets: 500
```

- `mode: global` — current behavior, unchanged
- `mode: per_flop` — per-flop clustering and training
- `flops: all` — all 1,755 canonical flops
- `flops: [list]` — specific flops for testing
- `flop.buckets` is ignored in per-flop mode (fixed flop = no flop bucketing needed)
- `output_dir` — directory for per-flop bucket files

### Phase 1: Per-Flop Clustering

For each flop in `flops`:
1. Enumerate all 46 possible turn cards for this flop
2. Run river clustering (equity-based, per-flop): cluster combos on each (flop+turn+river) board
3. Run turn clustering (potential-aware EMD): histogram over river buckets for each (flop+turn) board, cluster with k-means
4. Save to `{output_dir}/{canonical_flop_key}/turn.buckets` and `river.buckets`

Output structure:
```
per_flop_buckets/
  QhTh7d/
    turn.buckets
    turn.centroids
    river.buckets
    river.centroids
  AsKs2d/
    turn.buckets
    ...
```

Parallelizable: each flop is independent. Can distribute across cores/machines.

### Phase 2: Per-Flop Blueprint Training

For each flop in `flops`:
1. Load the global blueprint's preflop strategy (from `training.preflop_blueprint` config path)
2. Load this flop's per-flop turn/river bucket files
3. Run MCCFR with:
   - **Preflop: locked** — use the global blueprint's average strategy, do NOT update regrets or strategy sums at preflop decision nodes
   - **Flop: fixed** — every deal uses this specific flop (random hole cards, random turn/river cards)
   - **Flop buckets: none** — each combo is distinct (~1000 valid combos per flop)
   - **Turn/river: per-flop buckets** — from Phase 1
   - Standard DCFR discounting, pruning, etc.
4. Save per-flop blueprint to `{output_dir}/{canonical_flop_key}/strategy.bin`

The locked preflop ensures ranges entering the flop are realistic (derived from the global blueprint's self-play, not from "knowing" which flop is coming).

### Phase 3: Tauri Integration

When the subgame solver opens a flop spot:
1. Check for per-flop data at `{per_flop_dir}/{canonical_flop_key}/`
2. If found:
   - Seed the range solver with the per-flop blueprint strategy (instead of global)
   - Use per-flop turn/river buckets for boundary rollout strategies
3. If not found: fall back to global blueprint and global buckets (current behavior)

### Training Config for Per-Flop Mode

```yaml
clustering:
  algorithm: potential_aware_emd
  mode: per_flop
  flops: all
  output_dir: ./local_data/per_flop
  preflop:
    buckets: 169
  turn:
    buckets: 500
  river:
    buckets: 500

training:
  preflop_blueprint: ./local_data/blueprints/global/latest
  lock_preflop: true
  time_limit_minutes: 30        # per flop
  lcfr_warmup_iterations: 10000000
  prune_after_iterations: 5000000
  optimizer: dcfr+
  dcfr_alpha: 1.5
  dcfr_gamma: 2.0
```

### Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Config location | Extends existing clustering config | Run from single training config per user request |
| Flop bucketing in per-flop mode | None (each combo distinct) | Fixed flop = ~1000 combos, no abstraction needed |
| Preflop handling | Locked from global blueprint | Prevents "knows the flop" range distortion |
| Flop selection | `all` or explicit list | `all` for production, list for testing |
| Output structure | One directory per canonical flop | Simple, filesystem-based, easy to inspect |
| Parallelism | Each flop independent | Embarrassingly parallel across cores/machines |
| Fallback | Global blueprint if per-flop unavailable | Graceful degradation |

### What Doesn't Change

- Global blueprint training pipeline (`mode: global`)
- The MCCFR trainer core (same traversal, same regret updates)
- The range solver (no changes needed)
- The clustering algorithm (potential-aware EMD, same k-means)
