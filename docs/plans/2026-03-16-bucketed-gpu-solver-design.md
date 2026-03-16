# Bucketed GPU Solver — Design

## Overview

Redesign the GPU solver to operate on **bucketed hands** (500 or 1000 buckets) instead of 1326 concrete combos. This matches the Supremus architecture exactly and eliminates the CFV unit mismatch problem.

## Key Changes from Concrete Solver

| Dimension | Concrete (current) | Bucketed (new) |
|-----------|-------------------|----------------|
| Hands per node | 1326 | num_buckets (500/1000) |
| Regret arrays | [infosets × actions × 1326] | [infosets × actions × num_buckets] |
| Reach arrays | [nodes × 1326] | [nodes × num_buckets] |
| CFV arrays | [nodes × 1326] | [nodes × num_buckets] |
| Terminal showdown | hand strength comparison | equity_table[bucket_i × bucket_j] lookup |
| Card blocking | explicit per-hand | baked into equity table |
| CFV net input | 2 × 1326 + 52 + 13 + 3 = 2720 | 2 × num_buckets + 1 (pot) |
| CFV net output | 1326 | num_buckets |
| num_combinations | varies per board, causes unit issues | 1.0 (implicit, equity table normalized) |

## Architecture

### Bucket-Based Terminal Evaluation

**Showdown**: Instead of comparing hand strengths, use a precomputed **equity matrix** `E[num_buckets × num_buckets]` where `E[i][j]` = expected payoff when bucket i faces bucket j at showdown. Precomputed from the clustering pipeline by averaging hand-vs-hand equity within each bucket pair.

```
cfv_showdown[bucket_i] = sum_j(E[i][j] × opp_reach[j]) × half_pot
```

This is a matrix-vector multiply: `cfv = E × opp_reach × half_pot`. No card blocking needed — blocking is averaged into E during precomputation.

**Fold**: Even simpler — no card blocking:
```
cfv_fold[bucket_i] = ±half_pot × sum(opp_reach[j] for all j)
```

### Bucket-Based CFV Net

**Input**: `[2 × num_buckets + 1]` = OOP bucket reach + IP bucket reach + pot/stack
**Output**: `[num_buckets]` = counterfactual values per bucket

Much smaller than the 2720-dim concrete input. Network can be smaller too (Supremus uses 7×500 with 1000 buckets).

### Mapping Combos to Buckets

For a given board, the BucketFile maps each of 1326 combos to a bucket:
```
bucket = bucket_file.get_bucket(board_idx, combo_idx)
```

This mapping is needed to:
1. Convert initial 1326-dim range weights to num_buckets-dim reach: `reach[bucket] += range[combo]` for each combo mapping to that bucket.
2. Convert num_buckets-dim strategy back to 1326-dim for display.

### Equity Table Precomputation

For each showdown terminal in the tree, precompute the bucket-vs-bucket equity table:

```
For each canonical board:
  For each bucket pair (i, j):
    avg_equity = average over all (hand_a in bucket_i, hand_b in bucket_j) of:
      +half_pot if hand_a beats hand_b
      -half_pot if hand_a loses to hand_b
      0 if tie
    E[i][j] = avg_equity
```

This is done once per tree topology and uploaded to GPU. Size: `500 × 500 × 4 bytes = 1MB` (trivial).

### DCFR+ in Bucket Space

The entire DCFR+ loop operates on `num_buckets`-dimensional arrays:
- regret_match: same kernel, just `num_hands = num_buckets`
- forward_pass: same
- terminal_fold_eval: simplified (no card blocking)
- terminal_showdown_eval: matrix-vector multiply instead of O(n²) comparison
- backward_cfv: same
- update_regrets: same

### Leaf Evaluation

At depth boundaries, the solver has bucket-space reach for both players. The CFV net takes these directly — no encoding of board one-hot or rank presence needed (the network sees bucket distributions, not card identities). The board context is implicit in the bucket assignments.

CFV net input: `[OOP_reach(500) | IP_reach(500) | pot_normalized(1)] = 1001 features`
CFV net output: `[500 CFVs]`

This is 2.7x smaller than the 2720-dim concrete input. Inference is proportionally faster.

### No num_combinations Issue

In bucket space, the equity table is pre-normalized. The CFV values are directly in "expected payoff" units. No per-board num_combinations scaling needed. The unit mismatch that plagued the concrete solver disappears.

## Configuration

```yaml
buckets:
  path: "local_data/clusters_500bkt_v3"
  num_buckets: 500

# Or for 1000 buckets:
buckets:
  path: "local_data/clusters_1000bkt_v1"
  num_buckets: 1000
```

The `num_buckets` parameter flows through the entire pipeline: solver kernels, CFV net architecture, training data dimensions.

## Implementation Strategy

Reuse the existing GPU solver infrastructure. Most kernels are already parameterized by `num_hands` — just pass `num_buckets` instead. The main new work:

1. **Equity table precomputation** — compute bucket-vs-bucket equity from BucketFile + hand evaluator
2. **Showdown kernel** — replace hand strength comparison with equity matrix multiply
3. **Range-to-bucket mapping** — convert 1326 range weights to num_buckets reach
4. **CFV net architecture** — smaller input/output dimensions
5. **Leaf eval encoding** — bucket reach + pot instead of 2720-dim features
6. **Training pipeline** — generate bucket-space training data

## What We Keep

- All DCFR+ kernels (regret_match, forward_pass, backward_cfv, update_regrets, extract_strategy)
- Flat tree structure (nodes, infosets, CSR children)
- Batch solver architecture
- CudaNetInference (just different dimensions)
- Training pipeline structure (sample → solve → reservoir → train)
- Progressive resolving in Explorer
