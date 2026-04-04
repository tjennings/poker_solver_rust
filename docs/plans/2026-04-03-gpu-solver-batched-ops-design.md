# GPU Solver Batched Level Operations — Performance Optimization Design

*2026-04-03 — Rewrite forward/backward passes to use gather/scatter instead of per-node loops*

## Problem

The GPU solver issues ~600 individual burn tensor operations per DCFR iteration (one per node per action). Each is a GPU kernel launch with ~10-50μs overhead. For 500 iterations this produces 300,000 kernel launches → 2.6s, vs 0.3s for the CPU solver on the same river spot.

## Solution

Replace per-node CPU loops with batched gather/scatter operations that process all edges at a level in a single kernel launch. This is the actual GPUGT insight — the original implementation was a working prototype that dispatched per-node.

## Key Decisions

| Decision | Choice |
|----------|--------|
| Batch strategy | All edges at each depth level processed in single gather/scatter ops |
| Regret matching | Scatter-add + gather-divide over ALL edges at once (no per-node grouping) |
| Edge ordering | Sorted by depth so each level's edges are a contiguous range |
| Terminal eval | GPU-only fold eval via pre-computed card index tensors; batched showdown matmul |
| Index tensors | Pre-expanded to [batch, num_level_edges, num_hands] during setup |

## Pre-computed Index Tensors

Built once during `StreetSolver::new()`:

```rust
struct LevelIndices<B: Backend> {
    edge_range: Range<usize>,                    // contiguous range in edge arrays
    parent_gather_idx: Tensor<B, 3, Int>,        // [batch, level_edges, hands]
    child_scatter_idx: Tensor<B, 3, Int>,        // [batch, level_edges, hands]
    is_player0_edge: Tensor<B, 3>,               // [1, level_edges, 1]
    is_player1_edge: Tensor<B, 3>,               // [1, level_edges, 1]
}
```

Also pre-computed globally:
- `actions_per_edge: Tensor<B, 3>` — [1, num_edges, 1] — parent's action count per edge
- `edge_parent_idx: Tensor<B, 3, Int>` — [batch, num_edges, num_hands] — for global regret matching
- `card_to_hand_idx` — for GPU-native fold evaluation

## Batched Regret Matching

No per-node loop. Operates on all edges at once:

```
clipped = regrets.clamp_min(0.0)
denom_nodes = zeros.scatter(1, parent_idx, clipped)     // sum per node
denom_edges = denom_nodes.gather(1, parent_idx)          // broadcast to edges
uniform = 1.0 / actions_per_edge                          // pre-computed
strategy = where(denom_edges > 0, clipped / denom_edges, uniform)
```

## Forward Pass (per level, top-down)

```
parent_reach = reach.gather(1, parent_gather_idx)
strategy = batched_regret_match(level_regrets)
opp_mask = is_opponent_edge(player)
child_reach = parent_reach * (strategy * opp_mask + (1 - opp_mask))
reach = reach.scatter(1, child_scatter_idx, child_reach)
```

~3 kernel launches per level.

## Backward Pass (per level, bottom-up)

```
evaluate_terminals_batched(level)

child_cfv = cfv.gather(1, child_scatter_idx)
strategy = batched_regret_match(level_regrets)
trav_mask = is_traverser_edge(player)
weighted_cfv = child_cfv * (strategy * trav_mask + (1 - trav_mask))
cfv = cfv.scatter(1, parent_gather_idx, weighted_cfv)

parent_cfv = cfv.gather(1, parent_gather_idx)
instant_regret = (child_cfv - parent_cfv) * trav_mask
regrets[edge_range] += instant_regret
strategy_sum[edge_range] += strategy * trav_mask
```

~8 kernel launches per level.

## Terminal Evaluation (GPU-only)

**Fold:** Pre-compute `card_to_hand_scatter: [52, max_hands_per_card]` on GPU. Fold eval:
1. `total_reach = opp_reach.sum(hand_dim)`
2. `card_reach = scatter_add(opp_reach, card_indices)` → [batch, 52]
3. `blocking = gather(card_reach, hand_card1) + gather(card_reach, hand_card2) - same_hand_correction`
4. `cfv = payoff * (total_reach - blocking)`

All GPU, no CPU↔GPU transfer.

**Showdown:** Already GPU matmul. Batch all showdown nodes at a level.

## Performance Target

| Metric | Current | Target |
|--------|---------|--------|
| Kernel launches/iter | ~600 | ~50 |
| River solve (19 hands, 500 iter) | 2.6s | ≤0.3s |
| CPU↔GPU transfers/iter | ~20 (fold eval) | 0 |

## Scope

Rewrite `solver.rs` forward/backward/regret_match + `terminal.rs` fold eval + `tensors.rs` index setup. The `extract.rs` topology extraction and `lib.rs` public API stay the same. All 79 existing tests must continue passing.
