# Subgame Solver Warm-Start from Blueprint Strategy

**Date:** 2026-03-22

## Problem

The `CfvSubgameSolver` initializes `strategy_sum` to all zeros, meaning it starts from a uniform strategy and must rediscover the blueprint's strategy from scratch in ~200 iterations with noisy rollout leaf evaluations. This produces worse strategies than the blueprint itself — defeating the purpose of subgame solving.

## Design

Warm-start `strategy_sum` with the blueprint's action probabilities so the solver refines from the blueprint rather than from uniform.

### Approach

After constructing the solver, if `CbvContext` is available:

1. Walk the subgame tree and abstract tree in parallel from `abstract_node_idx`
2. At each decision node, match subgame actions to abstract tree actions:
   - Fold, check, call: match directly
   - Bet/raise sizes: match by size value. Extra subgame-only raises get zero initial weight.
3. For each matched action, for each combo:
   - `bucket = all_buckets.get_bucket(street, combo, board)`
   - `probs = strategy.get_action_probs(decision_idx, bucket)`
   - `strategy_sum[layout.slot(node, combo) + action] += prob * warmup_weight`
4. `warmup_weight` is a scalar (default ~10) controlling how strongly the blueprint prior influences the average strategy. Equivalent to N "virtual iterations."
5. At chance nodes, both trees advance to their single child.

### Action Matching

The subgame tree may have a superset of the blueprint's actions (same actions plus additional raise sizes). Blueprint actions appear in the same order, with extra raises interleaved or appended. Matching by action type + size:

- `Fold` → `Fold`
- `Check` → `Check`
- `Call` → `Call`
- `Bet(X)` → `Bet(X)` (match by size)
- `Raise(X)` → `Raise(X)` (match by size)
- `AllIn` → `AllIn`
- Extra subgame actions with no blueprint match → zero weight (solver discovers if valuable)

### Key Details

- The parallel walk assumes identical tree structure for shared actions. Both trees use the same game rules (pot, stacks, streets).
- `warmup_weight` should be configurable but not exposed to the UI initially.
- The warm-start only affects `strategy_sum`, not `regret_sum`. Regrets start at zero and accumulate normally.
- This is applied once during construction, before any CFR iterations.
