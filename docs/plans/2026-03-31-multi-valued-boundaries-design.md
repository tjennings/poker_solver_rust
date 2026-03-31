# Multi-Valued Boundary States for Depth-Limited Subgame Solving

## Summary

Replace the current single-valued, dynamically-refreshed boundary evaluator with fixed multi-valued boundary states (K=4 continuation strategies). Based on Brown & Sandholm NeurIPS 2018 "Depth-Limited Solving for Imperfect-Information Games" — the same approach used by Pluribus.

## Problem

The current boundary evaluator recomputes CFVs every N iterations using the solver's evolving reach probabilities. This creates a destabilizing feedback loop: noisy strategy → bad reach → bad boundary values → corrupted regrets → worse strategy. More refreshes = worse results.

## Solution

At each boundary node, precompute 4 sets of CFVs representing 4 opponent continuation strategies:
- **Continuation 0:** Blueprint strategy (unbiased)
- **Continuation 1:** Fold-biased (multiply fold probability by `bias_factor`, renormalize)
- **Continuation 2:** Call-biased (same)
- **Continuation 3:** Raise-biased (same)

Model the opponent's choice among these as K=4 opponent decision actions at each boundary node. CFR treats them as normal opponent decisions and converges to the mix that most exploits P1. P1's strategy becomes robust to all 4 continuations.

All values are precomputed once at construction time and never change during the solve.

## Boundary Evaluator Changes

Extend the `BoundaryEvaluator` trait:

```rust
trait BoundaryEvaluator {
    fn num_continuations(&self) -> usize { 1 }

    fn compute_cfvs(
        &self,
        player: usize,
        pot: i32,
        remaining_stack: f64,
        opponent_reach: &[f32],
        num_hands: usize,
        boundary_index: usize,
        continuation_index: usize,
    ) -> Vec<f32>;
}
```

The `SolveBoundaryEvaluator` precomputes all `4 × boundaries × 2 players × hands` CFVs at construction time using the `RolloutLeafEvaluator` with 4 bias configurations. Returns from cache on each call.

## Range Solver Internal Changes

When `num_continuations() > 1` at a boundary node:
- Replace the single boundary terminal with an opponent decision node with K children
- Each child is a terminal with precomputed CFVs for that continuation strategy
- Memory: K × num_hands per boundary (e.g., 4 × 750 × 19 = ~57K floats, ~228KB — negligible)

During CFR, the opponent accumulates regrets at continuation-choice nodes and converges to the exploitation-maximizing mix. P1's strategy must be robust to all K continuations.

DCFR discounting applies uniformly to all nodes including continuation choices (already handled — discounting happens every iteration in the range solver).

## Rollout Precomputation

1. Run 4 rollouts per boundary at construction time:
   - Unbiased (blueprint), fold-biased, call-biased, raise-biased
   - Bias factor from Tauri settings (`rollout_bias_factor`)
2. Use blueprint reach probabilities (from seeding), not solver's evolving strategy
3. Store all values as fixed terminals — no `flush_boundary_caches()` during solving
4. Cost: 4× single rollout upfront (~160s for 19 boundaries). Solve itself is pure fixed-game CFR.

## What Gets Removed

- `flush_boundary_caches()` calls during the solve loop
- Dynamic boundary recomputation logic
- The feedback loop between solver strategy and boundary values

## Testing

1. Unit: small game with boundary, verify K=4 produces opponent choice nodes
2. Integration: flop solve with K=4, verify convergence and differentiated strategies
3. Comparison: K=1 vs K=4 on same spot, K=4 should be no worse
4. Manual: compare solved strategy against blueprint — strong hands bet, weak hands check/fold

## Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| K value | 4 (hardcoded) | Matches Pluribus, validated in literature |
| Bias factor | From settings (`rollout_bias_factor`) | Already configurable in Tauri |
| Where to model K strategies | Inside range-solver (approach A) | Built for this problem |
| Leaf value refresh | Never (fixed at construction) | Literature is clear: fixed values = stable convergence |
| Bias targets | fold, call, raise | Standard 3 action types in poker |
