# Multi-Valued Leaf Evaluation for Depth-Limited Subgame Solving

**Goal:** Replace single-valued CBV leaf evaluation with multi-valued continuation strategy rollouts at depth boundaries. The opponent chooses which continuation strategy to follow, forcing the solver to produce strategies robust to all opponent responses. This is the approach Pluribus used with only 200 blueprint buckets.

**Problem:** A single CBV per bucket averages away strategic variation at boundaries. Flush draws, made hands, and air in the same bucket all get one value. The solver can't distinguish them, leading to degenerate strategies (shoving everything).

## Architecture

### Virtual Choice Node

Add a virtual opponent decision node at the **root** of the subgame tree, before any real actions. The opponent chooses one of K=4 continuation strategies that will be used at ALL depth boundaries in the subtree. This is a single choice for the entire hand — the opponent commits to one continuation philosophy.

The 4 continuation strategies are derived from the blueprint:
1. **Unbiased** — blueprint strategy as-is
2. **Fold-biased** — multiply fold/check probabilities by 10, renormalize
3. **Call-biased** — multiply call probabilities by 10, renormalize
4. **Raise-biased** — multiply bet/raise probabilities by 10, renormalize

The bias factor (10x) is configurable.

### Rollout-Based Leaf Evaluation

At each depth boundary during CFR iteration, for each combo, compute the expected value by rolling out the remaining streets using the selected continuation strategy:

1. **For each possible next-street card** (or a sample of N cards):
   - Extend the board with the new card
   - Look up the bucket for each player's hand on the new board
   - Walk the abstract tree's subtree from this point, using the continuation strategy's action probabilities at each decision node
   - At showdown/fold terminals, compute the payoff
   - If another street remains (turn→river), repeat: deal river card, look up river bucket, continue
2. **Average** the payoffs across all dealt cards

This produces one value per combo per continuation strategy. The virtual choice node then selects among the 4 values.

### Data Requirements

The rollout evaluator needs at leaf evaluation time:
- **Blueprint strategy** (`BlueprintV2Strategy`) — action probabilities per (decision_node, bucket)
- **Abstract game tree** (`GameTree`) — tree structure from the boundary onward
- **Bucket files** (`AllBuckets`) — to map concrete (board, combo) → bucket on each new street
- **Abstract node index** — where in the abstract tree the boundary corresponds to

All of these are already available in `CbvContext` (add `BlueprintV2Strategy`).

### Solver Integration

The `CfvSubgameSolver` traversal changes:

```
Original tree:           With virtual choice node:

  Root (OOP decision)      Choice (IP picks k∈{0,1,2,3})
   ├─ Check                 ├─ k=0: Root (OOP) with unbiased leaves
   ├─ Bet                   ├─ k=1: Root (OOP) with fold-biased leaves
   └─ AllIn                 ├─ k=2: Root (OOP) with call-biased leaves
                            └─ k=3: Root (OOP) with raise-biased leaves
```

Each iteration:
1. Build strategy snapshot
2. Propagate ranges to boundaries (once, shared across all k)
3. For each traverser (OOP, IP):
   a. For each k in 0..K:
      - Compute leaf values at all boundaries using continuation strategy k
      - Run CFR traversal of the full subgame tree
      - Accumulate regrets for the real tree nodes AND the virtual choice node
   b. Weight the choice node's regret by the traverser's reaching probability
4. Apply DCFR discounting

The choice node has its own regret and strategy sums, separate from the real tree.

### Rollout Implementation

```rust
fn rollout_from_boundary(
    combo: [Card; 2],
    board: &[Card],
    abstract_tree: &GameTree,
    abstract_node: u32,       // where in the abstract tree this boundary is
    strategy: &BlueprintV2Strategy,
    buckets: &AllBuckets,
    continuation_k: u8,       // which biased strategy to use
    bias_factor: f64,         // 10.0 default
    player: u8,               // which player's value to compute
    num_rollout_cards: Option<usize>,  // None = exhaustive, Some(N) = sample N
) -> f64
```

For **exhaustive** rollout (all possible next cards): iterate over all remaining deck cards, weight equally. This is exact but O(48) per boundary per combo for flop→turn, O(48×47) for flop→turn→river.

For **sampled** rollout: deal N random next cards, average. Faster but noisier.

Recommended: exhaustive for turn→river (47 cards), sampled for flop→turn→river (too many combinations). Configurable via `num_rollout_cards`.

### Biasing the Strategy

Given blueprint action probabilities `[p_fold, p_call, p_raise1, p_raise2, ...]`:

```rust
fn bias_strategy(probs: &[f32], bias_type: BiasType, factor: f64) -> Vec<f32> {
    let mut biased = probs.to_vec();
    for (i, p) in biased.iter_mut().enumerate() {
        let action_type = classify_action(i); // fold, call, or raise
        if action_type == bias_type {
            *p *= factor as f32;
        }
    }
    // Renormalize
    let sum: f32 = biased.iter().sum();
    if sum > 0.0 {
        for p in &mut biased { *p /= sum; }
    }
    biased
}
```

### Analytics

Diagnostics emitted during solving:

1. **Per-combo 4-value breakdown** at boundaries (logged for sample combos):
   ```
   [leaf audit] 3s4s boundary=0: unbiased=2.1 fold=4.8 call=-1.2 raise=-3.1
   ```

2. **Choice node strategy** after convergence:
   ```
   [choice audit] opponent continuation mix: unbiased=35% fold=15% call=30% raise=20%
   ```

3. **Rollout timing**:
   ```
   [rollout] 1176 combos × 41 boundaries × 4 strategies: 2.3s (parallelized)
   ```

### Performance

Per leaf evaluation (called every `leaf_eval_interval` iterations):
- 1176 combos × 41 boundaries × 4 strategies × 47 turn cards × ~8 tree nodes = ~72M operations
- With rayon parallelism across combos: ~1-3 seconds
- With `leaf_eval_interval=10` on 2000 iterations: ~200 evaluations × 2s = ~400s total

Per CFR iteration (excluding leaf evaluation):
- 4x traversals (one per k) instead of 1
- Each traversal is the same cost as current (~0.5s with parallelized showdowns)
- Total per iteration: ~2s

### Configuration

```yaml
subgame:
  continuation_strategies: 4          # K value (1 = single-valued, 4 = Pluribus)
  bias_factor: 10.0                   # how much to bias fold/call/raise
  rollout_samples: null               # null = exhaustive, N = sample N cards
  leaf_eval_interval: 10              # re-evaluate leaves every N iterations
```

### What Changes

| Component | Change |
|-----------|--------|
| `CbvContext` | Add `BlueprintV2Strategy` |
| `CfvSubgameSolver` | Add virtual choice node with K branches |
| `LeafEvaluator` trait | New `RolloutLeafEvaluator` implementation |
| `postflop.rs` | Wire rollout evaluator when blueprint is loaded |
| `train_with_leaf_interval` | Already supports interval; used as-is |

### What Doesn't Change

- The real game tree structure (same actions, same terminals)
- The `SubgameStrategy` output format
- The showdown equity computation
- The range propagation logic
- The regret matching at real decision nodes

## Testing

- Unit: rollout with known tree produces correct expected value
- Unit: biased strategy renormalizes correctly
- Unit: virtual choice node accumulates regrets independently
- Integration: solver with 4 continuation strategies converges to mixed strategy (not pure shove)
- Diagnostic: 4-value breakdown shows flush draws valued higher under fold-biased than call-biased
