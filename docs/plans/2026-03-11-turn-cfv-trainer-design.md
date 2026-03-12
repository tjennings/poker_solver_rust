# Turn CFV Trainer Design

## Goal

Build a turn CFV network trainer following the Supremus/DeepStack approach: solve random turn subgames using the trained river CFV network as a leaf evaluator at river transitions, then train a turn neural network on the resulting counterfactual values.

## Background (Supremus Paper, Section 4.3)

The CFV network cascade trains from the bottom up:
1. **River network** — trained on 50M subgames solved to showdown (already built in `crates/cfvnet/`)
2. **Turn network** — trained on 20M subgames, using the river CFV net at leaf nodes ← **this design**
3. **Flop network** — trained on 5M subgames, using the turn CFV net at leaf nodes (future work)

Each subgame is solved with 4,000 iterations of DCFR+. The neural net is evaluated **dynamically each CFR iteration** — as strategies evolve, the ranges at leaf nodes change, and the net is re-queried with updated ranges.

## Architecture

```
cfvnet turn datagen pipeline
  │
  ├── Sampler: sample 4-card board, pot, stacks, R(S,p) ranges
  │     (reuse sampler.rs with board_size parameter)
  │
  ├── Solver: CfvSubgameSolver (NEW, in core)
  │     ├── GameTree::build_subgame(Turn, depth_limit=1) → DepthBoundary at river
  │     ├── Per iteration:
  │     │     1. Regret-match → current strategy
  │     │     2. Propagate ranges through tree → (oop_range, ip_range) at each leaf
  │     │     3. LeafEvaluator::evaluate(leaf_info) → cfvs[node][combo]
  │     │     4. Normal CFR traversal with per-node leaf values
  │     └── Output: root CFVs per combo (training targets)
  │
  ├── Storage: reuse storage.rs (generalize board field to variable length)
  │
  ├── Encoding: INPUT_SIZE adjusts for 4 board cards
  │
  └── Training: reuse model/training.rs, network.rs, loss.rs as-is
```

## New Components

### 1. `LeafEvaluator` Trait (in `crates/core/src/blueprint_v2/`)

```rust
/// Evaluates counterfactual values at depth boundary nodes.
///
/// Called once per CFR iteration per boundary node with the current
/// reach-weighted ranges at that node. Returns pot-relative CFVs
/// for all combos from the traverser's perspective.
pub trait LeafEvaluator {
    fn evaluate(
        &self,
        board: &[Card],           // current board (e.g., 4 turn cards)
        pot: f64,                 // pot at this boundary node
        effective_stack: f64,     // remaining stack at this boundary
        oop_range: &[f64],        // OOP reach probabilities at this node
        ip_range: &[f64],         // IP reach probabilities at this node
        traverser: u8,            // which player's CFVs to return (0=OOP, 1=IP)
    ) -> Vec<f64>;               // pot-relative CFVs indexed by combo
}
```

### 2. `CfvSubgameSolver` (in `crates/core/src/blueprint_v2/cfv_subgame_solver.rs`)

Parallel DCFR solver with dynamic leaf evaluation. Same flat-buffer infrastructure as `SubgameCfrSolver` but with a fundamentally different iteration loop.

**Key differences from `SubgameCfrSolver`:**
- Takes `Box<dyn LeafEvaluator>` instead of `Vec<f64>` leaf values
- Stores `leaf_cfvs: Vec<Vec<f64>>` — indexed by `[boundary_node_idx][combo_idx]`, recomputed each iteration
- New `propagate_ranges()` method walks the tree with current strategy to compute range vectors at each `DepthBoundary` node
- Leaf CFVs recomputed each iteration before traversal

**Iteration loop:**
```
for iteration in 0..N:
    strategy_snapshot = regret_match(regret_sums)

    // Phase 1: compute ranges at all leaf nodes
    for traverser in 0..2:
        ranges_at_leaves = propagate_ranges(strategy_snapshot, traverser)
        for each boundary_node:
            leaf_cfvs[traverser][boundary_node] = evaluator.evaluate(
                board, pot, stack, oop_range, ip_range, traverser
            )

        // Phase 2: normal CFR traversal using per-node leaf values
        parallel_traverse(strategy_snapshot, leaf_cfvs[traverser])

    // Phase 3: DCFR discounting
    dcfr_discount(regret_sums, strategy_sums, iteration)
```

**Range propagation:** Walk the tree from root, tracking reach probability vectors for both players. At Decision nodes, multiply each combo's reach by its strategy probability for the taken action. At Chance nodes, pass through. At DepthBoundary terminals, record the accumulated reach vectors as the "ranges" at that node.

**Pot-relative leaf values and scaling:** The `LeafEvaluator` returns pot-relative CFVs. The solver stores these directly. During CFR traversal, `DepthBoundary` nodes use `leaf_cfvs[node][combo] * half_pot` as the terminal value — different boundary nodes have different pots, so this correctly handles check-check vs bet-call paths.

### 3. `RiverNetEvaluator` (in `crates/cfvnet/`)

Implements `LeafEvaluator` by averaging the river CFV network's output over all possible river cards:

```rust
impl LeafEvaluator for RiverNetEvaluator {
    fn evaluate(&self, board, pot, stack, oop_range, ip_range, traverser) -> Vec<f64> {
        let remaining = remaining_deck(board);  // 46 cards for turn board
        let mut cfvs = vec![0.0; 1326];
        let mut counts = vec![0usize; 1326];

        for river_card in remaining {
            let full_board = append(board, river_card);  // 5-card board
            let net_out = self.model.forward(
                full_board, pot, stack, oop_range, ip_range, traverser
            );
            for i in 0..1326 {
                if !blocked_by_board(i, &full_board) {
                    cfvs[i] += net_out[i];
                    counts[i] += 1;
                }
            }
        }

        for i in 0..1326 {
            if counts[i] > 0 { cfvs[i] /= counts[i] as f64; }
        }
        cfvs
    }
}
```

**Batching optimization:** The 46 river card evaluations can be batched into a single forward pass for efficiency.

### 4. `ExactRiverEvaluator` (in `crates/core/`, for testing)

Implements `LeafEvaluator` by solving each river card exactly using `SubgameCfrSolver`:

```rust
impl LeafEvaluator for ExactRiverEvaluator {
    fn evaluate(&self, board, pot, stack, oop_range, ip_range, traverser) -> Vec<f64> {
        let remaining = remaining_deck(board);
        let mut cfvs = vec![0.0; num_combos];
        let mut counts = vec![0usize; num_combos];

        for river_card in remaining {
            let full_board = append(board, river_card);
            let hands = SubgameHands::enumerate(&full_board);
            let tree = GameTree::build_subgame(River, pot, ..., None);
            let mut solver = SubgameCfrSolver::new(tree, hands, &full_board, opp_reach, leaf);
            solver.train(1000);
            // Extract per-combo CFVs from the solved strategy
            // ... average into cfvs
        }
        cfvs
    }
}
```

This is slow (46 river solves per call) but provides ground-truth for validation.

## Reused Components (from river trainer)

| Component | Changes needed |
|-|-|
| `sampler.rs` | Parameterize `sample_board()` to take board size (4 for turn, 5 for river) |
| `range_gen.rs` | None — works with any board size (hand strength eval adapts to card count) |
| `storage.rs` | Generalize board field to variable length (4 or 5 cards) |
| `generate.rs` | Replace `range-solver` call with `CfvSubgameSolver` + `RiverNetEvaluator` |
| `dataset.rs` | Adjust `INPUT_SIZE` for 4 board cards (2659 vs 2660) |
| `network.rs` | Parameterize `INPUT_SIZE` (architecture identical) |
| `training.rs` | None — training loop is street-agnostic |
| `loss.rs` | None |
| `config.rs` | Add `river_model_path`, `street` parameter, turn bet sizes |

## Validation

### #1 — Train/Val Huber Loss (built into training)
Same as river: validation split during training, report Huber loss each epoch. Target: train ≤ 0.008, val ≤ 0.010 (matching Supremus Table 1).

### #2 — `compare-exact` CLI Subcommand
Compare a trained model against exact solutions on random spots. Run on demand after training to measure real-world accuracy.

```
cfvnet compare-exact --model ./turn_model/ --river-model ./river_model/ \
    --num_spots 100 --threads 8 --config turn.yaml
```

For each spot:
1. Sample a random turn situation (board, pot, stacks, ranges)
2. Solve the turn using `CfvSubgameSolver` + `ExactRiverEvaluator` (exact river solves via `SubgameCfrSolver`, no neural net)
3. Query the turn model for predicted CFVs
4. Compute per-spot and aggregate metrics: MAE, RMSE, max error, mBB

This validates end-to-end accuracy including cascaded neural net error. Slow (~minutes per spot due to 46 exact river solves per leaf per iteration) but provides ground truth.

### #3 — `compare-net` CLI Subcommand
Compare a trained model against neural-net-backed solves on random spots. Faster than #2, measures the model's fit to its own training distribution.

```
cfvnet compare-net --model ./turn_model/ --river-model ./river_model/ \
    --num_spots 500 --threads 8 --config turn.yaml
```

For each spot:
1. Sample a random turn situation
2. Solve the turn using `CfvSubgameSolver` + `RiverNetEvaluator` (same pipeline as datagen)
3. Query the turn model for predicted CFVs
4. Compute per-spot and aggregate metrics: MAE, RMSE, max error, mBB

The delta between #2 and #3 metrics isolates the river net's error contribution at leaf nodes vs the turn model's own approximation error.

## Performance Considerations

- **Per iteration cost:** ~10-20 boundary nodes × 46 river cards = ~500-900 net forward passes (batchable)
- **Per sample cost:** 4,000 iterations × ~500 net calls = ~2M forward passes
- **Datagen throughput:** With batching and parallelism, targeting ~1-5 samples/second
- **Training data:** 20M samples (per Supremus), generating ~4M samples/day at 1 sample/sec
- **Range propagation:** O(num_nodes × num_combos) per iteration — lightweight compared to net inference

## Future: Flop Trainer

The same architecture extends to flop:
- `TurnNetEvaluator` implements `LeafEvaluator` using the trained turn model
- Sampler generates 3-card boards
- `CfvSubgameSolver` with `GameTree::build_subgame(Flop, depth_limit=1)`
- 47 possible turn cards per leaf (vs 46 river cards)
- Everything else identical
