# Vectorize CfvSubgameSolver Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use hex:executing-plans to implement this plan task-by-task.

**Goal:** Replace per-combo tree traversal in CfvSubgameSolver with a single-pass vectorized traversal that processes all 1326 combos simultaneously at each node, matching the range solver's approach.

**Architecture:** Replace `parallel_traverse` + per-combo `cfr_traverse` with a new `cfr_traverse_vectorized` that walks the tree once per traverser, carrying reach vectors of length `num_combos` and computing CFVs for all combos at each node. Regret and strategy updates happen inline at each decision node. No separate `build_strategy_snapshot` pass — regret matching done inline at each node.

**Tech Stack:** Rust, existing `GameTree`/`GameNode` types, `CfvLayout` for flat buffer access

---

### Task 1: Add `cfr_traverse_vectorized` — the core vectorized traversal

**Files:**
- Modify: `crates/core/src/blueprint_v2/cfv_subgame_solver.rs`

This replaces the per-combo `cfr_traverse` + `traverse_as_traverser` + `traverse_as_opponent` with a single function that processes all combos at each node.

**Step 1: Add the vectorized traversal method**

Add this as a method on `CfvSubgameSolver` (not on `CfvCfrCtx` — we need `&mut self` for regret/strategy updates). Place it after `propagate_ranges_recursive`, before the `train` method.

The function signature:

```rust
/// Vectorized CFR traversal: walks the tree once, processing all combos
/// simultaneously at each node. Returns CFVs for all combos.
///
/// At traverser's decision nodes: computes regret-matched strategy inline,
/// recurses into children, computes node CFVs, updates regrets and strategy sums.
/// At opponent's decision nodes: weights child values by opponent strategy.
/// At terminals: computes fold/showdown/boundary values for all combos.
fn cfr_traverse_vectorized(
    &mut self,
    node_idx: usize,
    traverser: u8,
    reach_traverser: &[f64],   // length = num_combos
    reach_opponent: &[f64],    // length = num_combos
    snapshot: &[f64],          // strategy snapshot (read-only)
) -> Vec<f64>                  // returns CFVs for all combos
```

Logic per node type:

**Terminal nodes:**
```rust
GameNode::Terminal { kind, pot, .. } => {
    let half_pot = *pot / 2.0;
    let n = self.hands.combos.len();
    let mut cfvs = vec![0.0; n];
    match kind {
        TerminalKind::Fold { winner } => {
            let sign = if *winner == traverser { 1.0 } else { -1.0 };
            for i in 0..n {
                cfvs[i] = sign * half_pot;
            }
        }
        TerminalKind::Showdown => {
            // Compute showdown value for all combos at once
            for hero in 0..n {
                cfvs[hero] = showdown_value_single(
                    hero, &self.hands, &self.equity_matrix,
                    &self.opp_reach_totals, half_pot,
                );
            }
        }
        TerminalKind::DepthBoundary => {
            let ordinal = self.node_to_boundary[node_idx];
            if ordinal != usize::MAX {
                for i in 0..n {
                    cfvs[i] = self.leaf_cfvs[ordinal]
                        .get(i).copied().unwrap_or(0.0) * half_pot;
                }
            }
        }
    }
    cfvs
}
```

**Chance nodes:**
```rust
GameNode::Chance { child, .. } => {
    self.cfr_traverse_vectorized(
        *child as usize, traverser,
        reach_traverser, reach_opponent, snapshot,
    )
}
```

**Decision nodes (traverser's turn):**
```rust
// 1. Regret-match to get strategy for ALL combos at this node
//    (inline, no separate snapshot needed for this node)
let num_actions = actions.len();
let n = self.hands.combos.len();

// Read strategy from snapshot for all combos
// snapshot layout: bases[node_idx] + combo_idx * num_actions + action
let node_base = self.layout.bases[node_idx];

// 2. Recurse into each child with updated reach
let mut action_cfvs: Vec<Vec<f64>> = Vec::with_capacity(num_actions);
for (a, &child_idx) in children.iter().enumerate() {
    // Build child reach: reach_traverser[i] * strategy[i][a]
    let mut child_reach = vec![0.0; n];
    for i in 0..n {
        child_reach[i] = reach_traverser[i]
            * snapshot[node_base + i * num_actions + a];
    }
    let child_cfv = self.cfr_traverse_vectorized(
        child_idx as usize, traverser,
        &child_reach, reach_opponent, snapshot,
    );
    action_cfvs.push(child_cfv);
}

// 3. Compute node value: weighted sum of action CFVs
let mut node_cfvs = vec![0.0; n];
for i in 0..n {
    for a in 0..num_actions {
        let strat = snapshot[node_base + i * num_actions + a];
        node_cfvs[i] += strat * action_cfvs[a][i];
    }
}

// 4. Update regrets and strategy sums
for i in 0..n {
    let base = node_base + i * num_actions;
    for a in 0..num_actions {
        self.regret_sum[base + a] +=
            reach_opponent[i] * (action_cfvs[a][i] - node_cfvs[i]);
        let strat = snapshot[node_base + i * num_actions + a];
        self.strategy_sum[base + a] += reach_traverser[i] * strat;
    }
}

node_cfvs
```

**Decision nodes (opponent's turn):**
```rust
// Same recurse pattern but update opponent reach, not traverser reach
let mut action_cfvs: Vec<Vec<f64>> = Vec::with_capacity(num_actions);
for (a, &child_idx) in children.iter().enumerate() {
    let mut child_opp_reach = vec![0.0; n];
    for i in 0..n {
        child_opp_reach[i] = reach_opponent[i]
            * snapshot[node_base + i * num_actions + a];
    }
    let child_cfv = self.cfr_traverse_vectorized(
        child_idx as usize, traverser,
        reach_traverser, &child_opp_reach, snapshot,
    );
    action_cfvs.push(child_cfv);
}

// Node value is strategy-weighted sum (using opponent's strategy)
let mut node_cfvs = vec![0.0; n];
for i in 0..n {
    for a in 0..num_actions {
        let strat = snapshot[node_base + i * num_actions + a];
        node_cfvs[i] += strat * action_cfvs[a][i];
    }
}
// No regret/strategy update at opponent nodes
node_cfvs
```

**Important:** Since `cfr_traverse_vectorized` takes `&mut self` (for regret/strategy updates), but also needs to read from `self.tree.nodes` (immutable), you'll need to clone the node data or extract what you need before the mutable operations. The cleanest approach: at the top of the function, extract node info (player, children indices, kind) into local variables before any `&mut self` calls.

```rust
// Extract node info to avoid borrow conflicts
let node = &self.tree.nodes[node_idx];
let (node_kind, node_player, node_children, node_actions_len, node_pot) = match node {
    GameNode::Terminal { kind, pot, .. } => (0u8, 0u8, vec![], 0, *pot),
    GameNode::Chance { child, .. } => (1u8, 0u8, vec![*child], 0, 0.0),
    GameNode::Decision { player, children, actions, .. } => {
        (2u8, *player, children.clone(), actions.len(), 0.0)
    }
};
```

**Step 2: Extract `showdown_value_single` as a free function**

Move the showdown computation out of `CfvCfrCtx` so the vectorized traversal can use it. It takes the same inputs but as explicit parameters:

```rust
fn showdown_value_single(
    hero_combo: usize,
    hands: &SubgameHands,
    equity_matrix: &[Vec<f64>],
    opp_reach_totals: &[f64],
    half_pot: f64,
) -> f64 {
    let hero_cards = hands.combos[hero_combo];
    let opp_reach_total = opp_reach_totals[hero_combo];
    if opp_reach_total <= 0.0 {
        return 0.0;
    }
    let mut equity_sum = 0.0;
    for (j, eq_row) in equity_matrix[hero_combo].iter().enumerate() {
        if cards_overlap(hero_cards, hands.combos[j]) {
            continue;
        }
        equity_sum += eq_row;
    }
    let avg_equity = equity_sum / opp_reach_total;
    (2.0 * avg_equity - 1.0) * half_pot
}
```

**Step 3: Verify compilation**

Run: `cargo check -p poker-solver-core 2>&1`
Expected: compiles (new code not called yet)

**Step 4: Commit**

```
feat(cfv-solver): add cfr_traverse_vectorized for all-combo single-pass traversal
```

---

### Task 2: Replace `train_with_leaf_interval` to use vectorized traversal

**Files:**
- Modify: `crates/core/src/blueprint_v2/cfv_subgame_solver.rs`

**Step 1: Rewrite the iteration loop**

Replace the body of `train_with_leaf_interval` (the `for iter_idx in 0..iterations` loop) with:

```rust
for iter_idx in 0..iterations {
    self.iteration += 1;

    // Build strategy snapshot (regret-matching for all nodes/combos).
    let snapshot = self.build_strategy_snapshot();

    // Determine whether to re-evaluate leaf boundaries.
    let should_eval = if leaf_eval_interval == 0 {
        true
    } else {
        self.iteration == 1
            || self.iteration % leaf_eval_interval == 0
            || iter_idx == iterations - 1
    };

    // Propagate ranges and evaluate boundaries if needed.
    if should_eval && !self.boundary_info.boundaries.is_empty() {
        let (oop_ranges, ip_ranges) = self.propagate_ranges(&snapshot);
        for traverser in 0..2u8 {
            for (b_idx, &(_, pot, invested)) in
                self.boundary_info.boundaries.iter().enumerate()
            {
                let eff_stack = self.starting_stack
                    - invested[0].max(invested[1]);
                self.leaf_cfvs[b_idx] = self.evaluator.evaluate(
                    &self.hands.combos,
                    &self.board,
                    pot,
                    eff_stack,
                    &oop_ranges[b_idx],
                    &ip_ranges[b_idx],
                    traverser,
                );
            }
        }
    }

    // Vectorized CFR traversal for each traverser.
    let n = self.hands.combos.len();
    let reach_init = vec![1.0; n];

    for traverser in 0..2u8 {
        let _cfvs = self.cfr_traverse_vectorized(
            self.tree.root as usize,
            traverser,
            &reach_init,
            &reach_init,
            &snapshot,
        );
    }

    // DCFR discounting.
    let iter_u64 = u64::from(self.iteration);
    if self.dcfr.should_discount(iter_u64) {
        self.dcfr.discount_regrets(&mut self.regret_sum, iter_u64);
        self.dcfr
            .discount_strategy_sums(&mut self.strategy_sum, iter_u64);
    }
}
```

Key changes:
- No more `parallel_traverse` call
- No more `CfvCfrCtx` creation
- Leaf evaluation uses single propagation for both traversers
- `cfr_traverse_vectorized` replaces the entire combo-level parallelism

**Step 2: Remove dead code**

Remove (or mark `#[allow(dead_code)]` for now):
- `CfvCfrCtx` struct and its `ParallelCfr` + `CfvCfrCtx` impl blocks
- The `use crate::cfr::parallel::{add_into, parallel_traverse, ParallelCfr};` import (if no longer used)

Keep `build_strategy_snapshot` and `propagate_ranges` — still needed.

**Step 3: Run tests**

Run: `cargo test -p poker-solver-core 2>&1`
Expected: All existing tests pass — `root_cfvs_returns_correct_length`, `root_cfvs_are_finite_and_bounded`, `cfv_solver_with_exact_evaluator_on_turn`, `propagate_ranges_sums_correctly`, etc.

Run: `cargo test -p cfvnet 2>&1`
Expected: All cfvnet tests pass.

**Step 4: Commit**

```
perf(cfv-solver): replace parallel_traverse with vectorized single-pass CFR
```

---

### Task 3: Performance verification

**Step 1: Build release and run turn datagen**

```bash
cargo run -p cfvnet --release --features cuda -- generate \
  -c sample_configurations/turn_cfvnet.yaml \
  -o /tmp/turn_perf_test.bin \
  --num-samples 100 \
  --backend cuda
```

**Expected:** Each situation should complete in seconds, not minutes. The progress bar should show meaningful throughput.

**Step 2: Compare output quality**

Run a small compare to verify the solver still produces reasonable CFVs:

```bash
cargo run -p cfvnet --release -- generate \
  -c sample_configurations/turn_cfvnet.yaml \
  -o /tmp/turn_cpu_test.bin \
  --num-samples 10 \
  --backend ndarray
```

Verify with datagen-eval:
```bash
cargo run -p cfvnet --release -- datagen-eval -d /tmp/turn_cpu_test.bin
```

**Step 3: Commit any fixes**

```
fix(cfv-solver): address issues from vectorized traversal performance test
```

---

### Task 4: Optimize allocations (if needed)

If performance testing shows allocation overhead from `Vec<f64>` allocations per node in the recursive traversal, add buffer reuse:

**Files:**
- Modify: `crates/core/src/blueprint_v2/cfv_subgame_solver.rs`

**Optimization options:**
1. Pre-allocate a CFV buffer pool and pass mutable slices instead of returning new Vecs
2. Use a scratch buffer pattern (like range solver's `ScratchBuffers`)
3. Change return type from `Vec<f64>` to writing into a caller-provided `&mut [f64]`

This task is conditional — only needed if Task 3 shows allocation is still a bottleneck.
