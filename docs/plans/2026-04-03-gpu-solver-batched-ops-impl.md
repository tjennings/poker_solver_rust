# GPU Solver Batched Operations Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use hex:executing-plans to implement this plan task-by-task.

**Goal:** Rewrite GPU solver's forward/backward passes from per-node CPU loops to batched gather/scatter operations, reducing kernel launches from ~600/iter to ~50/iter.

**Architecture:** Pre-compute expanded index tensors during setup. Each level's edges become a contiguous slice. Forward/backward passes use `gather` + `scatter` on entire levels at once. Regret matching uses scatter-add/gather-divide over all edges globally.

**Tech Stack:** Rust, burn 0.16 (wgpu), gather/scatter tensor ops

**Design doc:** `docs/plans/2026-04-03-gpu-solver-batched-ops-design.md`

---

## Task 1: Pre-computed Level Index Tensors

Add `LevelIndices` struct and pre-compute expanded gather/scatter index tensors during `StreetSolver::new()`. This is additive — doesn't change any existing functions yet.

**Files:**
- Modify: `crates/gpu-range-solver/src/tensors.rs`

**Step 1: Write the failing test**

Add to the existing `#[cfg(test)] mod tests` in tensors.rs:

```rust
#[test]
fn level_indices_cover_all_edges() {
    let game = make_river_game();
    let topo = extract_topology(&game);
    let term = extract_terminal_data(&game, &topo);
    let device = Default::default();
    let num_hands = game.private_cards(0).len().max(game.private_cards(1).len());

    let solver = super::StreetSolver::<TestBackend>::new(&topo, &term, 1, num_hands, &device);

    // All edges should appear in exactly one level's index
    let mut seen = vec![false; solver.num_edges];
    for li in &solver.level_indices {
        for e in li.edge_range.clone() {
            assert!(!seen[e], "edge {e} appears in multiple levels");
            seen[e] = true;
        }
    }
    assert!(seen.iter().all(|&s| s), "not all edges covered by level_indices");
}

#[test]
fn level_parent_gather_idx_shape() {
    let game = make_river_game();
    let topo = extract_topology(&game);
    let term = extract_terminal_data(&game, &topo);
    let device = Default::default();
    let num_hands = game.private_cards(0).len().max(game.private_cards(1).len());

    let solver = super::StreetSolver::<TestBackend>::new(&topo, &term, 1, num_hands, &device);

    for li in &solver.level_indices {
        let n_level_edges = li.edge_range.len();
        let shape = li.parent_gather_idx.shape();
        assert_eq!(shape.dims, [1, n_level_edges, num_hands]);
    }
}

#[test]
fn actions_per_edge_tensor_matches_topology() {
    let game = make_river_game();
    let topo = extract_topology(&game);
    let term = extract_terminal_data(&game, &topo);
    let device = Default::default();
    let num_hands = game.private_cards(0).len().max(game.private_cards(1).len());

    let solver = super::StreetSolver::<TestBackend>::new(&topo, &term, 1, num_hands, &device);

    let ape_data: Vec<f32> = solver.actions_per_edge.clone().squeeze::<2>(0)
        .narrow(1, 0, 1).squeeze::<1>(1).into_data().to_vec().unwrap();
    for e in 0..solver.num_edges {
        let parent = topo.edge_parent[e];
        let expected = topo.node_num_actions[parent] as f32;
        assert!((ape_data[e] - expected).abs() < 0.01,
            "actions_per_edge[{e}] = {} but parent node {} has {} actions",
            ape_data[e], parent, expected);
    }
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test -p gpu-range-solver -- level_indices_cover_all`
Expected: Compilation error — `level_indices` field doesn't exist

**Step 3: Implement**

Add to tensors.rs, before the `StreetSolver` struct:

```rust
/// Pre-computed index tensors for one depth level's edges.
/// Enables batched gather/scatter over all edges at a level in a single kernel.
pub struct LevelIndices<B: Backend> {
    /// Contiguous range of edges in the (sorted-by-depth) edge arrays.
    pub edge_range: std::ops::Range<usize>,

    /// Parent node indices expanded for gather on dim 1.
    /// Shape: [batch, num_level_edges, num_hands] — all values along dim 2 are the same.
    pub parent_gather_idx: Tensor<B, 3, Int>,

    /// Child node indices expanded for scatter on dim 1.
    /// Shape: [batch, num_level_edges, num_hands]
    pub child_scatter_idx: Tensor<B, 3, Int>,

    /// Per-edge: 1.0 if parent is player 0's decision node, 0.0 otherwise.
    /// Shape: [1, num_level_edges, 1]
    pub is_player0_edge: Tensor<B, 3>,

    /// Per-edge: 1.0 if parent is player 1's decision node, 0.0 otherwise.
    /// Shape: [1, num_level_edges, 1]
    pub is_player1_edge: Tensor<B, 3>,

    /// Per-edge: 1.0 if parent is a chance node, 0.0 otherwise.
    /// Shape: [1, num_level_edges, 1]
    pub is_chance_edge: Tensor<B, 3>,
}
```

Add new fields to `StreetSolver`:

```rust
    /// Pre-computed per-level gather/scatter index tensors.
    pub level_indices: Vec<LevelIndices<B>>,

    /// Per-edge: number of actions at the edge's parent node.
    /// Shape: [1, num_edges, 1] — broadcast-ready for division.
    pub actions_per_edge: Tensor<B, 3>,

    /// Per-edge: parent node index (expanded for global gather/scatter).
    /// Shape: [batch, num_edges, num_hands]
    pub edge_parent_expanded: Tensor<B, 3, Int>,

    /// Edges sorted by depth: maps new position -> original edge index.
    /// (If edges are already sorted by depth via BFS order, this may be identity.)
    pub edge_depth_order: Vec<usize>,
```

In `StreetSolver::new()`, after the existing setup code, add the index precomputation:

```rust
    // Sort edges by parent depth for level-contiguous ranges
    let node_depth: Vec<usize> = {
        let mut d = vec![0usize; num_nodes];
        for (depth, nodes) in topo.level_nodes.iter().enumerate() {
            for &nid in nodes { d[nid] = depth; }
        }
        d
    };

    // Group edges by parent depth
    let max_depth = topo.max_depth;
    let mut edges_by_depth: Vec<Vec<usize>> = vec![Vec::new(); max_depth + 1];
    for e in 0..num_edges {
        let parent_depth = node_depth[topo.edge_parent[e]];
        edges_by_depth[parent_depth].push(e);
    }

    // Build sorted edge arrays and level ranges
    let mut sorted_edges: Vec<usize> = Vec::with_capacity(num_edges);
    let mut level_ranges: Vec<std::ops::Range<usize>> = Vec::new();
    for depth_edges in &edges_by_depth {
        let start = sorted_edges.len();
        sorted_edges.extend(depth_edges);
        level_ranges.push(start..sorted_edges.len());
    }

    // Re-order edge_parent and edge_child arrays according to sorted order
    let sorted_parent: Vec<i32> = sorted_edges.iter().map(|&e| topo.edge_parent[e] as i32).collect();
    let sorted_child: Vec<i32> = sorted_edges.iter().map(|&e| topo.edge_child[e] as i32).collect();

    // Rebuild edge_parent/edge_child tensors in sorted order
    let edge_parent = Tensor::<B, 1, Int>::from_data(
        burn::tensor::TensorData::new(sorted_parent.clone(), [num_edges]), device,
    );
    let edge_child = Tensor::<B, 1, Int>::from_data(
        burn::tensor::TensorData::new(sorted_child.clone(), [num_edges]), device,
    );

    // Rebuild node_first_edge and node_num_actions mappings for sorted edge order
    // ... (recalculate based on sorted_edges)

    // Build actions_per_edge: [1, num_edges, 1]
    let ape_data: Vec<f32> = sorted_edges.iter()
        .map(|&e| topo.node_num_actions[topo.edge_parent[e]] as f32)
        .collect();
    let actions_per_edge = Tensor::<B, 3>::from_data(
        burn::tensor::TensorData::new(ape_data, [1, num_edges, 1]), device,
    );

    // Build expanded parent index: [batch, num_edges, num_hands]
    let parent_expanded_data: Vec<i32> = sorted_parent.iter()
        .flat_map(|&p| std::iter::repeat(p).take(num_hands))
        .collect();
    let single_batch_parent = Tensor::<B, 3, Int>::from_data(
        burn::tensor::TensorData::new(parent_expanded_data, [1, num_edges, num_hands]), device,
    );
    let edge_parent_expanded = single_batch_parent.expand([batch_size as i32, num_edges as i32, num_hands as i32]);

    // Build per-level LevelIndices
    let level_indices: Vec<LevelIndices<B>> = level_ranges.iter().enumerate().map(|(depth, range)| {
        let n = range.len();
        if n == 0 {
            // Empty level (no edges at this depth)
            return LevelIndices {
                edge_range: range.clone(),
                parent_gather_idx: Tensor::zeros([batch_size, 0, num_hands], device),
                child_scatter_idx: Tensor::zeros([batch_size, 0, num_hands], device),
                is_player0_edge: Tensor::zeros([1, 0, 1], device),
                is_player1_edge: Tensor::zeros([1, 0, 1], device),
                is_chance_edge: Tensor::zeros([1, 0, 1], device),
            };
        }

        // Parent/child indices for this level's edges
        let level_parents: Vec<i32> = sorted_parent[range.clone()].to_vec();
        let level_children: Vec<i32> = sorted_child[range.clone()].to_vec();

        // Expand to [batch, n, num_hands]
        let parent_data: Vec<i32> = level_parents.iter()
            .flat_map(|&p| std::iter::repeat(p).take(num_hands))
            .collect();
        let child_data: Vec<i32> = level_children.iter()
            .flat_map(|&c| std::iter::repeat(c).take(num_hands))
            .collect();

        let parent_idx = Tensor::<B, 3, Int>::from_data(
            burn::tensor::TensorData::new(parent_data, [1, n, num_hands]), device,
        ).expand([batch_size as i32, n as i32, num_hands as i32]);

        let child_idx = Tensor::<B, 3, Int>::from_data(
            burn::tensor::TensorData::new(child_data, [1, n, num_hands]), device,
        ).expand([batch_size as i32, n as i32, num_hands as i32]);

        // Player/chance classification masks: [1, n, 1]
        let mut p0_mask = vec![0.0f32; n];
        let mut p1_mask = vec![0.0f32; n];
        let mut chance_mask = vec![0.0f32; n];
        for (i, &orig_e) in sorted_edges[range.clone()].iter().enumerate() {
            let parent_node = topo.edge_parent[orig_e];
            match topo.node_type[parent_node] {
                NodeType::Player { player: 0 } => p0_mask[i] = 1.0,
                NodeType::Player { player: 1 } => p1_mask[i] = 1.0,
                NodeType::Chance => chance_mask[i] = 1.0,
                _ => {}
            }
        }

        LevelIndices {
            edge_range: range.clone(),
            parent_gather_idx: parent_idx,
            child_scatter_idx: child_idx,
            is_player0_edge: Tensor::from_data(
                burn::tensor::TensorData::new(p0_mask, [1, n, 1]), device,
            ),
            is_player1_edge: Tensor::from_data(
                burn::tensor::TensorData::new(p1_mask, [1, n, 1]), device,
            ),
            is_chance_edge: Tensor::from_data(
                burn::tensor::TensorData::new(chance_mask, [1, n, 1]), device,
            ),
        }
    }).collect();
```

**Important:** When edges are re-sorted, the `regrets`, `strategy_sum` tensors' edge dimension must follow the same order. Since they start as zeros, the sort order doesn't matter for initialization. But `node_first_edge` must be recomputed for the new edge ordering.

Also need to re-map: the `fold_node_ids`, `showdown_node_ids` etc. stay the same (they index nodes, not edges). But `node_first_edge` needs updating since edges are now sorted by parent depth.

Recompute `node_first_edge` from the sorted edge order:
```rust
    let mut new_node_first_edge = vec![0usize; num_nodes];
    let mut new_node_edge_count = vec![0usize; num_nodes];
    for &orig_e in &sorted_edges {
        let parent = topo.edge_parent[orig_e];
        new_node_edge_count[parent] += 1;
    }
    let mut offset = 0;
    for n in 0..num_nodes {
        new_node_first_edge[n] = offset;
        offset += new_node_edge_count[n];
    }
```

**Step 4: Run tests**

Run: `cargo test -p gpu-range-solver`
Expected: All old tests + 3 new tests pass

**Step 5: Commit**

```bash
git add crates/gpu-range-solver/src/tensors.rs
git commit -m "feat(gpu-range-solver): pre-computed level index tensors for batched gather/scatter"
```

---

## Task 2: Batched Regret Matching

Replace the per-node `regret_match()` with a global version that operates on all edges at once using scatter-add and gather-divide.

**Files:**
- Modify: `crates/gpu-range-solver/src/solver.rs`

**Step 1: Write the failing test**

```rust
#[test]
fn batched_regret_match_global_matches_per_node() {
    // Build a solver, set known regrets, compare global vs per-node results
    let game = make_river_game();
    let solver = make_solver(&game);
    let device: <B as burn::tensor::backend::Backend>::Device = Default::default();

    // Set some non-zero regrets
    let mut regrets_data = vec![0.0f32; solver.num_edges * solver.num_hands];
    for e in 0..solver.num_edges {
        for h in 0..solver.num_hands {
            regrets_data[e * solver.num_hands + h] = (e as f32 * 0.5 - 3.0) + h as f32 * 0.1;
        }
    }
    let regrets = Tensor::<B, 3>::from_data(
        burn::tensor::TensorData::new(regrets_data, [1, solver.num_edges, solver.num_hands]),
        &device,
    );

    let strategy = super::regret_match_global::<B>(
        &regrets, &solver.edge_parent_expanded, &solver.actions_per_edge,
        solver.num_nodes, &device,
    );

    let strat_data: Vec<f32> = strategy.into_data().to_vec().unwrap();

    // Verify: for each node's edges, strategy should sum to 1.0 per hand
    for n in 0..solver.num_nodes {
        let n_actions = solver.node_num_actions[n];
        if n_actions == 0 { continue; }
        let first = solver.node_first_edge[n];
        for h in 0..solver.num_hands {
            let sum: f32 = (0..n_actions)
                .map(|a| strat_data[(first + a) * solver.num_hands + h])
                .sum();
            assert!((sum - 1.0).abs() < 1e-4,
                "node {n} hand {h}: strategy sums to {sum}, expected 1.0");
        }
    }
}
```

**Step 2: Run test to verify it fails**

Expected: Compilation error — `regret_match_global` doesn't exist

**Step 3: Implement**

```rust
/// Batched regret matching over ALL edges at once.
///
/// Instead of looping per-node, uses scatter-add to sum positive regrets per node,
/// then gather-divide to normalize per edge. This is the GPUGT approach.
///
/// Input:
///   regrets: [batch, num_edges, num_hands]
///   edge_parent_idx: [batch, num_edges, num_hands] (Int) — parent node for each edge
///   actions_per_edge: [1, num_edges, 1] — parent's action count per edge
///   num_nodes: total number of nodes
///
/// Output: strategy [batch, num_edges, num_hands]
pub fn regret_match_global<B: Backend>(
    regrets: &Tensor<B, 3>,
    edge_parent_idx: &Tensor<B, 3, Int>,
    actions_per_edge: &Tensor<B, 3>,
    num_nodes: usize,
    device: &B::Device,
) -> Tensor<B, 3> {
    let [batch, num_edges, num_hands] = regrets.dims();

    // 1. Clip regrets to >= 0
    let clipped = regrets.clone().clamp_min(0.0);

    // 2. Scatter-add clipped regrets to nodes: denom[parent] = sum of clipped regrets
    let denom_nodes = Tensor::<B, 3>::zeros([batch, num_nodes, num_hands], device)
        .scatter(1, edge_parent_idx.clone(), clipped.clone());

    // 3. Gather denom back to edges: each edge gets its parent's total
    let denom_edges = denom_nodes.gather(1, edge_parent_idx.clone());

    // 4. Normalize: strategy = clipped / denom (or uniform if denom == 0)
    let uniform = actions_per_edge.clone().recip(); // 1/num_actions per edge
    let is_zero = denom_edges.clone().lower_equal_elem(0.0).float();
    let safe_denom = denom_edges.clamp_min(1e-30);
    let normalized = clipped / safe_denom;

    // Blend: where denom was zero, use uniform
    let is_zero_expanded = is_zero; // already [batch, edges, hands]
    let uniform_expanded = uniform.expand([batch as i32, num_edges as i32, num_hands as i32]);
    normalized * (Tensor::ones([batch, num_edges, num_hands], device) - is_zero_expanded.clone())
        + uniform_expanded * is_zero_expanded
}
```

**Step 4: Run tests**

Run: `cargo test -p gpu-range-solver -- batched_regret_match_global`
Expected: PASS

**Step 5: Commit**

```bash
git commit -m "feat(gpu-range-solver): batched regret matching via scatter-add/gather-divide"
```

---

## Task 3: Batched Forward Pass

Replace the per-node loop in `forward_pass` with batched gather/scatter using `LevelIndices`.

**Files:**
- Modify: `crates/gpu-range-solver/src/solver.rs`

**Step 1: Write the failing test**

```rust
#[test]
fn batched_forward_pass_matches_original() {
    let game = make_river_game();
    let device: <B as burn::tensor::backend::Backend>::Device = Default::default();

    use range_solver::interface::Game;
    let initial_weights: [Vec<f32>; 2] = [
        game.initial_weights(0).to_vec(),
        game.initial_weights(1).to_vec(),
    ];

    // Run original forward pass
    let mut solver_old = make_solver(&game);
    super::initialize_reach(&mut solver_old, &initial_weights[1], &device);
    super::forward_pass(&mut solver_old, 0, &device);
    let old_reach: Vec<f32> = solver_old.reach.into_data().to_vec().unwrap();

    // Run batched forward pass
    let mut solver_new = make_solver(&game);
    super::initialize_reach(&mut solver_new, &initial_weights[1], &device);
    super::forward_pass_batched(&mut solver_new, 0, &device);
    let new_reach: Vec<f32> = solver_new.reach.into_data().to_vec().unwrap();

    // Compare
    for i in 0..old_reach.len() {
        assert!((old_reach[i] - new_reach[i]).abs() < 1e-5,
            "reach[{i}]: old={} new={}", old_reach[i], new_reach[i]);
    }
}
```

**Step 2: Run test to verify it fails**

Expected: `forward_pass_batched` doesn't exist

**Step 3: Implement**

```rust
/// Batched forward pass using gather/scatter.
///
/// For each level top-down:
/// 1. Gather parent reach for all edges at this level
/// 2. Compute strategy via batched regret matching
/// 3. For opponent edges: child_reach = parent_reach * strategy
///    For traverser edges: child_reach = parent_reach (counterfactual)
///    For chance edges: child_reach = parent_reach / chance_factor * card_mask
/// 4. Scatter child reach to child nodes
pub fn forward_pass_batched<B: Backend>(
    solver: &mut crate::tensors::StreetSolver<B>,
    player: usize,
    device: &B::Device,
) {
    let [batch, _num_nodes, num_hands] = solver.reach.dims();

    for li in &solver.level_indices {
        let n = li.edge_range.len();
        if n == 0 { continue; }

        // 1. Gather parent reach: [batch, n_level_edges, num_hands]
        let parent_reach = solver.reach.clone().gather(1, li.parent_gather_idx.clone());

        // 2. Get strategy for this level's edges
        let level_regrets = solver.regrets.clone().narrow(1, li.edge_range.start, n);
        let level_parent_idx = solver.edge_parent_expanded.clone().narrow(1, li.edge_range.start, n);
        let level_ape = solver.actions_per_edge.clone().narrow(1, li.edge_range.start, n);
        let strategy = regret_match_global::<B>(
            &level_regrets, &level_parent_idx, &level_ape, solver.num_nodes, device,
        );

        // 3. Compute child reach based on node type
        let is_opp = if player == 0 {
            li.is_player1_edge.clone()
        } else {
            li.is_player0_edge.clone()
        };
        let is_trav = if player == 0 {
            li.is_player0_edge.clone()
        } else {
            li.is_player1_edge.clone()
        };

        // Opponent: parent_reach * strategy
        // Traverser: parent_reach (no strategy multiply)
        // Chance: parent_reach / chance_factor (handle separately if needed)
        let opp_expanded = is_opp.expand([batch as i32, n as i32, num_hands as i32]);
        let trav_expanded = is_trav.expand([batch as i32, n as i32, num_hands as i32]);
        let chance_expanded = li.is_chance_edge.clone()
            .expand([batch as i32, n as i32, num_hands as i32]);

        // For non-chance edges: opp gets strategy multiply, traverser gets 1.0
        let multiplier = strategy.clone() * opp_expanded.clone()
            + trav_expanded.clone()
            + chance_expanded.clone(); // chance gets 1.0 here, we handle division below

        let child_reach = parent_reach * multiplier;

        // TODO: chance node card blocking would go here (future optimization)
        // For now, chance edges get parent_reach * 1.0 (no division or card blocking)
        // This matches the original behavior for river-only games (no chance nodes)

        // 4. Scatter to child nodes
        solver.reach = solver.reach.clone()
            .scatter(1, li.child_scatter_idx.clone(), child_reach);
    }
}
```

**Note:** Chance node handling (division by chance_factor + card blocking) is handled in a later step. For now the batched forward pass works correctly for river games (no chance nodes).

**Step 4: Run tests**

Run: `cargo test -p gpu-range-solver -- batched_forward_pass_matches`
Expected: PASS

**Step 5: Commit**

```bash
git commit -m "feat(gpu-range-solver): batched forward pass via gather/scatter"
```

---

## Task 4: Batched Backward Pass

Replace the per-node loop in `backward_pass` with batched gather/scatter.

**Files:**
- Modify: `crates/gpu-range-solver/src/solver.rs`

**Step 1: Write the failing test**

```rust
#[test]
fn batched_backward_pass_matches_original() {
    let game = make_river_game();
    let device: <B as burn::tensor::backend::Backend>::Device = Default::default();

    use range_solver::interface::Game;
    let initial_weights: [Vec<f32>; 2] = [
        game.initial_weights(0).to_vec(),
        game.initial_weights(1).to_vec(),
    ];

    // Run original: init + forward + backward
    let mut solver_old = make_solver(&game);
    super::initialize_reach(&mut solver_old, &initial_weights[1], &device);
    super::forward_pass(&mut solver_old, 0, &device);
    super::backward_pass(&mut solver_old, 0, &device);
    let old_cfv: Vec<f32> = solver_old.cfv.clone().into_data().to_vec().unwrap();
    let old_regrets: Vec<f32> = solver_old.regrets.clone().into_data().to_vec().unwrap();

    // Run batched
    let mut solver_new = make_solver(&game);
    super::initialize_reach(&mut solver_new, &initial_weights[1], &device);
    super::forward_pass_batched(&mut solver_new, 0, &device);
    super::backward_pass_batched(&mut solver_new, 0, &device);
    let new_cfv: Vec<f32> = solver_new.cfv.clone().into_data().to_vec().unwrap();
    let new_regrets: Vec<f32> = solver_new.regrets.clone().into_data().to_vec().unwrap();

    // Compare root CFV
    let num_hands = solver_old.num_hands;
    for h in 0..num_hands {
        assert!((old_cfv[h] - new_cfv[h]).abs() < 1e-4,
            "root cfv[{h}]: old={} new={}", old_cfv[h], new_cfv[h]);
    }

    // Compare regrets
    for i in 0..old_regrets.len() {
        assert!((old_regrets[i] - new_regrets[i]).abs() < 1e-4,
            "regrets[{i}]: old={} new={}", old_regrets[i], new_regrets[i]);
    }
}
```

**Step 2: Run test to verify it fails**

**Step 3: Implement**

```rust
/// Batched backward pass using gather/scatter.
///
/// For each level bottom-up:
/// 1. Evaluate terminal nodes at this depth (fold + showdown)
/// 2. Gather child CFVs for all edges at this level
/// 3. Compute strategy and weight child CFVs (traverser: by strategy, opponent: by 1.0)
/// 4. Scatter-add weighted CFVs to parent nodes
/// 5. Compute regret updates for traverser edges
pub fn backward_pass_batched<B: Backend>(
    solver: &mut crate::tensors::StreetSolver<B>,
    player: usize,
    device: &B::Device,
) {
    let [batch, _num_nodes, num_hands] = solver.reach.dims();

    // Build terminal evaluation context (same as original)
    let opp = player ^ 1;
    let node_depth = build_node_depth(&solver.level_nodes, solver.num_nodes);
    let showdown_outcomes = build_showdown_outcomes::<B>(&solver.showdown_data, player, device);
    let term_ctx = TerminalEvalContext { /* ... same as original ... */ };

    // Zero CFV tensor
    solver.cfv = Tensor::zeros([batch, solver.num_nodes, num_hands], device);

    // Process levels bottom-up
    for depth_idx in (0..solver.level_indices.len()).rev() {
        let li = &solver.level_indices[depth_idx];

        // 1. Evaluate terminal nodes at this depth (reuse existing function)
        {
            let reach_clone = solver.reach.clone();
            evaluate_terminals_at_depth(
                &mut solver.cfv, &reach_clone, &term_ctx,
                depth_idx, &node_depth, &showdown_outcomes, device,
            );
        }

        let n = li.edge_range.len();
        if n == 0 { continue; }

        // 2. Gather child CFVs: [batch, n_level_edges, num_hands]
        let child_cfv = solver.cfv.clone().gather(1, li.child_scatter_idx.clone());

        // 3. Get strategy and weight by node type
        let level_regrets = solver.regrets.clone().narrow(1, li.edge_range.start, n);
        let level_parent_idx = solver.edge_parent_expanded.clone().narrow(1, li.edge_range.start, n);
        let level_ape = solver.actions_per_edge.clone().narrow(1, li.edge_range.start, n);
        let strategy = regret_match_global::<B>(
            &level_regrets, &level_parent_idx, &level_ape, solver.num_nodes, device,
        );

        let is_trav = if player == 0 {
            li.is_player0_edge.clone()
        } else {
            li.is_player1_edge.clone()
        };
        let trav_expanded = is_trav.expand([batch as i32, n as i32, num_hands as i32]);

        // Traverser: weight by strategy. Others: weight = 1.0
        let weight = strategy.clone() * trav_expanded.clone()
            + (Tensor::ones([batch, n, num_hands], device) - trav_expanded.clone());
        let weighted_cfv = child_cfv.clone() * weight;

        // 4. Scatter-add to parent nodes
        solver.cfv = solver.cfv.clone()
            .scatter(1, li.parent_gather_idx.clone(), weighted_cfv);

        // 5. Regret update for traverser edges
        let parent_cfv = solver.cfv.clone().gather(1, li.parent_gather_idx.clone());
        let instant_regret = (child_cfv - parent_cfv) * trav_expanded.clone();

        // Add to cumulative regrets
        let old_regrets = solver.regrets.clone().narrow(1, li.edge_range.start, n);
        let new_regrets = old_regrets + instant_regret;
        solver.regrets = solver.regrets.clone().slice_assign(
            [0..batch, li.edge_range.start..li.edge_range.end, 0..num_hands],
            new_regrets,
        );

        // Update strategy_sum for traverser edges
        let trav_strategy = strategy * trav_expanded;
        let old_strat = solver.strategy_sum.clone().narrow(1, li.edge_range.start, n);
        let new_strat = old_strat + trav_strategy;
        solver.strategy_sum = solver.strategy_sum.clone().slice_assign(
            [0..batch, li.edge_range.start..li.edge_range.end, 0..num_hands],
            new_strat,
        );
    }
}
```

**Step 4: Run tests**

Run: `cargo test -p gpu-range-solver -- batched_backward_pass_matches`
Expected: PASS

**Step 5: Commit**

```bash
git commit -m "feat(gpu-range-solver): batched backward pass via gather/scatter"
```

---

## Task 5: Wire Batched Passes into Solve Loop

Replace the original `forward_pass`/`backward_pass` calls in `run_iteration`, `compute_exploitability`, and `best_response_backward` with the batched versions.

**Files:**
- Modify: `crates/gpu-range-solver/src/solver.rs`

**Step 1: Write the failing test**

```rust
#[test]
fn batched_gpu_solve_matches_cpu() {
    // Same as existing gpu_solve_matches_cpu_convergence but using batched passes
    let game = make_river_game();
    let mut cpu_game = make_river_game();
    let cpu_expl = range_solver::solve(&mut cpu_game, 200, 0.0, false);

    let config = crate::GpuSolverConfig {
        max_iterations: 200,
        target_exploitability: 0.0,
        print_progress: false,
    };
    let result = crate::gpu_solve_game(&game, &config);

    assert!((result.exploitability - cpu_expl).abs() < 0.5,
        "GPU {:.4} vs CPU {:.4}", result.exploitability, cpu_expl);
}
```

**Step 2: Implement**

Replace calls in `run_iteration`:
```rust
pub fn run_iteration<B: Backend>(...) {
    let opp = player ^ 1;
    initialize_reach(solver, &initial_weights[opp], device);
    forward_pass_batched(solver, player, device);  // was: forward_pass
    backward_pass_batched(solver, player, device);  // was: backward_pass
}
```

Also rewrite `best_response_backward` to use the same batched gather/scatter approach (replacing its per-node loop).

For `compute_exploitability`, rewrite the best-response forward pass and backward pass to use batched operations with `max_dim` instead of strategy-weighted sum at traverser nodes.

**Step 3: Run ALL existing tests**

Run: `cargo test -p gpu-range-solver`
Expected: All 79 tests pass

**Step 4: Commit**

```bash
git commit -m "feat(gpu-range-solver): wire batched passes into solve loop"
```

---

## Task 6: GPU-only Fold Evaluation

Replace the CPU-round-trip fold evaluation with GPU-native operations using pre-computed card index tensors.

**Files:**
- Modify: `crates/gpu-range-solver/src/terminal.rs`
- Modify: `crates/gpu-range-solver/src/tensors.rs` (add card index tensors)

**Step 1: Write the failing test**

```rust
#[test]
fn gpu_fold_eval_matches_cpu_fold_eval() {
    // Use known opponent reach, compare GPU-only vs CPU-round-trip fold eval
    // ... (compare evaluate_fold vs evaluate_fold_gpu with same inputs)
}
```

**Step 2: Implement**

Pre-compute in `StreetSolver::new()`:
- `hand_card1_idx: Tensor<B, 1, Int>` — [num_hands] card1 for each hand
- `hand_card2_idx: Tensor<B, 1, Int>` — [num_hands] card2 for each hand
- `same_hand_correction: Tensor<B, 2>` — [num_hands] for each player, the opp hand index weight

```rust
/// GPU-only fold evaluation. No CPU round-trip.
pub fn evaluate_fold_gpu<B: Backend>(
    opp_reach: &Tensor<B, 2>,         // [batch, num_opp_hands]
    hand_card1: &Tensor<B, 2, Int>,    // [1, num_player_hands] -> card1 indices
    hand_card2: &Tensor<B, 2, Int>,    // [1, num_player_hands] -> card2 indices
    same_hand_opp_idx: &Tensor<B, 2, Int>, // [1, num_player_hands] -> opp idx or sentinel
    payoff: f32,
    device: &B::Device,
) -> Tensor<B, 2> {
    let [batch, num_opp] = opp_reach.dims();

    // 1. total reach
    let total = opp_reach.clone().sum_dim(1); // [batch, 1]

    // 2. per-card reach via scatter
    let card_idx = /* build [batch, num_opp, 1] index of card1 for each opp hand */;
    let card_reach = Tensor::zeros([batch, 52], device)
        .scatter(1, card1_idx, opp_reach.clone())
        /* + scatter for card2 */;

    // 3. blocking per player hand
    let block1 = card_reach.gather(1, hand_card1); // [batch, num_player]
    let block2 = card_reach.gather(1, hand_card2);
    let same_correction = opp_reach.gather(1, same_hand_opp_idx); // [batch, num_player]
    let blocking = block1 + block2 - same_correction;

    // 4. cfv = payoff * (total - blocking)
    (total.expand(blocking.shape()) - blocking) * payoff
}
```

**Step 3: Run tests**

**Step 4: Commit**

```bash
git commit -m "feat(gpu-range-solver): GPU-only fold evaluation (no CPU round-trip)"
```

---

## Task 7: Performance Verification and Cleanup

Run the benchmark comparison, remove old per-node functions, clean up.

**Files:**
- Modify: `crates/gpu-range-solver/src/solver.rs` (remove old functions)

**Step 1: Benchmark**

```bash
time cargo run -p poker-solver-trainer --release -- gpu-range-solve \
  --oop-range "QQ+,AKs" --ip-range "JJ-99,AQs" \
  --flop "Qs Jh 2c" --turn "8d" --river "3s" \
  --pot 100 --effective-stack 100 --iterations 500

time cargo run -p poker-solver-trainer --release -- range-solve \
  --oop-range "QQ+,AKs" --ip-range "JJ-99,AQs" \
  --flop "Qs Jh 2c" --turn "8d" --river "3s" \
  --pot 100 --effective-stack 100 --iterations 500
```

Target: GPU ≤ CPU time (≤0.3s).

**Step 2: Remove old per-node functions**

Delete: `forward_pass` (old), `backward_pass` (old), `regret_match` (old, per-node version).
Rename: `forward_pass_batched` → `forward_pass`, `backward_pass_batched` → `backward_pass`.

**Step 3: Run all tests**

Run: `cargo test -p gpu-range-solver`
Expected: All tests pass

**Step 4: Clippy**

Run: `cargo clippy -p gpu-range-solver`
Expected: Clean

**Step 5: Commit**

```bash
git commit -m "perf(gpu-range-solver): remove per-node loops, batched ops only"
```

---

## Summary

| Task | What | Kernel launches/iter |
|------|------|---------------------|
| 1 | Pre-computed index tensors | (setup only) |
| 2 | Batched regret matching | All edges at once |
| 3 | Batched forward pass | ~3 per level |
| 4 | Batched backward pass | ~8 per level |
| 5 | Wire into solve loop | - |
| 6 | GPU-only fold eval | Eliminates CPU round-trip |
| 7 | Benchmark + cleanup | Target: ≤0.3s |

**Critical path:** Tasks 1→2→3→4→5→7. Task 6 can be done in parallel after Task 1.
