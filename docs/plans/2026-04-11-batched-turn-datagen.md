# Batched Turn Datagen Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use hex:executing-plans to implement this plan task-by-task.

**Goal:** Batch many turn games per GPU kernel launch by using one canonical topology and per-batch fold payoffs. Eliminates per-game kernel compilation and utilizes all GPU SMs.

**Architecture:** Build one canonical turn tree at startup (highest SPR → no bet-size collapsing). Make fold payoffs per-batch in the CUDA kernel (same pattern as showdown outcomes). Create `GpuBatchSolver` once, batch 256 games per `prepare_batch` + `run_iterations` cycle. The orchestrator collects K situations, builds specs in parallel, batch-solves, batch-evaluates boundaries, writes records.

**Tech Stack:** Rust, CUDA (cudarc/nvrtc), gpu-range-solver, cfvnet.

---

## Parallelization Map

```
Task 1 (per-batch fold payoffs in kernel) → Task 2 (batched orchestrator)
```

Sequential — Task 2 depends on Task 1.

---

### Task 1: Per-Batch Fold Payoffs

Currently fold payoffs are `[num_folds]` — shared across all games in a batch. Since
different games have different pot/stack, fold payoffs must be per-batch: `[B * num_folds]`.
This follows the existing pattern for showdown outcomes which are already `[B * num_showdowns * H * H]`.

**Files:**
- Modify: `crates/gpu-range-solver/src/kernels.rs` (fold payoff indexing)
- Modify: `crates/gpu-range-solver/src/batch.rs` (SubgameSpec, prepare_batch, new)
- Modify: `crates/gpu-range-solver/src/solver.rs` (MegaTerminalData, build_mega_terminal_data)
- Test: `crates/gpu-range-solver/src/batch.rs`

#### Step 1: Add fold payoffs to SubgameSpec

In `batch.rs`, add fold payoffs to `SubgameSpec`:

```rust
pub struct SubgameSpec {
    pub initial_weights: [Vec<f32>; 2],
    pub showdown_outcomes_p0: Vec<f32>,
    pub showdown_outcomes_p1: Vec<f32>,
    /// Per-game fold payoffs for P0 traversal: `[num_folds]`.
    pub fold_payoffs_p0: Vec<f32>,
    /// Per-game fold payoffs for P1 traversal: `[num_folds]`.
    pub fold_payoffs_p1: Vec<f32>,
}
```

Update `SubgameSpec::from_game()` to populate these from `build_mega_terminal_data`:

```rust
SubgameSpec {
    initial_weights,
    showdown_outcomes_p0: mtd.showdown_outcomes_p0,
    showdown_outcomes_p1: mtd.showdown_outcomes_p1,
    fold_payoffs_p0: mtd.fold_payoffs_p0,
    fold_payoffs_p1: mtd.fold_payoffs_p1,
}
```

#### Step 2: Change kernel fold payoff indexing to per-batch

In `kernels.rs`, the hand-parallel kernel `HAND_PARALLEL_KERNEL_SOURCE` at the fold eval
section (line ~501):

```cuda
// OLD:
float payoff = (player == 0) ? fold_payoffs_p0[fi] : fold_payoffs_p1[fi];

// NEW:
float payoff = (player == 0)
    ? fold_payoffs_p0[bid * num_folds + fi]
    : fold_payoffs_p1[bid * num_folds + fi];
```

Do the same change in the mega-kernel `CFR_MEGA_KERNEL_SOURCE` (line ~825) for consistency.

#### Step 3: Upload batched fold payoffs in prepare_batch

In `batch.rs`, `prepare_batch()` currently uploads initial_weights and showdown_outcomes
per batch. Add fold payoffs to this:

```rust
// Build batched fold payoffs: [B * num_folds] for each player
let num_folds = self.num_folds;
let mut batched_fold_p0 = vec![0.0f32; batch_size * num_folds];
let mut batched_fold_p1 = vec![0.0f32; batch_size * num_folds];
for (b, spec) in specs.iter().enumerate() {
    let base = b * num_folds;
    let src_len = spec.fold_payoffs_p0.len().min(num_folds);
    batched_fold_p0[base..base + src_len].copy_from_slice(&spec.fold_payoffs_p0[..src_len]);
    batched_fold_p1[base..base + src_len].copy_from_slice(&spec.fold_payoffs_p1[..src_len]);
}
self.active_d_fold_payoffs_p0 = Some(upload_or_dummy_f32(&self.stream, &batched_fold_p0)?);
self.active_d_fold_payoffs_p1 = Some(upload_or_dummy_f32(&self.stream, &batched_fold_p1)?);
```

Add `active_d_fold_payoffs_p0/p1: Option<CudaSlice<f32>>` fields to `GpuBatchSolver`.

In `run_iterations()`, pass the active (per-batch) fold payoffs to the kernel instead of
the topology-level ones:

```rust
// OLD:
builder.arg(&self.d_fold_payoffs_p0);
builder.arg(&self.d_fold_payoffs_p1);

// NEW:
let fold_p0 = self.active_d_fold_payoffs_p0.as_ref().unwrap_or(&self.d_fold_payoffs_p0);
let fold_p1 = self.active_d_fold_payoffs_p1.as_ref().unwrap_or(&self.d_fold_payoffs_p1);
builder.arg(fold_p0);
builder.arg(fold_p1);
```

This maintains backward compatibility: `solve_batch` (which doesn't set active fold payoffs)
still uses the topology-level ones. The orchestrator's `prepare_batch` sets per-batch ones.

#### Step 4: Remove fold payoff dependency from GpuBatchSolver::new()

The constructor currently uploads fold payoffs from a dummy `build_mega_terminal_data` call.
Keep this as the default fallback (for backward compat with `solve_batch`), but the active
per-batch payoffs from `prepare_batch` will override them in `run_iterations`.

No code change needed here — the `unwrap_or` pattern in Step 3 handles this.

#### Step 5: Test — batched fold payoffs match single

```rust
#[test]
fn batched_fold_payoffs_matches_single() {
    // Build a river game, solve once with solve_batch (shared fold payoffs).
    // Then solve with prepare_batch (per-batch fold payoffs from SubgameSpec).
    // Results should match.
    let (game, topo, term) = make_test_game();
    let num_hands = ...;
    let spec = SubgameSpec::from_game(&game, &topo, &term, num_hands);

    // solve_batch path (shared fold payoffs)
    let mut solver1 = GpuBatchSolver::new(&topo, &term, 1, num_hands, 100).unwrap();
    let result1 = solver1.solve_batch(&[spec.clone()]).unwrap();

    // prepare_batch path (per-batch fold payoffs)
    let mut solver2 = GpuBatchSolver::new(&topo, &term, 4, num_hands, 100).unwrap();
    solver2.prepare_batch(&[spec.clone()]).unwrap();
    solver2.run_iterations(0, 100).unwrap();
    let result2 = solver2.extract_results().unwrap();

    // Compare
    for (a, b) in result1[0].strategy_sum.iter().zip(&result2[0].strategy_sum) {
        assert!((a - b).abs() < 1e-4, "mismatch: {a} vs {b}");
    }
}
```

**Run:** `cargo test -p gpu-range-solver batched_fold`
**Expected:** PASS.

#### Step 6: Test — multi-game batch with different fold payoffs

```rust
#[test]
fn multi_game_batch_different_payoffs() {
    // Create 4 specs with different fold payoffs (simulating different pot/stack).
    // Verify each produces different strategy_sum (confirming per-batch payoffs work).
    let (game, topo, term) = make_test_game();
    let num_hands = ...;
    let mut specs = Vec::new();
    for scale in &[1.0f32, 2.0, 0.5, 3.0] {
        let mut spec = SubgameSpec::from_game(&game, &topo, &term, num_hands);
        // Scale fold payoffs to simulate different pot sizes.
        for v in &mut spec.fold_payoffs_p0 { *v *= scale; }
        for v in &mut spec.fold_payoffs_p1 { *v *= scale; }
        specs.push(spec);
    }

    let mut solver = GpuBatchSolver::new(&topo, &term, 4, num_hands, 100).unwrap();
    solver.prepare_batch(&specs).unwrap();
    solver.run_iterations(0, 100).unwrap();
    let results = solver.extract_results().unwrap();

    assert_eq!(results.len(), 4);
    // Different payoff scales should produce different strategies.
    assert!(results[0].strategy_sum != results[1].strategy_sum);
}
```

#### Step 7: Commit

```bash
git add crates/gpu-range-solver/src/{kernels.rs,batch.rs,solver.rs}
git commit -m "feat(gpu-solver): per-batch fold payoffs for batched turn datagen

Fold payoffs indexed by [bid * num_folds + fi] in kernel, matching the
existing per-batch showdown outcomes pattern. SubgameSpec now includes
fold_payoffs_p0/p1. prepare_batch uploads batched payoffs."
```

---

### Task 2: Canonical Topology + Batched Orchestrator

Rewrite `run_gpu_turn` to build one canonical topology at startup, create `GpuBatchSolver`
once, and batch K games per kernel launch.

**Files:**
- Modify: `crates/cfvnet/src/datagen/domain/pipeline.rs` (run_gpu_turn)
- Modify: `crates/cfvnet/src/datagen/domain/game_tree.rs` (canonical tree builder)
- Test: `crates/cfvnet/src/datagen/domain/pipeline.rs`

**Depends on:** Task 1.

#### Step 1: Add canonical tree builder

In `game_tree.rs`, add a function that builds the maximum-SPR turn tree to get the
canonical (largest) topology:

```rust
/// Build a canonical turn tree using the maximum SPR configuration.
/// Returns the topology and terminal data that all batched games will share.
/// Uses SPR=100 (very deep stack) so no bet sizes collapse to allin.
pub fn build_canonical_turn_tree(
    bet_sizes: &[Vec<f64>],
) -> Option<(PostFlopGame, TreeTopology, TerminalData)> {
    // Use a dummy board, large stack, uniform ranges.
    let board = [0u8, 4, 8, 12]; // arbitrary non-conflicting cards
    let pot = 100.0;
    let effective_stack = 10000.0; // SPR=100, no bets collapse
    let mut ranges = [[0.0f32; 1326]; 2];
    // Set uniform range for all non-conflicting hands.
    for c0 in 0u8..52 {
        for c1 in (c0 + 1)..52 {
            if board.contains(&c0) || board.contains(&c1) { continue; }
            let idx = range_solver::card::card_pair_to_index(c0, c1);
            ranges[0][idx] = 1.0;
            ranges[1][idx] = 1.0;
        }
    }

    let game = build_turn_game(&board, pot, effective_stack, &ranges, bet_sizes)?;
    let topo = gpu_range_solver::extract::extract_topology(&game);
    let term = /* build terminal data from game + topo */;
    Some((game, topo, term))
}
```

The key insight: the topology (node/edge structure) from this canonical tree is the same
for ALL turn games with these bet sizes, regardless of actual pot/stack/board/ranges.
Different games only differ in fold payoffs (from pot/stack) and initial weights (from ranges).

**IMPORTANT**: The `TerminalData` from the canonical tree has dummy fold payoffs. These
are ignored — per-batch fold payoffs from `SubgameSpec` override them. But the structural
data (fold_node_ids, fold_depths, showdown_node_ids, showdown_depths, card data) is correct
and shared across all games.

Card data (player_card1/2, opp_card1/2, same_hand_idx) depends on which hands are in
each player's range. For the canonical tree, use ALL non-board-conflicting hands.
Per-game card conflicts are handled by zero initial_weights for conflicting hands.

**WAIT**: Card data actually varies per board — different boards block different hands,
changing the card arrays. The canonical tree uses a fixed board, so its card arrays are
for that specific board. Games with different boards have different blocked hands.

This is a problem. The fold eval kernel uses card data (player_card1/2, opp_card1/2) to
compute blocking effects. If we use the canonical tree's card data for all games, blocked
hands won't be correct for different boards.

**Solution**: Card data must also be per-batch, OR we use a fixed num_hands=1326 and let
zero initial_weights handle card conflicts. The kernel's fold eval uses card data to
determine which opponent hands are blocked — if an opponent hand has zero weight
(from initial_weights), it contributes nothing regardless of card data accuracy.

Actually, looking at the fold eval kernel more carefully: it computes
`sum(opponent_reach[non_blocked_hands])` to get the total reach at each fold node,
using `player_card1/2` and `opp_card1/2` to identify blocked hands. If we use the
canonical tree's card arrays (which include ALL C(48,2) hands for the canonical board),
and a different board's game has different blocked hands... the card arrays are wrong.

The simplest fix: **use num_hands = 1326 (all possible pairs)** and set initial_weights
to zero for board-conflicting hands. The card data arrays become board-independent
since they enumerate ALL 1326 pairs. A hand (c0,c1) conflicts with board card b if
c0==b or c1==b — this is handled by zero initial_weights, not by the card arrays.

The card arrays (`player_card1/2`, `opp_card1/2`, `same_hand_idx`) with 1326 hands
simply list all C(52,2) pairs and their blocking relationships. These are truly universal.

This means: canonical topology uses num_hands=1326, card arrays enumerate all pairs,
per-game initial_weights have zeros for board-conflicting hands.

#### Step 2: Rewrite `run_gpu_turn` with batched loop

The new structure:

```rust
fn run_gpu_turn(config, output_path) -> Result<(), String> {
    // 1. Load BoundaryNet.
    let evaluator = GpuBoundaryEvaluator::load(...)?;

    // 2. Build canonical topology.
    let (canon_game, topo, term) = build_canonical_turn_tree(&bet_sizes)?;
    let num_hands = 1326; // universal hand count
    let boundary_node_ids: Vec<usize> = topo.showdown_nodes.clone();

    // 3. Create GPU solver ONCE.
    let batch_size = config.datagen.gpu_batch_size.unwrap_or(256);
    let mut solver = GpuBatchSolver::new(&topo, &term, batch_size, num_hands, max_iterations)?;
    solver.set_leaf_injection(&leaf_ids, &leaf_depths)?;

    // 4. Generate in batches.
    let mut batch_sits: Vec<Situation> = Vec::with_capacity(batch_size);

    for sit in &mut sit_gen {
        // Build SubgameSpec for this situation (per-game weights + fold payoffs).
        let spec = build_turn_subgame_spec(&sit, &topo, &term, num_hands);
        batch_sits.push(sit);
        batch_specs.push(spec);

        if batch_specs.len() == batch_size {
            // Solve entire batch at once.
            solve_and_write_batch(
                &evaluator, &mut solver, &topo, &turn_term,
                &batch_specs, &batch_sits, &boundary_node_ids,
                num_hands, &writer, &pb,
            )?;
            batch_sits.clear();
            batch_specs.clear();
        }
    }
    // Handle remaining partial batch.
    if !batch_specs.is_empty() { ... }
}
```

The `solve_and_write_batch` function:

```rust
fn solve_and_write_batch(...) {
    solver.prepare_batch(&specs)?;

    // Initial boundary eval for all games in batch.
    let initial_ss = vec![0.0f32; topo.num_edges * num_hands * specs.len()];
    // ... evaluate boundaries for each game, upload batched leaf CFVs ...

    // Solve with periodic re-evaluation.
    let mut iter = 0;
    while iter < max_iterations {
        let end = (iter + leaf_eval_interval).min(max_iterations);
        solver.run_iterations(iter, end)?;
        iter = end;

        if iter < max_iterations {
            let mid_results = solver.extract_results()?;
            // Re-evaluate boundaries for ALL games in batch.
            for (b, (spec, sit)) in specs.iter().zip(sits).enumerate() {
                // compute_reach_at_nodes for game b using mid_results[b].strategy_sum
                // evaluate_boundaries_batched for game b
                // collect into batched leaf_cfv arrays
            }
            solver.update_leaf_cfvs(&all_p0, &all_p1)?;
        }
    }

    // Extract and write.
    let results = solver.extract_results()?;
    for (b, result) in results.iter().enumerate() {
        let evs = compute_evs_from_strategy_sum(&topo, &term, &result.strategy_sum, ...);
        // Build and write TrainingRecords.
    }
}
```

#### Step 3: Build per-game SubgameSpec from situation

A helper that builds a `SubgameSpec` for a situation against the canonical topology:

```rust
fn build_turn_subgame_spec(
    sit: &Situation,
    topo: &TreeTopology,
    term: &TerminalData,
    num_hands: usize, // 1326
) -> SubgameSpec {
    // Initial weights: sit.ranges mapped to 1326-space, zero for board-conflicting hands.
    let mut weights = [[0.0f32; 1326]; 2];
    for &(c0, c1) in &all_hand_pairs {
        if conflicts_with_board(c0, c1, &sit.board) { continue; }
        let idx = card_pair_to_index(c0, c1);
        weights[0][idx] = sit.ranges[0][idx];
        weights[1][idx] = sit.ranges[1][idx];
    }

    // Fold payoffs scaled by this game's pot/stack.
    let fold_payoffs = compute_fold_payoffs(topo, term, sit.pot, sit.effective_stack);

    // Showdown outcomes: zeros for turn boundary nodes (leaf injection handles them).
    let showdown_outcomes = vec![0.0; topo.showdown_nodes.len() * num_hands * num_hands];

    SubgameSpec {
        initial_weights: [weights[0].to_vec(), weights[1].to_vec()],
        showdown_outcomes_p0: showdown_outcomes.clone(),
        showdown_outcomes_p1: showdown_outcomes,
        fold_payoffs_p0,
        fold_payoffs_p1,
    }
}
```

#### Step 4: Handle batched leaf CFVs

The current `update_leaf_cfvs` takes `[num_leaves * num_hands]` per player.
For batched games: `[B * num_leaves * num_hands]`. The kernel's leaf injection
already indexes by `bid` (from the stride loop fix): `leaf_cfv[bid * num_leaves * H + li * H + h]`.

Wait — check this. Read the current leaf injection kernel code to verify it uses `bid`.

If the current kernel reads `leaf_cfv[li * H + h]` (NOT per-batch), we need to change
it to `leaf_cfv[bid * num_leaves * H + li * H + h]` as part of this task.

#### Step 5: Handle batched boundary evaluation

The boundary evaluator currently processes one request at a time. For the batch, we have
K games each needing boundary evaluation. Two options:

a) Loop over games, call `evaluate_boundaries_batched` per game, collect results.
b) Batch all games' boundaries into one mega-batch.

Option (a) is simpler and the boundary eval is not the bottleneck (3s/sample, and with
batching the DCFR time drops from 20s to ~80ms, making boundary eval the new bottleneck
anyway). Start with (a) — can optimize later.

#### Step 6: Remove timing instrumentation

Remove the `[sample N]` timing eprintln from the orchestrator (it was diagnostic).
Replace with aggregate throughput logging (samples/sec every 1000 samples).

#### Step 7: Test — batched turn datagen produces records

```rust
#[cfg(feature = "gpu-turn-datagen")]
#[test]
fn gpu_turn_batched_pipeline_produces_records() {
    // Similar to existing test but with gpu_batch_size: Some(4)
    // to verify batching works.
}
```

#### Step 8: Commit

```bash
git add crates/cfvnet/src/datagen/domain/{pipeline.rs,game_tree.rs}
git commit -m "feat(cfvnet): batched turn datagen with canonical topology

Build one canonical turn tree at startup, create GpuBatchSolver once,
batch 256 games per kernel launch. Per-batch fold payoffs and leaf
CFVs. Eliminates per-game kernel compilation and utilizes all GPU SMs."
```

---

## Performance Expectations

With batch_size=256 on a modern GPU:

| Component | Before (1 game) | After (256 games) | Speedup |
|-----------|-----------------|-------------------|---------|
| Kernel compile | 3s/game | 3s total (once) | 256x |
| DCFR (300 iter) | 20s | ~100ms/game | ~200x |
| Boundary eval | 3s | 3s (sequential) | 1x |
| Tree build | 13ms | 13ms | 1x |
| **Total/game** | **~25s** | **~3.1s** | **~8x** |

The boundary eval becomes the bottleneck at 3s/game after batching. Future optimization:
batch all games' boundary requests into one ORT call (option b from Step 5).

## Key Files

| File | Changes |
|------|---------|
| `crates/gpu-range-solver/src/kernels.rs` | fold_payoffs[bid * num_folds + fi] |
| `crates/gpu-range-solver/src/batch.rs` | SubgameSpec + fold payoffs, batched upload |
| `crates/cfvnet/src/datagen/domain/pipeline.rs` | Batched orchestrator loop |
| `crates/cfvnet/src/datagen/domain/game_tree.rs` | Canonical tree builder |
