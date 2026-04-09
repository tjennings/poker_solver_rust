# GPU Turn Datagen Fixes Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use hex:executing-plans to implement this plan task-by-task.

**Goal:** Fix three issues in the GPU turn datagen pipeline: remove the zero-CFV fallback (require BoundaryNet model), implement reach-based boundary re-evaluation using average strategy from strategy_sum, and fix the 1024-hand CUDA thread limit with a stride loop.

**Architecture:** Task 1 adds `compute_reach_at_nodes()` in gpu-range-solver (pure CPU, factors out the forward-walk from `compute_evs_from_strategy_sum`). Task 2 modifies the CUDA kernel to use a stride loop for hand processing, lifting the 1024-thread limit. Task 3 wires reach-based re-evaluation into the orchestrator and removes the zero-CFV fallback.

**Tech Stack:** Rust, CUDA (cudarc/nvrtc), gpu-range-solver crate, cfvnet crate.

---

## Parallelization Map

```
Task 1 (reach extraction)  ──┐
                              ├── Task 3 (orchestrator fixes)
Task 2 (stride loop kernel) ─┘
```

Tasks 1 and 2 are independent. Task 3 depends on both.

---

### Task 1: Add `compute_reach_at_nodes()` to gpu-range-solver

Factor out the forward-walk reach computation from `compute_evs_from_strategy_sum()` into
a reusable function that returns per-player reach at specified node IDs.

**Files:**
- Modify: `crates/gpu-range-solver/src/batch.rs:497-633`
- Test: `crates/gpu-range-solver/src/batch.rs` (test module)

#### Step 1: Write the failing test

Add to the `#[cfg(test)] mod tests` block in `batch.rs`:

```rust
#[test]
fn compute_reach_at_nodes_matches_evs_function() {
    // Build a river game (same as existing tests).
    let (game, topo, term) = make_test_game();
    let num_hands = game.private_cards(0).len().max(game.private_cards(1).len());

    // Solve to get strategy_sum.
    let mut solver = GpuBatchSolver::new(&topo, &term, 1, num_hands, 100).unwrap();
    let spec = SubgameSpec::from_game(&game, &topo, &term, num_hands);
    let results = solver.solve_batch(&[spec.clone()]).unwrap();

    // Pick some internal nodes (non-root, non-terminal).
    let target_nodes: Vec<usize> = topo.level_nodes.iter()
        .flat_map(|level| level.iter())
        .filter(|&&n| topo.node_num_actions[n] > 0 && n > 0)
        .take(3)
        .copied()
        .collect();

    if target_nodes.is_empty() { return; }

    let reach = compute_reach_at_nodes(
        &topo,
        &results[0].strategy_sum,
        &spec.initial_weights,
        num_hands,
        &target_nodes,
    );

    // Verify: reach values are non-negative, and at least some are positive.
    for player in 0..2 {
        assert_eq!(reach[player].len(), target_nodes.len() * num_hands);
        assert!(reach[player].iter().all(|&r| r >= 0.0 && r.is_finite()));
        assert!(reach[player].iter().any(|&r| r > 0.0),
            "player {player} should have some positive reach");
    }
}
```

**Run:** `cargo test -p gpu-range-solver compute_reach_at_nodes`
**Expected:** FAIL — `compute_reach_at_nodes` not found.

#### Step 2: Implement `compute_reach_at_nodes()`

Add this public function to `batch.rs`, **above** `compute_evs_from_strategy_sum`:

```rust
/// Compute per-player reach probabilities at specific nodes using the average strategy.
///
/// Normalizes `strategy_sum` into an average strategy, then forward-walks the tree
/// to compute reach at each node. Returns `[oop_reach, ip_reach]` where each is
/// `[target_nodes.len() * num_hands]` — reach values at the target nodes for the
/// opponent of each player (i.e., oop_reach[i] is P1's reach at node i when P0 traverses).
///
/// This is the same forward walk as `compute_evs_from_strategy_sum`, but extracts
/// reach at interior nodes instead of computing EVs at terminals.
pub fn compute_reach_at_nodes(
    topo: &TreeTopology,
    strategy_sum: &[f32],
    initial_weights: &[Vec<f32>; 2],
    num_hands: usize,
    target_nodes: &[usize],
) -> [Vec<f32>; 2] {
    // ... (implementation below)
}
```

The implementation reuses the normalization and forward-walk logic from
`compute_evs_from_strategy_sum` lines 509-633. Specifically:

1. **Copy lines 509-572** (build sorted edges, normalize strategy_sum to avg_strategy,
   build node→edge mapping). This is identical setup code.

2. **Copy lines 576-633** (per-player forward walk computing reach at all nodes).
   Stop after the reach propagation — do NOT include the terminal evaluation or
   backward pass (lines 635+).

3. **After the reach walk, extract reach at target nodes:**
   ```rust
   let mut result = [
       vec![0.0f32; target_nodes.len() * h],
       vec![0.0f32; target_nodes.len() * h],
   ];
   for player in 0..2 {
       // reach was computed for this player's traversal
       for (ti, &node_id) in target_nodes.iter().enumerate() {
           for hand in 0..h {
               result[player][ti * h + hand] = reach[node_id * h + hand];
           }
       }
   }
   ```

   **Important**: In the forward walk (copied from `compute_evs_from_strategy_sum`),
   when `player=0` traverses, `reach` holds P1's (opponent's) reach. When `player=1`
   traverses, `reach` holds P0's reach. So `result[0]` = P1 reach (opponent for P0
   traversal), `result[1]` = P0 reach (opponent for P1 traversal).

   For the BoundaryNet, we need OOP range and IP range at boundaries. OOP=P0, IP=P1.
   So: `oop_reach_at_boundary = result[1]` (P0's reach from P1's traversal) and
   `ip_reach_at_boundary = result[0]` (P1's reach from P0's traversal).

4. **DRY refactor (optional but recommended)**: Extract the shared setup code (lines
   509-572) into a private helper `fn build_avg_strategy_and_edge_map(...)` that both
   `compute_reach_at_nodes` and `compute_evs_from_strategy_sum` call. This avoids
   duplicating ~60 lines. If this is too risky, just duplicate — correctness over DRY.

**Run:** `cargo test -p gpu-range-solver compute_reach_at_nodes`
**Expected:** PASS.

#### Step 3: Add `compute_reach_at_nodes` to public exports

In `crates/gpu-range-solver/src/lib.rs`, add to the `pub use batch::` line:

```rust
pub use batch::{compute_evs_from_strategy_sum, compute_reach_at_nodes, GpuBatchSolver, ...};
```

**Run:** `cargo test -p gpu-range-solver` — all tests pass (88 existing + 1 new).

#### Step 4: Commit

```bash
git add crates/gpu-range-solver/src/batch.rs crates/gpu-range-solver/src/lib.rs
git commit -m "feat(gpu-solver): add compute_reach_at_nodes for boundary re-evaluation

Factor out forward-walk reach computation from compute_evs_from_strategy_sum.
Returns per-player reach at specified nodes using the average strategy from
strategy_sum. Used by the turn datagen orchestrator for reach-based boundary
re-evaluation."
```

---

### Task 2: Fix 1024-hand CUDA thread limit with stride loop

Change the hand-parallel kernel from one-thread-per-hand to a stride loop pattern,
allowing blocks with fewer threads than hands.

**Files:**
- Modify: `crates/gpu-range-solver/src/kernels.rs` (HAND_PARALLEL_KERNEL_SOURCE)
- Modify: `crates/gpu-range-solver/src/batch.rs` (launch config)
- Test: `crates/gpu-range-solver/src/batch.rs`

#### Step 1: Write the failing test

```rust
#[test]
fn solve_batch_handles_more_than_1024_hands() {
    // Create a game where num_hands > 1024 by using a 4-card board (turn).
    // C(48,2) = 1128 hands for a turn game.
    let (game, topo, term) = make_turn_test_game(); // use existing make_turn_game or similar
    let num_hands = game.private_cards(0).len().max(game.private_cards(1).len());

    // This should be > 1024 for a turn game.
    if num_hands <= 1024 {
        // Skip if this particular game has fewer hands.
        return;
    }

    let mut solver = GpuBatchSolver::new(&topo, &term, 1, num_hands, 50).unwrap();
    let spec = SubgameSpec::from_game(&game, &topo, &term, num_hands);
    let results = solver.solve_batch(&[spec]).unwrap();

    // Verify we get a result with non-zero strategy_sum.
    assert!(!results[0].strategy_sum.is_empty());
    assert!(results[0].strategy_sum.iter().any(|&v| v != 0.0));
}
```

**Run:** `cargo test -p gpu-range-solver solve_batch_handles_more_than_1024`
**Expected:** FAIL — CUDA launch error (block size > 1024).

#### Step 2: Modify launch config to cap block size

In `batch.rs`, in `run_iterations()` and the `solve_batch` wrapper (which calls
`run_iterations` internally), change the launch config:

```rust
// OLD:
let cfg = LaunchConfig {
    grid_dim: (batch_size as u32, 1, 1),
    block_dim: (self.num_hands as u32, 1, 1),
    shared_mem_bytes: self.shared_mem_bytes,
};

// NEW:
let block_threads = (self.num_hands as u32).min(1024);
let cfg = LaunchConfig {
    grid_dim: (batch_size as u32, 1, 1),
    block_dim: (block_threads, 1, 1),
    shared_mem_bytes: self.shared_mem_bytes,
};
```

The kernel already receives `H` (num_hands) as a parameter, so it knows the real hand count.

#### Step 3: Add stride loops to the CUDA kernel

In `kernels.rs`, the hand-parallel kernel uses `tid` (= `threadIdx.x`) to identify
which hand to process. Every place that does `if (tid < H) { ... per-hand work ... }`
needs to become a stride loop:

```cuda
// OLD:
if (tid < H) {
    for (int n = 0; n < N; n++) {
        reach[bid * NH + n * H + tid] = 0.0f;
        cfv[bid * NH + n * H + tid] = 0.0f;
    }
}

// NEW:
for (int h = tid; h < H; h += blockDim.x) {
    for (int n = 0; n < N; n++) {
        reach[bid * NH + n * H + h] = 0.0f;
        cfv[bid * NH + n * H + h] = 0.0f;
    }
}
```

There are **5 locations** to update (from the grep for `tid < H`):

1. **Line ~431** (DCFR discount): `regrets[idx]` and `strategy_sum[idx]`
2. **Line ~445** (zero reach/cfv): zero all nodes
3. **Line ~453** (root reach): set from initial_weights
4. **Line ~468** (forward pass): regret match + reach propagation
5. **Line ~573** (leaf injection): overwrite cfv at leaf nodes

For each, replace `if (tid < H) { ... use tid ... }` with
`for (int h = tid; h < H; h += blockDim.x) { ... use h instead of tid ... }`.

**Critical**: The fold eval section uses `__syncthreads()` and shared memory
(`s_card_reach[52]`). This section operates on cards (0..52), not hands, so it
does NOT need a stride loop change — it already uses `tid < 52` which is always
within bounds. Verify this does not need modification.

**Also update the cooperative load** of topology into shared memory (lines ~392-403)
— this already uses `for (int i = tid; i < E; i += blockDim.x)` stride pattern,
so it handles arbitrary block sizes correctly. No change needed here.

#### Step 4: Run tests

**Run:** `cargo test -p gpu-range-solver`
**Expected:** All existing tests pass (stride loop is transparent when H <= 1024),
plus the new >1024 test passes.

#### Step 5: Commit

```bash
git add crates/gpu-range-solver/src/kernels.rs crates/gpu-range-solver/src/batch.rs
git commit -m "fix(gpu-solver): support >1024 hands with kernel stride loop

Cap CUDA block size at 1024 threads. Kernel uses stride loop pattern
(for h = tid; h < H; h += blockDim.x) to process all hands even when
num_hands > blockDim.x. Enables GPU turn solving for turn games with
C(48,2) = 1128 hands."
```

---

### Task 3: Reach-based boundary re-evaluation + remove zero-CFV fallback

Wire `compute_reach_at_nodes` into the orchestrator's re-evaluation loop. Remove
the zero-CFV fallback path — require a BoundaryNet model for turn GPU datagen.

**Files:**
- Modify: `crates/cfvnet/src/datagen/domain/pipeline.rs` (run_gpu_turn)
- Modify: `crates/cfvnet/src/datagen/gpu_boundary_eval.rs` (adjust if needed)
- Test: `crates/cfvnet/src/datagen/domain/pipeline.rs`

**Depends on:** Tasks 1 and 2.

#### Step 1: Remove zero-CFV fallback

In `pipeline.rs` `run_gpu_turn()`, find the section that conditionally loads the
boundary evaluator:

```rust
#[cfg(feature = "gpu-turn-datagen")]
let boundary_evaluator = ...;
#[cfg(not(feature = "gpu-turn-datagen"))]
let boundary_evaluator: Option<...> = None;
```

Replace with: **always require `gpu-turn-datagen` feature and a model path**.
The entire `run_gpu_turn()` method should be gated on `#[cfg(feature = "gpu-turn-datagen")]`.
If `river_model_path` is not provided, return an error:

```rust
#[cfg(feature = "gpu-turn-datagen")]
fn run_gpu_turn(config: &CfvnetConfig, output_path: &Path) -> Result<(), String> {
    let model_path = config.game.river_model_path.as_deref()
        .ok_or("river_model_path is required for GPU turn datagen")?;
    let evaluator = GpuBoundaryEvaluator::load(Path::new(model_path))?;
    // ... no Option, no None, no fallback
}
```

Update the dispatch in `run_gpu()`:
```rust
if board_size == 4 {
    #[cfg(feature = "gpu-turn-datagen")]
    return Self::run_gpu_turn(config, output_path);
    #[cfg(not(feature = "gpu-turn-datagen"))]
    return Err("GPU turn datagen requires --features gpu-turn-datagen".into());
}
```

Remove all `#[cfg(feature = "gpu-turn-datagen")]` / `#[cfg(not(...))]` conditionals
inside `run_gpu_turn` — the entire function is now behind the feature gate.

#### Step 2: Replace root-range evaluation with reach-based evaluation

Replace the `evaluate_and_upload_boundaries()` helper. Instead of using `sit.ranges`
(root ranges), it should accept `strategy_sum` and compute reach at boundary nodes.

New signature:

```rust
fn evaluate_and_upload_boundaries_from_reach(
    evaluator: &GpuBoundaryEvaluator,
    solver: &mut GpuBatchSolver,
    topo: &TreeTopology,
    strategy_sum: &[f32],
    initial_weights: &[Vec<f32>; 2],
    num_hands: usize,
    board: &[u8; 4],
    hand_cards: &[(u8, u8)],
    boundary_node_ids: &[usize],
) -> Result<(), String> {
    use gpu_range_solver::compute_reach_at_nodes;

    let reach = compute_reach_at_nodes(
        topo, strategy_sum, initial_weights, num_hands, boundary_node_ids,
    );

    // reach[0] = IP (P1) reach at boundaries (from P0 traversal)
    // reach[1] = OOP (P0) reach at boundaries (from P1 traversal)
    // BoundaryNet expects: oop_reach (P0) and ip_reach (P1) in 1326-combo space.
    let num_boundaries = boundary_node_ids.len();
    let mut oop_reach_1326 = vec![0.0f32; num_boundaries * NUM_COMBOS];
    let mut ip_reach_1326 = vec![0.0f32; num_boundaries * NUM_COMBOS];

    for (bi, _node_id) in boundary_node_ids.iter().enumerate() {
        for (hi, &(c0, c1)) in hand_cards.iter().enumerate() {
            let combo_idx = range_solver::card::card_pair_to_index(c0, c1);
            // OOP (P0) reach = from P1's traversal (index 1)
            oop_reach_1326[bi * NUM_COMBOS + combo_idx] = reach[1][bi * num_hands + hi];
            // IP (P1) reach = from P0's traversal (index 0)
            ip_reach_1326[bi * NUM_COMBOS + combo_idx] = reach[0][bi * num_hands + hi];
        }
    }

    let request = BoundaryEvalRequest {
        board: *board,
        pot: /* from context */,
        effective_stack: /* from context */,
        oop_reach: oop_reach_1326,
        ip_reach: ip_reach_1326,
        num_boundaries,
    };

    let results = evaluate_boundaries_batched(evaluator, &[request], hand_cards)?;
    solver.update_leaf_cfvs(&results[0].leaf_cfv_p0, &results[0].leaf_cfv_p1)
        .map_err(|e| format!("upload leaf cfvs: {e}"))
}
```

#### Step 3: Update the solve loop to use reach-based re-evaluation

Replace the commented-out TODO block (lines 640-649) with actual re-evaluation.
Also need to call `solver.extract_results()` mid-solve to get `strategy_sum`:

```rust
// Initial boundary evaluation using root ranges (strategy_sum is zero).
let initial_strategy_sum = vec![0.0f32; topo.num_edges * num_hands];
evaluate_and_upload_boundaries_from_reach(
    &evaluator, &mut solver, &turn_topo, &initial_strategy_sum,
    &spec.initial_weights, num_hands, &board_4, &hand_cards,
    &boundary_node_ids,
)?;

let mut iter = 0u32;
while iter < max_iterations {
    let end = (iter + leaf_eval_interval).min(max_iterations);
    solver.run_iterations(iter, end)
        .map_err(|e| format!("run_iterations: {e}"))?;
    iter = end;

    if iter < max_iterations && leaf_eval_interval > 0 {
        // Download strategy_sum for reach computation.
        let mid_results = solver.extract_results()
            .map_err(|e| format!("mid-solve extract: {e}"))?;

        evaluate_and_upload_boundaries_from_reach(
            &evaluator, &mut solver, &turn_topo, &mid_results[0].strategy_sum,
            &spec.initial_weights, num_hands, &board_4, &hand_cards,
            &boundary_node_ids,
        )?;
    }
}
```

**Important**: `extract_results()` downloads `strategy_sum` but does NOT reset
the solver state. The GPU regrets/strategy_sum buffers are untouched. The next
`run_iterations()` call continues from where it left off. Verify this by checking
that `extract_results` only does `clone_dtoh` (read-only download), not any writes.

#### Step 4: Update the test

The existing `gpu_turn_pipeline_produces_records` test used the zero-CFV fallback.
Now that it's removed, the test needs the `gpu-turn-datagen` feature and a model.
Since we can't guarantee a model in CI, gate the test:

```rust
#[cfg(feature = "gpu-turn-datagen")]
#[test]
fn gpu_turn_pipeline_produces_records() {
    let model_path = "../../local_data/models/cfvnet_river_py_v2/model.onnx";
    if !std::path::Path::new(model_path).exists() {
        eprintln!("Skipping: ONNX model not found at {model_path}");
        return;
    }
    // ... rest of test with river_model_path set to model_path
}
```

#### Step 5: Run full test suite

```bash
cargo test -p gpu-range-solver   # 89+ tests pass
cargo test -p cfvnet --lib       # 210 tests pass
```

#### Step 6: Validate end-to-end

```bash
cargo run -p cfvnet --release --features gpu-turn-datagen -- generate \
  -c sample_configurations/turn_gpu_datagen.yaml \
  -o /tmp/turn_gpu_reach_test \
  --num-samples 5
```

**Expected:** Completes, writes 10 records. Should see boundary re-evaluation
happening every `leaf_eval_interval` iterations in logs.

#### Step 7: Commit

```bash
git add crates/cfvnet/src/datagen/domain/pipeline.rs crates/cfvnet/src/datagen/gpu_boundary_eval.rs
git commit -m "feat(cfvnet): reach-based boundary re-evaluation, remove zero-CFV fallback

Use compute_reach_at_nodes to extract average-strategy reach at boundary
nodes mid-solve. Feed actual reach (not root ranges) to BoundaryNet for
re-evaluation. Remove the zero-CFV fallback — river_model_path is now
required for GPU turn datagen."
```

---

## Key Reference Files

| File | What to read |
|------|-------------|
| `crates/gpu-range-solver/src/batch.rs:497-633` | `compute_evs_from_strategy_sum` — forward walk to factor out |
| `crates/gpu-range-solver/src/batch.rs:467-491` | `extract_results` — read-only download of strategy_sum |
| `crates/gpu-range-solver/src/kernels.rs:362-650` | Hand-parallel kernel — stride loop targets |
| `crates/cfvnet/src/datagen/domain/pipeline.rs:560-690` | `run_gpu_turn` orchestrator |
| `crates/cfvnet/src/datagen/domain/pipeline.rs:828-870` | `evaluate_and_upload_boundaries` — replace with reach-based |
| `crates/cfvnet/src/datagen/gpu_boundary_eval.rs:103` | `evaluate_boundaries_batched` — already accepts per-boundary reach |

## Implementation Notes

1. **Reach player mapping**: `compute_reach_at_nodes` returns `[p0_traversal_reach, p1_traversal_reach]`. In each traversal, reach holds the OPPONENT's reach. So `result[0]` = P1 (IP) reach, `result[1]` = P0 (OOP) reach. Map accordingly when building `BoundaryEvalRequest`.

2. **Initial evaluation**: Before any iterations, `strategy_sum` is all zeros. `compute_reach_at_nodes` will normalize to uniform strategy (1/n_actions), producing reach = initial_weights propagated through uniform play. This is a reasonable starting point.

3. **`extract_results()` is read-only**: It calls `clone_dtoh` on `state.strategy_sum` — a GPU→CPU copy that does not modify GPU state. Safe to call mid-solve.

4. **Stride loop correctness**: The fold eval section uses `s_card_reach[52]` shared memory with `__syncthreads()`. When `blockDim.x < H`, some threads handle multiple hands via the stride loop. The `__syncthreads()` still synchronizes all threads in the block. The fold eval section iterates over cards (0..52), not hands, and uses `threadIdx.x < 52` — this is unaffected by the stride loop change since `blockDim.x` will always be >= 52 (we'd never have fewer than 52 threads in a block for a poker game).
