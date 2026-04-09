# GPU Turn Datagen Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use hex:executing-plans to implement this plan task-by-task.

**Goal:** Generate turn CFV training data on GPU using DCFR with periodic BoundaryNet re-evaluation at river boundaries via ORT TensorRT execution provider.

**Architecture:** Extend the existing `DomainPipeline::run_gpu()` in `cfvnet` to support turn trees (board_size=4). Refactor `GpuBatchSolver` for incremental solving (run N iterations → re-evaluate boundaries → run N more). BoundaryNet inference uses the existing `ort` crate with TensorRT EP. Input construction and output reduction happen on CPU; the hot DCFR loop stays on GPU.

**Tech Stack:** Rust, cudarc (CUDA), ort 2.0.0-rc.9 (ONNX Runtime + TensorRT EP), gpu-range-solver crate, cfvnet crate.

---

## Parallelization Map

```
Task 1 (incremental solver) ──┐
                               ├── Task 3 (orchestrator) ── Task 4 (integration test)
Task 2 (boundary evaluator) ──┘
```

Tasks 1 and 2 are independent and can be implemented in parallel.

---

### Task 1: Incremental Solving API on GpuBatchSolver

Refactor `GpuBatchSolver` to support running DCFR iterations in chunks, with the ability to
download reach, update leaf CFVs, and resume. The existing `solve_batch()` stays intact as a
convenience wrapper.

**Files:**
- Modify: `crates/gpu-range-solver/src/kernels.rs:412` (CUDA kernel iteration loop)
- Modify: `crates/gpu-range-solver/src/batch.rs:68-354` (GpuBatchSolver)
- Modify: `crates/gpu-range-solver/src/gpu.rs` (GpuHandParallelState visibility)
- Test: `crates/gpu-range-solver/src/batch.rs` (add tests in a `#[cfg(test)]` module)

#### Step 1: Add `start_iteration` parameter to CUDA kernel

In `crates/gpu-range-solver/src/kernels.rs`, the hand-parallel kernel `cfr_solve` at line 308
has parameters ending with:

```cuda
int max_depth,
int max_iterations,
int num_folds,
```

**Change to:**

```cuda
int max_depth,
int start_iteration,
int end_iteration,
int num_folds,
```

And the iteration loop at line 412:

```cuda
for (int iter = 0; iter < max_iterations; iter++) {
```

**Change to:**

```cuda
for (int iter = start_iteration; iter < end_iteration; iter++) {
```

The DCFR discount formula at lines 417-425 already uses `iter` directly, so it will
correctly use the absolute iteration number. No changes needed there.

**Also update the mega-kernel** (`CFR_MEGA_KERNEL_SOURCE`) with the same parameter change
for consistency, even though the orchestrator uses the hand-parallel kernel.

#### Step 2: Update `solve_batch()` to pass new kernel params

In `crates/gpu-range-solver/src/batch.rs:284`:

```rust
let max_iter_i32 = self.max_iterations as i32;
```

**Change to:**

```rust
let start_iter_i32 = 0i32;
let end_iter_i32 = self.max_iterations as i32;
```

And update the kernel arg push (line 329) from `&max_iter_i32` to `&start_iter_i32`
followed by `&end_iter_i32`.

**Run test:** `cargo test -p gpu-range-solver` — existing tests should still pass since
`solve_batch` now passes `(0, max_iterations)` which is equivalent to the old behavior.

#### Step 3: Persist solver state across calls

Currently `GpuHandParallelState` is created inside `solve_batch()` and dropped when it
returns. We need to keep it alive for incremental solving.

Add a field to `GpuBatchSolver`:

```rust
// In struct GpuBatchSolver:
state: Option<GpuHandParallelState>,
current_batch_size: usize,
```

Initialize as `None` and `0` in `new()`.

#### Step 4: Add `prepare_batch()` method

Extract the setup portion of `solve_batch()` (lines 226-270) into a new method:

```rust
/// Upload per-subgame data and initialize solver state. Call before `run_iterations()`.
pub fn prepare_batch(&mut self, specs: &[SubgameSpec]) -> Result<(), Box<dyn std::error::Error>> {
    // Same validation and upload logic as solve_batch lines 226-270
    // Store state and d_initial_weights, d_showdown_outcomes on self
    // (or re-upload each time — simpler)
    let batch_size = specs.len();
    assert!(batch_size <= self.max_batch);
    
    let state = GpuHandParallelState::new(&self.stream, batch_size, self.num_nodes, self.num_edges, self.num_hands)?;
    self.state = Some(state);
    self.current_batch_size = batch_size;
    
    // Upload initial_weights, showdown_outcomes (same as solve_batch)
    // Store d_initial_weights, d_showdown_outcomes_p0/p1 on self
    Ok(())
}
```

Note: `d_initial_weights` and `d_showdown_outcomes_p0/p1` also need to be stored on the
struct (or re-uploaded each iteration run). Simplest: store as `Option<CudaSlice<f32>>` fields.

#### Step 5: Add `run_iterations()` method

```rust
/// Run DCFR iterations [start, end) on the prepared batch. State persists between calls.
pub fn run_iterations(&mut self, start: u32, end: u32) -> Result<(), Box<dyn std::error::Error>> {
    let state = self.state.as_mut().expect("call prepare_batch first");
    let batch_size = self.current_batch_size;
    
    let cfg = LaunchConfig {
        grid_dim: (batch_size as u32, 1, 1),
        block_dim: (self.num_hands as u32, 1, 1),
        shared_mem_bytes: self.shared_mem_bytes,
    };
    
    // Same kernel launch as solve_batch, but with start/end iteration
    let start_i32 = start as i32;
    let end_i32 = end as i32;
    // ... push args, launch kernel ...
    
    self.stream.synchronize()?;
    Ok(())
}
```

#### Step 6: Add `download_reach()` and `update_leaf_cfvs()` methods

```rust
/// Download reach probabilities from GPU. Returns [B * N * H] flat array.
pub fn download_reach(&self) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let state = self.state.as_ref().expect("call prepare_batch first");
    Ok(self.stream.clone_dtoh(&state.reach)?)
}

/// Upload new leaf CFVs for boundary re-evaluation.
/// p0 and p1 are [num_leaves * H] each.
pub fn update_leaf_cfvs(&mut self, p0: &[f32], p1: &[f32]) -> Result<(), Box<dyn std::error::Error>> {
    self.stream.clone_htod_copy_into(p0, &mut self.d_leaf_cfv_p0)?;
    self.stream.clone_htod_copy_into(p1, &mut self.d_leaf_cfv_p1)?;
    Ok(())
}
```

#### Step 7: Add `set_leaf_injection()` method

For turn solving, the leaf node IDs and depths must be set once after construction:

```rust
/// Configure leaf injection nodes for turn solving.
/// node_ids and depths come from ChanceDecomposition.turn_leaf_node_ids.
pub fn set_leaf_injection(&mut self, node_ids: &[i32], depths: &[i32]) -> Result<(), Box<dyn std::error::Error>> {
    self.d_leaf_node_ids = self.stream.clone_htod(node_ids)?;
    self.d_leaf_depths = self.stream.clone_htod(depths)?;
    self.num_leaves = node_ids.len();
    // Pre-allocate leaf CFV buffers: [num_leaves * num_hands] per player
    let size = node_ids.len() * self.num_hands;
    self.d_leaf_cfv_p0 = self.stream.clone_htod(&vec![0.0f32; size])?;
    self.d_leaf_cfv_p1 = self.stream.clone_htod(&vec![0.0f32; size])?;
    Ok(())
}
```

Add `num_leaves: usize` field to struct (default 0). Update `run_iterations()` to pass
`self.num_leaves as i32` instead of hardcoded `0i32`.

#### Step 8: Add `extract_results()` method

```rust
/// Download strategy sums and return per-subgame results.
pub fn extract_results(&self) -> Result<Vec<SubgameResult>, Box<dyn std::error::Error>> {
    let state = self.state.as_ref().expect("call prepare_batch first");
    let strategy_sum_all: Vec<f32> = self.stream.clone_dtoh(&state.strategy_sum)?;
    let eh = self.num_edges * self.num_hands;
    Ok((0..self.current_batch_size)
        .map(|b| SubgameResult {
            strategy_sum: strategy_sum_all[b * eh..(b + 1) * eh].to_vec(),
        })
        .collect())
}
```

#### Step 9: Refactor `solve_batch()` to use new methods

```rust
pub fn solve_batch(&mut self, specs: &[SubgameSpec]) -> Result<Vec<SubgameResult>, Box<dyn std::error::Error>> {
    if specs.is_empty() { return Ok(Vec::new()); }
    self.prepare_batch(specs)?;
    self.run_iterations(0, self.max_iterations)?;
    self.extract_results()
}
```

#### Step 10: Write test — incremental matches batch

```rust
#[cfg(test)]
mod tests {
    use super::*;
    // Use the existing make_river_game() from lib.rs tests or build a test helper.

    #[test]
    fn incremental_matches_batch() {
        // Build a river game and topology
        let (game, topo, term) = setup_test_game(); // helper
        let num_hands = ...;
        let max_iter = 150u32;
        
        // Batch solve (baseline)
        let mut solver1 = GpuBatchSolver::new(&topo, &term, 1, num_hands, max_iter).unwrap();
        let spec = SubgameSpec::from_game(&game, &topo, &term, num_hands);
        let batch_result = solver1.solve_batch(&[spec.clone()]).unwrap();
        
        // Incremental solve (50+50+50)
        let mut solver2 = GpuBatchSolver::new(&topo, &term, 1, num_hands, max_iter).unwrap();
        solver2.prepare_batch(&[spec]).unwrap();
        solver2.run_iterations(0, 50).unwrap();
        solver2.run_iterations(50, 100).unwrap();
        solver2.run_iterations(100, 150).unwrap();
        let incr_result = solver2.extract_results().unwrap();
        
        // Compare strategy_sum within tolerance
        for (a, b) in batch_result[0].strategy_sum.iter().zip(&incr_result[0].strategy_sum) {
            assert!((a - b).abs() < 1e-4, "mismatch: batch={a} incr={b}");
        }
    }
}
```

**Run:** `cargo test -p gpu-range-solver incremental_matches_batch`

**Expected:** PASS — both paths produce identical results.

#### Step 11: Write test — leaf injection with incremental solve

```rust
#[test]
fn turn_leaf_injection_incremental() {
    // Build a turn game with boundaries
    let (game, topo, term) = setup_turn_game(); // 4-card board
    let decomp = decompose_at_chance(&topo);
    let num_hands = ...;
    
    let mut solver = GpuBatchSolver::new(&decomp.turn_topo, &term_turn, 1, num_hands, 100).unwrap();
    
    // Set leaf injection from decomposition
    let leaf_ids: Vec<i32> = decomp.turn_leaf_node_ids.iter().map(|&x| x as i32).collect();
    let leaf_depths: Vec<i32> = leaf_ids.iter().map(|&id| decomp.turn_topo.node_depth[id as usize] as i32).collect();
    solver.set_leaf_injection(&leaf_ids, &leaf_depths).unwrap();
    
    // Set dummy leaf CFVs (zeros)
    let size = leaf_ids.len() * num_hands;
    solver.update_leaf_cfvs(&vec![0.0; size], &vec![0.0; size]).unwrap();
    
    let spec = SubgameSpec::from_game_turn(...); // build from turn game
    solver.prepare_batch(&[spec]).unwrap();
    solver.run_iterations(0, 50).unwrap();
    
    // Download reach — should have valid values
    let reach = solver.download_reach().unwrap();
    assert!(reach.iter().any(|&r| r > 0.0));
    
    // Update leaf CFVs (simulate boundary re-eval)
    solver.update_leaf_cfvs(&vec![0.1; size], &vec![0.1; size]).unwrap();
    solver.run_iterations(50, 100).unwrap();
    
    let results = solver.extract_results().unwrap();
    assert!(!results[0].strategy_sum.is_empty());
}
```

#### Step 12: Commit

```bash
git add crates/gpu-range-solver/src/kernels.rs crates/gpu-range-solver/src/batch.rs crates/gpu-range-solver/src/gpu.rs
git commit -m "feat(gpu-solver): add incremental solving API for turn datagen

Split solve_batch into prepare_batch/run_iterations/extract_results.
Add set_leaf_injection, update_leaf_cfvs, download_reach methods.
CUDA kernel now accepts start/end iteration for partial runs."
```

---

### Task 2: BoundaryNet GPU Evaluator

Create the BoundaryNet evaluation pipeline: ORT session with TensorRT EP, batched inference,
input construction and output reduction on CPU.

**Files:**
- Modify: `crates/cfvnet/Cargo.toml` (add ort CUDA/TRT features)
- Create: `crates/cfvnet/src/datagen/gpu_boundary_eval.rs`
- Modify: `crates/cfvnet/src/datagen/mod.rs` (add module)
- Test: in `gpu_boundary_eval.rs`

#### Step 1: Update Cargo.toml features

In `crates/cfvnet/Cargo.toml`, update the `ort` dependency to include CUDA and TRT features.
Create a combined feature flag:

```toml
ort = { version = "=2.0.0-rc.9", optional = true, default-features = false, features = ["ndarray", "half", "copy-dylibs"] }

[features]
onnx = ["ort"]
onnx-gpu = ["ort", "ort/cuda", "ort/tensorrt"]
gpu-turn-datagen = ["gpu-datagen", "onnx-gpu"]
```

This keeps `--features onnx` unchanged (CPU only) and adds `--features gpu-turn-datagen`
for the full GPU pipeline.

**Run:** `cargo check -p cfvnet --features gpu-turn-datagen` — should compile.

#### Step 2: Create GpuBoundaryEvaluator struct

Create `crates/cfvnet/src/datagen/gpu_boundary_eval.rs`:

```rust
#[cfg(feature = "gpu-turn-datagen")]
use ort::session::{Session, builder::GraphOptimizationLevel};

use crate::eval::boundary_evaluator::encode_boundary_inference_input;

/// BoundaryNet evaluator using ORT with TensorRT execution provider.
/// Performs batched inference for boundary re-evaluation during turn DCFR.
pub struct GpuBoundaryEvaluator {
    session: Session,
}

impl GpuBoundaryEvaluator {
    /// Load ONNX model with TensorRT EP (falls back to CUDA EP, then CPU).
    pub fn load(model_path: &std::path::Path) -> Result<Self, String> {
        let session = Session::builder()
            .map_err(|e| format!("ORT session builder: {e}"))?
            .with_execution_providers([
                ort::execution_providers::TensorRTExecutionProvider::default().build(),
                ort::execution_providers::CUDAExecutionProvider::default().build(),
            ])
            .map_err(|e| format!("ORT execution providers: {e}"))?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| format!("ORT optimization: {e}"))?
            .commit_from_file(model_path)
            .map_err(|e| format!("ORT load model: {e}"))?;
        
        Ok(Self { session })
    }

    /// Batched forward pass. input: [N, INPUT_SIZE], output: [N, 1326].
    pub fn infer_batch(&self, input: &[f32], num_rows: usize) -> Result<Vec<f32>, String> {
        use ort::value::Tensor;
        let input_size = crate::eval::boundary_evaluator::INPUT_SIZE;
        assert_eq!(input.len(), num_rows * input_size);
        
        let tensor = Tensor::from_array(([num_rows as i64, input_size as i64], input.to_vec()))
            .map_err(|e| format!("tensor creation: {e}"))?;
        let outputs = self.session
            .run(ort::inputs![tensor].map_err(|e| format!("inputs: {e}"))?)
            .map_err(|e| format!("session run: {e}"))?;
        let view = outputs[0]
            .try_extract_tensor::<f32>()
            .map_err(|e| format!("extract output: {e}"))?;
        Ok(view.iter().copied().collect())
    }
}
```

#### Step 3: Add boundary evaluation function

This function takes reach at boundary nodes + situation info, builds all BoundaryNet inputs
(48 rivers × 2 players × N boundaries × B subgames), runs inference, and reduces to leaf CFVs.

```rust
/// Boundary evaluation request for one subgame.
pub struct BoundaryEvalRequest {
    pub board: [u8; 4],           // turn board cards
    pub pot: f32,
    pub effective_stack: f32,
    pub oop_reach: Vec<f32>,      // [num_boundaries * 1326] reach at boundary nodes
    pub ip_reach: Vec<f32>,       // [num_boundaries * 1326]
    pub num_boundaries: usize,
}

/// Result: leaf CFVs ready for upload to GPU solver.
pub struct BoundaryEvalResult {
    pub leaf_cfv_p0: Vec<f32>,    // [num_boundaries * num_hands]
    pub leaf_cfv_p1: Vec<f32>,    // [num_boundaries * num_hands]
}

const OUTPUT_SIZE: usize = 1326;

/// Evaluate all boundary nodes for a batch of subgames.
pub fn evaluate_boundaries_batched(
    evaluator: &GpuBoundaryEvaluator,
    requests: &[BoundaryEvalRequest],
    num_hands: usize,
    hand_cards: &[(u8, u8)],      // private cards for indexing (num_hands entries)
) -> Result<Vec<BoundaryEvalResult>, String> {
    // 1. Identify valid river cards per board
    // 2. Build all inputs: for each request, each boundary, each player, each river
    // 3. Run batched inference
    // 4. Reduce: reach-weighted average over rivers, denormalize, map to hand indices
    // ... (implementation details below)
}
```

**Input construction (CPU, parallelized with rayon):**

For each request (subgame), each boundary node, each player (0, 1), each of ~48 valid rivers:
1. Build 5-card board = `request.board` + `river_card`
2. Get the player's range from `oop_reach`/`ip_reach` at this boundary, zero out hands
   conflicting with the river card
3. Call `encode_boundary_inference_input(oop_range, ip_range, board5, pot, stack, player)`
4. Append to flat input buffer

**Output reduction (CPU):**

For each boundary node, for each player, for each hand h:
```
weight_sum = 0
cfv_sum = 0
for each valid river r:
    if hand h does not conflict with r:
        opp_weight = sum of opponent reach for hands not conflicting with r
        cfv_sum += opp_weight * net_output[row_for(b,p,r)][combo_index(h)]
        weight_sum += opp_weight
leaf_cfv[h] = (cfv_sum / weight_sum) * (pot + effective_stack)
```

Map from 1326-combo space back to game's hand index space using `hand_cards`.

#### Step 4: Register module

In `crates/cfvnet/src/datagen/mod.rs`, add:

```rust
#[cfg(feature = "gpu-turn-datagen")]
pub mod gpu_boundary_eval;
```

#### Step 5: Write test — evaluator loads and runs

```rust
#[cfg(test)]
#[cfg(feature = "gpu-turn-datagen")]
mod tests {
    use super::*;

    #[test]
    fn evaluator_loads_onnx_model() {
        // Skip if no model available
        let model_path = std::path::Path::new("../../local_data/cfvnet/river/v2/best.onnx");
        if !model_path.exists() { return; }
        
        let eval = GpuBoundaryEvaluator::load(model_path).unwrap();
        
        // Single-row inference sanity check
        let input = vec![0.0f32; crate::eval::boundary_evaluator::INPUT_SIZE];
        let output = eval.infer_batch(&input, 1).unwrap();
        assert_eq!(output.len(), 1326);
        assert!(output.iter().all(|v| v.is_finite()));
    }
}
```

**Run:** `cargo test -p cfvnet --features gpu-turn-datagen evaluator_loads`

#### Step 6: Write test — boundary evaluation produces valid results

```rust
#[test]
fn boundary_eval_produces_valid_cfvs() {
    let model_path = std::path::Path::new("../../local_data/cfvnet/river/v2/best.onnx");
    if !model_path.exists() { return; }
    
    let eval = GpuBoundaryEvaluator::load(model_path).unwrap();
    
    // Build a simple request: 1 subgame, 1 boundary, uniform ranges
    let request = BoundaryEvalRequest {
        board: [0, 5, 10, 15],  // Arbitrary non-conflicting cards
        pot: 100.0,
        effective_stack: 200.0,
        oop_reach: vec![1.0 / 1326.0; 1326],
        ip_reach: vec![1.0 / 1326.0; 1326],
        num_boundaries: 1,
    };
    
    let hand_cards: Vec<(u8, u8)> = ...; // build from board
    let results = evaluate_boundaries_batched(&eval, &[request], hand_cards.len(), &hand_cards).unwrap();
    
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].leaf_cfv_p0.len(), hand_cards.len());
    assert_eq!(results[0].leaf_cfv_p1.len(), hand_cards.len());
    // CFVs should be finite
    assert!(results[0].leaf_cfv_p0.iter().all(|v| v.is_finite()));
}
```

#### Step 7: Commit

```bash
git add crates/cfvnet/Cargo.toml crates/cfvnet/src/datagen/gpu_boundary_eval.rs crates/cfvnet/src/datagen/mod.rs
git commit -m "feat(cfvnet): add GPU boundary evaluator with TensorRT EP

ORT session with TensorRT → CUDA → CPU fallback chain.
Batched inference for 48-river boundary evaluation.
Reach-weighted averaging and denormalization on CPU."
```

---

### Task 3: Turn Datagen Orchestrator

Extend `DomainPipeline::run_gpu()` to handle turn trees with periodic boundary re-evaluation.
This is the main integration task that connects the incremental solver (Task 1) with the
boundary evaluator (Task 2).

**Files:**
- Modify: `crates/cfvnet/src/datagen/domain/pipeline.rs:232-429` (run_gpu method)
- Test: `crates/cfvnet/src/datagen/domain/pipeline.rs` (gpu_tests module)

**Depends on:** Tasks 1 and 2.

#### Step 1: Remove the river-only restriction

In `pipeline.rs:245-247`:

```rust
if board_size < 5 {
    return Err("GPU datagen currently supports river (board_size=5) only".into());
}
```

**Replace with:** a branch that dispatches to `run_gpu_turn()` for board_size=4:

```rust
if board_size == 4 {
    return Self::run_gpu_turn(config, output_path);
}
```

#### Step 2: Implement `run_gpu_turn()` — setup phase

```rust
#[cfg(feature = "gpu-turn-datagen")]
fn run_gpu_turn(config: &CfvnetConfig, output_path: &Path) -> Result<(), String> {
    use gpu_range_solver::extract::{extract_terminal_data, extract_topology, decompose_at_chance};
    use gpu_range_solver::{GpuBatchSolver, SubgameSpec, compute_evs_from_strategy_sum};
    use crate::datagen::gpu_boundary_eval::{GpuBoundaryEvaluator, evaluate_boundaries_batched, BoundaryEvalRequest};
    
    let model_path = config.game.river_model_path.as_deref()
        .ok_or("river_model_path required for GPU turn datagen")?;
    let evaluator = GpuBoundaryEvaluator::load(std::path::Path::new(model_path))?;
    
    // ... situation generation setup (same as existing run_gpu) ...
    // ... writer setup ...
    // ... progress bar ...
}
```

#### Step 3: Implement the main solve loop

The core orchestration loop. For each situation:

```rust
for sit in &mut sit_gen {
    let game = match builder.build(&sit, &mut rng) {
        Some(g) => g,
        None => { pb.inc(1); continue; }
    };
    
    let topo = extract_topology(game.inner());
    let decomp = decompose_at_chance(&topo);
    let term = extract_terminal_data(game.inner(), &decomp.turn_topo);
    let num_hands = game.inner().private_cards(0).len()
        .max(game.inner().private_cards(1).len());
    
    if num_hands > 1024 {
        // CPU fallback (same as existing)
        ...
        continue;
    }
    
    // Build solver with turn topology
    let mut solver = GpuBatchSolver::new(
        &decomp.turn_topo, &term, 1, num_hands, max_iterations,
    ).map_err(|e| format!("GPU solver init: {e}"))?;
    
    // Set up leaf injection from decomposition
    let leaf_ids: Vec<i32> = decomp.turn_leaf_node_ids.iter().map(|&x| x as i32).collect();
    let leaf_depths: Vec<i32> = leaf_ids.iter()
        .map(|&id| decomp.turn_topo.node_depth[id as usize] as i32).collect();
    solver.set_leaf_injection(&leaf_ids, &leaf_depths)
        .map_err(|e| format!("leaf injection setup: {e}"))?;
    
    let spec = SubgameSpec::from_game(game.inner(), &decomp.turn_topo, &term, num_hands);
    solver.prepare_batch(&[spec.clone()])
        .map_err(|e| format!("prepare_batch: {e}"))?;
    
    let hand_cards: Vec<(u8, u8)> = game.inner().private_cards(0).to_vec(); // or max of both
    let leaf_eval_interval = config.datagen.leaf_eval_interval.max(1);
    
    // Initial boundary evaluation
    let board_arr: [u8; 4] = sit.board_cards()[..4].try_into().unwrap();
    evaluate_and_upload_boundaries(
        &evaluator, &solver, &sit, &board_arr, &decomp,
        num_hands, &hand_cards, &leaf_ids,
    )?;
    
    // Solve with periodic re-evaluation
    let mut iter = 0u32;
    while iter < max_iterations {
        let end = (iter + leaf_eval_interval).min(max_iterations);
        solver.run_iterations(iter, end)
            .map_err(|e| format!("run_iterations: {e}"))?;
        iter = end;
        
        if iter < max_iterations {
            // Re-evaluate boundaries with updated reach
            evaluate_and_upload_boundaries(
                &evaluator, &solver, &sit, &board_arr, &decomp,
                num_hands, &hand_cards, &leaf_ids,
            )?;
        }
    }
    
    // Extract results and write records (same EV extraction as existing run_gpu)
    let results = solver.extract_results()
        .map_err(|e| format!("extract: {e}"))?;
    let evs = compute_evs_from_strategy_sum(
        &decomp.turn_topo, &term, &results[0].strategy_sum, &spec.initial_weights, num_hands,
    );
    
    // Build and write training records (same as existing run_gpu lines 348-418)
    ...
}
```

#### Step 4: Implement `evaluate_and_upload_boundaries()` helper

```rust
fn evaluate_and_upload_boundaries(
    evaluator: &GpuBoundaryEvaluator,
    solver: &GpuBatchSolver,
    sit: &Situation,
    board: &[u8; 4],
    decomp: &ChanceDecomposition,
    num_hands: usize,
    hand_cards: &[(u8, u8)],
    leaf_ids: &[i32],
) -> Result<(), String> {
    // Download reach from GPU
    let reach = solver.download_reach()
        .map_err(|e| format!("download reach: {e}"))?;
    
    // Extract reach at boundary nodes into 1326-indexed arrays
    let num_boundaries = leaf_ids.len();
    let mut oop_reach_1326 = vec![0.0f32; num_boundaries * 1326];
    let mut ip_reach_1326 = vec![0.0f32; num_boundaries * 1326];
    
    for (bi, &node_id) in leaf_ids.iter().enumerate() {
        let node = node_id as usize;
        // reach layout: [B * N * H], B=1 for single-game
        for (hi, &(c0, c1)) in hand_cards.iter().enumerate() {
            let idx = range_solver::card::card_pair_to_index(c0, c1);
            // OOP reach (player 0 traversal: opponent=player1 reach stored)
            // The reach at a node represents opponent reach for the traversing player
            // Need to understand the layout from the solver's perspective
            oop_reach_1326[bi * 1326 + idx] = reach[node * num_hands + hi];
        }
        // IP reach from a second traversal or alternating storage...
        // (Implementation detail: depends on solver's reach storage convention)
    }
    
    let request = BoundaryEvalRequest {
        board: *board,
        pot: sit.pot as f32,
        effective_stack: sit.effective_stack as f32,
        oop_reach: oop_reach_1326,
        ip_reach: ip_reach_1326,
        num_boundaries,
    };
    
    let results = evaluate_boundaries_batched(evaluator, &[request], num_hands, hand_cards)?;
    
    // Upload leaf CFVs
    solver.update_leaf_cfvs(&results[0].leaf_cfv_p0, &results[0].leaf_cfv_p1)
        .map_err(|e| format!("upload leaf cfvs: {e}"))
}
```

**Important note for the implementer:** The reach array from the GPU solver stores reach for
the OPPONENT of the traversing player. The DCFR kernel alternates between player 0 and
player 1 traversals, overwriting reach each time. The reach after `run_iterations()` reflects
the LAST traversal's opponent reach.

For boundary evaluation, we need both OOP and IP reach at boundary nodes. Two options:
1. Run a final forward pass for each player and download both (requires kernel support)
2. Use the initial weights as proxy ranges (simpler, what the CPU path does)

Start with option 2 (initial weights as ranges) and upgrade to option 1 later. This matches
the CPU `NeuralNetEvaluator` which uses root ranges.

**Actually, re-reading the CPU domain solver** at `solver.rs:128-141`: the evaluator is called
with `game` which carries the current ranges (not root). The `NeuralNetEvaluator` calls
`game.reach_at_boundary(ordinal, player)` or similar to get current reach. Check the domain
`Game` wrapper and `NeuralNetEvaluator` for the exact pattern.

#### Step 5: Write test — GPU turn pipeline produces records

Add to `gpu_tests` module in pipeline.rs:

```rust
#[cfg(feature = "gpu-turn-datagen")]
#[test]
fn gpu_turn_pipeline_produces_records() {
    range_solver::set_force_sequential(true);
    let tmp = NamedTempFile::new().unwrap();
    let config = CfvnetConfig {
        game: GameConfig {
            initial_stack: 200,
            board_size: 4,
            river_model_path: Some("../../local_data/cfvnet/river/v2/best.onnx".into()),
            ..Default::default()
        },
        datagen: DatagenConfig {
            num_samples: 3,
            mode: "domain".into(),
            solver_iterations: 50,
            seed: Some(42),
            backend: "gpu".into(),
            leaf_eval_interval: 25,
            ..Default::default()
        },
        training: Default::default(),
        evaluation: Default::default(),
    };
    
    // Skip if model not available
    if !std::path::Path::new(config.game.river_model_path.as_ref().unwrap()).exists() {
        return;
    }
    
    DomainPipeline::run(&config, tmp.path()).unwrap();
    
    let mut reader = BufReader::new(std::fs::File::open(tmp.path()).unwrap());
    let mut count = 0;
    while let Ok(rec) = read_record(&mut reader) {
        assert_eq!(rec.board.len(), 4);
        assert!(rec.pot > 0.0);
        assert!(rec.game_value.is_finite());
        for &cfv in &rec.cfvs {
            assert!(cfv.is_finite());
        }
        count += 1;
    }
    assert!(count >= 2, "expected at least 2 records, got {count}");
}
```

#### Step 6: Commit

```bash
git add crates/cfvnet/src/datagen/domain/pipeline.rs
git commit -m "feat(cfvnet): GPU turn datagen with BoundaryNet re-evaluation

Extend run_gpu to handle turn trees via incremental DCFR solving
with periodic BoundaryNet boundary re-evaluation. Uses ORT with
TensorRT EP for batched boundary inference."
```

---

### Task 4: End-to-End Integration & Validation

Verify the full pipeline works from CLI with a real config.

**Files:**
- Create: `sample_configurations/turn_gpu_datagen.yaml`
- Test: manual CLI run + Python validation

#### Step 1: Create sample config

```yaml
game:
  initial_stack: 200
  board_size: 4
  river_model_path: "local_data/cfvnet/river/v2/best.onnx"
  bet_sizes:
    - [0.33, 0.67, 1.0]

datagen:
  street: "turn"
  mode: "domain"
  backend: "gpu"
  num_samples: 100
  solver_iterations: 200
  leaf_eval_interval: 50
  seed: 42
  turn_output: "local_data/cfvnet/turn/gpu_test"
  per_file: 1000
  pot_intervals: [[10, 50], [50, 200]]
  spr_intervals: [[1.0, 3.0], [3.0, 8.0]]
```

#### Step 2: Run from CLI

```bash
cargo run -p cfvnet --release --features gpu-turn-datagen -- generate \
  --config sample_configurations/turn_gpu_datagen.yaml
```

**Expected:** Completes without error, writes binary records to output dir.

#### Step 3: Validate output in Python

Use the existing Python dataloader to verify records are readable and have reasonable values:

```python
# Quick validation script
from cfvnet.data import TrainingDataset
ds = TrainingDataset("local_data/cfvnet/turn/gpu_test")
print(f"Records: {len(ds)}")
sample = ds[0]
print(f"Board size: {len(sample['board'])}")
print(f"CFV range: [{sample['cfvs'].min():.4f}, {sample['cfvs'].max():.4f}]")
print(f"Game value: {sample['game_value']:.4f}")
assert len(sample['board']) == 4  # Turn board
```

#### Step 4: Compare GPU vs CPU for a few situations

Run the same config with `backend: "cpu"` and compare CFVs. They won't be identical
(GPU uses float32, CPU may differ in solving path) but should be in the same ballpark:

```bash
# CPU reference
cargo run -p cfvnet --release -- generate \
  --config sample_configurations/turn_gpu_datagen.yaml \
  --output local_data/cfvnet/turn/cpu_test
```

Compare mean absolute CFV difference between GPU and CPU records.

#### Step 5: Commit config and any fixes

```bash
git add sample_configurations/turn_gpu_datagen.yaml
git commit -m "feat: add sample config for GPU turn datagen

Validated: GPU pipeline produces correct turn training records
with periodic BoundaryNet re-evaluation."
```

---

## Key Reference Files

| File | What's there |
|------|-------------|
| `crates/gpu-range-solver/src/batch.rs` | `GpuBatchSolver`, `solve_batch()`, `compute_evs_from_strategy_sum()` |
| `crates/gpu-range-solver/src/kernels.rs:308-650` | Hand-parallel CUDA kernel `cfr_solve` |
| `crates/gpu-range-solver/src/extract.rs` | `extract_topology()`, `decompose_at_chance()`, `ChanceDecomposition` |
| `crates/cfvnet/src/datagen/domain/pipeline.rs:232` | `DomainPipeline::run_gpu()` — extend this |
| `crates/cfvnet/src/datagen/domain/solver.rs:120-141` | CPU `Solver::step()` — reference for re-eval pattern |
| `crates/cfvnet/src/datagen/domain/evaluator.rs` | `BoundaryEvaluator` trait, `BoundaryCfvs`, `SolveStrategy` |
| `crates/cfvnet/src/datagen/domain/neural_net_evaluator.rs` | CPU `NeuralNetEvaluator` — reference for reach extraction |
| `crates/cfvnet/src/eval/boundary_evaluator.rs:26` | `encode_boundary_inference_input()` — reuse for input construction |
| `crates/cfvnet/src/eval/river_net_evaluator.rs:372` | `evaluate_boundaries()` — reference for 48-river batching pattern |
| `crates/cfvnet/src/datagen/storage.rs:43` | `write_record()` — binary format (no changes needed) |
| `crates/cfvnet/src/config.rs:142` | `DatagenConfig` — all config fields already exist |

## Implementation Notes

1. **Reach convention**: The GPU solver's `reach` array stores opponent reach for the
   current traversing player. After `run_iterations()`, this reflects the last traversal.
   Check `crates/cfvnet/src/datagen/domain/neural_net_evaluator.rs` for how the CPU path
   extracts per-player reach at boundaries.

2. **1326-index mapping**: BoundaryNet works in 1326-combo space. The GPU solver works in
   per-game hand index space (variable length). Use `card_pair_to_index(c0, c1)` from
   `range_solver::card` to convert between them.

3. **EV normalization**: GPU solver EVs are pre-divided by `num_combinations`. Multiply by
   `term.num_combinations` before normalizing for training records. See `pipeline.rs:361-373`.

4. **Feature flags**: Use `#[cfg(feature = "gpu-turn-datagen")]` to gate all new code.
   The entire pipeline compiles away when the feature is off.
