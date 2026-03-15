# Explorer GPU Resolving — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use hex:executing-plans to implement this plan task-by-task.

**Goal:** Wire the GPU solver + trained model stack into the Tauri Explorer for interactive heads-up resolving with progressive strategy display.

**Architecture:** New `GpuResolver` module in the gpu-solver crate handles model loading, street detection, tree building, and solving. Tauri commands in the Explorer call into GpuResolver. Frontend detects GPU mode and updates strategy display progressively.

**Tech Stack:** Rust, cudarc, CudaNetInference (cuBLAS), Tauri, existing Explorer frontend

---

## Task 1: GpuModelStack — Load and Manage Models

**Files:**
- Create: `crates/gpu-solver/src/resolve.rs`
- Modify: `crates/gpu-solver/src/lib.rs`

**Step 1: Define types**

```rust
pub struct ModelStackConfig {
    pub river_hidden_layers: usize,
    pub river_hidden_size: usize,
    pub turn_hidden_layers: usize,
    pub turn_hidden_size: usize,
    pub flop_hidden_layers: usize,
    pub flop_hidden_size: usize,
    pub preflop_hidden_layers: usize,
    pub preflop_hidden_size: usize,
}

pub struct GpuModelStack {
    gpu: GpuContext,
    river: CudaNetInference,
    turn: CudaNetInference,
    flop: CudaNetInference,
    preflop: CudaNetInference,
}
```

**Step 2: Load from directory**

```rust
impl GpuModelStack {
    pub fn load(dir: &Path, config: &ModelStackConfig) -> Result<Self, String> {
        let gpu = GpuContext::new(0)?;
        let max_batch = 50_000; // generous max for single-spot resolving

        let river = CudaNetInference::load_from_burn(
            &dir.join("river"), &gpu,
            config.river_hidden_layers, config.river_hidden_size, max_batch,
        )?;
        // ... same for turn, flop, preflop

        Ok(Self { gpu, river, turn, flop, preflop })
    }
}
```

**Step 3: Test** — load models from a directory (can use randomly-initialized models).

**Step 4: Commit**

```bash
git commit -am "feat(gpu-solver): add GpuModelStack for loading model sets"
```

---

## Task 2: GpuResolver — Single-Position Resolving

**Files:**
- Modify: `crates/gpu-solver/src/resolve.rs`

**Step 1: Define game state and result types**

```rust
pub struct GameState {
    pub board: Vec<u8>,          // 0, 3, 4, or 5 cards
    pub oop_range: Vec<f32>,     // 1326 combo weights
    pub ip_range: Vec<f32>,
    pub pot: i32,
    pub effective_stack: i32,
    pub bet_sizes_oop: String,   // e.g. "50%,100%,a"
    pub bet_sizes_ip: String,
}

pub struct ResolveResult {
    pub strategy: Vec<f32>,       // [num_actions * num_hands] for acting player
    pub actions: Vec<String>,     // action labels
    pub evs: Vec<f32>,           // per-hand EVs
    pub iterations: u32,
    pub player: u8,              // 0=OOP, 1=IP
    pub num_hands: usize,
}
```

**Step 2: Implement resolve**

```rust
impl GpuModelStack {
    pub fn resolve(
        &mut self,
        state: &GameState,
        max_iterations: u32,
    ) -> Result<ResolveResult, String> {
        match state.board.len() {
            0 => self.resolve_preflop(state, max_iterations),
            3 => self.resolve_flop(state, max_iterations),
            4 => self.resolve_turn(state, max_iterations),
            5 => self.resolve_river(state, max_iterations),
            n => Err(format!("Invalid board size: {n}")),
        }
    }

    fn resolve_river(&mut self, state: &GameState, iters: u32) -> Result<ResolveResult, String> {
        // Build river game from state
        // Build FlatTree
        // Create GpuSolver
        // Solve with `iters` iterations (no leaf model needed)
        // Extract strategy at root
    }

    fn resolve_turn(&mut self, state: &GameState, iters: u32) -> Result<ResolveResult, String> {
        // Build turn game with depth boundaries
        // Create TurnBatchSolverCuda (single spot) with river model
        // Solve
        // Extract strategy
    }

    fn resolve_flop(&mut self, state: &GameState, iters: u32) -> Result<ResolveResult, String> {
        // Same pattern with flop tree + turn model
    }

    fn resolve_preflop(&mut self, state: &GameState, iters: u32) -> Result<ResolveResult, String> {
        // Preflop tree + flop model (enumerate 22100 flops per boundary)
        // OR: use preflop auxiliary model directly at leaves (faster)
    }
}
```

**Step 3: Implement resolve_river first** — simplest case, no leaf model.

**Step 4: Test** — resolve a known river position, verify strategy matches cpu range-solver.

**Step 5: Implement resolve_turn and resolve_flop** — use TurnBatchSolverCuda/FlopBatchSolverCuda with single spot.

**Step 6: Commit**

```bash
git commit -am "feat(gpu-solver): add GpuResolver for single-position resolving"
```

---

## Task 3: Progressive Resolving

**Files:**
- Modify: `crates/gpu-solver/src/resolve.rs`

**Step 1: Add progressive resolve method**

```rust
impl GpuModelStack {
    pub fn resolve_progressive(
        &mut self,
        state: &GameState,
        checkpoints: &[u32],       // e.g. [500, 1000, 2000, 4000]
        on_checkpoint: impl FnMut(ResolveResult),
    ) -> Result<ResolveResult, String> {
        // Run solver in segments:
        // solve 500 iters → extract strategy → callback
        // solve 500 more (to 1000) → extract → callback
        // solve 1000 more (to 2000) → extract → callback
        // solve 2000 more (to 4000) → extract → callback
    }
}
```

The solver needs to support "solve N more iterations from current state" without resetting. Check if GpuSolver supports incremental solving — the solve loop already accumulates regrets/strategy_sum, so running more iterations just continues from where it left off.

Actually, the current `solve()` always starts from zero regrets. Need to add `solve_more(additional_iters)` that continues without resetting.

**Step 2: Add incremental solving to BatchGpuSolver**

Add `solve_incremental(&mut self, additional_iters: u32) -> Result<SolveResult, String>` that runs more iterations without reinitializing regrets/strategy_sum.

**Step 3: Test** — resolve 500 iters, then 500 more. Verify final result equals solving 1000 from scratch.

**Step 4: Commit**

```bash
git commit -am "feat(gpu-solver): add progressive resolving with checkpoints"
```

---

## Task 4: Tauri Commands

**Files:**
- Modify: `crates/tauri-app/src/exploration.rs`
- Modify: `crates/tauri-app/Cargo.toml`

**Step 1: Add gpu-solver dependency**

```toml
[dependencies]
poker-solver-gpu = { path = "../gpu-solver", features = ["cuda"], optional = true }

[features]
gpu = ["poker-solver-gpu"]
```

**Step 2: Add Tauri commands**

```rust
#[tauri::command]
async fn load_gpu_model_set(
    path: String,
    config: ModelStackConfig,
    state: tauri::State<'_, AppState>,
) -> Result<(), String> {
    let stack = GpuModelStack::load(Path::new(&path), &config)?;
    state.gpu_model_stack.lock().unwrap().replace(stack);
    Ok(())
}

#[tauri::command]
async fn gpu_resolve(
    game_state: GameState,
    max_iterations: u32,
    state: tauri::State<'_, AppState>,
) -> Result<ResolveResult, String> {
    let mut stack = state.gpu_model_stack.lock().unwrap();
    let stack = stack.as_mut().ok_or("No GPU model loaded")?;
    stack.resolve(&game_state, max_iterations)
}

#[tauri::command]
async fn is_gpu_model_loaded(
    state: tauri::State<'_, AppState>,
) -> bool {
    state.gpu_model_stack.lock().unwrap().is_some()
}
```

**Step 3: Add to AppState**

```rust
struct AppState {
    // existing fields...
    gpu_model_stack: Mutex<Option<GpuModelStack>>,
}
```

**Step 4: Test** — build the Tauri app with GPU feature, verify commands compile.

**Step 5: Commit**

```bash
git commit -am "feat(explorer): add Tauri commands for GPU resolving"
```

---

## Task 5: Dev Server Commands

**Files:**
- Modify: `crates/devserver/src/main.rs`

Mirror the Tauri commands as HTTP endpoints for browser-based development:

```rust
POST /api/load_gpu_model_set
POST /api/gpu_resolve
GET  /api/is_gpu_model_loaded
```

**Commit:**

```bash
git commit -am "feat(devserver): add GPU resolve HTTP endpoints"
```

---

## Task 6: Frontend GPU Mode

**Files:**
- Modify: `frontend/src/invoke.ts`
- Modify: `frontend/src/` (relevant strategy display components)

**Step 1: Add invoke wrappers**

```typescript
export async function loadGpuModelSet(path: string, config: ModelStackConfig): Promise<void>;
export async function gpuResolve(state: GameState, maxIterations: number): Promise<ResolveResult>;
export async function isGpuModelLoaded(): Promise<boolean>;
```

**Step 2: GPU mode detection**

When a GPU model set is loaded, the Explorer switches to GPU mode:
- Strategy matrix updates show "Resolving..." spinner during GPU solve
- Iteration counter shows progress (500/4000, 1000/4000, etc.)
- Strategy display updates at each checkpoint

**Step 3: Progressive update UI**

Use the progressive resolve API:
- Start resolve, show spinner
- At each checkpoint, update strategy matrix
- Show iteration count and time elapsed

**Step 4: Commit**

```bash
git commit -am "feat(frontend): add GPU resolve mode with progressive display"
```

---

## Task 7: Update Beans

```bash
beans update poker_solver_rust-822y -s completed
git add .beans/ && git commit -m "beans: Phase 5 complete — Explorer GPU integration"
```
