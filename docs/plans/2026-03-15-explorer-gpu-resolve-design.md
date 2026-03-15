# Phase 5: Explorer GPU Resolving — Design

## Overview

Wire the GPU solver + trained model stack into the Tauri Explorer for interactive heads-up resolving. Users load a model set (river/turn/flop/preflop models) as the GPU equivalent of a strategy bundle. Navigating the game tree triggers GPU re-solving at each decision point with progressive strategy display.

## User Experience

1. **Load model set** — User selects a directory containing trained models (river.mpk.gz, turn.mpk.gz, flop.mpk.gz, preflop.mpk.gz) via the existing file picker. Models load to GPU via CudaNetInference.

2. **Set starting state** — User configures ranges, pot, stack (same as existing range-solve setup).

3. **Navigate and resolve** — At each decision point:
   - GPU solver builds a one-street lookahead tree
   - Resolves with the appropriate model at leaf nodes
   - Progressive display: strategy appears after ~500 iterations (~1.6s), refines to configurable max iterations
   - Strategy matrix, action frequencies, EVs update live

4. **Off-tree actions** — When the opponent plays a bet size not in the abstraction, re-solve from the actual state using safe resolving. No action translation needed.

## Architecture

```
Explorer UI (frontend)
    │
    ▼
Tauri Commands (exploration.rs)
    │
    ├─ load_gpu_model_set(path) → loads 4 models to GPU
    ├─ gpu_resolve(game_state, iterations) → solves, returns strategy
    ├─ gpu_resolve_progressive(game_state, callback_iterations) → streams updates
    │
    ▼
GpuResolver (new crate module)
    │
    ├─ ModelStack: 4 CudaNetInference instances (river/turn/flop/preflop)
    ├─ Determines current street from game state
    ├─ Builds one-street lookahead tree with depth boundaries
    ├─ Selects appropriate leaf evaluator model
    ├─ Runs GpuSolver with leaf eval
    ├─ Returns strategy + EVs
    │
    ▼
Existing GPU Solver Infrastructure
    (BatchGpuSolver, TurnBatchSolverCuda, CudaNetInference, etc.)
```

## Model Stack

```rust
pub struct GpuModelStack {
    river: CudaNetInference,
    turn: CudaNetInference,
    flop: CudaNetInference,
    preflop: CudaNetInference,
    gpu: GpuContext,
}

impl GpuModelStack {
    pub fn load(dir: &Path, config: ModelStackConfig) -> Result<Self, String>;

    pub fn resolve(
        &mut self,
        state: &GameState,
        max_iterations: u32,
        progress_callback: Option<&dyn Fn(u32, &Strategy)>,  // (iterations_done, current_strategy)
    ) -> Result<ResolveResult, String>;
}
```

## Street Selection

Given a game state (board cards, ranges, pot, stack, action history):
- **Preflop** (0 board cards) → preflop model at leaves, resolve preflop betting
- **Flop** (3 board cards) → flop tree with turn model at leaves
- **Turn** (4 board cards) → turn tree with river model at leaves
- **River** (5 board cards) → river tree, no leaf model needed (pure terminal eval)

## Progressive Display

The solver runs in a background thread. At configurable iteration checkpoints (e.g., 500, 1000, 2000, 4000), it extracts the current strategy and sends it to the frontend:

```rust
pub struct ResolveResult {
    pub strategy: Vec<f32>,      // per-hand action probabilities
    pub evs: Vec<f32>,           // per-hand expected values
    pub iterations: u32,         // iterations completed
    pub exploitability: Option<f32>,
}
```

The frontend updates the strategy matrix each time a checkpoint arrives. The user sees strategies stabilize as iterations increase.

## Configurable Iterations

Default checkpoints: 500, 1000, 2000, 4000. User can configure max iterations and whether to show progressive updates.

## Integration with Existing Explorer

The Explorer's Tauri commands are in `crates/tauri-app/src/exploration.rs`. Add new commands alongside the existing bundle-loading commands:

```rust
#[tauri::command]
async fn load_gpu_model_set(path: String, config: ModelStackConfig) -> Result<(), String>;

#[tauri::command]
async fn gpu_resolve(game_state: GameState) -> Result<ResolveResult, String>;

#[tauri::command]
async fn is_gpu_model_loaded() -> bool;
```

The frontend detects whether a GPU model set or a strategy bundle is loaded and renders accordingly.

## Key Design Decisions

1. **Model set as "bundle"** — same UX pattern, different data. User picks a directory.
2. **Progressive display** — strategy appears fast, refines in background.
3. **Configurable iterations** — user controls accuracy/speed tradeoff.
4. **Street auto-detection** — resolver picks the right model based on board card count.
5. **Single-spot resolving** — one position at a time (not batched). Interactive use doesn't need batching.
6. **CudaNetInference for all models** — no burn in the hot path, pure cudarc/cuBLAS.
