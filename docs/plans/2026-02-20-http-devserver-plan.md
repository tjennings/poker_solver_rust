# HTTP Dev Server Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add an HTTP API server that mirrors the Tauri command layer, plus a frontend invoke wrapper, enabling browser-based debugging of the Explorer UI.

**Architecture:** Refactor Tauri commands into core functions taking `&ExplorationState`, then wrap them in both Tauri handlers (existing) and Axum POST handlers (new crate). Frontend gets an `invoke.ts` shim that detects Tauri vs browser and routes accordingly.

**Tech Stack:** Axum 0.8, tokio, tower-http (CORS), serde_json. Frontend: TypeScript fetch wrapper.

---

### Task 1: Refactor Tauri commands to separate core logic from Tauri-specific wrappers

**Files:**
- Modify: `crates/tauri-app/src/exploration.rs`
- Modify: `crates/tauri-app/src/lib.rs`

The Tauri commands currently use `tauri::State<'_, ExplorationState>` which implements `Deref<Target = ExplorationState>`. Refactor each command into:
1. A core function that takes `&ExplorationState` (or `Arc<ExplorationState>`) and returns the same `Result<T, String>`
2. A thin `#[tauri::command]` wrapper that extracts state and delegates

This makes the core functions callable from both Tauri and Axum.

**Step 1: Refactor sync commands**

For each sync command, extract the body into a `_core` function. Example pattern:

```rust
// Core function — no Tauri dependency
pub fn get_strategy_matrix_core(
    state: &ExplorationState,
    position: ExplorationPosition,
    threshold: Option<f32>,
    street_histories: Option<Vec<Vec<String>>>,
) -> Result<StrategyMatrix, String> {
    // ... existing body, unchanged ...
}

// Tauri wrapper — thin delegation
#[tauri::command]
pub fn get_strategy_matrix(
    state: State<'_, ExplorationState>,
    position: ExplorationPosition,
    threshold: Option<f32>,
    street_histories: Option<Vec<Vec<String>>>,
) -> Result<StrategyMatrix, String> {
    get_strategy_matrix_core(&state, position, threshold, street_histories)
}
```

Apply this pattern to these sync commands:
- `get_strategy_matrix` → `get_strategy_matrix_core`
- `get_available_actions` → `get_available_actions_core`
- `is_bundle_loaded` → `is_bundle_loaded_core`
- `get_bundle_info` → `get_bundle_info_core`
- `canonicalize_board` → `canonicalize_board_core`
- `get_computation_status` → `get_computation_status_core`
- `is_board_cached` → `is_board_cached_core`
- `get_combo_classes` → `get_combo_classes_core`

`list_agents` takes no state — leave as-is.

**Step 2: Refactor async commands**

For async commands, the core function takes `Arc<ExplorationState>` (since it needs `'static` for spawn_blocking):

```rust
pub async fn load_bundle_core(
    state: Arc<ExplorationState>,
    path: String,
) -> Result<BundleInfo, String> {
    // ... existing body, replacing state.field with state.field ...
}

#[tauri::command]
pub async fn load_bundle(
    state: State<'_, ExplorationState>,
    path: String,
) -> Result<BundleInfo, String> {
    // State doesn't impl Clone, but ExplorationState fields are Arc-wrapped.
    // We need to restructure: make ExplorationState Clone or pass individual Arc fields.
    // Simplest: make the core fn take &ExplorationState and use spawn_blocking differently.
    load_bundle_core(&state, path).await
}
```

Actually, since `tauri::async_runtime::spawn_blocking` captures the closure, the async commands clone individual `Arc` fields before spawning. The core function can still take `&ExplorationState` — the Arc cloning happens inside. Apply to:
- `load_bundle` → `load_bundle_core`
- `load_preflop_solve` → `load_preflop_solve_core`
- `solve_preflop_live` → `solve_preflop_live_core`
- `load_subgame_source` → `load_subgame_source_core`

**Step 3: Handle `start_bucket_computation` specially**

This command takes `AppHandle` for event emission. Refactor to accept an optional callback:

```rust
pub fn start_bucket_computation_core(
    state: &ExplorationState,
    board: Vec<String>,
    on_progress: Option<Box<dyn Fn(usize, usize, &str) + Send + 'static>>,
) -> Result<String, String> {
    // ... existing body, replacing app.emit with on_progress callback ...
}
```

The Tauri wrapper passes a closure that calls `app.emit()`. The Axum handler passes `None`.

**Step 4: Export core functions from lib.rs**

Add all `_core` functions to the `pub use exploration::{ ... }` block.

**Step 5: Verify build**

Run: `cargo build -p poker-solver-tauri`
Expected: Compiles. Existing behavior unchanged.

**Step 6: Commit**

```bash
git add crates/tauri-app/
git commit -m "refactor(tauri): extract core functions from Tauri command wrappers"
```

---

### Task 2: Create the devserver crate with Axum

**Files:**
- Create: `crates/devserver/Cargo.toml`
- Create: `crates/devserver/src/main.rs`
- Modify: `Cargo.toml` (workspace members)

**Step 1: Add crate to workspace**

In root `Cargo.toml`, add `"crates/devserver"` to workspace members.

**Step 2: Create Cargo.toml**

```toml
[package]
name = "poker-solver-devserver"
version = "0.1.0"
edition = "2021"

[dependencies]
axum = "0.8"
tokio = { version = "1", features = ["full"] }
tower-http = { version = "0.6", features = ["cors"] }
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
poker-solver-tauri = { path = "../tauri-app" }
```

**Step 3: Create main.rs with all routes**

```rust
use std::sync::Arc;
use axum::{Router, Json, extract::State as AxumState, routing::post, http::Method};
use tower_http::cors::{CorsLayer, Any};
use serde::Deserialize;

use poker_solver_tauri::ExplorationState;

type AppState = Arc<ExplorationState>;

#[tokio::main]
async fn main() {
    let state = Arc::new(ExplorationState::default());

    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods([Method::POST])
        .allow_headers(Any);

    let app = Router::new()
        .route("/api/load_bundle", post(handle_load_bundle))
        .route("/api/load_preflop_solve", post(handle_load_preflop_solve))
        .route("/api/solve_preflop_live", post(handle_solve_preflop_live))
        .route("/api/load_subgame_source", post(handle_load_subgame_source))
        .route("/api/get_strategy_matrix", post(handle_get_strategy_matrix))
        .route("/api/get_available_actions", post(handle_get_available_actions))
        .route("/api/is_bundle_loaded", post(handle_is_bundle_loaded))
        .route("/api/get_bundle_info", post(handle_get_bundle_info))
        .route("/api/canonicalize_board", post(handle_canonicalize_board))
        .route("/api/start_bucket_computation", post(handle_start_bucket_computation))
        .route("/api/get_computation_status", post(handle_get_computation_status))
        .route("/api/is_board_cached", post(handle_is_board_cached))
        .route("/api/list_agents", post(handle_list_agents))
        .route("/api/get_combo_classes", post(handle_get_combo_classes))
        .layer(cors)
        .with_state(state);

    let addr = "0.0.0.0:3001";
    println!("Dev server listening on {addr}");
    let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}
```

Each handler follows this pattern:

```rust
// For commands with params:
#[derive(Deserialize)]
struct LoadBundleParams {
    path: String,
}

async fn handle_load_bundle(
    AxumState(state): AxumState<AppState>,
    Json(params): Json<LoadBundleParams>,
) -> Result<Json<poker_solver_tauri::BundleInfo>, (axum::http::StatusCode, String)> {
    poker_solver_tauri::load_bundle_core(&state, params.path)
        .await
        .map(Json)
        .map_err(|e| (axum::http::StatusCode::BAD_REQUEST, e))
}

// For commands with no params:
async fn handle_is_bundle_loaded(
    AxumState(state): AxumState<AppState>,
) -> Json<bool> {
    Json(poker_solver_tauri::is_bundle_loaded_core(&state))
}
```

Write all 14 handlers following this pattern. For `start_bucket_computation`, pass `None` for the on_progress callback.

**Step 4: Verify build**

Run: `cargo build -p poker-solver-devserver`
Expected: Compiles.

**Step 5: Smoke test**

Run: `cargo run -p poker-solver-devserver &`
Then: `curl -X POST http://localhost:3001/api/is_bundle_loaded`
Expected: `false`

Kill the server after testing.

**Step 6: Commit**

```bash
git add crates/devserver/ Cargo.toml
git commit -m "feat: add poker-solver-devserver crate with Axum HTTP API"
```

---

### Task 3: Create frontend invoke wrapper

**Files:**
- Create: `frontend/src/invoke.ts`
- Modify: `frontend/src/Explorer.tsx` (change import)
- Modify: `frontend/src/Simulator.tsx` (change import)

**Step 1: Create invoke.ts**

```typescript
const DEV_SERVER_URL = 'http://localhost:3001';

function isTauri(): boolean {
  return '__TAURI__' in window;
}

export async function invoke<T>(cmd: string, args?: Record<string, unknown>): Promise<T> {
  if (isTauri()) {
    const { invoke: tauriInvoke } = await import('@tauri-apps/api/core');
    return tauriInvoke<T>(cmd, args);
  }

  const res = await fetch(`${DEV_SERVER_URL}/api/${cmd}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(args ?? {}),
  });

  if (!res.ok) {
    const text = await res.text();
    throw new Error(text);
  }

  return res.json();
}
```

**Step 2: Update Explorer.tsx import**

Change line 2 from:
```typescript
import { invoke } from '@tauri-apps/api/core';
```
to:
```typescript
import { invoke } from './invoke';
```

**Step 3: Update Simulator.tsx import**

Same change:
```typescript
import { invoke } from './invoke';
```

**Step 4: Handle non-Tauri environment for dialog**

The Explorer uses `import { open } from '@tauri-apps/plugin-dialog'` for the file picker. In browser mode, this won't work. Add a fallback that uses `window.prompt()`:

In Explorer.tsx, find the `handleLoadDataset` function and update:

```typescript
const handleLoadDataset = useCallback(async () => {
  try {
    let path: string | null = null;
    if ('__TAURI__' in window) {
      const { open } = await import('@tauri-apps/plugin-dialog');
      path = await open({ directory: true, title: 'Select Dataset Directory' });
    } else {
      path = window.prompt('Enter dataset directory path:');
    }
    if (path) {
      loadSource(path);
    }
  } catch (e) {
    setError(String(e));
  }
}, [loadSource]);
```

**Step 5: Verify TypeScript compiles**

Run: `cd frontend && npx tsc --noEmit`
Expected: Clean.

**Step 6: Commit**

```bash
git add frontend/src/invoke.ts frontend/src/Explorer.tsx frontend/src/Simulator.tsx
git commit -m "feat(frontend): add invoke wrapper with fetch fallback for browser mode"
```

---

### Task 4: End-to-end test

**Step 1: Start the devserver**

Run: `cargo run -p poker-solver-devserver &`

**Step 2: Start the frontend**

Run: `cd frontend && npm run dev`

**Step 3: Test in browser**

Open `http://localhost:5173` in a browser. Verify:
1. Page loads without errors (check console)
2. `is_bundle_loaded` returns false (no errors in console)
3. Enter a preflop bundle path via prompt → strategy matrix loads
4. Navigate actions within the preflop tree

**Step 4: Test with curl**

```bash
# Load a preflop bundle
curl -X POST http://localhost:3001/api/load_bundle \
  -H 'Content-Type: application/json' \
  -d '{"path":"preflop_hu_25bb"}'

# Check it loaded
curl -X POST http://localhost:3001/api/is_bundle_loaded

# Get strategy matrix
curl -X POST http://localhost:3001/api/get_strategy_matrix \
  -H 'Content-Type: application/json' \
  -d '{"position":{"board":[],"history":[],"pot":3,"stacks":[49,48],"stack_p1":49,"stack_p2":48,"to_act":0,"num_players":2,"active_players":[true,true]}}'
```

**Step 5: Commit any fixes, then final commit**

```bash
git commit -m "test: verify devserver end-to-end integration"
```
