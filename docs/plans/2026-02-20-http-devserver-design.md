# HTTP Dev Server — Design

## Goal

Enable browser-based debugging of the Explorer UI by adding an HTTP API that mirrors the Tauri command layer, so Claude (or any developer) can launch the frontend in a headless browser and interact with it programmatically.

## Architecture

```
poker-solver-devserver (Axum)     Vite dev server
        :3001                          :5173
    POST /api/*  ←── fetch ──  invoke() wrapper
         ↓                         ↓
  ExplorationState           React Explorer UI
  (same as Tauri)            (same components)
```

### Backend: `crates/devserver/`

New binary crate with Axum. Wraps the 14 exploration commands from `crates/tauri-app/src/exploration.rs` as POST endpoints.

The Tauri commands take `State<'_, ExplorationState>` — the devserver holds the same state via `Arc<ExplorationState>` in Axum's state extractor.

Key challenge: Tauri commands use `tauri::State` which is a thin wrapper around `Arc`. The command functions need to be callable from both Tauri and Axum. Two options:
1. Extract core logic into shared functions that take `&ExplorationState` directly
2. Duplicate the thin handler layer in the devserver

Option 2 is simpler — the handlers are just deserialization + calling the same underlying functions + serialization. The core logic already lives in helper functions (`get_strategy_matrix_bundle`, `get_strategy_matrix_preflop`, etc).

**Routes:**

| Route | Tauri Command |
|-|-|
| POST /api/load_bundle | load_bundle(path) |
| POST /api/load_preflop_solve | load_preflop_solve(path) |
| POST /api/solve_preflop_live | solve_preflop_live(stack_depth, iterations) |
| POST /api/load_subgame_source | load_subgame_source(blueprint_path) |
| POST /api/get_strategy_matrix | get_strategy_matrix(position, threshold?, street_histories?) |
| POST /api/get_available_actions | get_available_actions(position) |
| POST /api/is_bundle_loaded | is_bundle_loaded() |
| POST /api/get_bundle_info | get_bundle_info() |
| POST /api/canonicalize_board | canonicalize_board(cards) |
| POST /api/start_bucket_computation | start_bucket_computation(board) |
| POST /api/get_computation_status | get_computation_status() |
| POST /api/is_board_cached | is_board_cached(board) |
| POST /api/list_agents | list_agents() |
| POST /api/get_combo_classes | get_combo_classes(position, hand) |

CORS: Allow `http://localhost:5173`. Port: 3001.

### Frontend: `invoke` wrapper

Replace `import { invoke } from '@tauri-apps/api/core'` with a local `src/invoke.ts` that detects `window.__TAURI__` and falls back to fetch.

### Refactoring needed

The Tauri command functions currently use `tauri::State<'_, ExplorationState>`. To share them with Axum, refactor each to take `&ExplorationState` directly, then have thin Tauri and Axum wrappers that extract state and delegate.

### Not in scope

- Simulation commands (not needed for Explorer debugging)
- File picker dialog (browser users pass paths directly)
- Authentication (localhost dev-only tool)
- Production deployment
