# Strategy Explorer

Desktop app and browser-based UI for browsing trained strategies interactively using a 13x13 hand matrix. Two tabs: **Explorer** (game tree navigation) and **Simulator** (hand simulation).

## Running

### Tauri Desktop App

```bash
cd frontend && npm install && cd ..
cd crates/tauri-app && cargo tauri dev
```

### Browser via Dev Server

No Tauri build required — useful for faster UI iteration:

```bash
cargo run -p poker-solver-devserver &   # HTTP API on :3001
cd frontend && npm run dev              # Vite on :5173
# Open http://localhost:5173
```

The frontend auto-detects Tauri vs browser via `window.__TAURI__` and uses `fetch()` in browser mode. File picker falls back to `window.prompt()` — enter absolute paths.

Test endpoints directly:
```bash
curl -X POST http://localhost:3001/api/is_bundle_loaded -H 'Content-Type: application/json' -d '{}'
```

## Loading Strategies

Open the hamburger menu to choose a strategy source:

### Trained Bundle
Select a strategy bundle directory (output from `train` command). Displays metadata: stack depth, bet sizes, info set count, training iterations.

### Rule-Based Agents
Agent TOML configs from `agents/*.toml` are listed automatically. Each agent maps `HandClass` variants to action frequencies. Select one to explore its strategy.

### Preflop Bundle
Load a preflop solve output (from `solve-preflop` command). Displays the preflop strategy with 169 canonical hands.

## Explorer Tab

### Preflop

The 13x13 hand matrix shows action probabilities for every starting hand. Each cell displays a color-coded bar:
- Blue = fold
- Green = call/check
- Red/graduated = bet/raise (lighter for small sizes, darker for large/all-in)

Click an action button to advance down the game tree.

Click a cell to expand combo-level detail — shows the hand class breakdown (e.g. how many combos of AKs are Flush, Pair, etc.) at the current board state.

### Postflop

When the game reaches the flop, enter board cards (e.g. `Ac Th 4d`). The app:
1. Canonicalizes the board (suit isomorphism) and establishes a suit mapping
2. Computes EHS2 buckets for all 169 canonical hands (progress bar shown)
3. Displays the strategy matrix for that board

Continue through turn and river by entering additional cards. The suit mapping from flop canonicalization is applied to turn/river cards automatically.

### Navigation

- **Action buttons** — click to advance to a child node
- **History strip** — shows the full action sequence at the top; click any point to rewind
- **Available actions** — displayed for the current decision point with probabilities

## Simulator Tab

Hand simulation interface for testing strategies against each other.

## Key Files

| File | Purpose |
|-|-|
| `crates/tauri-app/src/exploration.rs` | All exploration commands (Tauri wrappers + `_core` variants) |
| `crates/devserver/src/main.rs` | HTTP mirror of Tauri API for browser debugging |
| `frontend/src/Explorer.tsx` | Explorer UI component |
| `frontend/src/Simulator.tsx` | Simulator UI component |
| `frontend/src/invoke.ts` | Invoke wrapper (auto-detects Tauri vs fetch) |
| `frontend/src/types.ts` | TypeScript type definitions |
| `agents/*.toml` | Rule-based agent configs |

## API Commands

The explorer uses these backend commands (available as Tauri commands or HTTP `POST /api/{name}`):

| Command | Description |
|-|-|
| `load_bundle` | Load a trained strategy bundle |
| `load_preflop_solve` | Load a preflop strategy bundle |
| `solve_preflop_live` | Solve preflop in real-time |
| `load_subgame_source` | Load a blueprint for subgame solving |
| `get_strategy_matrix` | Get strategy for a position (returns 13x13 matrix) |
| `get_available_actions` | Get actions at current position |
| `get_bundle_info` | Get loaded bundle metadata |
| `is_bundle_loaded` | Check if any strategy is loaded |
| `start_bucket_computation` | Start async EHS2 bucket computation for a board |
| `is_board_cached` | Check if bucket computation is complete |
| `get_computation_status` | Get bucket computation progress |
| `canonicalize_board` | Canonicalize board cards via suit isomorphism |
| `list_agents` | List available agent TOML configs |
| `get_combo_classes` | Get combo-level hand class breakdown for a cell |
