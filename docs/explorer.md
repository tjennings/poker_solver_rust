# Strategy Explorer

Desktop app and browser-based UI for browsing trained strategies interactively using a 13x13 hand matrix. Two tabs: **Explorer** (game tree navigation) and **Simulator** (hand simulation).

## Running

### Tauri Desktop App

```bash
cd frontend && npm install && cd ..
cd crates/tauri-app && cargo tauri dev
```

### Browser via Dev Server

No Tauri build required -- useful for faster UI iteration:

```bash
cargo run -p poker-solver-devserver &   # HTTP API on :3001
cd frontend && npm run dev              # Vite on :5173
# Open http://localhost:5173
```

The frontend auto-detects Tauri vs browser via `window.__TAURI__` and uses `fetch()` in browser mode. File picker falls back to `window.prompt()` -- enter absolute paths.

Test endpoints directly:
```bash
curl -X POST http://localhost:3001/api/is_bundle_loaded -H 'Content-Type: application/json' -d '{}'
```

## Loading Strategies

Open the hamburger menu to choose a strategy source:

### Blueprint V2 Bundle
Select a blueprint_v2 strategy bundle directory (output from `train-blueprint` command). Displays metadata: stack depth, bet sizes, info set count, training iterations.

**Snapshot selection**: If a blueprint has multiple training snapshots (`snapshot_0000/`, `snapshot_0001/`, etc.), a second picker appears showing each snapshot with its iteration count and training time. The latest snapshot is pre-selected. Blueprints with only one snapshot load directly without the extra step.

### Rule-Based Agents
Agent TOML configs from `agents/*.toml` are listed automatically. Each agent maps `HandClass` variants to action frequencies. Select one to explore its strategy.

## Explorer Tab

### Preflop

The 13x13 hand matrix shows action probabilities for every starting hand. Each cell displays a color-coded bar:
- Blue = fold
- Green = call/check
- Red/graduated = bet/raise (lighter for small sizes, darker for large/all-in)

Click an action button to advance down the game tree.

Click a cell to expand combo-level detail -- shows the hand class breakdown (e.g. how many combos of AKs are Flush, Pair, etc.) at the current board state.

### Postflop

When the game reaches the flop, enter board cards (e.g. `Ac Th 4d`). The app:
1. Canonicalizes the board (suit isomorphism) and establishes a suit mapping
2. Computes EHS2 buckets for all 169 canonical hands (progress bar shown)
3. Displays the strategy matrix for that board

Continue through turn and river by entering additional cards. The suit mapping from flop canonicalization is applied to turn/river cards automatically.

### Navigation

- **Action buttons** -- click to advance to a child node
- **History strip** -- shows the full action sequence at the top; click any point to rewind
- **Available actions** -- displayed for the current decision point with probabilities

## Simulator Tab

Hand simulation interface for testing strategies against each other.

## Remote Backend

WarpGTO can connect to a remote backend for GPU-accelerated solving. This is useful when the solver machine has a powerful GPU but you want to use the desktop UI on another machine.

### Setup

1. On the remote machine, start the devserver:
   ```bash
   cargo run -p poker-solver-devserver --release
   ```
   The server listens on `http://0.0.0.0:3001`.

2. On your local machine, open WarpGTO and go to **Settings**.

3. Enter the remote machine's URL (e.g., `http://192.168.1.50:3001`) in the **Remote Backend URL** field.

4. A green dot indicates a successful connection. Leave the field empty to return to local mode.

### Notes

- File paths (bundle loading, cache directory) refer to the **remote machine's** filesystem. Type paths manually when in remote mode.
- All solver commands (exploration, postflop, simulation) are routed to the remote backend. Window management stays local.
- Simulation events (progress, completion) stream over WebSocket (`/ws/events`) in remote mode.
- No authentication is required — the server is intended for trusted LAN use.

## Key Files

| File | Purpose |
|-|-|
| `crates/tauri-app/src/exploration.rs` | All exploration commands (Tauri wrappers + `_core` variants) |
| `crates/devserver/src/main.rs` | HTTP mirror of Tauri API for browser debugging |
| `frontend/src/Explorer.tsx` | Explorer UI component |
| `frontend/src/Simulator.tsx` | Simulator UI component |
| `frontend/src/invoke.ts` | Invoke wrapper (routes to Tauri IPC, remote HTTP, or local devserver) |
| `frontend/src/events.ts` | Event listener abstraction (Tauri events or WebSocket) |
| `frontend/src/types.ts` | TypeScript type definitions |
| `agents/*.toml` | Rule-based agent configs |

## API Commands

The explorer uses these backend commands (available as Tauri commands or HTTP `POST /api/{name}`):

| Command | Description |
|-|-|
| `load_bundle` | Load a trained strategy bundle (auto-detects blueprint_v2 format) |
| `load_blueprint_v2` | Load a blueprint_v2 strategy bundle (optional `snapshot` param to pick a specific snapshot) |
| `list_snapshots` | List available snapshots in a blueprint directory (returns name, iterations, elapsed time) |
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
