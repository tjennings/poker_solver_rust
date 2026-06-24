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

The Explorer loads both legacy HU `blueprint_v2` bundles and universal dense
blueprint bundles (`docs/blueprint_format.md`). When opening a directory the
loader detects `blueprint.json` (universal) before `config.yaml` (legacy) and
routes through the unified loader (`blueprint_universal::loader`):

- **Universal HU** bundles render through the existing HU views — a
  `BlueprintV2Strategy` is reconstructed from the universal rows (bitwise-identical
  to the legacy strategy) using the `config.yaml` retained inside the bundle to
  rebuild the game tree.
- **Universal MP** bundles (eager and lazy sparse) load read-only: the bundle
  info panel reports format kind, player count, seats, stacks, and bucket counts
  from the manifest. Full N-player browsing (seat selection, hand grids) is a
  later phase, so HU-only views report "MP browsing not yet supported" for MP
  bundles.

Universal bundles may be placed directly at the selected directory, under
`final/`, under a direct `snapshot_NNNN/`, or under the native trainer layout
`snapshot_NNNN/universal/`. The `load_blueprint_v2` snapshot path also detects
that nested universal layout before parsing `config.yaml`, so MP configs using
`game.num_players` are not treated as HU `BlueprintV2Config` files.

Bundle and snapshot listings report the format kind and player count for each
entry. Snapshot listings mark `snapshot_NNNN/universal/blueprint.json` as
loadable and read iteration metadata from either `iterations` or `iteration`.

### Blueprint V2 Bundle
Select a blueprint_v2 strategy bundle directory (output from `train-blueprint` command). Displays metadata: stack depth, bet sizes, info set count, training iterations.

**Snapshot selection**: If a blueprint has multiple training snapshots (`snapshot_0000/`, `snapshot_0001/`, etc.), a second picker appears showing each snapshot with its iteration count and training time. The latest snapshot is pre-selected. Blueprints with only one snapshot load directly without the extra step. MP lazy sparse snapshots are Explorer-loadable when `snapshots.format` is `universal` or `both`, which writes `snapshot_NNNN/universal/blueprint.json`.

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

When using Blueprint, Subgame, or Exact strategy sources, the matrix is source-specific for the current game state. Solved Subgame and Exact sources keep their solved matrix cache anchored to the street state where the solve started, so taking actions or rewinding within that solved subtree continues to show the solved strategy instead of falling back to the blueprint matrix. Player labels in the Explorer are always seat positions (`BB`/`SB`).

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
| `load_bundle` | Load a trained strategy bundle (auto-detects universal `blueprint.json` vs legacy `blueprint_v2` `config.yaml`) |
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

## Boundary Evaluation

The Settings view controls depth-boundary evaluators for Subgame and Exact solves. The first non-Exact street after the current root street becomes the cut point; later streets are disabled because they cannot be reached by the depth-limited solve.

Available boundary modes:

- `Exact`: no neural or subtree cut at that street.
- `CFVNet`: evaluate the boundary with the legacy river-model adapter. On a 4-card turn boundary this averages river-model outputs over valid river runouts.
- `Direct CFVNet`: evaluate the boundary board directly with an ONNX model. This is the mode for direct turn-boundary checkpoints trained on 4-card turn boards.
- `Exact Subtree`: solve the downstream subtree exactly with DCFR.

Turn and river CFVNet boundaries are supported; flop CFVNet is intentionally disabled. For turn-boundary CFVNet checkpoints trained directly on 4-card turn boards, set the turn boundary to `Direct CFVNet` and choose the ONNX file. The current local direct checkpoint path is `local_data/models/turn_boundary_cfvnet_v2/best.onnx`; the companion `.onnx.data` file must stay beside it but is local data and should not be committed.

The legacy `CFVNet` option remains available temporarily for compatibility and can be removed after the direct path is confirmed in the Explorer.

## Boundary Tracing

When solving hybrid (cfvnet) spots, you can enable per-boundary trace logging to capture what the boundary evaluator sees and produces at each depth boundary.

### Enabling from Settings

1. Open **Settings** in the explorer UI
2. Under **Boundary Tracing (debug)**, set:
   - **Boundaries to trace**: ordinal indices (e.g. `42`, `0,42,100`, or `all`)
   - **Iterations to trace**: `last` (default), `all`, or specific indices (e.g. `0,49,99`)
3. Leave "Boundaries to trace" empty to disable tracing (zero performance cost)

### Output

Trace files are written to `./local_data/logs/boundary_<ord>.txt` (created automatically).

Each record has this format:

```
[iter=50 boundary=42 board=JdTh9dQc pot=88 stack=78 spr=0.89]
Spot: sb:2bb,bb:10bb,sb:22bb,bb:call|Jd9d7d|bb:check,sb:bet44%,bb:call|3c
OOP range (168 combos): AA:6/6:1.00 KK:6/6:0.98 AKs:4/4:0.90 ...
IP range (197 combos):  QQ:6/6:1.00 AKs:4/4:0.80 ...
OOP CFVs (chips): AA:+3.45 KK:+2.10 AKs:+1.80 ...
IP CFVs  (chips): ...
Strategy at preceding decision (node #1234, OOP to act):
  Actions: [check, bet33%, allin]
  AA:  [0.00, 0.30, 0.70]
  KK:  [0.20, 0.80, 0.00]
  AKo: [0.80, 0.20, 0.00]
  ...
---
```

- **combos** = distinct card pairs with nonzero weight (not 169 hand classes)
- **spr** = stack / pot, rounded to 2 decimals
- **Strategy section** = per-hand-class probability over actions at the nearest ancestor decision node, sorted by descending reach weight
- `---` separates records within a file
