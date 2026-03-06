# Postflop Explorer Design

## Goal

Add a "Postflop Solver" mode to the Tauri explorer app that lets users configure ranges, pot/stacks, and bet sizes, then solve and explore postflop strategies street-by-street using the `range-solver` crate. Strategies update live during the solve, and range filtering carries forward between streets.

## Architecture

The postflop explorer is a new frontend component (`PostflopExplorer.tsx`) with dedicated Tauri backend commands wrapping the `range-solver` crate. It shares matrix rendering and action color utilities with the existing `Explorer.tsx` but has its own flow and state.

### Entry Point

The existing Explorer's "no data loaded" state currently shows a single "Load Dataset" card. This becomes two cards **stacked vertically**, splitting the available space:

- **Top:** Load Dataset (existing folder icon) — opens file picker, loads a bundle
- **Bottom:** Postflop Solver (puzzle piece icon) — enters the postflop-only flow

Clicking "Postflop Solver" replaces both cards with the postflop flow.

### User Flow

1. **Config card** appears in the action strip showing default game settings
2. **Flop input** appears next to the config card in the action strip (text input: `Ah Kd 7c`)
3. Entering 3 valid cards triggers the **solve** — live-updating 13x13 matrix with progress bar
4. **Navigate** the solved tree by clicking actions in the action strip (same as existing explorer)
5. When a street closes (call), a **turn input** appears as the next block in the action strip
6. Enter turn card → re-solve with filtered ranges and updated pot/stacks
7. Same for **river**

The action strip grows left to right: `[Config] [Flop: Ah Kd 7c] [Check] [Bet 75%] [Call] [Turn: 5s] ...`

### Defaults (3-bet pot)

| Setting | Default |
|-|-|
| Hero range (OOP) | `QQ+,AKs,AKo` |
| Villain range (IP) | `JJ-66,AQs-ATs,AQo,KQs` |
| Pot | 30 |
| Effective stack | 170 |
| Bet sizes | 25%, 33%, 75% bets; all-in raises |

### Config Modal

Clicking the config card opens a modal with:
- Two text areas for ranges (PioSOLVER syntax)
- Two number inputs (pot, effective stack)
- Four text inputs for bet sizes (OOP bet, OOP raise, IP bet, IP raise)
- Parse on submit, show errors inline

## Backend

### State

`PostflopState` holds:
- `Option<PostFlopGame>` — the current solved game
- Current config (ranges, pot, stacks, bet sizes)
- Filtered ranges per player (updated between streets)
- Solve progress (current iteration, target, exploitability)
- Background solve thread handle

Lives in Tauri managed state alongside the existing `ExplorationState`.

### Commands

| Command | Input | Output | Description |
|-|-|-|-|
| `postflop_set_config` | ranges, pot, stacks, bet sizes | Config summary | Validate and store config |
| `postflop_solve_street` | board cards (3-5 cards) | Solve started ack | Build game with current ranges, start async solve on background thread |
| `postflop_get_progress` | — | iteration, exploitability, strategy matrix | Poll for live updates (~every 2s). Returns current matrix snapshot. |
| `postflop_play_action` | action index | strategy matrix at new node | Navigate solved tree, return strategy at target node |
| `postflop_close_street` | action history for the street | filtered ranges, updated pot/stacks | Walk action history, apply strategy weights to filter ranges, prepare for next street |

### Solve Execution

- `postflop_solve_street` spawns a background thread that calls `solve_step()` in a loop
- Between steps, it snapshots the current strategy matrix into shared state
- `postflop_get_progress` reads the latest snapshot without blocking the solver
- Frontend polls every ~2 seconds, updates the matrix and progress bar
- When solve completes (target iterations or exploitability reached), progress reports completion

### Range Filtering Between Streets

When a street closes (e.g., OOP checks, IP bets 75%, OOP calls):

1. Walk the solved tree through the action history
2. At each decision node, multiply the acting player's range weights by their strategy frequency for the chosen action
3. `filtered_range[hand] = original_range[hand] * product(strategy_weight[hand] at each action taken by that player)`
4. Non-acting player's range passes through unchanged at non-fold nodes
5. Construct new `Range` objects with filtered weights for the next street's solve
6. Update pot and stacks based on the action sequence

This is the standard PioSOLVER approach. The range-solver provides per-hand strategy weights via `game.strategy()`.

## Frontend

### New Files

- `PostflopExplorer.tsx` — main component for the postflop-only flow
- `matrix-utils.ts` — extracted shared utilities from Explorer.tsx (matrix rendering, action colors, formatEV, cell detail panel)

### Component State Machine

```
INIT → CONFIG_SET → SOLVING → SOLVED → NAVIGATING
                                 |
                          (street closes)
                                 |
                     SOLVING (next street, filtered ranges)
```

### Shared Code Extraction

Extract from `Explorer.tsx` into `matrix-utils.ts`:
- `getActionColor()`, `sortedBetActions()`, `displayOrderIndices()`, `displayOrderActions()`
- `formatEV()`, `formatActionLabel()`
- `matrixToHandIndex()`
- Matrix grid rendering component (13x13 grid with action probability bars)
- Cell detail panel (action probabilities, combo groups)

Both `Explorer.tsx` and `PostflopExplorer.tsx` import from `matrix-utils.ts`.

### Live Matrix Updates

- During solve: matrix renders with current strategy, updates every ~2s
- Progress bar above matrix shows: iteration count, exploitability value
- Matrix cells show action probability bars (same visual style as existing explorer)
- When solve completes, progress bar shows "Converged" or final stats

### Action Strip Blocks

- **Config block:** Condensed summary (e.g., "30 pot / 170 eff"), click to open modal
- **Board input blocks:** Text input accepting card notation (e.g., `Ah Kd 7c`), styled as action blocks
- **Action history blocks:** Same rendering as existing explorer (fold/check/call/bet/raise with colors)

## Dev Server

Mirror all 5 new postflop commands as POST endpoints in `crates/devserver/src/main.rs`:
- `POST /api/postflop_set_config`
- `POST /api/postflop_solve_street`
- `POST /api/postflop_get_progress`
- `POST /api/postflop_play_action`
- `POST /api/postflop_close_street`

Same `PostflopState` shared via `Arc<Mutex<>>` in Axum app state.

## Testing

- **Backend unit tests:** Config parsing, range filtering math, action navigation
- **Integration test:** Set config → solve flop → navigate actions → close street → solve turn → verify filtered ranges
- **Manual testing via dev server:** Use browser + curl to test each command independently
