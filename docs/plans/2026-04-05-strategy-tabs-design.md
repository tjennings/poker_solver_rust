# Strategy Source Tabs — Explorer UI

## Problem

The explorer currently shows only the blueprint strategy at each node. Users want to compare the blueprint against subgame-solved and exact-solved strategies side by side. The existing solve button is buried in the action strip and only supports one solve mode.

## Design

### Tab Bar

Three text tabs above the entire strategy container (matrix + combo detail sidebar), left-aligned, visible on flop and later streets only:

```
Blueprint   Subgame [solve]   Exact [solve]
```

- **Blueprint** — always available, no solve button. Shows blueprint strategy.
- **Subgame [solve]** — range-solver DCFR with depth-bounding. Clicking `[solve]` starts the solve; while running, becomes `[cancel]`.
- **Exact [solve]** — range-solver DCFR to showdown, no depth-bounding or abstraction. Same solve/cancel pattern.

Active tab is highlighted (cyan text, matching existing accent). Inactive tabs are dimmed. Tab selection persists across node navigation.

### What the tabs control

Everything below the tab bar reflects the active tab's strategy source:
- 13x13 hand matrix (action probabilities, colors)
- Combo detail sidebar (per-combo breakdown, EV)
- Any EV overlays on cells

The action strip above the tabs is shared — it shows the game tree position regardless of which strategy source is active.

### Backend — Dual Solve State

Extend `GameSessionState` to hold two independent solve states:

```rust
pub struct GameSessionState {
    pub session: RwLock<Option<GameSession>>,   // blueprint (always loaded)
    pub subgame_solve: Arc<SolveState>,         // depth-bounded solve cache
    pub exact_solve: Arc<SolveState>,           // full-depth solve cache
}
```

The existing `SolveState` struct is reused as-is — it already has iteration tracking, cancellation, matrix snapshots, and a solve cache (`HashMap<Vec<usize>, CachedSolveNode>`).

### Backend — Command Changes

**`game_solve(mode: String, ...)`** — add `mode` parameter (`"subgame"` or `"exact"`). Selects which `SolveState` to write to and configures the solver accordingly:
- `"subgame"`: uses depth-bounded solving (existing behavior, with leaf evaluator if available)
- `"exact"`: configures range-solver with no depth limit, solves to showdown

**`game_cancel_solve(mode: String)`** — cancels the specific solve by mode.

**`game_get_state(source: String)`** — add `source` parameter (`"blueprint"`, `"subgame"`, or `"exact"`). Controls which strategy populates the matrix in the response:
- `"blueprint"`: reads from loaded blueprint (existing behavior)
- `"subgame"`: reads from `subgame_solve.solve_cache` if current node is in the solved subtree, falls back to empty/unavailable
- `"exact"`: reads from `exact_solve.solve_cache`, same pattern

Navigation commands (`game_play_action`, `game_back`, `game_deal_card`) remain unchanged — they update tree position only. The frontend re-fetches state with the active source after navigation.

### Frontend — State

```typescript
const [activeSource, setActiveSource] = useState<"blueprint" | "subgame" | "exact">("blueprint");
const [subgameSolving, setSubgameSolving] = useState(false);
const [exactSolving, setExactSolving] = useState(false);
```

On tab click: set `activeSource`, call `game_get_state(source)`, update matrix.

On navigation: call `game_get_state(activeSource)` instead of default.

On solve click: call `game_solve(mode)`, start polling `game_get_state(mode)` for progress.

### Frontend — Tab Rendering

Tab bar renders between action strip and strategy container. Simplified text links with `[solve]` / `[cancel]` inline:

```
Blueprint   Subgame [solve]   Exact [solve]
```

When solving:
```
Blueprint   Subgame [cancel] 142/1000 0.3%   Exact [solve]
```

Progress bar at the bottom of the strategy container shows detailed progress for whichever solve is active.

### Solve cache lifecycle

- Each solve cache is independent. Starting a subgame solve doesn't clear the exact cache, and vice versa.
- Navigating outside the solved subtree shows "no solution available" for that tab (matrix empty or greyed out).
- Starting a new solve on the same tab replaces that tab's cache.
- Loading a new blueprint or starting a new game clears both solve caches.

### Removed

- Existing SOLVE button in the action strip (vertical text button) — removed entirely. Solve triggers only from tab bar.

### Not in scope

- Preflop tabs (blueprint only on preflop)
- Saving/exporting solved strategies
- BoundaryNet integration (separate feature, future work)

## File Changes

| File | Change |
|------|--------|
| `crates/tauri-app/src/game_session.rs` | Add second `SolveState`, add `mode`/`source` params to commands |
| `crates/tauri-app/src/exploration.rs` | Expose exact-solve config option |
| `frontend/src/GameExplorer.tsx` | Add tab bar, remove solve button from action strip, wire tab state |
| `frontend/src/Explorer.tsx` | No changes (matrix component is data-driven) |
| `frontend/src/game-types.ts` | Add source field to relevant types if needed |
| `frontend/src/App.css` | Tab bar styles |
| `crates/devserver/src/main.rs` | Update API routes for new command signatures |
