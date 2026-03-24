# Unified Backend-Driven Game State — Design

**Date:** 2026-03-24

## Problem

The frontend currently juggles two player conventions (V2 tree: player 0=SB; range-solver: player 0=OOP/BB), two matrix sources (blueprint vs solver), two separate components (Explorer for preflop, PostflopExplorer for postflop), and assembles game state from scattered pieces. Every new code path risks swapping OOP/IP. The frontend makes decisions about whose turn it is, what position label to show, when to switch modes, and how to propagate ranges — all of which are game-logic concerns that belong in the backend.

## Design

### Principle

**Backend owns all game state. Frontend is display-only.**

One render loop: `render(state) → user clicks → backend updates → new state → re-render`.

No preflop vs postflop distinction in the frontend. No blueprint vs solver mode. No player convention logic. The backend returns position labels as strings ("BB", "SB"), not indices.

### Backend: `GameSession`

A stateful session tracking the full game from preflop through river:

```rust
pub struct GameSession {
    // Blueprint source
    tree: Arc<V2GameTree>,
    strategy: Arc<BlueprintV2Strategy>,
    decision_map: Arc<Vec<u32>>,
    cbv_context: Option<Arc<CbvContext>>,
    hand_evs: Option<Arc<Vec<[[f64; 169]; 2]>>>,

    // Current position
    node_idx: u32,
    board: Vec<String>,
    action_history: Vec<ActionRecord>,

    // Ranges — always: oop = BB, ip = SB
    oop_weights: Vec<f32>,  // 1326
    ip_weights: Vec<f32>,   // 1326

    // Solver (None until solve requested)
    subgame: Option<SubgameSolveState>,
}

pub struct ActionRecord {
    pub action_id: String,
    pub label: String,       // "Bet 66%", "Call", "Check"
    pub position: String,    // "BB" or "SB"
    pub pot: i32,
    pub stack: i32,
    pub street: String,
}
```

The session normalizes player conventions internally. V2 tree `player == tree.dealer` → SB/IP, `player != tree.dealer` → BB/OOP. This mapping lives in ONE place in the backend.

### One return type: `GameState`

Every query returns the same shape:

```rust
pub struct GameState {
    pub street: String,              // "Preflop", "Flop", "Turn", "River"
    pub position: String,            // "BB" or "SB" — who acts next
    pub board: Vec<String>,
    pub pot: i32,
    pub stacks: [i32; 2],           // [BB stack, SB stack]
    pub matrix: Option<StrategyMatrix>,  // 13x13 grid, actions, EVs, weights
    pub actions: Vec<ActionInfo>,    // available actions at this node
    pub action_history: Vec<ActionRecord>,
    pub is_terminal: bool,
    pub is_chance: bool,             // waiting for next card
    pub solve: Option<SolveStatus>,  // if solving: iteration, max, exploitability, elapsed
}
```

### Commands

| Command | Input | Purpose |
|---------|-------|---------|
| `game_new(blueprint_path)` | Path to blueprint bundle | Load blueprint, start at preflop root |
| `game_get_state()` | — | Return current GameState (used for polling during solve too) |
| `game_play_action(action_id)` | Action ID string | Take action, update ranges, advance tree |
| `game_deal_card(card)` | Card string e.g. "Ah" | Deal next street card, advance past chance node |
| `game_back()` | — | Undo last action |
| `game_solve(params)` | Solver config (iters, exploitability, etc.) | Run subgame solver from current position |

### Frontend: One Component

```tsx
function GameExplorer() {
  const [state, setState] = useState<GameState | null>(null);

  const playAction = async (actionId: string) => {
    await invoke('game_play_action', { action_id: actionId });
    setState(await invoke('game_get_state'));
  };

  // Renders: matrix grid, action cards from state.actions,
  // breadcrumbs from state.action_history, board cards,
  // solve progress from state.solve
}
```

One `<StrategyMatrix>` component renders the 13x13 grid for preflop, flop, turn, river — all the same. One `<ActionCard>` component renders the available actions. The action history is a list of `<ActionRecord>` cards rendered from `state.action_history`.

### What Gets Removed

**Backend:**
- `PostFlopGame` / range-solver full-depth path (`solve_full_depth`, `build_game`, `capture_matrix_snapshot`, `build_strategy_matrix`)
- `game: Mutex<Option<PostFlopGame>>` from PostflopState
- `SolverChoice::FullDepth` dispatch branch
- Separate blueprint matrix building in `get_strategy_matrix_v2` (absorbed into GameSession)
- `get_preflop_ranges_core` / `blueprint_propagate_ranges` as separate functions (range tracking is built into action progression)
- All OOP/IP player convention logic scattered across multiple functions

**Frontend:**
- `PostflopExplorer.tsx` (merged into unified component)
- `Explorer.tsx` preflop-specific matrix/action handling (replaced by unified component)
- `blueprintMode`, `blueprintHistory`, `handleBlueprintAction`, `handleAction` — all separate code paths
- `blueprintToPostflopMatrix` conversion
- `fetchBlueprintMatrix` direct calls
- All `matrix.player === matrix.dealer` logic
- Frontend-managed `actionHistory` state (backend owns it now)

### What Stays

- The V2 game tree, strategy, decision_map — the blueprint data
- `CfvSubgameSolver` — the depth-limited subgame solver
- `snapshot_from_subgame`, `build_matrix_from_snapshot` — matrix building from subgame results
- Street transition logic (deal card → advance past chance node)
- Board card picker UI
- Matrix grid rendering component (simplified — no mode switching)
- Solve progress display

### Migration Path

This is a large refactor. Suggested order:

1. Build `GameSession` backend with the 6 commands, backed by the V2 tree
2. Build new unified frontend component that only talks to those commands
3. Wire it up, verify preflop + postflop blueprint navigation works
4. Re-integrate subgame solver into `game_solve`
5. Remove old Explorer.tsx, PostflopExplorer.tsx, and dead backend code
