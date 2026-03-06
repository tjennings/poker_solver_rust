# Range Tracking Across Betting Rounds

**Date:** 2026-03-05
**Status:** Approved
**Bean:** poker_solver_rust-vlzf

## Problem

The strategy explorer shows action probabilities at each decision node, but doesn't track how the player's range narrows through the action sequence. Hands that would have folded at earlier decision points still appear at full size. There's a threshold-based binary filter (`is_hand_in_range`) but it doesn't compute cumulative reaching probabilities and has no visual weight scaling.

## Design Decisions

| Decision | Choice |
|-|-|
| Visual display | Bar height scaling (proportional to reaching probability) |
| Existing threshold filter | Remove — not used |
| Granularity | 169 canonical hands preflop → expand to combos at postflop |
| Manual editing | Both players editable |
| Edit persistence | Sticky at their node — edits preserved on navigation |
| Card removal in UI | Show combo-level detail at postflop (blocked combos visible) |
| Solver handoff format | `PlayerRange` struct passed directly |

## Core Data Structure

```rust
/// Per-player range: reaching probabilities for 169 canonical hands.
/// Shared between explorer UI and subgame solver.
pub struct PlayerRange {
    pub hands: [f64; 169],           // reaching probability (0.0–1.0)
    pub source: RangeSource,
    pub overrides: HashSet<usize>,   // indices of manually edited hands
}

pub enum RangeSource {
    Computed,  // auto-derived from strategy replay
    Edited,    // computed + manual overrides
    Manual,    // fully user-specified
}
```

`PlayerRange` is the shared language between explorer and solver. No translation layer.

## Range Computation

### Preflop (169-hand level)

1. Start: all 169 hands at 1.0 (full range)
2. Each action by a player multiplies that player's range by the action's probability from the strategy
3. Example: SB opens → SB range × `strategy[raise]` per hand. BB 3-bets → BB range × `strategy[raise]` per hand
4. Manual overrides replace the computed value at their node and propagate downstream

### Postflop Transition

1. 169-hand ranges expand to combo-level for display (card removal applied)
2. UI shows combo survival count per cell; clicking expands to combo detail (existing `ComboClassPanel`)
3. Solver receives `PlayerRange` directly, expands to combos internally

## Node-Level Override Storage

```rust
pub struct RangeHistory {
    pub snapshots: Vec<RangeSnapshot>,
}

pub struct RangeSnapshot {
    pub p1_range: PlayerRange,
    pub p2_range: PlayerRange,
    pub node_index: usize,  // corresponds to history item index
}
```

- Navigating forward: recompute from last snapshot + strategy
- Rewinding: restore snapshot at that node (including sticky edits)

## Frontend Changes

### Bar Height Scaling
- Each cell's strategy bars render at height proportional to `reaching_probability` for the viewing player
- Full range (1.0) = full cell height. 50% reach = half height. Below ~1% = hidden
- Colors remain at full saturation — only height changes

### Range Editing
- Click+drag on cells to set weight (0.0–1.0)
- Edited cells get a visual indicator (corner dot or colored border)
- Player toggle to switch between editing P1/P2 range

### Removals
- Delete `threshold` state variable and `_setThreshold`
- Remove `filtered` field from `MatrixCell` and related CSS
- Remove `is_hand_in_range` / `action_meets_threshold` backend functions

### New State
```typescript
interface PlayerRange {
    hands: number[];           // [169] reaching probabilities
    source: 'computed' | 'edited' | 'manual';
    overrides: Set<number>;    // manually edited indices
}

// Held in Explorer component state
p1Range: PlayerRange
p2Range: PlayerRange
rangeHistory: RangeSnapshot[]  // snapshots at each decision point
```

## Backend Changes

### New in `poker_solver_core`
- `PlayerRange` struct in a new `range` module (or in `game/`)
- `compute_reaching_range(blueprint, config, board, street_histories, current_history, player) -> [f64; 169]`
  - Refactored from `is_hand_in_range`: returns full array instead of per-hand boolean
  - Multiplies action probabilities cumulatively across all streets

### Modified `get_strategy_matrix`
- Returns `reaching_weights: Vec<f64>` (length 169) alongside existing `cells` data
- Computed via `compute_reaching_range` for the viewing player
- Frontend uses these weights for bar height scaling

### New endpoint
- `update_range` — accepts manual range edits from frontend, stores in explorer state
- `get_ranges` — returns current P1/P2 ranges at the current node

## Solver Handoff

The future real-time postflop subgame solver accepts:
```rust
struct SubgameInput {
    p1_range: PlayerRange,
    p2_range: PlayerRange,
    board: Vec<Card>,
    pot: u32,
    stacks: [u32; 2],
}
```

The solver:
1. Reads `player_range.hands[169]`
2. Expands to combo-level (card removal applied)
3. Normalizes weights
4. Uses as initial reach probabilities for CFR traversal

## Files Affected

| File | Change |
|-|-|
| `crates/core/src/range.rs` (new) | `PlayerRange`, `RangeSource`, `compute_reaching_range` |
| `crates/tauri-app/src/exploration.rs` | Add `reaching_weights` to matrix response, new endpoints, remove threshold filter |
| `crates/devserver/src/main.rs` | Mirror new endpoints |
| `frontend/src/types.ts` | `PlayerRange` interface, updated `StrategyMatrix` |
| `frontend/src/Explorer.tsx` | Range state, bar height scaling, editing UI, remove threshold |
| `frontend/src/App.css` | Bar height scaling styles, edit indicators |
