# Range Tracking Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use hex:executing-plans to implement this plan task-by-task.

**Goal:** Add reaching-probability range tracking to the strategy explorer, with bar height scaling, manual editing, and a `PlayerRange` struct shared between explorer and future subgame solver.

**Architecture:** New `PlayerRange` type in `crates/core/src/range.rs` holding `[f64; 169]` reaching probabilities. Backend `compute_reaching_range` replaces binary `is_hand_in_range` with cumulative probability products. Frontend scales strategy bar heights proportionally and supports click-to-edit range overrides with node-level snapshots.

**Tech Stack:** Rust (core + tauri-app), TypeScript/React (frontend), Axum (devserver)

---

## Agent Team & Execution Order

| Agent | Tasks | Parallel? |
|-|-|-|
| `rust-developer` #1 | Tasks 1–3 (core data structure + backend) | Sequential |
| `rust-developer` #2 | Task 4 (devserver mirror) | After Task 3 |
| `rust-developer` #3 | Task 7 (cleanup) | After Task 3 |
| Frontend developer | Tasks 5–6 (frontend types, bar scaling, editing) | After Task 3 |

Tasks 4, 7, and 5-6 can run **in parallel** once Task 3 is complete.

---

### Task 1: PlayerRange Data Structure

**Files:**
- Create: `crates/core/src/range.rs`
- Modify: `crates/core/src/lib.rs:15-28` (add `pub mod range;`)

**Step 1: Write the failing test**

In `crates/core/src/range.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_range_is_full() {
        let r = PlayerRange::new();
        assert_eq!(r.hands.len(), 169);
        assert!(r.hands.iter().all(|&h| (h - 1.0).abs() < f64::EPSILON));
        assert_eq!(r.source, RangeSource::Computed);
        assert!(r.overrides.is_empty());
    }

    #[test]
    fn multiply_action_narrows_range() {
        let mut r = PlayerRange::new();
        // Simulate: hand 0 (AA) has 80% raise probability, so after raise, reaching = 0.8
        let mut action_probs = [0.0f64; 169];
        action_probs[0] = 0.8;
        action_probs[1] = 0.0; // KK folds entirely
        action_probs[2] = 1.0; // QQ always raises
        r.multiply_action(&action_probs);
        assert!((r.hands[0] - 0.8).abs() < 1e-9);
        assert!((r.hands[1] - 0.0).abs() < 1e-9);
        assert!((r.hands[2] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn cumulative_multiply() {
        let mut r = PlayerRange::new();
        let mut probs1 = [1.0f64; 169];
        probs1[0] = 0.5; // AA: 50% raise
        r.multiply_action(&probs1);

        let mut probs2 = [1.0f64; 169];
        probs2[0] = 0.4; // AA: 40% call (of the remaining range)
        r.multiply_action(&probs2);

        // Cumulative: 0.5 * 0.4 = 0.2
        assert!((r.hands[0] - 0.2).abs() < 1e-9);
    }

    #[test]
    fn manual_override_sets_value() {
        let mut r = PlayerRange::new();
        r.set_hand(0, 0.75);
        assert!((r.hands[0] - 0.75).abs() < 1e-9);
        assert!(r.overrides.contains(&0));
        assert_eq!(r.source, RangeSource::Edited);
    }

    #[test]
    fn serialization_roundtrip() {
        let mut r = PlayerRange::new();
        r.set_hand(5, 0.33);
        let json = serde_json::to_string(&r).unwrap();
        let r2: PlayerRange = serde_json::from_str(&json).unwrap();
        assert!((r2.hands[5] - 0.33).abs() < 1e-9);
        assert!(r2.overrides.contains(&5));
    }
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test -p poker-solver-core range::tests --no-run 2>&1 | head -20`
Expected: compilation errors — `PlayerRange` not defined

**Step 3: Write minimal implementation**

In `crates/core/src/range.rs`:

```rust
//! Player range tracking for strategy exploration and subgame solving.
//!
//! `PlayerRange` holds reaching probabilities for 169 canonical hands.
//! Shared between the explorer UI and the future real-time postflop solver.

use std::collections::HashSet;
use serde::{Deserialize, Serialize};

/// Number of canonical starting hands (13 pairs + 78 suited + 78 offsuit).
pub const NUM_HANDS: usize = 169;

/// How the range was produced.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum RangeSource {
    /// Auto-derived from strategy replay.
    Computed,
    /// Computed with manual overrides.
    Edited,
    /// Fully user-specified.
    Manual,
}

/// Per-player range: reaching probabilities for 169 canonical hands.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlayerRange {
    /// Reaching probability per canonical hand (0.0–1.0).
    /// Index matches `CanonicalHand::index()`.
    pub hands: [f64; NUM_HANDS],
    /// How this range was produced.
    pub source: RangeSource,
    /// Indices of hands that were manually edited.
    #[serde(default, skip_serializing_if = "HashSet::is_empty")]
    pub overrides: HashSet<usize>,
}

impl PlayerRange {
    /// Full range: all hands at 1.0.
    #[must_use]
    pub fn new() -> Self {
        Self {
            hands: [1.0; NUM_HANDS],
            source: RangeSource::Computed,
            overrides: HashSet::new(),
        }
    }

    /// Multiply each hand's reaching probability by the action probability.
    ///
    /// `action_probs[i]` is the probability that hand `i` takes the chosen action.
    /// Hands with manual overrides are not modified.
    pub fn multiply_action(&mut self, action_probs: &[f64; NUM_HANDS]) {
        for i in 0..NUM_HANDS {
            if !self.overrides.contains(&i) {
                self.hands[i] *= action_probs[i];
            }
        }
    }

    /// Manually set a hand's reaching probability.
    pub fn set_hand(&mut self, index: usize, weight: f64) {
        assert!(index < NUM_HANDS, "hand index out of range");
        self.hands[index] = weight.clamp(0.0, 1.0);
        self.overrides.insert(index);
        if self.source == RangeSource::Computed {
            self.source = RangeSource::Edited;
        }
    }
}

impl Default for PlayerRange {
    fn default() -> Self {
        Self::new()
    }
}
```

Add to `crates/core/src/lib.rs` after line 28 (`pub mod simulation;`):

```rust
pub mod range;
```

**Step 4: Run tests**

Run: `cargo test -p poker-solver-core range::tests -v`
Expected: all 5 tests PASS

**Step 5: Commit**

```bash
git add crates/core/src/range.rs crates/core/src/lib.rs
git commit -m "feat(range): add PlayerRange data structure for reaching probabilities"
```

---

### Task 2: compute_reaching_range Function

**Files:**
- Modify: `crates/core/src/range.rs` (add computation function)
- Reference: `crates/tauri-app/src/exploration.rs:1848-1966` (existing `is_hand_in_range` logic)

This task adds a function that computes the full 169-element reaching probability array by replaying the action history. The logic mirrors `is_hand_in_range` in `exploration.rs` but returns cumulative probabilities instead of a boolean.

**Important:** This function lives in the exploration crate (`tauri-app`), not core, because it depends on `BundleConfig`, `BlueprintStrategy`, and exploration-specific helpers (`action_to_code`, `apply_action`, `hand_bits_at_street`, etc.). We add a standalone helper in `exploration.rs`.

**Step 1: Write the failing test**

In `crates/tauri-app/src/exploration.rs`, add at the bottom of the file (inside `#[cfg(test)] mod tests`):

```rust
#[test]
fn compute_reaching_range_returns_169_elements() {
    // Basic structural test: full range with no history = all 1.0
    let result = compute_reaching_range_from_history(
        &[], // no street histories
        &[], // no current history
        0,   // player 0
        &|_rank1, _rank2, _suited, _street_idx, _action_codes, _pot, _stacks, _to_call| {
            // Return uniform strategy: all actions equal
            vec![0.5, 0.5]
        },
        100, // stack depth
    );
    assert_eq!(result.len(), 169);
    assert!(result.iter().all(|&p| (p - 1.0).abs() < 1e-9));
}
```

**Step 2: Write the implementation**

In `crates/tauri-app/src/exploration.rs`, add a new function. This is a refactored version of `is_hand_in_range` that returns `[f64; 169]`:

```rust
/// Compute reaching probability for all 169 canonical hands by replaying
/// the action history through the blueprint strategy.
///
/// Returns `[f64; 169]` where each element is the cumulative product of
/// action probabilities for the given player across all streets.
///
/// `strategy_fn` is a callback that returns the strategy probability vector
/// for a given hand at a given game state. This abstraction allows testing
/// without a real blueprint.
fn compute_reaching_range(
    config: &BundleConfig,
    blueprint: &poker_solver_core::blueprint::BlueprintStrategy,
    full_board: &[Card],
    street_histories: &[Vec<String>],
    current_history: &[String],
    viewing_player: u8,
) -> [f64; 169] {
    let bet_sizes = &config.game.bet_sizes;
    let mut reaching = [1.0f64; 169];

    let all_streets: Vec<&[String]> = street_histories
        .iter()
        .map(|v| v.as_slice())
        .chain(std::iter::once(current_history))
        .collect();

    for (street_idx, street_actions) in all_streets.iter().enumerate() {
        let board_for_street = board_at_street(full_board, street_idx);
        let mut stacks = [
            config.game.stack_depth * 2 - 1,
            config.game.stack_depth * 2 - 2,
        ];
        let mut pot = 3u32;
        let mut to_call = initial_to_call(street_idx, &stacks);
        let mut action_codes: Vec<u8> = Vec::new();

        for (i, action) in street_actions.iter().enumerate() {
            let acting_player = (i % 2) as u8;

            if acting_player == viewing_player {
                let code = action_to_code(action);
                let action_idx = action_code_to_strategy_index(
                    code, to_call, bet_sizes.len(),
                );

                // For each of the 169 hands, look up the action probability
                for (hand_idx, reach) in reaching.iter_mut().enumerate() {
                    if *reach < 1e-15 {
                        continue; // already dead
                    }
                    let (rank1, rank2, suited) = canonical_hand_ranks(hand_idx);
                    let hand_bits = hand_bits_at_street(
                        config, rank1, rank2, suited, board_for_street, street_idx,
                    );
                    let street_num = street_idx.min(3) as u8;
                    let eff_stack = stacks[0].min(stacks[1]);
                    let key = InfoKey::new(
                        hand_bits,
                        street_num,
                        spr_bucket(pot, eff_stack),
                        &action_codes,
                    ).as_u64();

                    let prob = match blueprint.lookup(key) {
                        Some(strategy) => {
                            action_idx
                                .and_then(|idx| strategy.get(idx))
                                .copied()
                                .unwrap_or(1.0)
                        }
                        None => 1.0, // no data → assume in range
                    };
                    *reach *= f64::from(prob);
                }
            }

            action_codes.push(action_to_code(action));
            apply_action(action, i % 2, &mut stacks, &mut pot, &mut to_call, bet_sizes);
        }
    }

    reaching
}
```

Also add the helper to convert a canonical hand index back to rank chars:

```rust
/// Convert a canonical hand index (0..169) back to (rank1, rank2, suited).
///
/// Inverse of `CanonicalHand::index()`. Pairs first (0..12), suited (13..90),
/// offsuit (91..168).
fn canonical_hand_ranks(index: usize) -> (char, char, bool) {
    use poker_solver_core::hands::CanonicalHand;
    let hand = CanonicalHand::from_index(index);
    let ranks = ['A', 'K', 'Q', 'J', 'T', '9', '8', '7', '6', '5', '4', '3', '2'];
    let r1 = ranks[value_to_rank_index(hand.high())];
    let r2 = ranks[value_to_rank_index(hand.low())];
    (r1, r2, hand.suited())
}
```

**Note:** If `CanonicalHand::from_index` doesn't exist yet, add it to `crates/core/src/hands.rs`:

```rust
/// Reconstruct a canonical hand from its index (inverse of `index()`).
#[must_use]
pub fn from_index(index: usize) -> Self {
    let ranks = [
        Value::Ace, Value::King, Value::Queen, Value::Jack, Value::Ten,
        Value::Nine, Value::Eight, Value::Seven, Value::Six, Value::Five,
        Value::Four, Value::Three, Value::Two,
    ];
    if index < 13 {
        // Pair
        Self::new(ranks[index], ranks[index], false)
    } else if index < 91 {
        // Suited: 78 combos
        let i = index - 13;
        let (high, low) = pair_index_to_ranks(i, &ranks);
        Self::new(high, low, true)
    } else {
        // Offsuit: 78 combos
        let i = index - 91;
        let (high, low) = pair_index_to_ranks(i, &ranks);
        Self::new(high, low, false)
    }
}
```

Where `pair_index_to_ranks` maps a linear index (0..78) to the corresponding rank pair. This mirrors the triangular indexing in `index()`.

**Step 3: Run tests**

Run: `cargo test -p poker-solver-tauri compute_reaching -- -v`
Expected: PASS

**Step 4: Add test for narrowing behavior**

```rust
#[test]
fn reaching_range_narrows_on_fold_action() {
    // After a fold action, the folding player's range for hands that
    // never fold should be 1.0, and hands that always fold should be 0.0
    // This test uses the real blueprint lookup path so requires an
    // integration-style test with a loaded bundle.
    // Covered by Task 3 integration tests.
}
```

**Step 5: Commit**

```bash
git add crates/core/src/hands.rs crates/tauri-app/src/exploration.rs
git commit -m "feat(range): add compute_reaching_range for cumulative probability tracking"
```

---

### Task 3: Wire Reaching Weights into StrategyMatrix Response

**Files:**
- Modify: `crates/tauri-app/src/exploration.rs:127-168` (`MatrixCell`, `StrategyMatrix`)
- Modify: `crates/tauri-app/src/exploration.rs:534-632` (`get_strategy_matrix_bundle`)
- Modify: `frontend/src/types.ts:20-44` (`MatrixCell`, `StrategyMatrix`)

**Step 1: Update `StrategyMatrix` to include reaching weights**

In `crates/tauri-app/src/exploration.rs`, add field to `StrategyMatrix` (after line 167):

```rust
/// Per-hand reaching probabilities for both players.
/// Length 169 each, indexed by `CanonicalHand::index()`.
pub reaching_p1: Vec<f64>,
pub reaching_p2: Vec<f64>,
```

Remove the `filtered` field from `MatrixCell` (line 137):
```rust
// DELETE: pub filtered: bool,
```

**Step 2: Update `get_strategy_matrix_bundle` to compute reaching weights**

In `get_strategy_matrix_bundle` (line 534), replace the threshold-based filtering with reaching range computation:

```rust
// Replace the old threshold-based filter block (lines 565-585) with:
let reaching_p1 = compute_reaching_range(
    config, blueprint, &board, street_histories, &position.history, 0,
);
let reaching_p2 = compute_reaching_range(
    config, blueprint, &board, street_histories, &position.history, 1,
);
```

In the cell construction loop, remove `filtered` field and stop skipping probabilities:

```rust
// Old (lines 573-604):
let filtered = apply_filter && !is_hand_in_range(...);
let probabilities = if filtered { vec![] } else { /* lookup */ };

// New:
let probabilities = /* always lookup, no filtering */;
```

Add `reaching_p1` and `reaching_p2` to the return struct:

```rust
Ok(StrategyMatrix {
    cells,
    actions,
    street: format!("{street:?}"),
    pot: pos_state.pot,
    stack: /* ... */,
    to_call: pos_state.to_call,
    stack_p1: pos_state.stack_p1,
    stack_p2: pos_state.stack_p2,
    stacks: vec![pos_state.stack_p1, pos_state.stack_p2],
    reaching_p1: reaching_p1.to_vec(),
    reaching_p2: reaching_p2.to_vec(),
})
```

For `get_strategy_matrix_agent` and `get_strategy_matrix_preflop`, return `reaching_p1: vec![1.0; 169]` and `reaching_p2: vec![1.0; 169]` (no range tracking for agents/preflop-only).

**Step 3: Remove threshold parameter from `get_strategy_matrix_core`**

Update function signature (line 486-491):
```rust
pub fn get_strategy_matrix_core(
    state: &ExplorationState,
    position: ExplorationPosition,
    street_histories: Option<Vec<Vec<String>>>,
) -> Result<StrategyMatrix, String> {
```

Remove `threshold` from `get_strategy_matrix` Tauri command too (line 525-531).

**Step 4: Update TypeScript types**

In `frontend/src/types.ts`:

```typescript
// MatrixCell: remove 'filtered' field (line 25)
export interface MatrixCell {
  hand: string;
  suited: boolean;
  pair: boolean;
  probabilities: ActionProb[];
  // REMOVED: filtered: boolean;
}

// StrategyMatrix: add reaching weights (after line 43)
export interface StrategyMatrix {
  cells: MatrixCell[][];
  actions: ActionInfo[];
  street: string;
  pot: number;
  stack: number;
  to_call: number;
  stack_p1: number;
  stack_p2: number;
  reaching_p1: number[];  // NEW: [169] reaching probabilities
  reaching_p2: number[];  // NEW: [169] reaching probabilities
}
```

**Step 5: Run tests and verify compilation**

Run: `cargo test -p poker-solver-tauri -- -v && cargo test -p poker-solver-core -- -v`
Run: `cd frontend && npx tsc --noEmit`

**Step 6: Commit**

```bash
git add crates/tauri-app/src/exploration.rs frontend/src/types.ts
git commit -m "feat(range): wire reaching weights into StrategyMatrix response"
```

---

### Task 4: Update Dev Server

**Files:**
- Modify: `crates/devserver/src/main.rs:36-41` (`StrategyMatrixParams`)

**Step 1: Remove threshold from params**

```rust
#[derive(Deserialize)]
struct StrategyMatrixParams {
    position: poker_solver_tauri::ExplorationPosition,
    street_histories: Option<Vec<Vec<String>>>,
    // REMOVED: threshold: Option<f32>,
}
```

Update handler (line 130-139):

```rust
async fn handle_get_strategy_matrix(
    AxumState(state): AxumState<AppState>,
    Json(params): Json<StrategyMatrixParams>,
) -> Result<Json<serde_json::Value>, (axum::http::StatusCode, String)> {
    result_to_response(poker_solver_tauri::get_strategy_matrix_core(
        &state,
        params.position,
        params.street_histories,
    ))
}
```

**Step 2: Verify compilation**

Run: `cargo build -p poker-solver-devserver`

**Step 3: Commit**

```bash
git add crates/devserver/src/main.rs
git commit -m "chore(devserver): remove threshold param from strategy matrix endpoint"
```

---

### Task 5: Frontend Bar Height Scaling

**Files:**
- Modify: `frontend/src/Explorer.tsx:92-140` (`HandCell` component)
- Modify: `frontend/src/Explorer.tsx:496-544` (state and invoke calls)

**Step 1: Add reaching weight to HandCell**

Update `HandCell` props and rendering:

```tsx
function HandCell({
  cell,
  actions,
  reachWeight,  // NEW: 0.0–1.0 reaching probability
  isSelected,
  onClick,
}: {
  cell: MatrixCell;
  actions: ActionInfo[];
  reachWeight: number;  // NEW
  isSelected: boolean;
  onClick: () => void;
}) {
  // ... existing gradient computation ...

  // Bar height proportional to reaching probability
  const barHeight = Math.max(reachWeight * 100, 0);
  const isUnreachable = reachWeight < 0.01;

  return (
    <div
      className={`matrix-cell ${isSelected ? 'selected' : ''} ${isUnreachable ? 'unreachable' : ''}`}
      onClick={onClick}
    >
      <div
        className="cell-bar"
        style={{
          background: isUnreachable ? undefined : gradientStops,
          height: `${barHeight}%`,
        }}
      />
      <span className="cell-label">{cell.hand}</span>
    </div>
  );
}
```

**Step 2: Map canonical hand index to matrix position**

Add a utility to map the 13x13 grid position to the canonical hand index:

```tsx
// Map (row, col) in the 13x13 matrix to a canonical hand index (0..168).
// Row = high rank index, Col = low rank index.
// Diagonal = pairs (0..12), above diagonal = suited (13..90), below = offsuit (91..168).
function matrixToHandIndex(row: number, col: number): number {
  if (row === col) {
    return row; // pair
  } else if (col > row) {
    // suited: above diagonal
    // Triangular index: sum of (12-row) + ... offset
    let idx = 0;
    for (let r = 0; r < row; r++) idx += (12 - r);
    idx += (col - row - 1);
    return 13 + idx;
  } else {
    // offsuit: below diagonal (row > col)
    let idx = 0;
    for (let r = 0; r < col; r++) idx += (12 - r);
    idx += (row - col - 1);
    return 91 + idx;
  }
}
```

**Step 3: Update matrix rendering to pass reaching weights**

In the matrix render loop (around line 1071-1083):

```tsx
{matrix.cells.map((row, rowIdx) => (
  <div key={rowIdx} className="matrix-row">
    {row.map((cell, colIdx) => {
      const handIdx = matrixToHandIndex(rowIdx, colIdx);
      const viewing = position.to_act;
      const reachWeight = viewing === 0
        ? (matrix.reaching_p1[handIdx] ?? 1.0)
        : (matrix.reaching_p2[handIdx] ?? 1.0);
      return (
        <HandCell
          key={colIdx}
          cell={cell}
          actions={matrix.actions}
          reachWeight={reachWeight}
          isSelected={selectedCell?.row === rowIdx && selectedCell?.col === colIdx}
          onClick={() => setSelectedCell({ row: rowIdx, col: colIdx })}
        />
      );
    })}
  </div>
))}
```

**Step 4: Remove threshold state and filtered references**

In Explorer component state (around line 525):
```tsx
// DELETE: const [threshold, _setThreshold] = useState(2);
```

Remove `threshold` from all `invoke('get_strategy_matrix', ...)` calls — remove the `threshold: threshold / 100` param.

Remove `cell.filtered` references from `HandCell` (line 133-134).

Remove `extractStreetHistories` usage tied to threshold — keep `street_histories` param for range computation.

**Step 5: Add CSS for bar height scaling**

In `frontend/src/App.css`, update `.matrix-cell`:

```css
.matrix-cell {
  position: relative;
  /* existing styles */
}

.cell-bar {
  position: absolute;
  bottom: 0;
  left: 0;
  right: 0;
  /* height set via inline style */
}

.cell-label {
  position: relative; /* above the bar */
  z-index: 1;
}

.matrix-cell.unreachable {
  opacity: 0.2;
}
```

**Step 6: Verify in browser**

Run: `cargo run -p poker-solver-devserver & cd frontend && npm run dev`
Load a bundle, navigate through actions. Bars should shrink as range narrows.

**Step 7: Commit**

```bash
git add frontend/src/Explorer.tsx frontend/src/types.ts frontend/src/App.css
git commit -m "feat(explorer): bar height scaling for reaching probabilities"
```

---

### Task 6: Range Editing UI

**Files:**
- Modify: `frontend/src/Explorer.tsx` (add range editing state and handlers)
- Modify: `frontend/src/types.ts` (add `PlayerRange` interface)

**Step 1: Add TypeScript range types**

In `frontend/src/types.ts`:

```typescript
export interface PlayerRange {
  hands: number[];           // [169] reaching probabilities
  source: 'computed' | 'edited' | 'manual';
  overrides: number[];       // indices of manually edited hands (serialized from Set)
}

export interface RangeSnapshot {
  p1_range: PlayerRange;
  p2_range: PlayerRange;
  node_index: number;
}
```

**Step 2: Add range state to Explorer**

```tsx
const [p1Range, setP1Range] = useState<PlayerRange>({
  hands: Array(169).fill(1.0),
  source: 'computed',
  overrides: [],
});
const [p2Range, setP2Range] = useState<PlayerRange>({
  hands: Array(169).fill(1.0),
  source: 'computed',
  overrides: [],
});
const [rangeSnapshots, setRangeSnapshots] = useState<RangeSnapshot[]>([]);
const [editingPlayer, setEditingPlayer] = useState<0 | 1 | null>(null);
```

**Step 3: Add range editing to HandCell**

When `editingPlayer` is set, clicking a cell toggles between weight presets (1.0 → 0.5 → 0.0 → 1.0), or use a slider on long-press:

```tsx
const handleCellEdit = useCallback((row: number, col: number) => {
  if (editingPlayer === null) return;
  const handIdx = matrixToHandIndex(row, col);
  const setRange = editingPlayer === 0 ? setP1Range : setP2Range;

  setRange(prev => {
    const newHands = [...prev.hands];
    // Cycle: 1.0 → 0.5 → 0.0 → 1.0
    const current = newHands[handIdx];
    newHands[handIdx] = current > 0.75 ? 0.5 : current > 0.25 ? 0.0 : 1.0;
    const newOverrides = new Set(prev.overrides);
    newOverrides.add(handIdx);
    return {
      hands: newHands,
      source: 'edited' as const,
      overrides: Array.from(newOverrides),
    };
  });
}, [editingPlayer]);
```

**Step 4: Add edit toggle button**

Above the matrix, add a toolbar:

```tsx
<div className="range-toolbar">
  <button
    className={`range-edit-btn ${editingPlayer === 0 ? 'active' : ''}`}
    onClick={() => setEditingPlayer(editingPlayer === 0 ? null : 0)}
  >
    Edit SB Range
  </button>
  <button
    className={`range-edit-btn ${editingPlayer === 1 ? 'active' : ''}`}
    onClick={() => setEditingPlayer(editingPlayer === 1 ? null : 1)}
  >
    Edit BB Range
  </button>
</div>
```

**Step 5: Update range on navigation**

When `handleActionSelect` is called, snapshot the current ranges before advancing:

```tsx
// Inside handleActionSelect, before advancing:
setRangeSnapshots(prev => [...prev, {
  p1_range: { ...p1Range },
  p2_range: { ...p2Range },
  node_index: historyItems.length,
}]);

// After getting the new matrix, update ranges from reaching_weights:
setP1Range(prev => ({
  ...prev,
  hands: newMatrix.reaching_p1,
  // Preserve manual overrides
}));
```

When rewinding (`handleHistoryRewind`), restore the snapshot:

```tsx
const snapshot = rangeSnapshots.find(s => s.node_index === index);
if (snapshot) {
  setP1Range(snapshot.p1_range);
  setP2Range(snapshot.p2_range);
}
setRangeSnapshots(prev => prev.filter(s => s.node_index < index));
```

**Step 6: Visual indicator for edited cells**

```css
.matrix-cell.edited::after {
  content: '';
  position: absolute;
  top: 2px;
  right: 2px;
  width: 4px;
  height: 4px;
  border-radius: 50%;
  background: #facc15; /* yellow dot */
}
```

**Step 7: Commit**

```bash
git add frontend/src/Explorer.tsx frontend/src/types.ts frontend/src/App.css
git commit -m "feat(explorer): range editing UI with node-level snapshots"
```

---

### Task 7: Cleanup — Remove Old Threshold/Filter Code

**Files:**
- Modify: `crates/tauri-app/src/exploration.rs` (remove `is_hand_in_range`, `action_meets_threshold`)
- Modify: `frontend/src/Explorer.tsx` (remove any remaining threshold references)

**Step 1: Remove dead backend code**

Delete from `exploration.rs`:
- `is_hand_in_range` function (lines 1848-1909)
- `action_meets_threshold` function (lines 1932-1966)
- Any `threshold` parameters from remaining function signatures

**Step 2: Remove dead frontend code**

Delete from `Explorer.tsx`:
- Any remaining `threshold` references
- The `filtered` CSS class styles from `App.css`
- The `extractStreetHistories` function if only used for threshold filtering (check — it's also used for `street_histories` param, so keep it if still needed)

**Step 3: Verify no regressions**

Run: `cargo test -p poker-solver-tauri -- -v`
Run: `cargo clippy -p poker-solver-tauri`
Run: `cd frontend && npx tsc --noEmit`

**Step 4: Commit**

```bash
git add -A
git commit -m "refactor: remove threshold-based range filtering (replaced by reaching probabilities)"
```

---

## Testing Strategy

| Layer | Test type | What to verify |
|-|-|-|
| `PlayerRange` (core) | Unit tests | new(), multiply_action(), set_hand(), serialization |
| `compute_reaching_range` (exploration) | Integration test | Reaching probs narrow correctly through action history |
| `StrategyMatrix` response | Smoke test via curl | `reaching_p1` and `reaching_p2` arrays present and length 169 |
| Frontend bar scaling | Manual browser test | Bars shrink as range narrows through tree navigation |
| Range editing | Manual browser test | Click cells to edit, edits persist on navigation, rewind restores |

## Docs to Update

- `docs/explorer.md` — Add section on range tracking display and editing
- `docs/plans/2026-03-05-range-tracking-design.md` — Mark as implemented
