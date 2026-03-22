# Blueprint Postflop Explorer — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use hex:executing-plans to implement this plan task-by-task.

**Goal:** Show the blueprint's postflop strategy immediately after a flop is dealt, navigable before and without running the subgame solver.

**Architecture:** Extend the existing `get_strategy_matrix_v2` backend function to perform real postflop bucket lookups (instead of returning uniform distributions). The frontend PostflopExplorer gains a "blueprint mode" that calls `get_strategy_matrix` for tree navigation, with a conversion layer to `PostflopStrategyMatrix` for rendering. When SOLVE is clicked, the display transitions to the solver's live-updating output.

**Tech Stack:** Rust (tauri-app crate), TypeScript/React (frontend), existing `AllBuckets`/`BlueprintV2Strategy` infrastructure.

---

## Context for Implementers

### Key Types

- **`StrategyMatrix`** (`exploration.rs:152`): 13×13 grid of `MatrixCell` with `ActionProb` probabilities. Returned by `get_strategy_matrix`. Has `street`, `pot`, `stack_p1`, `stack_p2`, `reaching_p1`, `reaching_p2`.
- **`PostflopStrategyMatrix`** (`postflop.rs:114`): 13×13 grid of `PostflopMatrixCell` with `f32[]` probabilities, `combo_count`, `ev`, `combos`, `weight`. Used by PostflopExplorer for rendering.
- **`AllBuckets`** (`core/blueprint_v2/mccfr.rs:88`): Unified bucket lookup. `get_bucket(street, [Card;2], &[Card]) -> u16`. Already loaded in `CbvContext`.
- **`CbvContext`** (`postflop.rs:742`): Has `all_buckets: Arc<AllBuckets>` and `strategy: Arc<BlueprintV2Strategy>`. Stored in `PostflopState.cbv_context`.
- **`BlueprintV2Strategy`** (`core/blueprint_v2/bundle.rs:36`): `get_action_probs(decision_idx, bucket) -> &[f32]`.

### Key Functions

- **`get_strategy_matrix_v2`** (`exploration.rs:907`): Builds `StrategyMatrix` from blueprint. Lines 1030-1042 are the postflop stub returning uniform distributions — this is what we fix.
- **`get_strategy_matrix_core`** (`exploration.rs:619`): Entry point that dispatches to `get_strategy_matrix_v2`. Currently does not have access to `PostflopState`.
- **`walk_v2_tree`** (`exploration.rs:733`): Walks abstract game tree following action history. Auto-skips chance nodes.
- **`build_matrix_from_snapshot`** (`postflop.rs:885`): Aggregates per-combo data into 13×13 cells — pattern to follow for combo averaging.
- **`toMatrixCell`** (`PostflopExplorer.tsx:25`): Converts `PostflopMatrixCell` → `MatrixCell` for rendering.

### Navigation Model

- **Blueprint mode**: Call `get_strategy_matrix` with `ExplorationPosition { board, history }`. The history uses action strings (`"f"`, `"c"`, `"r:0"`, `"r:A"`). `walk_v2_tree` auto-skips chance nodes.
- **Solver mode**: Call `postflop_play_action(action_index)`. The history uses numeric indices. This is the existing behavior.
- **Street transition**: When `get_strategy_matrix` returns `street: "Turn"` but `board.length === 3`, the frontend should prompt for a turn card. After card picked, re-call with updated board. The bucket lookups automatically use the new board cards.

---

## Task 1: Backend — Thread CbvContext into get_strategy_matrix_v2

**Files:**
- Modify: `crates/tauri-app/src/exploration.rs:619-650` (get_strategy_matrix_core, get_strategy_matrix)
- Modify: `crates/tauri-app/src/exploration.rs:907` (get_strategy_matrix_v2 signature)
- Modify: `crates/devserver/src/main.rs` (if get_strategy_matrix endpoint needs PostflopState)

**Step 1: Write a test that calls get_strategy_matrix_v2 with a postflop position**

Add a test in `exploration.rs` that loads a blueprint with bucket files, walks to a flop decision node, and asserts the returned probabilities are NOT uniform (i.e., not all `1/N`).

```rust
#[test]
fn test_postflop_strategy_matrix_uses_buckets() {
    // This test will initially fail because the postflop branch returns uniform distributions.
    // After Task 2, it should pass.
    // For now, just verify the plumbing — that CbvContext reaches get_strategy_matrix_v2.
}
```

**Note:** This test may need a real blueprint + bucket files. If test fixtures don't exist, create a minimal fixture or use `#[ignore]` initially.

**Step 2: Add `cbv_context: Option<&CbvContext>` parameter to `get_strategy_matrix_v2`**

```rust
fn get_strategy_matrix_v2(
    _config: &BlueprintV2Config,
    strategy: &BlueprintV2Strategy,
    tree: &V2GameTree,
    decision_map: &[u32],
    position: &ExplorationPosition,
    cbv_context: Option<&CbvContext>,  // NEW
) -> Result<StrategyMatrix, String> {
```

**Step 3: Update `get_strategy_matrix_core` to accept and pass through CbvContext**

Change signature to accept `postflop_state: Option<&PostflopState>`. Extract `CbvContext` from `postflop_state.cbv_context.read()` and pass to `get_strategy_matrix_v2`.

```rust
pub fn get_strategy_matrix_core(
    state: &ExplorationState,
    position: ExplorationPosition,
    _threshold: Option<f32>,
    _street_histories: Option<Vec<Vec<String>>>,
    postflop_state: Option<&PostflopState>,  // NEW
) -> Result<StrategyMatrix, String> {
    // ...
    StrategySource::BlueprintV2 { config, strategy, tree, decision_map, .. } => {
        let cbv_guard = postflop_state.and_then(|ps| {
            let guard = ps.cbv_context.read();
            // Need to clone the Arc to avoid holding the lock
            guard.clone()
        });
        get_strategy_matrix_v2(config, strategy, tree, decision_map, &position, cbv_guard.as_deref())
    }
}
```

**Step 4: Update Tauri command `get_strategy_matrix` to inject PostflopState**

```rust
#[tauri::command(rename_all = "snake_case")]
pub fn get_strategy_matrix(
    state: State<'_, ExplorationState>,
    postflop_state: State<'_, Arc<PostflopState>>,  // NEW
    position: ExplorationPosition,
    threshold: Option<f32>,
    street_histories: Option<Vec<Vec<String>>>,
) -> Result<StrategyMatrix, String> {
    get_strategy_matrix_core(&state, position, threshold, street_histories, Some(&postflop_state))
}
```

**Step 5: Update devserver endpoint if needed**

The devserver mirrors Tauri commands. If `get_strategy_matrix` endpoint doesn't pass PostflopState, add it.

**Step 6: Run tests and verify compilation**

```bash
cargo test -p poker-solver-tauri --lib
```

**Step 7: Commit**

```bash
git add crates/tauri-app/src/exploration.rs crates/devserver/src/main.rs
git commit -m "feat: thread CbvContext into get_strategy_matrix_v2 for postflop bucket lookups"
```

---

## Task 2: Backend — Real postflop bucket lookup in get_strategy_matrix_v2

**Files:**
- Modify: `crates/tauri-app/src/exploration.rs:1030-1042` (the postflop stub)

**Step 1: Write a test for postflop blueprint strategy extraction**

A unit test that verifies a specific postflop position returns non-uniform probabilities when buckets are available. Requires test fixtures (blueprint + bucket files). If no fixtures exist, create a helper that builds a minimal `AllBuckets` + `BlueprintV2Strategy` in-memory.

**Step 2: Replace the uniform distribution stub with real bucket lookups**

Replace lines 1030-1042 in `get_strategy_matrix_v2`:

```rust
} else {
    // Postflop: look up each combo's bucket, query blueprint strategy
    if let Some(ctx) = cbv_context {
        let board = parse_board(&position.board)
            .map_err(|e| format!("Invalid board for bucket lookup: {e}"))?;
        let street = street_from_board_len(board.len())?;
        let suits = [Suit::Spade, Suit::Heart, Suit::Diamond, Suit::Club];
        let v1 = char_to_value(rank1);
        let v2 = char_to_value(rank2);

        let mut total_prob = vec![0.0f64; actions.len()];
        let mut combo_count = 0u32;

        if pair {
            // Pairs: 6 combos (choose 2 from 4 suits)
            for si in 0..4 {
                for sj in (si + 1)..4 {
                    let c1 = Card::new(v1, suits[si]);
                    let c2 = Card::new(v1, suits[sj]);
                    if board.contains(&c1) || board.contains(&c2) { continue; }
                    let bucket = ctx.all_buckets.get_bucket(street, [c1, c2], &board);
                    let probs = ctx.strategy.get_action_probs(decision_idx as usize, bucket);
                    for (k, &p) in probs.iter().enumerate() {
                        if k < total_prob.len() { total_prob[k] += p as f64; }
                    }
                    combo_count += 1;
                }
            }
        } else if suited {
            // Suited: 4 combos (same suit)
            for &s in &suits {
                let c1 = Card::new(v1, s);
                let c2 = Card::new(v2, s);
                if board.contains(&c1) || board.contains(&c2) { continue; }
                let bucket = ctx.all_buckets.get_bucket(street, [c1, c2], &board);
                let probs = ctx.strategy.get_action_probs(decision_idx as usize, bucket);
                for (k, &p) in probs.iter().enumerate() {
                    if k < total_prob.len() { total_prob[k] += p as f64; }
                }
                combo_count += 1;
            }
        } else {
            // Offsuit: 12 combos (different suits)
            for &s1 in &suits {
                for &s2 in &suits {
                    if s1 == s2 { continue; }
                    let c1 = Card::new(v1, s1);
                    let c2 = Card::new(v2, s2);
                    if board.contains(&c1) || board.contains(&c2) { continue; }
                    let bucket = ctx.all_buckets.get_bucket(street, [c1, c2], &board);
                    let probs = ctx.strategy.get_action_probs(decision_idx as usize, bucket);
                    for (k, &p) in probs.iter().enumerate() {
                        if k < total_prob.len() { total_prob[k] += p as f64; }
                    }
                    combo_count += 1;
                }
            }
        }

        if combo_count > 0 {
            actions.iter().enumerate().map(|(k, a)| ActionProb {
                action: a.label.clone(),
                probability: (total_prob[k] / combo_count as f64) as f32,
            }).collect()
        } else {
            // All combos blocked by board
            actions.iter().map(|a| ActionProb {
                action: a.label.clone(),
                probability: 0.0,
            }).collect()
        }
    } else {
        // No bucket data available — fall back to uniform
        let n = actions.len();
        let uniform = 1.0 / n as f32;
        actions.iter().map(|a| ActionProb {
            action: a.label.clone(),
            probability: uniform,
        }).collect()
    }
};
```

**Step 3: Extend reaching probability computation for postflop**

The reach computation loop (lines 920-967) currently uses only preflop bucket indices. For postflop nodes, it needs to use `AllBuckets::get_bucket`. Extend the loop: when a node's street is postflop and `cbv_context` is available, enumerate combos per canonical hand and average the action probability for reach multiplication.

This is critical for blueprint multi-street navigation where actions on earlier streets narrow the range.

**Step 4: Run tests**

```bash
cargo test -p poker-solver-tauri --lib
```

**Step 5: Commit**

```bash
git add crates/tauri-app/src/exploration.rs
git commit -m "feat: postflop get_strategy_matrix uses real bucket lookups instead of uniform"
```

---

## Task 3: Frontend — Add blueprint mode and StrategyMatrix conversion

**Files:**
- Modify: `frontend/src/PostflopExplorer.tsx`
- Modify: `frontend/src/types.ts` (import StrategyMatrix if not already imported in PostflopExplorer)

**Step 1: Add conversion function**

In `PostflopExplorer.tsx`, add a function to convert `StrategyMatrix` → `PostflopStrategyMatrix`:

```typescript
import { StrategyMatrix } from './types';

/** Convert a blueprint StrategyMatrix to PostflopStrategyMatrix for rendering. */
function blueprintToPostflopMatrix(
  sm: StrategyMatrix,
  board: string[],
  player: number,
): PostflopStrategyMatrix {
  return {
    cells: sm.cells.map(row =>
      row.map(cell => ({
        hand: cell.hand,
        suited: cell.suited,
        pair: cell.pair,
        probabilities: cell.probabilities.map(ap => ap.probability),
        combo_count: cell.probabilities.some(ap => ap.probability > 0) ? 1 : 0,
        ev: null,
        combos: [],
        weight: cell.probabilities.some(ap => ap.probability > 0) ? 1.0 : 0.0,
      })),
    ),
    actions: sm.actions,
    player,
    pot: sm.pot,
    stacks: [sm.stack_p1, sm.stack_p2],
    board,
  };
}
```

**Step 2: Add blueprint mode state**

```typescript
// Blueprint navigation state
const [blueprintMode, setBlueprintMode] = useState(false);
const [blueprintHistory, setBlueprintHistory] = useState<string[]>([]);
const [solved, setSolved] = useState(false);
```

**Step 3: Commit**

```bash
git add frontend/src/PostflopExplorer.tsx
git commit -m "feat: add blueprint mode state and StrategyMatrix conversion"
```

---

## Task 4: Frontend — Fetch blueprint strategy after flop deal

**Files:**
- Modify: `frontend/src/PostflopExplorer.tsx` (board change effect, ~lines 170-204)

**Step 1: Replace cache check with blueprint strategy fetch**

Replace the `useEffect` that checks cache (lines 171-204) with one that fetches the blueprint strategy:

```typescript
useEffect(() => {
  if (!configSummary || boardCards.length < 3 || solving || solved) return;

  let cancelled = false;
  const fetchBlueprint = async () => {
    try {
      // Determine to_act from tree position (default: OOP=0 acts first postflop)
      const toAct = 0;
      const sm = await invoke<StrategyMatrix>('get_strategy_matrix', {
        position: {
          board: boardCards,
          history: blueprintHistory,
          pot: config.pot,
          stacks: [config.effective_stack, config.effective_stack],
          to_act: toAct,
          num_players: 2,
          active_players: [true, true],
        },
      });
      if (cancelled) return;
      const player = sm.street === 'Preflop' ? 0 : toAct;
      setMatrix(blueprintToPostflopMatrix(sm, boardCards, player));
      setBlueprintMode(true);
      setNeedsSolve(true); // Show SOLVE button
    } catch (e) {
      if (!cancelled) {
        setError(String(e));
        setNeedsSolve(true);
      }
    }
  };
  fetchBlueprint();
  return () => { cancelled = true; };
}, [boardCards.length, configSummary, blueprintHistory]); // eslint-disable-line react-hooks/exhaustive-deps
```

**Note:** The `ExplorationPosition` needs to include the **full** action history from the preflop explorer through the postflop actions. The `preflopHistory` prop contains preflop actions. The blueprint tree history needs to include both preflop and postflop action strings. Check how Explorer.tsx builds the history and replicate the same format.

**Step 2: Commit**

```bash
git add frontend/src/PostflopExplorer.tsx
git commit -m "feat: fetch and display blueprint strategy after flop deal"
```

---

## Task 5: Frontend — Blueprint action navigation

**Files:**
- Modify: `frontend/src/PostflopExplorer.tsx` (handleAction, handleNavigateBack)

**Step 1: Add blueprint action handler**

When in blueprint mode, clicking an action should:
1. Record the action's `id` in `blueprintHistory`
2. Add to `actionHistory` for UI rendering (same as existing)
3. Call `get_strategy_matrix` with updated history
4. If the response `street` field indicates a street transition (e.g., street is "Turn" but board has 3 cards), show the card picker instead

```typescript
const handleBlueprintAction = useCallback(async (actionId: string, actionIndex: number) => {
  setError(null);
  setSelectedCell(null);

  // Optimistically add action card
  if (matrix) {
    setActionHistory(prev => [...prev, {
      selectedId: String(actionIndex),
      actionIndex,
      position: matrix.player === 0 ? 'SB' : 'BB',
      stack: matrix.stacks[matrix.player],
      pot: matrix.pot,
      actions: matrix.actions,
      streetIndex: currentStreetIndex,
    }]);
  }
  setMatrix(null);

  const newHistory = [...blueprintHistory, actionId];
  setBlueprintHistory(newHistory);

  try {
    const sm = await invoke<StrategyMatrix>('get_strategy_matrix', {
      position: {
        board: boardCards,
        history: newHistory,
        pot: config.pot,
        stacks: [config.effective_stack, config.effective_stack],
        to_act: 0,
        num_players: 2,
        active_players: [true, true],
      },
    });

    // Detect street transition: tree advanced past a chance node
    const expectedStreet = boardCards.length === 3 ? 'Flop' : boardCards.length === 4 ? 'Turn' : 'River';
    if (sm.street !== expectedStreet) {
      // Need next card before we can show the matrix
      setAwaitingCard(true);
      return;
    }

    // Determine acting player from the tree (position in the returned matrix)
    // to_act is derived from the tree node, we can infer from pot/stack changes
    // For now, use the to_act we can derive
    setMatrix(blueprintToPostflopMatrix(sm, boardCards, 0 /* will need proper player */));
  } catch (e) {
    if (String(e).includes('terminal')) {
      setTerminal(true);
    } else {
      setBlueprintHistory(prev => prev.slice(0, -1));
      setActionHistory(prev => prev.slice(0, -1));
      setError(String(e));
    }
  }
}, [blueprintMode, blueprintHistory, matrix, boardCards, config, currentStreetIndex]);
```

**Step 2: Wire blueprint action handler into the action block**

In the current action block `onSelect`, dispatch to either `handleBlueprintAction` or the existing `handleAction` based on `blueprintMode`:

```typescript
{matrix && !terminal && !awaitingCard && (
  <ActionBlock
    position={matrix.player === 0 ? 'SB' : 'BB'}
    stack={matrix.stacks[matrix.player]}
    pot={matrix.pot}
    actions={matrix.actions}
    onSelect={(actionId) => {
      if (blueprintMode) {
        handleBlueprintAction(actionId, Number(actionId));
      } else {
        handleAction(Number(actionId));
      }
    }}
    isCurrent={true}
  />
)}
```

**Step 3: Add blueprint navigate-back**

When clicking a history action card in blueprint mode, truncate the blueprint history and re-fetch:

```typescript
const handleBlueprintNavigateBack = useCallback(async (historyIndex: number) => {
  setError(null);
  setSelectedCell(null);
  setTerminal(false);
  setAwaitingCard(false);

  // Truncate history
  setActionHistory(prev => prev.slice(0, historyIndex));
  const newHistory = blueprintHistory.slice(0, historyIndex);
  setBlueprintHistory(newHistory);
  setMatrix(null);

  try {
    const sm = await invoke<StrategyMatrix>('get_strategy_matrix', {
      position: {
        board: boardCards,
        history: newHistory,
        pot: config.pot,
        stacks: [config.effective_stack, config.effective_stack],
        to_act: 0,
        num_players: 2,
        active_players: [true, true],
      },
    });
    setMatrix(blueprintToPostflopMatrix(sm, boardCards, 0));
  } catch (e) {
    setError(String(e));
  }
}, [blueprintHistory, boardCards, config]);
```

**Step 4: Route navigate-back based on mode**

```typescript
onSelect={() => {
  if (blueprintMode) {
    handleBlueprintNavigateBack(i);
  } else {
    handleNavigateBack(i);
  }
}}
```

**Step 5: Commit**

```bash
git add frontend/src/PostflopExplorer.tsx
git commit -m "feat: blueprint tree navigation in PostflopExplorer"
```

---

## Task 6: Frontend — SOLVE transition

**Files:**
- Modify: `frontend/src/PostflopExplorer.tsx` (handleSolve, solve button rendering)

**Step 1: Modify handleSolve to transition from blueprint to solver mode**

When SOLVE is clicked:
1. Set `blueprintMode = false`
2. Clear blueprint-specific state
3. Proceed with existing solve logic (but don't clear action history — user stays at current position)

```typescript
const handleSolve = useCallback(() => {
  if (solving) {
    // Cancel
    invoke('postflop_cancel_solve').catch(() => {});
    if (pollRef.current) { clearInterval(pollRef.current); pollRef.current = null; }
    setSolving(false);
    setNeedsSolve(true);
    setBlueprintMode(true); // Back to blueprint mode
    return;
  }
  const cards = boardInput.trim().split(/\s+/);
  if (cards.length < 3) return;
  setError(null);
  setSolving(true);
  setNeedsSolve(false);
  setBlueprintMode(false);  // Switch to solver mode
  setActionHistory([]); // Reset action history for solver tree
  setBlueprintHistory([]);
  setMatrix(null);
  setProgress(null);
  initialExplRef.current = Infinity;
  // ... existing solve invocation code unchanged ...
}, [solving, boardInput, startPolling]);
```

**Step 2: Update solve button to show "Solved" state**

Replace the solve button rendering (lines 598-603):

```tsx
{/* Solve button */}
{(needsSolve || solving || solved) && (
  <div
    className={`action-block solve-block ${solving ? 'solving' : ''} ${solved ? 'solved' : ''}`}
    onClick={solved ? undefined : handleSolve}
    style={solved ? { cursor: 'default' } : undefined}
  >
    <span className="solve-label">
      {solved ? 'SOLVED' : solving ? 'CANCEL' : 'SOLVE'}
    </span>
  </div>
)}
```

**Step 3: Set solved=true when solve completes**

In the polling callback (startPolling), when `is_complete`:

```typescript
if (p.is_complete) {
  setSolving(false);
  setSolved(true);
  // ...
}
```

**Step 4: Add CSS for solved state**

Add to the stylesheet:

```css
.solve-block.solved {
  background: #16a34a;
  color: white;
  cursor: default;
}
```

**Step 5: Reset solved state on new board/reset**

In `handleReset` and flop picker `onConfirm`:
```typescript
setSolved(false);
setBlueprintMode(false);
setBlueprintHistory([]);
```

**Step 6: Commit**

```bash
git add frontend/src/PostflopExplorer.tsx frontend/src/styles/
git commit -m "feat: SOLVE transition from blueprint mode, green Solved indicator"
```

---

## Task 7: Frontend — Street progression with blueprint ranges

**Files:**
- Modify: `frontend/src/PostflopExplorer.tsx` (NextCardPicker confirm handler, ~lines 638-654)

**Step 1: Handle next street transition in blueprint mode**

When card is picked after a blueprint-mode street transition (no solve was run), use blueprint range propagation:

```typescript
onConfirm={(card) => {
  setShowNextCardPicker(false);
  setAwaitingCard(false);
  setError(null);

  if (solved) {
    // Solved mode: use existing close_street with solver ranges
    invoke<PostflopStreetResult>('postflop_close_street', { action_history: streetActions })
      .then(() => {
        const newBoard = [...boardCards, card];
        setBoardInput(newBoard.join(' '));
        setPriorStreetActions(prev => [...prev, streetActions]);
        setStreetActions([]);
        setCurrentStreetIndex(prev => prev + 1);
        setMatrix(null);
        setProgress(null);
        setSolved(false); // New street, not yet solved
        setNeedsSolve(true);
        setBlueprintMode(true); // Back to blueprint for new street
      })
      .catch((e) => { setError(String(e)); });
  } else {
    // Blueprint mode: update board and re-fetch blueprint strategy
    const newBoard = [...boardCards, card];
    setBoardInput(newBoard.join(' '));
    setCurrentStreetIndex(prev => prev + 1);
    setMatrix(null);
    // blueprintHistory continues — the tree already advanced past the chance node
    // The useEffect for boardCards change will re-fetch with the new board
  }
}}
```

**Step 2: Block street progression during solve**

When solving, the chance node handler already blocks (existing `blockedAtChanceRef` logic). Verify this still works correctly.

**Step 3: Commit**

```bash
git add frontend/src/PostflopExplorer.tsx
git commit -m "feat: blueprint range propagation for next street, block during solve"
```

---

## Task 8: Integration testing

**Step 1: Manual test the full flow**

Using the dev server:
1. Load a blueprint with bucket files
2. Navigate preflop actions to reach a flop
3. Deal a flop — verify blueprint strategy appears immediately
4. Click actions — verify blueprint tree navigation works
5. Click SOLVE — verify transition to solver with live updates
6. Wait for solve — verify "Solved" button appears
7. Navigate to next street — verify blueprint appears again with filtered ranges

```bash
cargo run -p poker-solver-devserver &
cd frontend && npm run dev
```

**Step 2: Commit any fixes**

```bash
git commit -m "fix: integration testing fixes for blueprint postflop explorer"
```

---

## Summary of file changes

| File | Changes |
|------|---------|
| `crates/tauri-app/src/exploration.rs` | Thread CbvContext into get_strategy_matrix_v2; replace postflop uniform stub with bucket lookups; extend reach computation for postflop |
| `crates/devserver/src/main.rs` | Pass PostflopState into get_strategy_matrix endpoint |
| `frontend/src/PostflopExplorer.tsx` | Blueprint mode state, conversion function, blueprint fetch on flop deal, blueprint navigation, SOLVE transition, Solved button, street progression |
| `frontend/src/styles/*.css` | `.solve-block.solved` style |
