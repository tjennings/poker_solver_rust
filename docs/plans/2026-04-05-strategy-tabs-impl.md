# Strategy Source Tabs Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use hex:executing-plans to implement this plan task-by-task.

**Goal:** Add three strategy source tabs (Blueprint, Subgame, Exact) above the strategy container in the explorer UI, with independent solve caches and inline solve buttons.

**Architecture:** Backend adds a second `SolveState` and a `mode` parameter to `game_solve_core` / `game_cancel_solve_core`, plus a `source` parameter to `game_get_state_core`. Frontend adds a tab bar component between the action strip and strategy container, tracks active source in React state, and re-fetches state with the active source on tab switch and navigation.

**Tech Stack:** Rust (tauri-app, devserver crates), TypeScript/React (frontend), CSS

**Design doc:** `docs/plans/2026-04-05-strategy-tabs-design.md`

---

### Task 1: Backend — Dual SolveState in GameSessionState

**Files:**
- Modify: `crates/tauri-app/src/game_session.rs:194-207`

**Step 1: Write the failing test**

```rust
#[test]
fn game_session_state_has_dual_solve_states() {
    let gss = GameSessionState::default();
    // Both solve states should be independent
    gss.subgame_solve.iteration.store(42, Ordering::Relaxed);
    gss.exact_solve.iteration.store(99, Ordering::Relaxed);
    assert_eq!(gss.subgame_solve.iteration.load(Ordering::Relaxed), 42);
    assert_eq!(gss.exact_solve.iteration.load(Ordering::Relaxed), 99);
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test -p poker-solver-tauri game_session_state_has_dual`
Expected: FAIL — no field `subgame_solve`

**Step 3: Change `GameSessionState` to hold two solve states**

In `crates/tauri-app/src/game_session.rs`, replace:

```rust
pub struct GameSessionState {
    pub session: RwLock<Option<GameSession>>,
    pub solve_state: Arc<SolveState>,
}

impl Default for GameSessionState {
    fn default() -> Self {
        Self {
            session: RwLock::new(None),
            solve_state: Arc::new(SolveState::default()),
        }
    }
}
```

With:

```rust
pub struct GameSessionState {
    pub session: RwLock<Option<GameSession>>,
    pub subgame_solve: Arc<SolveState>,
    pub exact_solve: Arc<SolveState>,
}

impl Default for GameSessionState {
    fn default() -> Self {
        Self {
            session: RwLock::new(None),
            subgame_solve: Arc::new(SolveState::default()),
            exact_solve: Arc::new(SolveState::default()),
        }
    }
}
```

Add a helper method to select the solve state by mode:

```rust
impl GameSessionState {
    /// Get the SolveState for the given mode ("subgame" or "exact").
    /// Defaults to subgame for unknown modes.
    pub fn solve_for(&self, mode: &str) -> &Arc<SolveState> {
        match mode {
            "exact" => &self.exact_solve,
            _ => &self.subgame_solve,
        }
    }
}
```

**Step 4: Fix all compilation errors**

Every reference to `session_state.solve_state` in the file must be updated. For now, change all existing references to `session_state.subgame_solve` to maintain current behavior. The commands will be updated in subsequent tasks to accept a mode parameter.

Key locations to update (search for `solve_state`):
- `game_get_state_core` (~line 1476): `let ss = &session_state.subgame_solve;`
- `game_play_action_core` (~line 1532): `let ss = &session_state.subgame_solve;`
- `game_back_core` (~line 1601): `let ss = &session_state.subgame_solve;`
- `game_solve_core` (~line 1642): `session_state.subgame_solve.solving.load(...)`
- `game_cancel_solve_core` (~line 2102): `session_state.subgame_solve.cancel.store(...)`
- Test functions: search for `solve_state` in `#[cfg(test)]` section

Also update `game_new` / session reset to clear BOTH solve states:
- Find where `solve_state.reset()` is called and add `exact_solve.reset()` alongside.

**Step 5: Run tests**

Run: `cargo test -p poker-solver-tauri`
Expected: PASS (all existing tests still work, just using subgame_solve)

**Step 6: Commit**

```bash
git add crates/tauri-app/src/game_session.rs
git commit -m "refactor: split solve_state into subgame_solve and exact_solve"
```

---

### Task 2: Backend — Add `mode` parameter to game_solve_core

**Files:**
- Modify: `crates/tauri-app/src/game_session.rs` (game_solve_core, game_solve, game_cancel_solve_core, game_cancel_solve)

**Step 1: Write the failing test**

```rust
#[test]
fn game_solve_core_accepts_mode_parameter() {
    let gss = make_test_session_state();
    // Should accept mode without error (even if solve fails for other reasons)
    let result = game_solve_core(&gss, Some("subgame".to_string()), None, None, None, None, None, None, None);
    // We just need it to not fail on the mode parameter itself
    assert!(result.is_ok() || result.unwrap_err().contains("postflop") || result.unwrap_err().contains("decision"));
}
```

**Step 2: Add `mode` parameter to `game_solve_core`**

Change the signature from:

```rust
pub fn game_solve_core(
    session_state: &GameSessionState,
    max_iterations: Option<u32>,
    ...
) -> Result<(), String> {
```

To:

```rust
pub fn game_solve_core(
    session_state: &GameSessionState,
    mode: Option<String>,
    max_iterations: Option<u32>,
    ...
) -> Result<(), String> {
```

Inside the function:
- Resolve mode: `let mode = mode.as_deref().unwrap_or("subgame");`
- Select solve state: `let ss = session_state.solve_for(mode);`
- Replace all `session_state.subgame_solve` references with `ss`
- For "exact" mode: set `eval_interval = 0` and disable depth bounding (configure the PostFlopGame to solve to showdown without leaf evaluator)

**Step 3: Update `game_solve` Tauri command**

```rust
#[tauri::command]
pub fn game_solve(
    session_state: tauri::State<'_, GameSessionState>,
    mode: Option<String>,
    max_iterations: Option<u32>,
    ...
) -> Result<(), String> {
    game_solve_core(&session_state, mode, max_iterations, ...)
}
```

**Step 4: Update `game_cancel_solve_core` and `game_cancel_solve`**

```rust
pub fn game_cancel_solve_core(session_state: &GameSessionState, mode: Option<String>) -> Result<(), String> {
    let ss = session_state.solve_for(mode.as_deref().unwrap_or("subgame"));
    ss.cancel.store(true, Ordering::Relaxed);
    Ok(())
}

#[tauri::command]
pub fn game_cancel_solve(
    session_state: tauri::State<'_, GameSessionState>,
    mode: Option<String>,
) -> Result<(), String> {
    game_cancel_solve_core(&session_state, mode)
}
```

**Step 5: Run tests and fix compilation**

Run: `cargo test -p poker-solver-tauri`
Expected: PASS

Run: `cargo build -p poker-solver-tauri`
Expected: compiles

**Step 6: Commit**

```bash
git add crates/tauri-app/src/game_session.rs
git commit -m "feat: add mode parameter to game_solve and game_cancel_solve"
```

---

### Task 3: Backend — Add `source` parameter to game_get_state_core

**Files:**
- Modify: `crates/tauri-app/src/game_session.rs` (game_get_state_core, game_get_state)

**Step 1: Write the failing test**

```rust
#[test]
fn game_get_state_core_returns_blueprint_by_default() {
    let gss = make_test_session_state();
    // With source="blueprint", should return blueprint strategy (no solve overlay)
    let state = game_get_state_core(&gss, None).unwrap();
    assert!(state.solve.is_none());
}
```

**Step 2: Add `source` parameter**

Change `game_get_state_core` signature:

```rust
pub fn game_get_state_core(
    session_state: &GameSessionState,
    source: Option<String>,
) -> Result<GameState, String> {
```

Logic:
- `source = None` or `"blueprint"`: return blueprint strategy (no solve overlay). This is just `session.get_state()` without the solve state override block.
- `source = "subgame"`: overlay from `session_state.subgame_solve` (current behavior)
- `source = "exact"`: overlay from `session_state.exact_solve`

```rust
let source = source.as_deref().unwrap_or("blueprint");
let mut state = session.get_state();

if source != "blueprint" {
    let ss = session_state.solve_for(source);
    let is_solving = ss.solving.load(Ordering::Relaxed);
    let iteration = ss.iteration.load(Ordering::Relaxed);

    if is_solving || iteration > 0 {
        // ... existing solve overlay logic using `ss` ...
    }
}
```

**Step 3: Update `game_get_state` Tauri command**

```rust
#[tauri::command]
pub fn game_get_state(
    session_state: tauri::State<'_, GameSessionState>,
    source: Option<String>,
) -> Result<GameState, String> {
    game_get_state_core(&session_state, source)
}
```

**Step 4: Update `game_play_action_core` and `game_back_core` to accept source**

These functions currently check `solve_state` to navigate within solved trees. They need to know WHICH solve cache to check. Add `source: Option<String>` parameter and select the right solve state.

For `game_play_action_core`:
```rust
pub fn game_play_action_core(
    session_state: &GameSessionState,
    action_id: &str,
    source: Option<String>,
) -> Result<GameState, String> {
    let source_str = source.as_deref().unwrap_or("blueprint");
    let ss = session_state.solve_for(source_str);
    // ... rest uses `ss` instead of `session_state.subgame_solve` ...
```

Same pattern for `game_back_core`. The Tauri commands `game_play_action` and `game_back` get an optional `source` parameter too.

**Step 5: Fix all callers and tests**

Search for all calls to `game_get_state_core`, `game_play_action_core`, `game_back_core` and add the new parameter (usually `None` for existing tests).

**Step 6: Run tests**

Run: `cargo test -p poker-solver-tauri`
Expected: PASS

**Step 7: Commit**

```bash
git add crates/tauri-app/src/game_session.rs
git commit -m "feat: add source parameter to game_get_state, play_action, and back"
```

---

### Task 4: Backend — Update devserver routes

**Files:**
- Modify: `crates/devserver/src/main.rs`

**Step 1: Add `mode` to `GameSolveParams`**

```rust
#[derive(Deserialize)]
struct GameSolveParams {
    mode: Option<String>,           // NEW
    max_iterations: Option<u32>,
    target_exploitability: Option<f32>,
    leaf_eval_interval: Option<u32>,
    rollout_bias_factor: Option<f64>,
    rollout_num_samples: Option<u32>,
    rollout_opponent_samples: Option<u32>,
    range_clamp_threshold: Option<f64>,
}
```

**Step 2: Add source param structs and update handlers**

```rust
#[derive(Deserialize)]
struct GameGetStateParams {
    source: Option<String>,
}

#[derive(Deserialize)]
struct GameCancelSolveParams {
    mode: Option<String>,
}

#[derive(Deserialize)]
struct GamePlayActionParams {
    action_id: String,
    source: Option<String>,
}

#[derive(Deserialize)]
struct GameBackParams {
    source: Option<String>,
}
```

Update handlers to pass the new parameters through to the `_core` functions.

**Step 3: Build and verify**

Run: `cargo build -p poker-solver-devserver`
Expected: compiles

**Step 4: Commit**

```bash
git add crates/devserver/src/main.rs
git commit -m "feat: update devserver routes for mode and source parameters"
```

---

### Task 5: Frontend — Add strategy source tabs

**Files:**
- Modify: `frontend/src/GameExplorer.tsx`
- Modify: `frontend/src/game-types.ts`
- Modify: `frontend/src/App.css`

**Step 1: Add activeSource state and solve tracking**

In `GameExplorer.tsx`, add new state variables alongside existing ones (~line 143):

```typescript
const [activeSource, setActiveSource] = useState<"blueprint" | "subgame" | "exact">("blueprint");
const [subgameSolving, setSubgameSolving] = useState(false);
const [exactSolving, setExactSolving] = useState(false);
```

**Step 2: Create tab bar component**

Add a `StrategyTabs` inline component rendered between the action strip and `matrix-container` (~line 693). Only render when `state.street !== "Preflop"`:

```typescript
const isPostflop = state && state.street !== "Preflop";

// ... between action strip and matrix-container:
{isPostflop && (
  <div className="strategy-tabs">
    <span
      className={`strategy-tab ${activeSource === 'blueprint' ? 'active' : ''}`}
      onClick={() => handleTabSwitch('blueprint')}
    >
      Blueprint
    </span>
    <span className="strategy-tab-group">
      <span
        className={`strategy-tab ${activeSource === 'subgame' ? 'active' : ''}`}
        onClick={() => handleTabSwitch('subgame')}
      >
        Subgame
      </span>
      <span
        className="strategy-tab-action"
        onClick={() => handleSolve('subgame')}
      >
        [{subgameSolving ? 'cancel' : 'solve'}]
      </span>
    </span>
    <span className="strategy-tab-group">
      <span
        className={`strategy-tab ${activeSource === 'exact' ? 'active' : ''}`}
        onClick={() => handleTabSwitch('exact')}
      >
        Exact
      </span>
      <span
        className="strategy-tab-action"
        onClick={() => handleSolve('exact')}
      >
        [{exactSolving ? 'cancel' : 'solve'}]
      </span>
    </span>
  </div>
)}
```

**Step 3: Implement handleTabSwitch**

```typescript
const handleTabSwitch = async (source: "blueprint" | "subgame" | "exact") => {
  setActiveSource(source);
  try {
    const s = await invoke<GameState>('game_get_state', { source });
    setState(s);
  } catch (e) {
    setError(String(e));
  }
};
```

**Step 4: Implement handleSolve**

Refactor the existing `solveBtn` onClick logic into a reusable function:

```typescript
const handleSolve = async (mode: "subgame" | "exact") => {
  const isSolving = mode === 'subgame' ? subgameSolving : exactSolving;
  const setSolving = mode === 'subgame' ? setSubgameSolving : setExactSolving;

  if (isSolving) {
    // Cancel
    try {
      setSolving(false);
      await invoke('game_cancel_solve', { mode });
      const s = await invoke<GameState>('game_get_state', { source: mode });
      setState(s);
    } catch (e) {
      setError(String(e));
    }
  } else {
    // Start solve
    try {
      setSolving(true);
      setActiveSource(mode); // switch to this tab
      const globalConfig = JSON.parse(localStorage.getItem('global_config') || '{}');
      await invoke('game_solve', {
        mode,
        maxIterations: globalConfig.solve_iterations ?? 200,
        targetExploitability: globalConfig.target_exploitability ?? 3.0,
        leafEvalInterval: mode === 'exact' ? 0 : (globalConfig.leaf_eval_interval ?? 10),
        rolloutBiasFactor: globalConfig.rollout_bias_factor ?? 10.0,
        rolloutNumSamples: globalConfig.rollout_num_samples ?? 3,
        rolloutOpponentSamples: globalConfig.rollout_opponent_samples ?? 8,
        rangeClampThreshold: globalConfig.range_clamp_threshold ?? 0.05,
      });
      // Poll for progress
      const pollId = setInterval(async () => {
        try {
          const s = await invoke<GameState>('game_get_state', { source: mode });
          setState(s);
          if (s.solve?.is_complete) {
            clearInterval(pollId);
            setSolving(false);
          }
        } catch {
          clearInterval(pollId);
          setSolving(false);
        }
      }, 500);
    } catch (e) {
      setError(String(e));
      setSolving(false);
    }
  }
};
```

**Step 5: Update navigation calls to pass source**

Find all calls to `game_get_state`, `game_play_action`, `game_back` in the frontend and pass `source: activeSource` (or the correct source for the context). Key locations:

- `game_play_action` calls: add `source: activeSource`
- `game_back` calls: add `source: activeSource`
- `game_get_state` calls (polling, refreshes): add `source: activeSource`
- `game_deal_card` calls: these reset to blueprint (deal card exits any solved subtree)

**Step 6: Remove the old SOLVE button**

Delete the `solveBtn` function and its usage in the action strip (lines ~548-615). The solve functionality now lives in the tab bar.

**Step 7: Add CSS for tab bar**

In `frontend/src/App.css`:

```css
.strategy-tabs {
  display: flex;
  gap: 1rem;
  padding: 0.4rem 0.5rem;
  font-size: 0.8rem;
  border-bottom: 1px solid #1a1a2e;
}

.strategy-tab {
  color: #666;
  cursor: pointer;
  transition: color 0.15s;
}

.strategy-tab:hover {
  color: #aaa;
}

.strategy-tab.active {
  color: #00d9ff;
}

.strategy-tab-group {
  display: flex;
  gap: 0.3rem;
  align-items: center;
}

.strategy-tab-action {
  color: #f59e0b;
  cursor: pointer;
  font-size: 0.7rem;
}

.strategy-tab-action:hover {
  color: #fbbf24;
}
```

**Step 8: Test in browser**

Run: `cargo run -p poker-solver-devserver & cd frontend && npm run dev`
- Load a blueprint, navigate to a flop spot
- Verify tabs appear
- Click Subgame [solve] — verify solve starts, progress shows
- Switch to Blueprint tab — verify blueprint strategy shows
- Switch back to Subgame — verify solved strategy shows
- Click Exact [solve] — verify second solve starts independently
- Navigate within solved tree — verify correct strategy follows

**Step 9: Commit**

```bash
git add frontend/src/GameExplorer.tsx frontend/src/App.css
git commit -m "feat: add strategy source tabs with dual solve support"
```

---

### Task 6: Backend — Exact solve mode (no depth bounding)

**Files:**
- Modify: `crates/tauri-app/src/game_session.rs` (game_solve_core)

**Step 1: Differentiate solve configuration by mode**

In `game_solve_core`, after resolving the mode, configure the solver differently for "exact":

```rust
let mode = mode.as_deref().unwrap_or("subgame");

// For exact mode: disable leaf evaluator, solve to showdown
let use_leaf_evaluator = mode != "exact";
let eval_interval = if mode == "exact" { 0 } else { leaf_eval_interval.unwrap_or(10) };
```

When building the `PostFlopGame` or configuring the solver, skip the `RolloutLeafEvaluator` / `CbvContext` when `use_leaf_evaluator` is false. The solver will expand the full game tree to showdown.

The exact solve will be slower but produce the true GTO strategy for that subgame.

**Step 2: Reject concurrent solves per mode only**

Currently: "A solve is already in progress" blocks any new solve. Change to only block if the SAME mode is already solving:

```rust
let ss = session_state.solve_for(mode);
if ss.solving.load(Ordering::Relaxed) {
    return Err(format!("A {mode} solve is already in progress"));
}
```

This allows subgame and exact solves to run concurrently.

**Step 3: Test**

Run: `cargo test -p poker-solver-tauri`
Expected: PASS

**Step 4: Commit**

```bash
git add crates/tauri-app/src/game_session.rs
git commit -m "feat: exact solve mode runs range-solver without depth bounding"
```

---

### Task 7: Full workspace build and test verification

**Step 1: Build entire workspace**

Run: `cargo build`
Expected: clean build

**Step 2: Run all tests**

Run: `cargo test`
Expected: all pass, < 1 minute

**Step 3: Run clippy**

Run: `cargo clippy`
Expected: no warnings in changed code

**Step 4: Frontend build check**

Run: `cd frontend && npm run build`
Expected: compiles without errors

**Step 5: Update docs**

Update `docs/explorer.md` with information about the strategy source tabs per CLAUDE.md instructions.

**Step 6: Final commit**

```bash
git commit -m "docs: update explorer docs for strategy source tabs"
```
