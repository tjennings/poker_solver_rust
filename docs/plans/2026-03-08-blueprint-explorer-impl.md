# Blueprint Explorer Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use hex:executing-plans to implement this plan task-by-task.

**Goal:** Add a "Load Strategy" flow that loads a trained blueprint for preflop navigation, then transitions into postflop range solving with blueprint-derived ranges and per-street caching.

**Architecture:** The Settings view gets blueprint directory + target exploitability fields. Explorer gets a two-button landing (Load Strategy / Range Solve). Load Strategy loads a BlueprintV2 bundle, navigates preflop via the existing strategy matrix UI, then on non-fold terminal derives 1326-combo ranges and hands off to the PostflopExplorer with a pre-populated config. Solved spots are cached to `{blueprint_dir}/spots/`.

**Tech Stack:** Rust (tauri-app crate, core crate), TypeScript/React (frontend), range-solver crate, Tauri dialog API

---

### Task 1: Settings View — Blueprint Directory + Target Exploitability

**Files:**
- Modify: `frontend/src/Settings.tsx`
- Modify: `frontend/src/types.ts`
- Modify: `frontend/src/invoke.ts` (no changes needed, already supports generic invoke)

**Context:** `Settings.tsx` is currently a stub ("Coming soon"). We need it to show two fields that persist via `localStorage`. The blueprint directory field uses the OS native directory picker in Tauri mode, or a text input in browser mode.

**Step 1: Add GlobalConfig type**

In `frontend/src/types.ts`, add:

```typescript
export interface GlobalConfig {
  blueprint_dir: string;
  target_exploitability: number;
}
```

**Step 2: Implement Settings component**

Replace `frontend/src/Settings.tsx` with a form containing:
- "Blueprint Directory" label + text display + "Browse" button (calls `@tauri-apps/plugin-dialog` `open({ directory: true })` in Tauri mode)
- "Target Exploitability" numeric input (default 3.0)
- Save button that writes to `localStorage` key `"global_config"`
- On mount, load from `localStorage`

**Step 3: Export a `useGlobalConfig` hook**

Create `frontend/src/useGlobalConfig.ts`:
- Reads/writes `localStorage` key `"global_config"`
- Returns `{ config, setConfig }` with `GlobalConfig` type
- Default: `{ blueprint_dir: '', target_exploitability: 3.0 }`

This hook will be consumed by Explorer and PostflopExplorer.

**Step 4: Verify it works**

Run: `cd frontend && npm run dev`
Open Settings tab, set a directory, change exploitability, reload — values persist.

**Step 5: Commit**

```
feat: implement Settings view with blueprint directory and target exploitability
```

---

### Task 2: Backend — `list_blueprints` Command

**Files:**
- Modify: `crates/tauri-app/src/lib.rs` (add export)
- Modify: `crates/tauri-app/src/exploration.rs` (add function)
- Modify: `crates/tauri-app/src/main.rs` (register command)
- Modify: `crates/devserver/src/main.rs` (add route)

**Context:** We need a command that scans a directory for blueprint bundles. Each subdirectory with a `config.yaml` is a potential blueprint. The existing `load_blueprint_v2_core` in `exploration.rs` already knows how to find `strategy.bin` files — reuse that logic.

**Step 1: Define the return type**

In `exploration.rs`, add:

```rust
#[derive(Debug, Clone, Serialize)]
pub struct BlueprintListEntry {
    pub name: String,
    pub path: String,
    pub stack_depth: f64,
    pub has_strategy: bool,
}
```

**Step 2: Implement `list_blueprints_core`**

```rust
pub fn list_blueprints_core(dir: String) -> Result<Vec<BlueprintListEntry>, String> {
    let base = PathBuf::from(&dir);
    if !base.is_dir() {
        return Ok(vec![]);
    }
    let mut entries = Vec::new();
    let read_dir = std::fs::read_dir(&base)
        .map_err(|e| format!("Cannot read directory: {e}"))?;
    for entry in read_dir.filter_map(Result::ok) {
        let path = entry.path();
        if !path.is_dir() { continue; }
        let config_path = path.join("config.yaml");
        if !config_path.exists() { continue; }
        let config = match v2_bundle::load_config(&path) {
            Ok(c) => c,
            Err(_) => continue,
        };
        let has_strategy = path.join("final/strategy.bin").exists()
            || path.join("strategy.bin").exists()
            || std::fs::read_dir(&path)
                .ok()
                .map(|rd| rd.filter_map(Result::ok)
                    .any(|e| e.file_name().to_str()
                        .is_some_and(|n| n.starts_with("snapshot_"))))
                .unwrap_or(false);
        entries.push(BlueprintListEntry {
            name: entry.file_name().to_string_lossy().to_string(),
            path: path.to_string_lossy().to_string(),
            stack_depth: config.game.stack_depth,
            has_strategy,
        });
    }
    entries.sort_by(|a, b| a.name.cmp(&b.name));
    Ok(entries)
}
```

**Step 3: Add Tauri command wrapper + devserver route**

Standard pattern: `#[tauri::command]` wrapper calling `list_blueprints_core`, register in `main.rs` `.invoke_handler()`, add matching route in devserver.

**Step 4: Test manually**

```bash
curl -X POST http://localhost:3001/api/list_blueprints -H 'Content-Type: application/json' -d '{"dir": "/path/to/runs"}'
```

**Step 5: Commit**

```
feat: add list_blueprints command to scan blueprint directory
```

---

### Task 3: Backend — `get_preflop_ranges` Command

**Files:**
- Modify: `crates/tauri-app/src/exploration.rs`
- Modify: `crates/tauri-app/src/lib.rs`
- Modify: `crates/tauri-app/src/main.rs`
- Modify: `crates/devserver/src/main.rs`

**Context:** Given a loaded BlueprintV2 source and a preflop action history, walk the tree and compute 1326-combo range weights for OOP and IP. At each decision node, multiply each canonical hand's weight by the strategy probability for the chosen action. Expand 169 → 1326 with uniform suit distribution.

**Step 1: Understand the 169 → 1326 mapping**

Each of the 169 canonical hands maps to multiple combos:
- Pairs (13): 6 combos each (AA has 6 suit combos)
- Suited (78): 4 combos each (AKs has 4 suit combos)
- Offsuit (78): 12 combos each (AKo has 12 suit combos)

The range solver uses `card_pair_to_index(c1, c2)` which returns an index in 0..1326. We need to map from canonical hand index (0..169) to the set of (c1, c2) pairs in that canonical hand, assigning each combo the same weight.

Use `poker_solver_core::hands::CanonicalHand` to get the index, and iterate all 52×51/2 card pairs to build the mapping.

**Step 2: Implement `get_preflop_ranges_core`**

```rust
#[derive(Debug, Clone, Serialize)]
pub struct PreflopRanges {
    pub oop_weights: Vec<f32>,  // [1326]
    pub ip_weights: Vec<f32>,   // [1326]
    pub pot: i32,               // pot at terminal node (in range-solver units)
    pub effective_stack: i32,   // effective stack at terminal
}
```

The function:
1. Read the loaded `BlueprintV2` source from `ExplorationState`
2. Walk the V2 tree following the preflop action history (reuse `walk_v2_tree` or a variant)
3. At each decision node along the path, get the strategy for the 169 canonical hands
4. For the acting player, multiply canonical hand weights by the chosen action's probability
5. At the terminal, expand 169 weights → 1326 combos
6. Return pot and effective stack from the `V2WalkState`

Key detail: both players' ranges start at 1.0 for all 169 hands. As we walk the tree, at each decision node only the acting player's range is filtered. The walk must track which player is acting (from the tree node).

**Step 3: Register command**

Standard Tauri command wrapper + devserver route.

**Step 4: Write a unit test**

Test with a known blueprint where we know the expected strategy probabilities. Load the toy blueprint, walk a simple line (SB raises, BB calls), verify the output ranges have values < 1.0 for hands that fold.

**Step 5: Commit**

```
feat: add get_preflop_ranges command for 169→1326 range expansion
```

---

### Task 4: Frontend — Explorer Landing with Load Strategy / Range Solve Buttons

**Files:**
- Modify: `frontend/src/Explorer.tsx`
- Modify: `frontend/src/types.ts`

**Context:** Currently the Explorer shows two small icon buttons (folder icon + plus icon) when no bundle is loaded. Replace these with two clearly labeled buttons: "Load Strategy" and "Range Solve". Load Strategy opens a blueprint selection modal. Range Solve opens the PostflopExplorer (existing behavior).

**Step 1: Add BlueprintListEntry type**

In `types.ts`:

```typescript
export interface BlueprintListEntry {
  name: string;
  path: string;
  stack_depth: number;
  has_strategy: boolean;
}
```

**Step 2: Replace the landing view**

In `Explorer.tsx`, find the `!bundleInfo && !loading && !showDatasetPicker && !showPostflop` conditional block (around line 1283). Replace the icon-only split buttons with:

```tsx
<div className="explorer-landing">
  <button className="landing-btn" onClick={handleLoadStrategy}>
    Load Strategy
  </button>
  <button className="landing-btn" onClick={() => setShowPostflop(true)}>
    Range Solve
  </button>
</div>
```

**Step 3: Add blueprint selection modal state**

Add state for the blueprint selection flow:
```typescript
const [showBlueprintPicker, setShowBlueprintPicker] = useState(false);
const [blueprints, setBlueprints] = useState<BlueprintListEntry[]>([]);
const [selectedBlueprint, setSelectedBlueprint] = useState<string | null>(null);
```

**Step 4: Implement `handleLoadStrategy`**

```typescript
const handleLoadStrategy = async () => {
  const globalConfig = JSON.parse(localStorage.getItem('global_config') || '{}');
  if (!globalConfig.blueprint_dir) {
    setError('Set Blueprint Directory in Settings first');
    return;
  }
  try {
    const list = await invoke<BlueprintListEntry[]>('list_blueprints', { dir: globalConfig.blueprint_dir });
    setBlueprints(list.filter(b => b.has_strategy));
    setShowBlueprintPicker(true);
  } catch (e) {
    setError(String(e));
  }
};
```

**Step 5: Add blueprint picker modal UI**

A modal showing the list of blueprints as clickable cards. Each shows `{name} ({stack_depth}BB)`. Clicking one calls `loadSource(blueprint.path)` (existing function that calls `load_bundle` → `load_blueprint_v2_core`).

**Step 6: Commit**

```
feat: Explorer landing with Load Strategy / Range Solve buttons
```

---

### Task 5: Frontend — Preflop Navigation with Fold Terminal Handling

**Files:**
- Modify: `frontend/src/Explorer.tsx`

**Context:** The existing Explorer already supports BlueprintV2 preflop navigation via `get_strategy_matrix` and action clicking. However, we need to handle the transition to postflop when hitting a non-fold terminal, and stop navigation (without adding an action card) on fold terminals.

The existing `handleAction` function in Explorer already navigates the blueprint tree. We need to detect when the new position is a terminal (fold vs non-fold) and handle accordingly.

**Step 1: Detect fold vs non-fold terminal**

After clicking an action, call `get_available_actions`. If it returns an empty array, the node is terminal. Check if the last action was "Fold" (from the action label).

- Fold terminal: do nothing extra — matrix stays, no new action card
- Non-fold terminal (call after raise): trigger postflop transition

**Step 2: Add postflop transition state**

```typescript
const [preflopTerminal, setPreflopTerminal] = useState<{
  history: string[];  // preflop action history
  pot: number;
  effectiveStack: number;
} | null>(null);
```

When a non-fold terminal is detected, set `preflopTerminal` with the current position data and show the flop picker.

**Step 3: Implement transition to PostflopExplorer**

When `preflopTerminal` is set and the user picks a flop, call `get_preflop_ranges` to get the 1326-combo weights, then pass them to PostflopExplorer as pre-populated config (new prop).

**Step 4: Test manually**

Load a blueprint, navigate preflop to a fold → verify matrix stays. Navigate to a call → verify flop picker appears.

**Step 5: Commit**

```
feat: preflop fold/call terminal handling with postflop transition
```

---

### Task 6: Frontend — PostflopExplorer Accepts Blueprint-Derived Config

**Files:**
- Modify: `frontend/src/PostflopExplorer.tsx`
- Modify: `frontend/src/types.ts`

**Context:** PostflopExplorer currently always shows its own config modal. When launched from Load Strategy, it should accept a pre-populated config (ranges, pot, stack, bet sizes from the blueprint) and skip the config modal. Target exploitability comes from global config.

**Step 1: Add `blueprintConfig` prop**

```typescript
interface PostflopExplorerProps {
  onBack: () => void;
  blueprintConfig?: {
    oop_range_weights: number[];  // 1326 combo weights
    ip_range_weights: number[];
    pot: number;
    effective_stack: number;
    oop_bet_sizes: string;
    oop_raise_sizes: string;
    ip_bet_sizes: string;
    ip_raise_sizes: string;
    blueprint_dir: string;        // for caching
  };
}
```

**Step 2: Use blueprintConfig when present**

When `blueprintConfig` is provided:
- Skip showing the config modal
- Use the blueprint's bet sizes and pot/stack
- Pass the 1326 weights as `filtered_oop_weights` / `filtered_ip_weights` to the solve command
- Read `target_exploitability` from global config
- Show the config card as read-only (click doesn't open modal)

**Step 3: Remove target_exploitability from PostflopConfig**

Move to global config. Update `PostflopConfig` in `types.ts` to remove `target_exploitability`. The solve call reads it from global config instead.

This also requires a backend change — `postflop_solve_street` already accepts `target_exploitability` as a parameter, so the frontend just reads from global config and passes it.

**Step 4: Commit**

```
feat: PostflopExplorer accepts blueprint-derived config
```

---

### Task 7: Backend — Bet Size Mapping (Blueprint → Range Solver)

**Files:**
- Modify: `crates/tauri-app/src/exploration.rs` (add helper)

**Context:** Blueprint's `action_abstraction` defines bet sizes as `Vec<Vec<f64>>` per street (outer vec = raise depth, inner vec = pot fractions). Range solver wants them as `BetSizeOptions` format strings like `"33%,67%,100%"`.

**Step 1: Implement conversion function**

```rust
/// Convert blueprint action abstraction sizes to range-solver format strings.
///
/// Returns `(bet_sizes_str, raise_sizes_str)` for one street.
/// Depth 0 → bet sizes, depth 1 → raise sizes. If only one depth, use same for both.
fn blueprint_sizes_to_range_solver(depths: &[Vec<f64>]) -> (String, String) {
    let format_depth = |sizes: &[f64]| -> String {
        sizes.iter()
            .map(|&f| {
                let pct = (f * 100.0).round() as u32;
                format!("{pct}%")
            })
            .collect::<Vec<_>>()
            .join(",")
    };

    let bet_str = depths.first().map(|d| format_depth(d)).unwrap_or_default();
    let raise_str = depths.get(1).map(|d| format_depth(d)).unwrap_or_else(|| bet_str.clone());

    // Always include all-in
    let add_allin = |s: String| if s.is_empty() { "a".to_string() } else { format!("{s},a") };
    (add_allin(bet_str), add_allin(raise_str))
}
```

**Step 2: Use in `get_preflop_ranges_core`**

Include the converted bet sizes in the `PreflopRanges` return struct so the frontend can pass them to PostflopExplorer.

**Step 3: Unit test**

```rust
#[test]
fn test_blueprint_sizes_to_range_solver() {
    let depths = vec![vec![0.33, 0.67, 1.0], vec![0.5, 1.0]];
    let (bet, raise) = blueprint_sizes_to_range_solver(&depths);
    assert_eq!(bet, "33%,67%,100%,a");
    assert_eq!(raise, "50%,100%,a");
}
```

**Step 4: Commit**

```
feat: blueprint bet size to range-solver format conversion
```

---

### Task 8: Backend — Solve Cache (Write)

**Files:**
- Modify: `crates/tauri-app/src/postflop.rs`

**Context:** After a solve completes, write the strategy buffers to a cache file. The cache key is a hash of config + board + action history. The file format is defined in the cache design doc.

**Step 1: Add cache key computation**

```rust
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

fn compute_cache_key(
    config: &PostflopConfig,
    board: &[String],
    prior_actions: &[Vec<usize>], // action histories from prior streets
) -> u64 {
    let mut hasher = DefaultHasher::new();
    config.oop_range.hash(&mut hasher);
    config.ip_range.hash(&mut hasher);
    config.pot.hash(&mut hasher);
    config.effective_stack.hash(&mut hasher);
    config.oop_bet_sizes.hash(&mut hasher);
    config.oop_raise_sizes.hash(&mut hasher);
    config.ip_bet_sizes.hash(&mut hasher);
    config.ip_raise_sizes.hash(&mut hasher);
    for card in board {
        card.hash(&mut hasher);
    }
    for street_actions in prior_actions {
        for &a in street_actions {
            a.hash(&mut hasher);
        }
    }
    hasher.finish()
}
```

**Step 2: Add cache write after solve**

In `postflop_solve_street_core`, after the solve thread finishes and stores the game, write the cache file if a `cache_dir` is provided. Add `cache_dir: Option<PathBuf>` to `PostflopState`.

The write happens at the end of the solve thread (after `finalize`):

```rust
if let Some(ref cache_dir) = shared.cache_dir.read().clone() {
    let spots_dir = cache_dir.join("spots");
    let _ = std::fs::create_dir_all(&spots_dir);
    let key = compute_cache_key(&config, &board, &prior_actions);
    let path = spots_dir.join(format!("{key:016x}.bin"));
    // Write header + storage buffers
    let _ = write_cache_file(&path, &game, exploitability, iteration);
}
```

**Step 3: Implement `write_cache_file`**

Write the binary format from the cache design doc. Access `game.storage1`, `game.storage2`, `game.storage_ip`, `game.storage_chance` — these are `pub(crate)` fields in range-solver. We'll need to either make them `pub` or add accessor methods.

Check if range-solver fields are accessible from tauri-app. If not, add public accessor methods to `PostFlopGame`:

```rust
// In range-solver/src/game/mod.rs
impl PostFlopGame {
    pub fn storage_buffers(&self) -> (&[u8], &[u8], &[u8], &[u8]) {
        (&self.storage1, &self.storage2, &self.storage_ip, &self.storage_chance)
    }
}
```

**Step 4: Unit test**

Test that after solving, a cache file appears in the expected location with the correct magic bytes.

**Step 5: Commit**

```
feat: write postflop solve results to cache file
```

---

### Task 9: Backend — Solve Cache (Read)

**Files:**
- Modify: `crates/tauri-app/src/postflop.rs`
- Modify: `range-solver/src/game/mod.rs` (add `set_storage_buffers` or similar)

**Context:** Before solving, check if a cache file exists. If so, rebuild the game from config, inject the cached storage buffers, and skip the solve.

**Step 1: Add `load_cache_file` function**

Reads the binary format, validates magic and version, returns the storage buffers + metadata.

```rust
struct CachedSolve {
    exploitability: f32,
    iterations: u32,
    storage1: Vec<u8>,
    storage2: Vec<u8>,
    storage_ip: Vec<u8>,
    storage_chance: Vec<u8>,
}
```

**Step 2: Add `set_storage_buffers` to PostFlopGame**

In `range-solver/src/game/mod.rs`:

```rust
impl PostFlopGame {
    pub fn set_storage_buffers(
        &mut self,
        s1: Vec<u8>,
        s2: Vec<u8>,
        s_ip: Vec<u8>,
        s_chance: Vec<u8>,
    ) {
        assert_eq!(s1.len(), self.storage1.len());
        assert_eq!(s2.len(), self.storage2.len());
        assert_eq!(s_ip.len(), self.storage_ip.len());
        assert_eq!(s_chance.len(), self.storage_chance.len());
        self.storage1 = s1;
        self.storage2 = s2;
        self.storage_ip = s_ip;
        self.storage_chance = s_chance;
    }
}
```

**Step 3: Add new command `postflop_check_cache_core`**

```rust
pub fn postflop_check_cache_core(
    state: &PostflopState,
    board: Vec<String>,
    prior_actions: Vec<Vec<usize>>,
) -> Result<Option<CacheInfo>, String>
```

Returns cache metadata (exploitability, iterations) if found, or None. The frontend can then decide whether to show "Cached" or the solve button.

**Step 4: Add new command `postflop_load_cached_core`**

```rust
pub fn postflop_load_cached_core(
    state: &Arc<PostflopState>,
    board: Vec<String>,
    prior_actions: Vec<Vec<usize>>,
) -> Result<PostflopStrategyMatrix, String>
```

Rebuilds game from config, injects cached buffers, finalizes, returns the strategy matrix. Sets `solve_complete = true`.

**Step 5: Register commands**

Add `postflop_check_cache` and `postflop_load_cached` to Tauri and devserver.

**Step 6: Unit test**

Round-trip test: solve a spot → verify cache file written → load from cache → verify matrix matches.

**Step 7: Commit**

```
feat: load postflop solve results from cache
```

---

### Task 10: Frontend — Cache Integration in PostflopExplorer

**Files:**
- Modify: `frontend/src/PostflopExplorer.tsx`

**Context:** When the user picks a flop (or advances to turn/river), check the cache before showing the solve button. If cached, load immediately and show metadata.

**Step 1: Add cache check after board selection**

After board cards are set, call `postflop_check_cache`. If it returns metadata:
- Show "Cached (exploitability: X% pot, N iterations)" instead of the solve button
- Auto-load via `postflop_load_cached`
- Set matrix from the result

If not cached, show the solve button as before.

**Step 2: Add re-solve option**

When viewing a cached solution, show a small "Re-solve" button that ignores the cache and runs a fresh solve (which overwrites the cache on completion).

**Step 3: Pass `prior_actions` to solve/cache calls**

The PostflopExplorer needs to track the full action history across streets (not just current street). When calling cache check or solve, pass the prior street action histories so the cache key is complete.

**Step 4: Test manually**

Solve a flop → verify cache info appears on reload. Pick a different flop → verify no cache. Re-solve → verify cache updates.

**Step 5: Commit**

```
feat: cache integration in PostflopExplorer
```

---

### Task 11: Integration Test — Full Load Strategy Flow

**Files:**
- Create: `crates/tauri-app/tests/blueprint_explorer_integration.rs` (or add to existing test file)

**Context:** End-to-end test of the Load Strategy flow: list blueprints → load one → get preflop ranges → build postflop config → solve (or cache hit).

**Step 1: Write integration test**

This requires a small test blueprint (the toy config from `sample_configurations/blueprint_v2_toy.yaml`). The test:
1. Calls `list_blueprints_core` on a test directory
2. Loads a blueprint via `load_blueprint_v2_core`
3. Navigates preflop to a terminal (e.g., SB raises, BB calls)
4. Calls `get_preflop_ranges_core` with the action history
5. Verifies OOP and IP weights sum to reasonable values
6. Verifies pot/stack are derived correctly

**Step 2: Commit**

```
test: integration test for Load Strategy flow
```

---

### Task 12: Move `target_exploitability` out of `PostflopConfig`

**Files:**
- Modify: `crates/tauri-app/src/postflop.rs` (remove from `PostflopConfig`, add as separate param)
- Modify: `crates/devserver/src/main.rs` (update param structs)
- Modify: `frontend/src/PostflopExplorer.tsx` (read from global config)
- Modify: `frontend/src/types.ts` (remove from `PostflopConfig`)

**Context:** Target exploitability moves to global config. `postflop_solve_street` already accepts it as a separate parameter. We just need to remove it from `PostflopConfig` and have the frontend pass it separately from global config.

**Step 1: Remove `target_exploitability` from `PostflopConfig`**

In `postflop.rs`, remove the field from the struct and from the `Default` impl. Update `postflop_set_config_core` to not reference it.

**Step 2: Update frontend**

In `PostflopExplorer.tsx`, remove `target_exploitability` from the config state. When calling `postflop_solve_street`, read from `useGlobalConfig()` instead.

**Step 3: Remove from config modal**

The game config modal in PostflopExplorer no longer shows target exploitability.

**Step 4: Verify tests pass**

Run: `cargo test -p poker-solver-tauri`

**Step 5: Commit**

```
refactor: move target_exploitability from PostflopConfig to global config
```
